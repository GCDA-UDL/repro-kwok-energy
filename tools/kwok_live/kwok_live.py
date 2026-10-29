"""Live KWOK experiment: the energy layer over a real KWOK cluster, scraped by Prometheus.

For each cluster size N this script

1. creates N fake nodes and one placeholder pod per node in a KWOK cluster
   (``kwokctl``, binary runtime) and waits until every node is Ready and every pod
   Running; the time this takes is recorded;
2. builds the energy exporter over the nodes the API server reports as Ready,
   each replaying a measured trace chosen by the node's profile label, and serves
   ``/metrics`` (one scrape target for the whole cluster);
3. runs Prometheus against it for ``--window`` seconds while sampling the CPU and
   memory of every process involved (KWOK, kube-apiserver, etcd, the exporter,
   Prometheus) and the host's own power (RAPL package+DRAM and, if present, the
   platform meter): the cost of simulating N nodes;
4. closes the loop: for a sample of nodes, the counter increase Prometheus stored
   (``node_energy_joules_total``, raw samples, reset-safe) is compared with an
   independent offline integration of the model over the same replayed signals
   and logical time;
5. deletes the nodes and pods.

Rows go to ``<out>/kwok_live.csv`` and ``<out>/closed_loop.csv``.

    python tools/kwok_live/kwok_live.py --coeffs data/coeffs/a4000_v2.json \
        --traces data/e2e_v2/a4000 --sizes 100,1000,5000,10000 --out data/kwok_live/a4000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from kwok_energy import gen_cluster  # noqa: E402
from kwok_energy.cli.run_exporter import _load_coeffs  # noqa: E402
from kwok_energy.energy_model import FeatureWindow, predict_power  # noqa: E402
from kwok_energy.exporter import EnergyExporter, start_server  # noqa: E402
from kwok_energy.features import add_gpu_util_ewma  # noqa: E402
from kwok_energy.platform_meter import PlatformMeterReader  # noqa: E402
from kwok_energy.rapl import MultiSocketRAPL  # noqa: E402
from kwok_energy.replay import load_map  # noqa: E402
from kwok_energy.trace import read as read_trace  # noqa: E402
from kwok_energy.trace import write as write_trace  # noqa: E402
from kwok_energy.validation import segmented_counter_increase  # noqa: E402

BENCH = Path(os.environ.get("KWOK_BENCH_HOME", Path.home() / "kwok-bench"))
BIN = BENCH / "bin"
CLUSTER = "kwok-energy"
CONTEXT = f"kwok-{CLUSTER}"
PROCS = {"kwok": "kwok-controller", "apiserver": "kube-apiserver", "etcd": "etcd",
         "prometheus": "prometheus"}


def sh(*cmd: str, check: bool = True, timeout: float = 900, inp: str | None = None) -> str:
    p = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, input=inp)
    if check and p.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}... failed: {p.stderr.strip()[:500]}")
    return p.stdout


def kubectl(*args: str, **kw) -> str:
    return sh(str(BIN / "kubectl"), "--context", CONTEXT, *args, **kw)


def ensure_cluster() -> None:
    clusters = sh(str(BIN / "kwokctl"), "get", "clusters", check=False)
    if CLUSTER not in clusters.split():
        sh(str(BIN / "kwokctl"), "create", "cluster", "--name", CLUSTER, "--runtime", "binary",
           timeout=1800)
    kubectl("get", "ns", "default")


def ready_counts() -> tuple[int, int]:
    nodes = json.loads(kubectl("get", "nodes", "-l", "kwok.x-k8s.io/node=fake", "-o", "json"))
    n_ready = sum(1 for n in nodes["items"] for c in n["status"].get("conditions", [])
                  if c["type"] == "Ready" and c["status"] == "True")
    pods = json.loads(kubectl("get", "pods", "-n", "default", "-o", "json"))
    p_run = sum(1 for p in pods["items"] if p["status"].get("phase") == "Running")
    return n_ready, p_run


def apply_chunked(docs: list[dict], chunk: int = 500) -> None:
    import yaml
    for i in range(0, len(docs), chunk):
        kubectl("apply", "-f", "-", inp=yaml.safe_dump_all(docs[i:i + chunk]), timeout=1800)


def cleanup() -> None:
    kubectl("delete", "pods", "-n", "default", "-l", "kwok-energy/node", "--wait=false",
            check=False, timeout=1800)
    kubectl("delete", "nodes", "-l", "kwok.x-k8s.io/node=fake", "--wait=false", check=False,
            timeout=1800)
    t = time.time()
    while time.time() - t < 900:
        n, p = ready_counts()
        if n == 0 and p == 0:
            return
        time.sleep(5)


# --------------------------------------------------------------------------- #
#  process + host power sampling
# --------------------------------------------------------------------------- #
def _pids(name: str) -> list[int]:
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                if (d / "comm").read_text().strip().startswith(name[:15]):
                    out.append(int(d.name))
            except OSError:
                pass
    return out


def _proc_stat(pid: int) -> tuple[float, float]:
    """(cpu seconds, rss MB) of one process."""
    f = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    ticks = os.sysconf("SC_CLK_TCK")
    cpu = (int(f[11]) + int(f[12])) / ticks
    rss = int(f[21]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    return cpu, rss


class CostSampler(threading.Thread):
    def __init__(self, interval: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_ev = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.rapl = MultiSocketRAPL()
        self.meter = PlatformMeterReader()

    def run(self) -> None:
        def snap():
            pkg, dram = self.rapl.read()
            s: dict[str, Any] = {"t": time.monotonic(), "pkg_uj": pkg, "dram_uj": dram}
            for k, name in PROCS.items():
                cpu = rss = 0.0
                for pid in _pids(name):
                    try:
                        c, r = _proc_stat(pid)
                        cpu, rss = cpu + c, rss + r
                    except OSError:
                        pass
                s[f"{k}_cpu_s"], s[f"{k}_rss_mb"] = cpu, rss
            c, r = _proc_stat(os.getpid())
            s["exporter_cpu_s"], s["exporter_rss_mb"] = c, r
            s["meter_w"] = self.meter.read_power_w() if self.meter.available else None
            return s
        self.samples.append(snap())
        while not self.stop_ev.wait(self.interval):
            self.samples.append(snap())
        self.samples.append(snap())

    def summary(self) -> dict[str, Any]:
        a, b = self.samples[0], self.samples[-1]
        dt = b["t"] - a["t"]
        out: dict[str, Any] = {"window_s": dt}
        for k in [*PROCS, "exporter"]:
            out[f"{k}_cpu_pct"] = 100 * (b[f"{k}_cpu_s"] - a[f"{k}_cpu_s"]) / dt
            out[f"{k}_rss_mb_max"] = max(s[f"{k}_rss_mb"] for s in self.samples)
        # Counters can wrap more than once over a long window: sum per-sample deltas.
        for key, zones in (("pkg", self.rapl.package_zones), ("dram", self.rapl.dram_zones)):
            js = [MultiSocketRAPL.joules(p[f"{key}_uj"], c[f"{key}_uj"], zones)
                  for p, c in zip(self.samples, self.samples[1:], strict=False)]
            out[f"host_{key}_w"] = (sum(js) / dt) if js and all(j is not None for j in js) else None
        m = [s["meter_w"] for s in self.samples if s["meter_w"] is not None]
        out["host_meter_w"] = sum(m) / len(m) if m else None
        return out


# --------------------------------------------------------------------------- #
#  Prometheus
# --------------------------------------------------------------------------- #
def start_prometheus(workdir: Path, port: int, exporter_port: int, scrape_s: int) -> subprocess.Popen:
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    (workdir / "prometheus.yml").write_text(
        f"global:\n  scrape_interval: {scrape_s}s\n  scrape_timeout: {max(1, scrape_s - 1)}s\n"
        "scrape_configs:\n  - job_name: kwok-energy\n"
        f"    static_configs:\n      - targets: ['127.0.0.1:{exporter_port}']\n")
    return subprocess.Popen(
        [str(BIN / "prometheus"), f"--config.file={workdir / 'prometheus.yml'}",
         f"--storage.tsdb.path={workdir / 'tsdb'}", f"--web.listen-address=127.0.0.1:{port}",
         "--log.level=warn"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prom(port: int, path: str, **params: str) -> Any:
    q = urllib.parse.urlencode(params)
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/{path}?{q}", timeout=120) as r:
        return json.loads(r.read())["data"]


def prom_scalar(port: int, expr: str) -> float | None:
    res = prom(port, "query", query=expr)["result"]
    return float(res[0]["value"][1]) if res else None


# --------------------------------------------------------------------------- #
#  one cluster size
# --------------------------------------------------------------------------- #
def run_size(n: int, args, traces: list[Path], platform_id: str) -> tuple[dict, list[dict]]:
    out = Path(args.out)
    cfg = {
        "node_count": n,
        "profiles": [{"name": t.stem, "count": c, "platform_id": platform_id,
                      "trace_path": str(t.resolve())}
                     for t, c in zip(traces, _split(n, len(traces)), strict=True) if c > 0],
        "exporter_target": f"127.0.0.1:{args.exporter_port}",
    }
    art = gen_cluster.generate(cfg)
    paths = gen_cluster.write_artifacts(art, out / f"artifacts_{n}")

    t0 = time.monotonic()
    apply_chunked(art.kwok_fake_nodes)
    apply_chunked(art.kwok_fake_pods)
    while True:
        nr, pr = ready_counts()
        if nr >= n and pr >= n:
            break
        if time.monotonic() - t0 > args.ready_timeout:
            raise TimeoutError(f"N={n}: only {nr} nodes Ready / {pr} pods Running")
        time.sleep(5)
    t_ready = time.monotonic() - t0

    # Exporter over the nodes the API server reports Ready (all of them, by now).
    source = load_map(paths.node_profile_map)
    exporter = EnergyExporter(source=source, coeffs_map=_load_coeffs(args.coeffs))
    server = start_server(exporter, port=args.exporter_port)
    tick_cost: list[float] = []
    tick_done_wall: list[float] = []  # wall-clock completion time of tick k (logical k*tick)
    stop = threading.Event()
    logical = {"t": 0.0}

    def ticker() -> None:
        while not stop.is_set():
            s = time.perf_counter()
            exporter.tick(logical["t"])
            tick_cost.append(time.perf_counter() - s)
            tick_done_wall.append(time.time())
            logical["t"] += args.tick
            stop.wait(max(0.0, args.tick - (time.perf_counter() - s)))
    th = threading.Thread(target=ticker, daemon=True)
    th.start()

    promp = start_prometheus(out / f"prom_{n}", args.prom_port, args.exporter_port, args.scrape)
    cost = CostSampler()
    cost.start()
    time.sleep(args.window)
    cost.stop_ev.set()
    cost.join()
    stop.set()
    th.join(timeout=30)
    n_ticks = len(tick_done_wall)

    row: dict[str, Any] = {"platform_id": platform_id, "nodes": n, "pods": n,
                           "time_to_ready_s": t_ready, "tick_s": args.tick,
                           "scrape_s": args.scrape,
                           "tick_cost_mean_s": sum(tick_cost) / len(tick_cost),
                           "tick_cost_max_s": max(tick_cost),
                           "ticks_late": sum(1 for c in tick_cost if c > args.tick)}
    row.update(cost.summary())
    row["series"] = prom_scalar(args.prom_port, "prometheus_tsdb_head_series")
    row["energy_series"] = prom_scalar(args.prom_port, "count(node_energy_joules_total)")
    row["scrape_duration_mean_s"] = prom_scalar(
        args.prom_port, f"avg_over_time(scrape_duration_seconds{{job='kwok-energy'}}[{int(args.window)}s])")
    row["scrape_duration_max_s"] = prom_scalar(
        args.prom_port, f"max_over_time(scrape_duration_seconds{{job='kwok-energy'}}[{int(args.window)}s])")
    row["scrape_up_ratio"] = prom_scalar(
        args.prom_port, f"avg_over_time(up{{job='kwok-energy'}}[{int(args.window)}s])")
    row["samples_per_scrape"] = prom_scalar(
        args.prom_port, "max_over_time(scrape_samples_scraped{job='kwok-energy'}[5m])")

    # Closed loop on a node sample. Prometheus stores raw counter samples stamped
    # with the scrape time; a sample taken at wall time ts holds the ticks completed
    # before ts. The offline side integrates the model independently (not through
    # the exporter) over the same replayed signals for exactly those ticks.
    import bisect
    coeffs = next(iter(_load_coeffs(args.coeffs).values()))
    loop_rows = []
    rng = random.Random(n)
    for node in rng.sample(source.nodes(), min(args.loop_nodes, n)):
        res = prom(args.prom_port, "query",
                   query=f"node_energy_joules_total{{node='{node}'}}[{int(args.window) + 60}s]")["result"]
        samples = [(float(t), float(v)) for t, v in res[0]["values"]] if res else []
        prom_inc = segmented_counter_increase(samples)
        cum = [0.0]  # cum[k] = energy after ticks 0..k-1 (tick 0 only initialises)
        for k in range(1, n_ticks):
            p = predict_power(FeatureWindow(current=source.signals_at(node, k * args.tick)),
                              coeffs).total
            cum.append(cum[-1] + p * args.tick)
        by_done = [0.0, *cum]  # by_done[m] = counter after m completed ticks

        def at(ts: float, table: list[float] = by_done) -> float:
            return table[min(len(table) - 1, bisect.bisect_right(tick_done_wall, ts))]
        off_inc = (at(samples[-1][0]) - at(samples[0][0])) if samples else 0.0
        loop_rows.append({"nodes": n, "node": node,
                          "profile": art.node_profile_map["nodes"][node]["profile"],
                          "prom_samples": len(samples), "prometheus_increase_j": prom_inc,
                          "offline_increase_j": off_inc,
                          "rel_diff": abs(prom_inc - off_inc) / off_inc if off_inc else None})
    row["closed_loop_max_rel_diff"] = max((r["rel_diff"] or 0.0) for r in loop_rows) if loop_rows else None

    promp.terminate()
    promp.wait(timeout=60)
    server.shutdown()
    cleanup()
    return row, loop_rows


def retau(traces: list[Path], session: Path, tau_s: float, outdir: Path) -> list[Path]:
    """Copies of the replay traces with the GPU history recomputed at the fitted tau."""
    by_seq = {int(r["seq_index"]): r["gpu_util_ewma"]
              for r in add_gpu_util_ewma(read_trace(session), tau_s)}
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for t in traces:
        rows = read_trace(t)
        for r in rows:
            r["gpu_util_ewma"] = by_seq[int(r["seq_index"])]
        write_trace(rows, outdir / t.name)
        out.append(outdir / t.name)
    return out


def _split(n: int, k: int) -> list[int]:
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--traces", required=True, help="dir with heldout_*_r1.csv traces to replay")
    ap.add_argument("--sizes", default="100,1000,5000,10000")
    ap.add_argument("--window", type=float, default=300.0)
    ap.add_argument("--tick", type=float, default=1.0)
    ap.add_argument("--scrape", type=int, default=15)
    ap.add_argument("--loop-nodes", type=int, default=20)
    ap.add_argument("--ready-timeout", type=float, default=3600.0)
    ap.add_argument("--exporter-port", type=int, default=19100)
    ap.add_argument("--prom-port", type=int, default=19090)
    ap.add_argument("--out", default="data/kwok_live")
    args = ap.parse_args()

    traces = sorted(Path(args.traces).glob("heldout_*_r1.csv"))
    if not traces:
        raise SystemExit("no heldout_*_r1.csv traces to replay")
    coeffs_json = json.loads(Path(args.coeffs).read_text())
    platform_id = coeffs_json["platform_id"]
    tau = coeffs_json["transform"].get("gpu_util_ewma_tau_s")
    if tau is not None:
        traces = retau(traces, Path(args.traces) / "session.csv", float(tau),
                       Path(args.out) / "replay")
    ensure_cluster()
    cleanup()
    rows, loops = [], []
    for n in [int(x) for x in args.sizes.split(",")]:
        print(f"== N={n} ==", file=sys.stderr, flush=True)
        r, lr = run_size(n, args, traces, platform_id)
        rows.append(r)
        loops += lr
        write_csv(rows, Path(args.out) / "kwok_live.csv")
        write_csv(loops, Path(args.out) / "closed_loop.csv")
        print(json.dumps(r, default=str), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
