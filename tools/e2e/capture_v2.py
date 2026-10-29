"""Measurement protocol v2: one continuous, randomised capture session per host.

What changes with respect to ``capture_traces.py`` (v1, the June 2026 dataset):

* **Continuous timeline.** A sampler thread records every interval of the session,
  including the set-up, ramp and gap periods between workloads (labelled
  ``_setup:<w>``, ``_ramp:<w>``, ``_teardown:<w>``, ``_gap``). The GPU-utilisation
  history feature (``gpu_util_ewma``) is computed online over this timeline, so the
  model sees the power-state tail a previous workload leaves behind.
* **Randomised order.** Each repetition runs the workloads in a seeded random
  order, so no workload always follows the same predecessor (in v1 the CPU-only
  held-out workload always followed a GPU workload).
* **Long windows.** Default 120 s measured per workload (v1: 20 s).
* **Whole node.** Every GPU is measured (v1: GPU 0 only) and, where the host
  exposes one, the platform power meter gives whole-node input power.
* **Real applications** as held-out workloads (ResNet-50 training, LLM serving,
  PostgreSQL/pgbench, a CPython build, training co-located with pgbench) next to
  the synthetic calibration sweeps. See ``tools/e2e/real/``.
* Disk and network rates are measured (v1 left them at 0).

Outputs (``--outdir``/<platform>/):
  cal_<w>_r<k>.csv, heldout_<w>_r<k>.csv   measured windows (trace schema v1.1)
  session.csv                              every interval of the session
  session_manifest.json                    host, versions, seed, order, timings,
                                           background-load snapshots

    python tools/e2e/capture_v2.py --platform a4000 --reps 5 --duration 120
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform as pyplatform
import random
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kwok_energy.collect import collect_snapshot, compute_interval  # noqa: E402
from kwok_energy.features import DEFAULT_TAU_S, GPUUtilEWMA  # noqa: E402
from kwok_energy.nvml import NVMLMultiReader  # noqa: E402
from kwok_energy.platform_meter import PlatformMeterReader  # noqa: E402
from kwok_energy.rapl import MultiSocketRAPL, RAPLReader  # noqa: E402
from kwok_energy.trace import write as write_trace  # noqa: E402

try:
    from mem_bandwidth import ContinuousIMCSampler, MemBandwidthSampler
except Exception:  # pragma: no cover - perf missing
    ContinuousIMCSampler = MemBandwidthSampler = None  # type: ignore[assignment,misc]

SCHEMA_VERSION = "1.1"
NODE = "bench-0"
REAL_DIR = Path(__file__).resolve().parent / "real"
BENCH_HOME = Path(os.environ.get("KWOK_BENCH_HOME", Path.home() / "kwok-bench"))

_DISK_RE = re.compile(r"^(nvme\d+n\d+|sd[a-z]+|vd[a-z]+|xvd[a-z]+)$")
_NET_SKIP = ("lo", "docker", "veth", "br-", "virbr", "cni", "flannel")


# --------------------------------------------------------------------------- #
#  host counters not covered by the v1 collector
# --------------------------------------------------------------------------- #
def read_disk_bytes() -> tuple[int, int]:
    rd = wr = 0
    try:
        for line in Path("/proc/diskstats").read_text().splitlines():
            p = line.split()
            if len(p) > 9 and _DISK_RE.match(p[2]):
                rd += int(p[5]) * 512
                wr += int(p[9]) * 512
    except OSError:
        pass
    return rd, wr


def read_net_bytes() -> tuple[int, int]:
    rx = tx = 0
    try:
        for line in Path("/proc/net/dev").read_text().splitlines()[2:]:
            name, data = line.split(":", 1)
            name = name.strip()
            if name.startswith(_NET_SKIP):
                continue
            f = data.split()
            rx += int(f[0])
            tx += int(f[8])
    except OSError:
        pass
    return rx, tx


def read_cpu_temp_c() -> float | None:
    best = None
    for d in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (d / "name").read_text().strip() != "coretemp":
                continue
            for t in d.glob("temp*_input"):
                v = float(t.read_text()) / 1000.0
                best = v if best is None else max(best, v)
        except (OSError, ValueError):
            continue
    return best


# --------------------------------------------------------------------------- #
#  sampler thread: the continuous session timeline
# --------------------------------------------------------------------------- #
class Sampler(threading.Thread):
    """Samples the node every ``interval`` s and labels each interval with the
    phase the main thread has set (``set_phase``)."""

    def __init__(self, platform_id: str, interval: float, tau_s: float,
                 membw: str = "continuous") -> None:
        super().__init__(daemon=True)
        self.platform_id = platform_id
        self.interval = interval
        self.rapl = RAPLReader()
        self.rapl_all = MultiSocketRAPL()  # every socket (v1 read socket 0 only)
        self.gpu = NVMLMultiReader()
        self.meter = PlatformMeterReader()
        self.bw: Any = None
        if membw == "continuous" and ContinuousIMCSampler is not None:
            bw = ContinuousIMCSampler(int(interval * 1000))
            self.bw = bw if bw.method != "unavailable" else None
        if self.bw is None and MemBandwidthSampler is not None:
            self.bw = MemBandwidthSampler()  # one perf process per interval
            self.bw.sample(0.2)  # resolve the perf method (imc / llc / none) up front
        self.ewma = GPUUtilEWMA(tau_s)
        self.rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._phase: tuple[str, str, int] = ("_gap", "_gap", 0)  # (label, role, rep)
        self._halt = threading.Event()  # not _stop: that name is a Thread method
        self.t0 = time.monotonic()
        self.error: BaseException | None = None

    def set_phase(self, label: str, role: str = "_gap", rep: int = 0) -> None:
        with self._lock:
            self._phase = (label, role, rep)

    def stop(self) -> None:
        self._halt.set()

    def _snap(self) -> dict[str, Any]:
        s = collect_snapshot(self.rapl, nvml_reader=self.gpu, platform_id=self.platform_id,
                             node=NODE)
        s["_disk_b"] = read_disk_bytes()
        s["_net_b"] = read_net_bytes()
        s["_pkg_all"], s["_dram_all"] = self.rapl_all.read()
        return s

    def run(self) -> None:  # pragma: no cover - needs the host
        try:
            prev = self._snap()
            prev_t = time.monotonic()
            seq = 0
            while not self._halt.is_set():
                with self._lock:
                    label, role, rep = self._phase
                if self.bw is not None:
                    proxy, method = self.bw.sample(self.interval)
                else:
                    time.sleep(self.interval)
                    proxy, method = 0.0, "unavailable"
                now = time.monotonic()
                cur = self._snap()
                cur["mem_bandwidth_proxy"] = proxy
                cur["mem_proxy_method"] = method
                dt = now - prev_t
                row = compute_interval(prev, cur, dt)
                row["schema_version"] = SCHEMA_VERSION
                row["workload"] = label
                row["run_id"] = f"r{rep}"
                row["session_t_s"] = now - self.t0
                row["t_end_s"] = now - self.t0
                row["t_start_s"] = prev_t - self.t0
                row["seq_index"] = seq
                (r0, w0), (r1, w1) = prev["_disk_b"], cur["_disk_b"]
                row["disk_read_bps"] = max(0, r1 - r0) / dt
                row["disk_write_bps"] = max(0, w1 - w0) / dt
                (x0, t0), (x1, t1) = prev["_net_b"], cur["_net_b"]
                row["net_rx_bps"] = max(0, x1 - x0) / dt
                row["net_tx_bps"] = max(0, t1 - t0) / dt
                # Package and DRAM over all sockets (overrides the socket-0 collector).
                for key, zones, raw in (("pkg", self.rapl_all.package_zones, "_pkg_all"),
                                        ("dram", self.rapl_all.dram_zones, "_dram_all")):
                    j = MultiSocketRAPL.joules(prev[raw], cur[raw], zones)
                    row[f"rapl_{key}_joules"] = j
                    row[f"rapl_{key}_watts_mean"] = None if j is None else j / dt
                    row[f"rapl_{key}_available"] = j is not None
                row["gpu_util_ewma"] = self.ewma.update(row["gpu_utilization"], dt)
                pw = self.meter.read_power_w() if self.meter.available else None
                row["platform_meter_available"] = pw is not None
                row["platform_meter_watts_mean"] = pw
                row["platform_meter_joules"] = None if pw is None else pw * dt
                diag = self.gpu.read_diagnostics() if self.gpu.available else {}
                row.update({k: diag.get(k) for k in
                            ("gpu_count", "gpu_sm_clock_mhz", "gpu_pstate", "gpu_temp_c")})
                row["cpu_temp_c"] = read_cpu_temp_c()
                row["_role"] = role
                with self._lock:
                    self.rows.append(row)
                prev, prev_t = cur, now
                seq += 1
        except BaseException as e:  # surface sampler crashes to the main thread
            self.error = e


# --------------------------------------------------------------------------- #
#  synthetic load kernels (spawned processes; the parent never touches CUDA)
# --------------------------------------------------------------------------- #
def _cpu_burn(stop_at: float, ready) -> None:
    ready.set()
    x = 1.0001
    while time.time() < stop_at:
        for _ in range(100000):
            x = (x * 1.0000001 + 1.0) ** 0.5


def _cpu_burst(stop_at: float, ready, period: float = 10.0, duty: float = 0.5) -> None:
    ready.set()
    x = 1.0001
    while time.time() < stop_at:
        if (time.time() % period) < period * duty:
            for _ in range(20000):
                x = (x * 1.0000001 + 1.0) ** 0.5
        else:
            time.sleep(0.05)


def _mem_burn(stop_at: float, ready) -> None:
    import numpy as np
    a = np.ones(32_000_000, dtype=np.float64)
    b = np.ones(32_000_000, dtype=np.float64)
    ready.set()
    while time.time() < stop_at:
        a = a + b * 1.0001


def _gpu_burn(stop_at: float, ready, frac: float = 1.0, device: int = 0) -> None:
    import torch
    if not torch.cuda.is_available():
        ready.set()
        return
    dev = torch.device(f"cuda:{device}")
    a = torch.randn(8192, 8192, device=dev)
    b = torch.randn(8192, 8192, device=dev)
    torch.cuda.synchronize(dev)
    ready.set()
    while time.time() < stop_at:
        t = time.time()
        a = (a @ b).clone()
        torch.cuda.synchronize(dev)
        busy = time.time() - t
        if frac < 1.0 and busy > 0:
            time.sleep(busy * (1.0 / frac - 1.0))


# --------------------------------------------------------------------------- #
#  workload definitions
# --------------------------------------------------------------------------- #
@dataclass
class Workload:
    name: str
    role: str  # "cal" | "heldout"
    start: Callable[[float], list]  # (stop_at) -> handles
    setup: Callable[[], None] = lambda: None
    teardown: Callable[[], None] = lambda: None
    ready_marker: str | None = None  # wait for this line on a handle's stdout
    kind: str = "synthetic"
    notes: str = ""
    extra: dict = field(default_factory=dict)


_CTX = mp.get_context("spawn")


def _procs(target, n: int, stop_at: float, *args) -> list:
    ps = []
    for _ in range(n):
        ev = _CTX.Event()
        p = _CTX.Process(target=target, args=(stop_at, ev, *args), daemon=True)
        p.ready = ev  # type: ignore[attr-defined]
        p.start()
        ps.append(p)
    return ps


def _wait_procs_ready(handles: list, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    for h in handles:
        ev = getattr(h, "ready", None)
        if ev is not None and not ev.wait(max(0.1, deadline - time.time())):
            raise TimeoutError("a load process never became ready")


def _popen(cmd: list[str], **kw) -> subprocess.Popen:
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True, **kw)


def _docker(*args: str, check: bool = True, timeout: float = 300) -> str:
    p = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout


def _wait_for(pred: Callable[[], bool], timeout: float, what: str) -> None:
    t = time.time()
    while time.time() - t < timeout:
        if pred():
            return
        time.sleep(1.0)
    raise TimeoutError(f"timed out waiting for {what}")


PG = "kwokbench-pg"
OLLAMA = "kwokbench-ollama"
LLM_MODEL = os.environ.get("KWOK_BENCH_LLM", "qwen2.5:3b")
# Recent Ollama builds need NVIDIA driver >= 550 for compute 7.5 GPUs and silently
# fall back to the CPU otherwise; pin an older image on such hosts (0.6.0 works
# with driver 535). The image used is recorded in the session manifest.
OLLAMA_IMAGE = os.environ.get("KWOK_BENCH_OLLAMA_IMAGE", "ollama/ollama:latest")


def _pg_up() -> None:
    _docker("rm", "-f", PG, check=False)
    _docker("run", "-d", "--rm", "--name", PG, "--shm-size=1g", "-e", "POSTGRES_PASSWORD=bench",
            "-v", "kwokbench-pg:/var/lib/postgresql/data", "postgres:16")
    _wait_for(lambda: subprocess.run(["docker", "exec", PG, "pg_isready", "-U", "postgres"],
                                     capture_output=True).returncode == 0, 120, "postgres")
    time.sleep(3.0)


def _pg_down() -> None:
    _docker("stop", "-t", "30", PG, check=False, timeout=120)


MEASURE_S = 120.0  # set from --duration in main()


def _pgbench(stop_at: float) -> subprocess.Popen:
    secs = int(MEASURE_S) + 20  # covers ramp + window; the container stop ends it
    return _popen(["docker", "exec", PG, "pgbench", "-U", "postgres", "-n", "-c", "32", "-j", "8",
                   "-T", str(secs), "-P", "30", "bench"])


def _ollama_up() -> None:
    _docker("rm", "-f", OLLAMA, check=False)
    _docker("run", "-d", "--rm", "--name", OLLAMA, "--gpus", "device=0", "--dns", "8.8.8.8",
            "-v", "kwokbench-ollama:/root/.ollama", "-p", "127.0.0.1:11434:11434",
            OLLAMA_IMAGE)
    import urllib.request

    def _ok() -> bool:
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False
    _wait_for(_ok, 120, "ollama")
    # Load the model onto the GPU before the measured window (one short request).
    subprocess.run([sys.executable, str(REAL_DIR / "llm_client.py"), "--warmup", "--model",
                    LLM_MODEL], check=True, timeout=600)


def _ollama_down() -> None:
    _docker("stop", "-t", "20", OLLAMA, check=False, timeout=120)


def _py(script: str, *args: str) -> subprocess.Popen:
    return _popen([sys.executable, "-u", str(REAL_DIR / script), *args])


def _cpython_dir() -> Path:
    return BENCH_HOME / "cpython"


def _build_loop(stop_at: float) -> subprocess.Popen:
    nproc = os.cpu_count() or 8
    script = (f"cd {_cpython_dir()} && while true; do make -j{nproc} -s >/dev/null 2>&1; "
              "make clean >/dev/null 2>&1; done")
    return _popen(["bash", "-c", script])


def _build_clean() -> None:
    subprocess.run(["make", "-s", "clean"], cwd=_cpython_dir(), capture_output=True, timeout=300)


def workloads(ncpu: int, ngpu: int, include_real: bool) -> list[Workload]:
    q = lambda f: max(1, int(round(ncpu * f)))  # noqa: E731
    W = [
        Workload("idle", "cal", lambda s: []),
        Workload("cpu_12", "cal", lambda s: _procs(_cpu_burn, q(0.125), s)),
        Workload("cpu_25", "cal", lambda s: _procs(_cpu_burn, q(0.25), s)),
        Workload("cpu_50", "cal", lambda s: _procs(_cpu_burn, q(0.5), s)),
        Workload("cpu_75", "cal", lambda s: _procs(_cpu_burn, q(0.75), s)),
        Workload("cpu_100", "cal", lambda s: _procs(_cpu_burn, ncpu, s)),
        Workload("mem_lo", "cal", lambda s: _procs(_mem_burn, 2, s)),
        Workload("mem_hi", "cal", lambda s: _procs(_mem_burn, max(4, ncpu // 4), s)),
        Workload("gpu_25", "cal", lambda s: _procs(_gpu_burn, 1, s, 0.25, 0)),
        Workload("gpu_50", "cal", lambda s: _procs(_gpu_burn, 1, s, 0.50, 0)),
        Workload("gpu_100", "cal", lambda s: _procs(_gpu_burn, 1, s, 1.0, 0)),
        Workload("gpu_all_100", "cal",
                 lambda s: sum((_procs(_gpu_burn, 1, s, 1.0, i) for i in range(ngpu)), [])),
        # synthetic held-out mixes (never seen at fit time)
        Workload("mixed_cpu_gpu", "heldout",
                 lambda s: _procs(_cpu_burn, q(0.5), s) + _procs(_gpu_burn, 1, s, 1.0, 0)),
        Workload("bursty_cpu", "heldout", lambda s: _procs(_cpu_burst, ncpu, s),
                 notes="all cores, 5 s on / 5 s off"),
        Workload("mem_gpu", "heldout",
                 lambda s: _procs(_mem_burn, max(4, ncpu // 4), s) + _procs(_gpu_burn, 1, s, 1.0, 0)),
    ]
    if include_real:
        W += [
            Workload("train_resnet50", "heldout",
                     lambda s: [_py("train_resnet.py", "--device", "cuda:0", "--workers", "8")],
                     ready_marker="READY", kind="real",
                     notes="PyTorch ResNet-50 training (torch only), synthetic ImageNet-size data with CPU augmentation, AMP, 8 loader workers"),
            Workload("llm_serve", "heldout",
                     lambda s: [_py("llm_client.py", "--model", LLM_MODEL, "--rate", "0.5")],
                     setup=_ollama_up, teardown=_ollama_down, ready_marker="READY", kind="real",
                     notes=f"Ollama {LLM_MODEL} on GPU 0, Poisson requests 0.5/s, 128 tokens"),
            Workload("pgbench", "heldout", lambda s: [_pgbench(s)],
                     setup=_pg_up, teardown=_pg_down, kind="real",
                     notes="PostgreSQL 16, pgbench TPC-B-like, scale 100, 32 clients, 8 threads"),
            Workload("build_cpython", "heldout", lambda s: [_build_loop(s)],
                     teardown=_build_clean, kind="real",
                     notes="CPython 3.12 parallel build loop (make -j nproc; make clean)"),
            Workload("colocated_train_pgbench", "heldout",
                     lambda s: [_py("train_resnet.py", "--device", "cuda:0", "--workers", "8"),
                                _pgbench(s)],
                     setup=_pg_up, teardown=_pg_down, ready_marker="READY", kind="real",
                     notes="train_resnet50 and pgbench on the same node"),
        ]
    return W


# --------------------------------------------------------------------------- #
#  running one workload
# --------------------------------------------------------------------------- #
def _wait_ready(handles: list, marker: str, timeout: float = 300.0) -> None:
    popens = [h for h in handles if isinstance(h, subprocess.Popen)]
    if not popens:
        return
    p = popens[0]
    _drain(popens[1:])
    seen = threading.Event()

    def _reader() -> None:
        assert p.stdout is not None
        for line in p.stdout:
            if marker in line:
                seen.set()
        # keep draining so the child never blocks on a full pipe
    threading.Thread(target=_reader, daemon=True).start()
    if not seen.wait(timeout):
        raise TimeoutError(f"workload never printed {marker!r}")


def _drain(handles: list) -> None:
    for h in handles:
        if isinstance(h, subprocess.Popen) and h.stdout is not None:
            threading.Thread(target=lambda s=h.stdout: [None for _ in s], daemon=True).start()


def _stop(handles: list) -> None:
    for h in handles:
        if isinstance(h, subprocess.Popen):
            if h.poll() is None:
                try:
                    os.killpg(h.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        else:
            if h.is_alive():
                h.terminate()
    deadline = time.time() + 15
    for h in handles:
        if isinstance(h, subprocess.Popen):
            try:
                h.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(h.pid, signal.SIGKILL)
                h.wait(timeout=10)
        else:
            h.join(timeout=max(0.1, deadline - time.time()))
            if h.is_alive():
                h.kill()
                h.join(timeout=5)


def run_workload(w: Workload, rep: int, sampler: Sampler, duration: float, ramp: float) -> dict:
    t = {"workload": w.name, "role": w.role, "rep": rep, "kind": w.kind}
    sampler.set_phase(f"_setup:{w.name}")
    t["setup_start"] = time.monotonic() - sampler.t0
    w.setup()
    sampler.set_phase(f"_ramp:{w.name}")
    stop_at = time.time() + duration + ramp + 300  # loads are stopped explicitly
    handles = w.start(stop_at)
    _wait_procs_ready(handles)
    if w.ready_marker:
        _wait_ready(handles, w.ready_marker)
    else:
        _drain(handles)
    time.sleep(ramp)
    sampler.set_phase(w.name, w.role, rep)
    t["measure_start"] = time.monotonic() - sampler.t0
    time.sleep(duration)
    t["measure_end"] = time.monotonic() - sampler.t0
    sampler.set_phase(f"_teardown:{w.name}")
    _stop(handles)
    w.teardown()
    sampler.set_phase("_gap")
    t["teardown_end"] = time.monotonic() - sampler.t0
    return t


# --------------------------------------------------------------------------- #
#  provenance
# --------------------------------------------------------------------------- #
def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # pragma: no cover
        return f"<{e}>"


def background_snapshot() -> dict:
    return {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "loadavg": _sh("cat /proc/loadavg").strip(),
        "top_processes": _sh("ps -eo user,pid,pcpu,pmem,etime,comm --sort=-pcpu | head -15"),
        "docker_ps": _sh("docker ps --format '{{.Names}} {{.Image}} {{.Status}}'"),
        "gpu_apps": _sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                        "--format=csv,noheader"),
        "logged_in": _sh("who"),
    }


def host_info(sampler: Sampler) -> dict:
    return {
        "hostname": pyplatform.node(),
        "kernel": pyplatform.release(),
        "cpu_model": _sh("lscpu | grep 'Model name' | head -1").split(":", 1)[-1].strip(),
        "ncpu": os.cpu_count(),
        "mem": _sh("free -g | head -2"),
        "gpus": _sh("nvidia-smi --query-gpu=index,name,driver_version,power.limit "
                    "--format=csv,noheader"),
        "rapl_source": sampler.rapl.source,
        "rapl_sockets": sampler.rapl_all.sockets,
        "rapl_dram_zones": len(sampler.rapl_all.dram_zones),
        "rapl_dram": sampler.rapl.dram_available,
        "gpu_count": sampler.gpu.device_count,
        "gpu_energy_method": sampler.gpu.energy_method,
        "platform_meter": {
            "available": sampler.meter.available,
            "model": sampler.meter.model,
            "accuracy": sampler.meter.accuracy,
            "average_interval_ms": sampler.meter.average_interval_ms,
        },
        "mem_proxy": sampler.bw.method if sampler.bw else "unavailable",
        "ollama_image": OLLAMA_IMAGE,
        "ollama_image_id": _sh(f"docker image inspect {OLLAMA_IMAGE} --format '{{{{.Id}}}}'").strip(),
        "ollama_version": _sh(f"docker run --rm --entrypoint /bin/ollama {OLLAMA_IMAGE} --version"
                              " 2>&1 | tail -1").strip(),
        "python": sys.version.split()[0],
        "torch": _sh(f"{sys.executable} -c 'import torch;print(torch.__version__)'").strip(),
        "git_sha": _sh(f"git -C {REPO} rev-parse HEAD").strip(),
        "git_dirty": bool(_sh(f"git -C {REPO} status --porcelain").strip()),
    }


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def _fmt(s: float) -> str:
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


def main() -> int:  # pragma: no cover - needs the host
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--platform", required=True)
    ap.add_argument("--outdir", default="data/e2e_v2")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--ramp", type=float, default=2.0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU_S)
    ap.add_argument("--seed", type=int, default=20261026)
    ap.add_argument("--only", default="", help="comma list of workload names")
    ap.add_argument("--no-real", action="store_true", help="synthetic workloads only")
    ap.add_argument("--settle", type=float, default=60.0, help="idle seconds before the first workload")
    ap.add_argument("--membw", choices=("continuous", "per-interval"), default="continuous",
                    help="memory-traffic proxy: one long-lived perf -I (default) or one perf per interval")
    args = ap.parse_args()
    global MEASURE_S
    MEASURE_S = args.duration

    out = Path(args.outdir) / args.platform
    if out.exists() and any(out.glob("*.csv")):
        print(f"refusing to overwrite existing traces in {out}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    sampler = Sampler(args.platform, args.interval, args.tau, args.membw)
    if sampler.rapl_all.sockets == 0 or sampler.rapl_all.read()[0].count(None):
        print("RAPL energy_uj is not readable: grant read access first", file=sys.stderr)
        return 2
    if not sampler.rapl_all.wrap_known:
        print("RAPL max_energy_range_uj is not readable: counter wraps (~every 20-30 min under "
              "load) cannot be corrected; grant read access first", file=sys.stderr)
        return 2
    ngpu = max(1, sampler.gpu.device_count)
    W = workloads(os.cpu_count() or 8, ngpu, include_real=not args.no_real)
    if args.only:
        keep = set(args.only.split(","))
        W = [w for w in W if w.name in keep]
    rng = random.Random(args.seed)
    plan = []
    for rep in range(1, args.reps + 1):
        order = list(W)
        rng.shuffle(order)
        plan += [(rep, w) for w in order]

    manifest: dict[str, Any] = {
        "protocol": "capture_v2", "schema_version": SCHEMA_VERSION, "platform_id": args.platform,
        "args": vars(args), "workloads": [{"name": w.name, "role": w.role, "kind": w.kind,
                                           "notes": w.notes} for w in W],
        "host": host_info(sampler), "background_start": background_snapshot(), "timings": [],
    }
    est = len(plan) * (args.duration + args.ramp + 8) + args.settle
    print(f"== capture_v2 platform={args.platform} jobs={len(plan)} ~{_fmt(est)} | "
          f"GPUs={ngpu} meter={sampler.meter.available} membw={manifest['host']['mem_proxy']} ==",
          file=sys.stderr)

    sampler.start()
    t_start = time.time()
    status = "ok"
    try:
        sampler.set_phase("_settle")
        time.sleep(args.settle)
        for i, (rep, w) in enumerate(plan, 1):
            if sampler.error:
                raise RuntimeError(f"sampler crashed: {sampler.error!r}")
            manifest["timings"].append(run_workload(w, rep, sampler, args.duration, args.ramp))
            el = time.time() - t_start
            print(f"   >>> {i}/{len(plan)} {w.name} r{rep} | elapsed {_fmt(el)} | "
                  f"ETA ~{_fmt(el / i * (len(plan) - i))}", file=sys.stderr, flush=True)
    except BaseException as e:
        status = f"aborted: {e!r}"
        raise
    finally:
        sampler.set_phase("_end")
        time.sleep(2 * args.interval)
        sampler.stop()
        sampler.join(timeout=30)
        if hasattr(sampler.bw, "close"):
            sampler.bw.close()
        manifest["status"] = status
        manifest["background_end"] = background_snapshot()
        manifest["wall_s"] = time.time() - t_start
        rows = sampler.rows
        for r in rows:
            r.pop("_role", None)
        write_trace(rows, out / "session.csv")
        groups: dict[tuple[str, str, int], list] = {}
        for r in rows:
            name = r["workload"]
            if name.startswith("_"):
                continue
            role = next(w.role for w in W if w.name == name)
            groups.setdefault((role, name, int(r["run_id"][1:])), []).append(r)
        for (role, name, rep), rs in sorted(groups.items()):
            base = rs[0]["session_t_s"] - rs[0]["dt_s"]  # per-file bounds start at 0 (I3)
            rows_out = []
            for r in rs:
                r2 = dict(r)
                r2["t_end_s"] = r["session_t_s"] - base
                r2["t_start_s"] = r2["t_end_s"] - r["dt_s"]
                rows_out.append(r2)
            prefix = "cal" if role == "cal" else "heldout"
            write_trace(rows_out, out / f"{prefix}_{name}_r{rep}.csv")
        (out / "session_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        print(f"\n{status}: {len(groups)} traces + session.csv -> {out} in "
              f"{_fmt(time.time() - t_start)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
