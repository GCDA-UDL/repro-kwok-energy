"""`probe` — quick vendor-agnostic energy capture (dev tool).

A thin CLI that drives the **src/ collectors** (`RAPLReader` + `make_gpu_reader`,
Intel/AMD/NVIDIA) to capture energy while a synthetic load — or your own program —
runs, and emits a JSON summary + per-sample CSV + a Markdown report. It reuses
`collect.collect_snapshot`/`compute_interval`; it does NOT reimplement any collector.

Linux only for CPU (RAPL/powercap); GPU works wherever the readers do.

    probe --duration 15 --load both --report report.md --json out.json --csv out.csv
    probe --cmd "python3 train.py --epochs 1" --report report.md

Run as root (or Docker --privileged) so RAPL energy_uj is readable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from multiprocessing import Process

from ..collect import collect_snapshot, compute_interval, make_gpu_reader
from ..rapl import RAPLReader

_COMPONENTS = {
    "package": ("rapl_pkg_joules", "rapl_pkg_watts_mean"),
    "dram": ("rapl_dram_joules", "rapl_dram_watts_mean"),
    "gpu": ("gpu_energy_joules", "gpu_watts_mean"),
}


# --- synthetic load --------------------------------------------------------
def _cpu_burn(stop_at: float) -> None:
    x = 1.0001
    while time.time() < stop_at:
        for _ in range(100000):
            x = (x * 1.0000001 + 1.0) ** 0.5


def _gpu_burn(stop_at: float) -> None:
    try:
        import torch  # CUDA and ROCm torch both expose torch.cuda
        if not torch.cuda.is_available():
            return
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        while time.time() < stop_at:
            a = (a @ b).clone()
            torch.cuda.synchronize()
    except Exception:
        return


def _start_load(kind: str, duration: float) -> list[Process]:
    stop = time.time() + duration
    ps: list[Process] = []
    if kind in ("cpu", "both"):
        ps += [Process(target=_cpu_burn, args=(stop,)) for _ in range(os.cpu_count() or 4)]
    if kind in ("gpu", "both"):
        ps.append(Process(target=_gpu_burn, args=(stop,)))
    for p in ps:
        p.start()
    return ps


# --- stats / report --------------------------------------------------------
def _pctl(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))]


def _summarize(rows: list[dict], duration: float, mode: str) -> dict:
    comp: dict[str, dict] = {}
    for name, (jkey, wkey) in _COMPONENTS.items():
        joules = [r[jkey] for r in rows if r.get(jkey) is not None]
        watts = [r[wkey] for r in rows if r.get(wkey) is not None]
        if not joules:
            continue
        e = float(sum(joules))
        comp[name] = {
            "energy_j": round(e, 4),
            "energy_wh": round(e / 3600.0, 6),
            "mean_w": round(sum(watts) / len(watts), 3) if watts else None,
            "min_w": round(min(watts), 3) if watts else None,
            "max_w": round(max(watts), 3) if watts else None,
            "p95_w": round(_pctl(watts, 0.95), 3) if watts else None,
            "samples": len(watts),
        }
    cpu_e = comp.get("package", {}).get("energy_j", 0.0) + comp.get("dram", {}).get("energy_j", 0.0)
    gpu_e = comp.get("gpu", {}).get("energy_j", 0.0)
    return {
        "host": {
            "host": platform.node(),
            "kernel": platform.release(),
            "cpu_cores": os.cpu_count(),
            "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        },
        "run": {"duration_s": round(duration, 3), "mode": mode},
        "components": comp,
        "totals": {
            "cpu_energy_j": round(cpu_e, 4),
            "gpu_energy_j": round(gpu_e, 4),
            "total_energy_j": round(cpu_e + gpu_e, 4),
            "avg_total_power_w": round((cpu_e + gpu_e) / duration, 3) if duration else None,
        },
    }


def _write_report(path: str, s: dict) -> None:
    h, t = s["host"], s["totals"]
    rows = ["| component | energy (J) | mean W | p95 W | max W | samples |",
            "|---|--:|--:|--:|--:|--:|"]
    for name, m in s["components"].items():
        rows.append(f"| {name} | {m['energy_j']} | {m['mean_w']} | {m['p95_w']} "
                    f"| {m['max_w']} | {m['samples']} |")
    md = f"""# Energy capture report

- **Host:** {h['host']} ({h['cpu_cores']} cores), kernel {h['kernel']}
- **When:** {h['timestamp_utc']}
- **Run:** {s['run']['mode']}, {s['run']['duration_s']} s

## Totals
| metric | value |
|---|--:|
| CPU energy | {t['cpu_energy_j']} J |
| GPU energy | {t['gpu_energy_j']} J |
| **Total** | **{t['total_energy_j']} J** |
| Avg power | {t['avg_total_power_w']} W |

## Components
{chr(10).join(rows)}

> CPU = RAPL package+DRAM (IT-component, not wall-plug); GPU via NVML or DRM hwmon.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)


# --- main loop -------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    rapl = RAPLReader()
    gpu = make_gpu_reader()
    print(f"# CPU source: {rapl.source} (available={rapl.available}); "
          f"GPU available={gpu.available}", file=sys.stderr)

    proc, loads = None, []
    if args.cmd:
        proc = subprocess.Popen(args.cmd, shell=True)
        mode = f"cmd:{args.cmd}"
    else:
        loads = _start_load(args.load, args.duration)
        mode = f"load:{args.load}"

    csv_fh = open(args.csv, "w", encoding="utf-8") if args.csv else None
    if csv_fh:
        csv_fh.write("t_s,package_w,dram_w,gpu_w\n")

    rows: list[dict] = []
    t0 = time.time()
    prev = collect_snapshot(rapl, nvml_reader=gpu)
    prev_t = t0

    def done() -> bool:
        return proc.poll() is not None if proc is not None else time.time() >= t0 + args.duration

    while not done():
        time.sleep(args.interval)
        now = time.time()
        cur = collect_snapshot(rapl, nvml_reader=gpu)
        row = compute_interval(prev, cur, now - prev_t)  # actual dt (not nominal)
        rows.append(row)
        if csv_fh:
            csv_fh.write(f"{now - t0:.3f},{row.get('rapl_pkg_watts_mean') or ''},"
                         f"{row.get('rapl_dram_watts_mean') or ''},{row.get('gpu_watts_mean') or ''}\n")
        prev, prev_t = cur, now
    duration = time.time() - t0

    if proc is not None:
        proc.wait()
    for p in loads:
        p.join(timeout=5)
    gpu.shutdown()
    if csv_fh:
        csv_fh.close()

    summary = _summarize(rows, duration, mode)
    print(json.dumps(summary["totals"], indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"# wrote {args.json}", file=sys.stderr)
    if args.report:
        _write_report(args.report, summary)
        print(f"# wrote {args.report}", file=sys.stderr)
    return 0 if (rapl.available or gpu.available) else 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick vendor-agnostic CPU+GPU energy capture.")
    ap.add_argument("--duration", type=float, default=10.0, help="seconds (ignored with --cmd)")
    ap.add_argument("--interval", type=float, default=0.2, help="sampling interval s")
    ap.add_argument("--load", choices=["none", "cpu", "gpu", "both"], default="none")
    ap.add_argument("--cmd", help="measure THIS program while it runs")
    ap.add_argument("--csv", help="per-sample CSV path")
    ap.add_argument("--json", help="summary JSON path")
    ap.add_argument("--report", help="Markdown report path")
    sys.exit(run(ap.parse_args()))


if __name__ == "__main__":
    main()
