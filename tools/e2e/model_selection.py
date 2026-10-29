"""Model selection on calibration data only: leave-one-workload-out (LOWO) CV.

For one capture_v2 bundle, every candidate configuration

* signal set   ``swept`` (calibration.SWEPT_FEATURES) or ``swept`` without GPU-memory
               utilisation (collinear with GPU utilisation in the matmul sweep)
* ridge alpha  0.01 .. 1000 (on standardised signals, intercept not penalised)
* history      none, or the GPU-utilisation EWMA with tau in 5 .. 80 s

is fitted on the calibration traces of all but one calibration workload and scored on
the five repetitions of the left-out workload. The score is the mean, over left-out
workloads, of the mean per-run |error| of the total (package + DRAM + GPU) energy.
Held-out traces are never read. The best configuration without history and the best
with history are written to ``<out>/selection.json`` for ``q1_analysis.py``.

    python tools/e2e/model_selection.py data/e2e_v2/a4000 data/e2e_v2/2080ti \
        --out data/q1/v2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "e2e"))

from kwok_energy import calibration as cal  # noqa: E402
from kwok_energy.energy_model import FeatureWindow, predict_power  # noqa: E402
from kwok_energy.features import add_gpu_util_ewma  # noqa: E402
from kwok_energy.trace import read as read_trace  # noqa: E402

from q1_analysis import COMP_KEYS, signals  # noqa: E402

SIGNAL_SETS = {
    "swept": cal.SWEPT_FEATURES,
    "swept_no_gmem": [f for f in cal.SWEPT_FEATURES if f != "gpu_mem_utilization"],
}
ALPHAS = [0.01, 1.0, 10.0, 100.0, 1000.0]
TAUS = [5.0, 10.0, 20.0, 40.0, 80.0]


def load(bundle: Path) -> tuple[str, dict[str, list[list[dict]]], list[dict]]:
    """Calibration runs grouped by workload, and the session timeline."""
    if (bundle / "raw").is_dir():  # an archived dataset bundle
        bundle = bundle / "raw"
    files = sorted(bundle.glob("cal_*.csv"))
    if not files:
        raise SystemExit(f"{bundle}: no calibration traces")
    groups: dict[str, list[list[dict]]] = defaultdict(list)
    for f in files:
        groups[f.stem.split("_", 1)[1].rsplit("_r", 1)[0]].append(read_trace(f))
    session = read_trace(bundle / "session.csv")
    platform = str(next(iter(groups.values()))[0][0]["platform_id"])
    return platform, groups, session


def set_ewma(groups: dict[str, list[list[dict]]], session: list[dict], tau: float) -> None:
    by_seq = {int(r["seq_index"]): r["gpu_util_ewma"]
              for r in add_gpu_util_ewma([dict(r) for r in session], tau)}
    for runs in groups.values():
        for run in runs:
            for r in run:
                r["gpu_util_ewma"] = by_seq[int(r["seq_index"])]


def run_energy_err(coeffs, run: list[dict]) -> dict[str, float]:
    e_meas: dict[str, float] = defaultdict(float)
    e_pred: dict[str, float] = defaultdict(float)
    for r in run:
        dt = float(r["dt_s"])
        p = predict_power(FeatureWindow(current=signals(r)), coeffs)
        for c, k in COMP_KEYS.items():
            pw = getattr(p, c)
            if pw is None or r.get(k) is None:
                continue
            e_meas[c] += float(r[k])
            e_pred[c] += pw * dt
    e_meas["total"] = sum(e_meas[c] for c in COMP_KEYS)
    e_pred["total"] = sum(e_pred[c] for c in COMP_KEYS)
    return {c: abs(e_pred[c] - e_meas[c]) / e_meas[c] for c in e_meas if e_meas[c] > 0}


def lowo(platform: str, groups: dict[str, list[list[dict]]], feats: list[str],
         alpha: float) -> dict[str, float]:
    """Mean over left-out workloads of the mean per-run |energy error|, per component."""
    per_wl: dict[str, list[float]] = defaultdict(list)
    for left in groups:
        train = [r for wl, runs in groups.items() if wl != left for run in runs for r in run]
        coeffs = cal.fit(train, platform_id=platform, ridge_alpha=alpha, features=feats)
        errs = [run_energy_err(coeffs, run) for run in groups[left]]
        for c in errs[0]:
            per_wl[c].append(float(np.mean([e[c] for e in errs])))
    return {c: 100 * float(np.mean(v)) for c, v in per_wl.items()}


def select(bundle: Path) -> tuple[str, list[dict], dict]:
    platform, groups, session = load(bundle)
    rows: list[dict] = []
    for sname, base in SIGNAL_SETS.items():
        for alpha in ALPHAS:
            for tau in [None, *TAUS]:
                feats = list(base) if tau is None else [*base, "gpu_util_ewma"]
                if tau is not None:
                    set_ewma(groups, session, tau)
                s = lowo(platform, groups, feats, alpha)
                rows.append({"platform": platform, "signal_set": sname, "alpha": alpha,
                             "tau_s": "" if tau is None else tau,
                             **{f"lowo_{c}_energy_abs_err": round(v, 3) for c, v in s.items()}})
                print(f"{platform:>7} {sname:<14} alpha={alpha:<7g} tau={tau!s:<5} "
                      f"total {s['total']:6.2f}%  pkg {s.get('package', float('nan')):6.2f}%  "
                      f"dram {s.get('dram', float('nan')):6.2f}%  gpu {s.get('gpu', float('nan')):6.2f}%",
                      flush=True)
    key = "lowo_total_energy_abs_err"
    best_base = min((r for r in rows if r["tau_s"] == ""), key=lambda r: r[key])
    best_hist = min((r for r in rows if r["tau_s"] != ""), key=lambda r: r[key])

    def cfg(r: dict) -> dict:
        feats = list(SIGNAL_SETS[r["signal_set"]])
        if r["tau_s"] != "":
            feats.append("gpu_util_ewma")
        return {"signal_set": r["signal_set"], "features": feats, "alpha": r["alpha"],
                "tau_s": None if r["tau_s"] == "" else r["tau_s"], "lowo_total": r[key]}

    return platform, rows, {"base": cfg(best_base), "hist": cfg(best_hist)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundles", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/q1/v2"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selection: dict[str, dict] = {}
    all_rows: list[dict] = []
    for b in args.bundles:
        platform, rows, chosen = select(b)
        selection[platform] = chosen
        all_rows += rows
        print(f"{platform}: selected {json.dumps(chosen)}")
    with (args.out / "model_selection.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
