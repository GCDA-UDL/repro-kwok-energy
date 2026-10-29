"""Comparator, transfer and whole-node analysis over one or more dataset bundles.

For every bundle (``dataset/<platform>_<date>/`` or a capture_v2 output directory)
this script fits, on the bundle's calibration traces only,

* ``energy_model``      the decomposed model, v1 features
* ``energy_model_hist`` the decomposed model with the GPU-history term
                        (only when the traces carry ``gpu_util_ewma``)

  With ``--selection`` (the output of ``model_selection.py``), ``energy_model`` and
  ``energy_model_hist`` use the per-platform configuration chosen by leave-one-
  workload-out CV on the calibration data, and ``energy_model_allsignals`` is the
  pre-specified configuration (every v1 signal, alpha 0.01), reported for comparison.
* ``util_linear``       calibrated whole-node ``a + b·u_cpu + c·u_gpu``
* ``fan2007``           calibrated Fan-Weber-Barroso CPU-utilisation curve
* ``tdp_parametric``    datasheet TDPs + one idle measurement (no fitting)
* ``constant``          mean calibration power

and evaluates every model on the bundle's held-out traces with two metrics:

* ``interval_mape``  mean |error| / measured over 1-s intervals (the harness metric)
* ``energy_err``     signed error of the energy integrated over each held-out run,
                     (E_pred - E_meas) / E_meas; reported as mean |.| over runs

With two or more bundles it also fits on one platform and predicts the other
(cross-platform transfer). Bundles with a platform meter add the whole-node
analysis: how much of the node's input power RAPL+NVML cover, and the error of
an affine node model ``P_node = alpha + beta·P_components`` fitted on calibration.
The evaluated history model of each v2 bundle is written to ``coeffs_<platform>.json``
(the coefficients behind the reported numbers, archived as the bundle's coeffs).

    python tools/e2e/q1_analysis.py dataset/a4000_20260620 dataset/2080ti_20260620 \
        --out data/q1/v1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from kwok_energy import baselines as bl  # noqa: E402
from kwok_energy import calibration as cal  # noqa: E402
from kwok_energy.energy_model import Coeffs, FeatureWindow, predict_power  # noqa: E402
from kwok_energy.features import add_gpu_util_ewma  # noqa: E402
from kwok_energy.schema import MODEL_INPUT_FIELDS  # noqa: E402
from kwok_energy.trace import read as read_trace  # noqa: E402

# Datasheet thermal design power per CPU socket / GPU board (vendor specifications).
NAMEPLATE = {
    "a4000": {"cpu_tdp_w": 2 * 100.0, "gpu_tdp_w": 140.0, "gpus": 2},   # 2x Xeon Silver 4210R, RTX A4000
    "2080ti": {"cpu_tdp_w": 205.0, "gpu_tdp_w": 250.0, "gpus": 4},     # Xeon W-3245, RTX 2080 Ti
}
COMP_KEYS = {"package": "rapl_pkg_joules", "dram": "rapl_dram_joules", "gpu": "gpu_energy_joules"}
PER_RUN: list[dict] = []  # one row per (model, held-out run, component), for paired statistics


# --------------------------------------------------------------------------- #
#  loading
# --------------------------------------------------------------------------- #
class Bundle:
    def __init__(self, path: Path, tau_s: float | None) -> None:
        self.path = path
        raw = path / "raw" if (path / "raw").is_dir() else path
        self.cal_files = sorted(raw.glob("cal_*.csv"))
        self.held_files = sorted(raw.glob("heldout_*.csv"))
        if not self.cal_files or not self.held_files:
            raise SystemExit(f"{path}: no cal_/heldout_ traces")
        self.cal = [read_trace(f) for f in self.cal_files]
        self.held = [read_trace(f) for f in self.held_files]
        first = self.cal[0][0]
        self.platform = str(first.get("platform_id") or path.name.split("_")[0])
        self.v2 = first.get("session_t_s") is not None
        self.has_meter = any(r.get("platform_meter_joules") is not None
                             for run in self.held for r in run)
        manifest = raw / "session_manifest.json"
        self.gpu_count = 1
        if manifest.is_file():
            self.gpu_count = int(json.loads(manifest.read_text())["host"].get("gpu_count") or 1)
        self.session = raw / "session.csv"
        self.tau_s = tau_s
        if self.v2 and tau_s is not None:
            self.set_tau(tau_s)

    def set_tau(self, tau_s: float) -> None:
        """Recompute the history feature on the continuous session timeline."""
        self.tau_s = tau_s
        rows = add_gpu_util_ewma(read_trace(self.session), tau_s)
        by_seq = {int(r["seq_index"]): r["gpu_util_ewma"] for r in rows}
        for run in self.cal + self.held:
            for r in run:
                r["gpu_util_ewma"] = by_seq[int(r["seq_index"])]

    @property
    def cal_rows(self) -> list[dict]:
        return [r for run in self.cal for r in run]


def measured_total_w(r: dict) -> float:
    return sum(float(r[k]) for k in COMP_KEYS.values() if r.get(k) is not None) / float(r["dt_s"])


def signals(r: dict) -> dict:
    return {k: v for k, v in r.items() if k in MODEL_INPUT_FIELDS}


# --------------------------------------------------------------------------- #
#  models
# --------------------------------------------------------------------------- #
class Wrapped:
    """Uniform ``predict(row) -> watts dict`` over the energy model and baselines."""

    def __init__(self, name: str, fn) -> None:
        self.name = name
        self.fn = fn

    def predict(self, r: dict) -> dict[str, float | None]:
        return self.fn(r)


def energy_model(coeffs) -> Any:
    def _f(r: dict) -> dict[str, float | None]:
        p = predict_power(FeatureWindow(current=signals(r)), coeffs)
        return {"package": p.package, "dram": p.dram, "gpu": p.gpu, "total": p.total}
    return _f


def baseline(m) -> Any:
    def _f(r: dict) -> dict[str, float | None]:
        return {"total": m.predict_power(FeatureWindow(current=signals(r))).total}
    return _f


def fit_selected(b: Bundle, cfg: dict):
    """Fit the decomposed model with a selected configuration (sets tau first)."""
    if cfg.get("tau_s") is not None:
        b.set_tau(float(cfg["tau_s"]))
    return cal.fit(b.cal_rows, platform_id=b.platform, ridge_alpha=float(cfg["alpha"]),
                   features=cfg["features"])


def fit_models(b: Bundle, sel: dict | None = None) -> list[Wrapped]:
    """Comparators and the decomposed models without history (history: ``fit_hist``)."""
    rows = b.cal_rows
    if sel:
        models = [Wrapped("energy_model_allsignals", energy_model(cal.fit(rows, platform_id=b.platform))),
                  Wrapped("energy_model", energy_model(fit_selected(b, sel["base"])))]
    else:
        models = [Wrapped("energy_model", energy_model(cal.fit(rows, platform_id=b.platform)))]
    models.append(Wrapped("util_linear", baseline(bl.fit_util_linear_total(rows))))
    models.append(Wrapped("fan2007", baseline(bl.fit_fan2007(rows))))
    idle = [r for run, f in zip(b.cal, b.cal_files, strict=True) if "_idle_" in f.name for r in run]
    cpu_idle = float(np.mean([(float(r["rapl_pkg_joules"]) + float(r.get("rapl_dram_joules") or 0))
                              / float(r["dt_s"]) for r in idle]))
    gpu_idle = float(np.mean([float(r["gpu_energy_joules"]) / float(r["dt_s"]) for r in idle])) / b.gpu_count
    np_ = NAMEPLATE.get(b.platform)
    if np_:
        models.append(Wrapped("tdp_parametric", baseline(bl.TDPParametricBaseline(
            cpu_idle_watts=cpu_idle, cpu_tdp_watts=np_["cpu_tdp_w"], gpu_idle_watts=gpu_idle,
            gpu_tdp_watts=np_["gpu_tdp_w"], gpu_count=b.gpu_count))))
    models.append(Wrapped("constant", baseline(bl.ConstantPowerBaseline(
        float(np.mean([measured_total_w(r) for r in rows]))))))
    return models


# --------------------------------------------------------------------------- #
#  evaluation
# --------------------------------------------------------------------------- #
def evaluate(model: Wrapped, runs: list[list[dict]], files: list[Path], *, train: str,
             test: str) -> list[dict]:
    """Per (workload, component): interval MAPE and per-run energy errors."""
    acc: dict[tuple[str, str], dict[str, list]] = defaultdict(lambda: {"ape": [], "err": []})
    for run, f in zip(runs, files, strict=True):
        wl = f.stem.split("_", 1)[1].rsplit("_r", 1)[0]
        e_meas: dict[str, float] = defaultdict(float)
        e_pred: dict[str, float] = defaultdict(float)
        for r in run:
            dt = float(r["dt_s"])
            pred = model.predict(r)
            meas = {c: (None if r.get(k) is None else float(r[k]) / dt) for c, k in COMP_KEYS.items()}
            meas["total"] = measured_total_w(r)
            for c, pw in pred.items():
                if pw is None or meas.get(c) is None:
                    continue
                if meas[c] > 1.0:  # MAPE floor (1 W), as in the harness
                    acc[(wl, c)]["ape"].append(abs(pw - meas[c]) / meas[c])
                e_meas[c] += meas[c] * dt
                e_pred[c] += pw * dt
        for c in e_pred:
            if e_meas[c] > 0:
                acc[(wl, c)]["err"].append((e_pred[c] - e_meas[c]) / e_meas[c])
                PER_RUN.append({"model": model.name, "train_platform": train, "test_platform": test,
                                "workload": wl, "run": f.stem, "component": c,
                                "measured_j": e_meas[c], "predicted_j": e_pred[c],
                                "energy_err_pct": 100 * (e_pred[c] - e_meas[c]) / e_meas[c]})
    out = []
    for (wl, c), v in sorted(acc.items()):
        errs = v["err"]
        out.append({
            "model": model.name, "train_platform": train, "test_platform": test, "workload": wl,
            "component": c, "n_intervals": len(v["ape"]), "n_runs": len(errs),
            "interval_mape": 100 * float(np.mean(v["ape"])) if v["ape"] else math.nan,
            "energy_abs_err": 100 * float(np.mean(np.abs(errs))) if errs else math.nan,
            "energy_signed_err": 100 * float(np.mean(errs)) if errs else math.nan,
            "energy_abs_err_ci95": _ci([100 * abs(e) for e in errs]),
        })
    return out


def _ci(x: list[float], n_boot: int = 2000, seed: int = 7) -> str:
    """Percentile bootstrap 95% CI of the mean, as ``lo..hi`` (empty if < 2 values)."""
    if len(x) < 2:
        return ""
    rng = random.Random(seed)
    means = sorted(float(np.mean([rng.choice(x) for _ in x])) for _ in range(n_boot))
    return f"{means[int(0.025 * n_boot)]:.2f}..{means[int(0.975 * n_boot)]:.2f}"


def fit_hist(b: Bundle, sel: dict | None = None):
    """Decomposed model with the history term (selected config, or all v1 signals)."""
    if sel:
        return fit_selected(b, sel["hist"])
    return cal.fit(b.cal_rows, platform_id=b.platform, features=cal.FIT_FEATURES_V2)


def write_coeffs(coeffs, b: Bundle, sel: dict | None, path: Path) -> None:
    """Serialise the evaluated history model with the lineage of its calibration traces."""
    payload = cal.to_payload(coeffs, b.tau_s)
    prov = payload["provenance"]
    # Fitted from pre-read rows, so record a combined hash over every calibration trace.
    prov["calibration_trace_sha256"] = cal.combined_trace_sha256(b.cal_files)
    prov["calibration_traces"] = len(b.cal_files)
    prov["ridge_alpha"] = float(sel["hist"]["alpha"]) if sel else 0.01
    # The provenance host is the machine that ran the fit; also record the measured platform.
    prov["fit_host"] = prov.pop("host", None)
    manifest = b.session.parent / "session_manifest.json"
    if manifest.is_file():
        h = json.loads(manifest.read_text(encoding="utf-8"))["host"]
        prov["platform_host"] = {k: h.get(k) for k in ("hostname", "cpu_model", "ncpu", "gpus",
                                                        "gpu_count", "kernel", "mem_proxy")}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                    newline="\n")


def node_level(b: Bundle, coeffs) -> list[dict]:
    """Whole-node analysis against the platform meter (v2 bundles that have one)."""
    def comp_w(r):
        return measured_total_w(r)

    cal_pairs = [(comp_w(r), float(r["platform_meter_joules"]) / float(r["dt_s"]))
                 for r in b.cal_rows if r.get("platform_meter_joules") is not None]
    x, y = np.array(cal_pairs).T
    beta, alpha = np.polyfit(x, y, 1)
    out = []
    for run, f in zip(b.held, b.held_files, strict=True):
        wl = f.stem.split("_", 1)[1].rsplit("_r", 1)[0]
        rs = [r for r in run if r.get("platform_meter_joules") is not None]
        if not rs:
            continue
        e_node = sum(float(r["platform_meter_joules"]) for r in rs)
        e_comp = sum(comp_w(r) * float(r["dt_s"]) for r in rs)
        e_pred = sum((alpha + beta * predict_power(FeatureWindow(current=signals(r)), coeffs).total)
                     * float(r["dt_s"]) for r in rs)
        out.append({"platform": b.platform, "workload": wl, "run": f.stem,
                    "coverage_pct": 100 * e_comp / e_node,
                    "node_energy_err_pct": 100 * (e_pred - e_node) / e_node,
                    "alpha_w": alpha, "beta": beta})
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summary(rows: list[dict], component: str | None = "total") -> list[dict]:
    """Mean over workloads of the energy metrics, per (model, train, test[, component])."""
    acc: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if component is None or r["component"] == component:
            acc[(r["model"], r["train_platform"], r["test_platform"], r["component"])].append(r)
    out = []
    for (m, tr, te, c), rs in acc.items():
        out.append({"model": m, "train_platform": tr, "test_platform": te, "component": c,
                    "workloads": len(rs),
                    "mean_interval_mape": float(np.mean([r["interval_mape"] for r in rs])),
                    "mean_energy_abs_err": float(np.mean([r["energy_abs_err"] for r in rs])),
                    "max_energy_abs_err": float(np.max([r["energy_abs_err"] for r in rs]))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bundles", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/q1"))
    ap.add_argument("--tau", type=float, default=None,
                    help="recompute gpu_util_ewma on the session timeline with this tau (v2)")
    ap.add_argument("--live-coeffs", type=Path, nargs="*", default=[],
                    help="coefficient files used by a live replay: held-out error as replayed "
                         "(history at the file's tau) -> live_coeffs.csv")
    ap.add_argument("--selection", type=Path, default=None,
                    help="selection.json from model_selection.py (per-platform configs)")
    args = ap.parse_args()

    selection = json.loads(args.selection.read_text()) if args.selection else {}
    bundles = [Bundle(p, args.tau) for p in args.bundles]
    rows: list[dict] = []
    node: list[dict] = []
    for b in bundles:
        sel = selection.get(b.platform)
        for m in fit_models(b, sel):
            rows += evaluate(m, b.held, b.held_files, train=b.platform, test=b.platform)
        if b.v2:
            hist = fit_hist(b, sel)
            args.out.mkdir(parents=True, exist_ok=True)
            write_coeffs(hist, b, sel, args.out / f"coeffs_{b.platform}.json")
            rows += evaluate(Wrapped("energy_model_hist", energy_model(hist)), b.held,
                             b.held_files, train=b.platform, test=b.platform)
            if b.has_meter:
                node += node_level(b, hist)
    # cross-platform transfer: decomposed model fitted on one host, applied to another
    # (the source platform's configuration; the target's history uses the source tau)
    for src in bundles:
        for dst in bundles:
            if src is dst:
                continue
            sel = selection.get(src.platform)
            if sel and src.v2 and dst.v2:
                coeffs = fit_selected(src, sel["hist"])
                if sel["hist"].get("tau_s") is not None:
                    dst.set_tau(float(sel["hist"]["tau_s"]))
            else:
                feats = cal.FIT_FEATURES_V2 if (src.v2 and dst.v2) else cal.FIT_FEATURES
                coeffs = cal.fit(src.cal_rows, platform_id=src.platform, features=feats)
            rows += evaluate(Wrapped("energy_model_transfer", energy_model(coeffs)), dst.held,
                             dst.held_files, train=src.platform, test=dst.platform)
    write_csv(rows, args.out / "per_workload.csv")
    write_csv(PER_RUN, args.out / "per_run.csv")
    s = summary(rows)
    write_csv(s, args.out / "summary.csv")
    write_csv(summary(rows, None), args.out / "summary_components.csv")
    write_csv(node, args.out / "node_level.csv")
    # Coefficient files loaded by a live replay, evaluated as the replay used them (the
    # history recomputed at the file's tau); after the outputs above, so they stay unchanged.
    live = []
    for path in args.live_coeffs:
        d = json.loads(path.read_text(encoding="utf-8"))
        b = next(x for x in bundles if x.platform == d["platform_id"])
        tau = d["transform"].get("gpu_util_ewma_tau_s")
        if tau is not None:
            b.set_tau(float(tau))
        c = Coeffs(platform_id=d["platform_id"], covered_components=frozenset(d["covered_components"]),
                   transform=d["transform"], components=d["components"],
                   provenance=d.get("provenance", {}))
        r = evaluate(Wrapped("energy_model_live", energy_model(c)), b.held, b.held_files,
                     train=b.platform, test=b.platform)
        cap = json.loads((b.session.parent / "session_manifest.json").read_text(encoding="utf-8"))
        sel_err = next(x["mean_energy_abs_err"] for x in s if x["model"] == "energy_model_hist"
                       and x["train_platform"] == b.platform and x["test_platform"] == b.platform)
        live.append({"platform": b.platform, "coeffs_file": path.name,
                     "coeffs_sha256": cal.hash_file(path), "tau_label_s": tau,
                     "capture_tau_s": cap.get("args", {}).get("tau"),
                     "heldout_total_abs_err": summary(r)[0]["mean_energy_abs_err"],
                     "selected_total_abs_err": sel_err})
    if live:
        write_csv(live, args.out / "live_coeffs.csv")
    for r in sorted(s, key=lambda r: (r["test_platform"], r["train_platform"], r["mean_energy_abs_err"])):
        print(f"{r['test_platform']:>7} <- {r['train_platform']:<7} {r['model']:<22} "
              f"interval MAPE {r['mean_interval_mape']:6.1f}%  energy |err| "
              f"{r['mean_energy_abs_err']:6.1f}% (max {r['max_energy_abs_err']:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
