"""Paper numbers and tables, generated from the analysis CSVs (no hand-copied values).

Reads ``q1_analysis.py`` / ``model_selection.py`` outputs (and, when present, the live
KWOK results) and writes

* ``paper/generated/values.tex``  one ``\\V{key}`` value per number quoted in the text
* ``paper/generated/tab_*.tex``   the result tables

Confidence intervals are percentile bootstraps over the repetitions of each held-out
workload (resampled within workload, so they measure run-to-run repeatability; the
workload set itself is fixed). Paired comparisons resample the per-run differences
between two models in the same way.

    python tools/e2e/paper_tables.py --analysis data/q1/v2 --out paper/generated
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

PLAT = {"a4000": "A", "2080ti": "B"}
PLAT_NAME = {"a4000": "A4000", "2080ti": "2080\\,Ti"}
MODELS = [  # (key, label) in table order
    ("energy_model_hist", "Decomposed + history (ours)"),
    ("energy_model", "Decomposed, no history"),
    ("util_linear", "Util-linear (calibrated)"),
    ("tdp_parametric", "TDP-parametric (datasheet)"),
    ("fan2007", "Fan et al.\\ (calibrated)"),
    ("constant", "Constant power"),
    ("energy_model_allsignals", "All signals (pre-specified)"),
]
MKEY = {"energy_model_hist": "hist", "energy_model": "base", "util_linear": "util",
        "tdp_parametric": "tdp", "fan2007": "fan", "constant": "const",
        "energy_model_allsignals": "all", "energy_model_transfer": "xfer"}
WORKLOADS = [  # (key, label, kind)
    ("mixed_cpu_gpu", "\\texttt{mixed\\_cpu\\_gpu}", "synthetic"),
    ("bursty_cpu", "\\texttt{bursty\\_cpu}", "synthetic"),
    ("mem_gpu", "\\texttt{mem\\_gpu}", "synthetic"),
    ("train_resnet50", "\\texttt{train\\_resnet50}", "real"),
    ("llm_serve", "\\texttt{llm\\_serve}", "real"),
    ("pgbench", "\\texttt{pgbench}", "real"),
    ("build_cpython", "\\texttt{build\\_cpython}", "real"),
    ("colocated_train_pgbench", "\\texttt{colocated}", "real"),
]
WKEY = {"mixed_cpu_gpu": "mixed", "bursty_cpu": "bursty", "mem_gpu": "memgpu",
        "train_resnet50": "resnet", "llm_serve": "llm", "pgbench": "pgbench",
        "build_cpython": "cpython", "colocated_train_pgbench": "coloc"}
COMPS = ["package", "dram", "gpu", "total"]


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f1(x: float) -> str:
    out = f"{x:.1f}"
    return "0.0" if out == "-0.0" else out


def fs(x: float) -> str:
    """Signed one-decimal value; a value that rounds to zero prints as 0.0."""
    out = f"{x:+.1f}"
    return "0.0" if out in ("+0.0", "-0.0") else out


def boot_workloads(by_wl: dict[str, dict[str, float]], n: int = 4000, seed: int = 13):
    """Percentile bootstrap over workloads of the mean per-workload value (generalisation)."""
    rng = random.Random(seed)
    per = [float(np.mean(list(r.values()))) for r in by_wl.values()]
    out = sorted(float(np.mean([rng.choice(per) for _ in per])) for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n)]


class Runs:
    """Per-run signed errors: (model, train, test, component) -> {workload: [err, ...]}."""

    def __init__(self, rows: list[dict]) -> None:
        self.d: dict[tuple, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for r in rows:
            k = (r["model"], r["train_platform"], r["test_platform"], r["component"])
            self.d[k][r["workload"]][r["run"]] = float(r["energy_err_pct"])

    def get(self, model: str, plat: str, comp: str = "total", train: str | None = None):
        return self.d[(model, train or plat, plat, comp)]


def mean_abs(by_wl: dict[str, dict[str, float]]) -> float:
    """Mean over workloads of the mean per-run |error|."""
    return float(np.mean([np.mean([abs(e) for e in runs.values()]) for runs in by_wl.values()]))


def boot(by_wl: dict[str, dict[str, float]], stat, n: int = 4000, seed: int = 11):
    """Stratified percentile bootstrap (resample runs within each workload)."""
    rng = random.Random(seed)
    keys = sorted(by_wl)
    vals = {w: list(by_wl[w].values()) for w in keys}
    out = sorted(stat({w: {str(i): rng.choice(vals[w]) for i in range(len(vals[w]))} for w in keys})
                 for _ in range(n))
    return out[int(0.025 * n)], out[int(0.975 * n)]


def paired(a: dict, b: dict) -> dict[str, dict[str, float]]:
    """Per-run |a| - |b| (same runs), for a paired bootstrap of the difference."""
    return {w: {r: abs(a[w][r]) - abs(b[w][r]) for r in a[w]} for w in a}


def mean_of(by_wl: dict[str, dict[str, float]]) -> float:
    return float(np.mean([np.mean(list(r.values())) for r in by_wl.values()]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--analysis", type=Path, default=Path("data/q1/v2"))
    ap.add_argument("--kwok-live", type=Path, default=Path("data/kwok_live/a4000"))
    ap.add_argument("--out", type=Path, default=Path("paper/generated"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    runs = Runs(read_csv(a.analysis / "per_run.csv"))
    per_wl = read_csv(a.analysis / "per_workload.csv")
    sel = json.loads((a.analysis / "selection.json").read_text())
    V: dict[str, str] = {}

    # --- headline table: total energy error per model x platform ------------------
    lines = []
    for m, label in MODELS:
        cells = []
        for p in PLAT:
            byw = runs.get(m, p)
            if not byw:
                cells.append("--")
                continue
            mu, (lo, hi) = mean_abs(byw), boot(byw, mean_abs)
            mx = max(np.mean([abs(e) for e in r.values()]) for r in byw.values())
            cells.append(f"{f1(mu)} [{f1(lo)}, {f1(hi)}] & {f1(mx)}")
            k = f"{PLAT[p]}.{MKEY[m]}"
            V[f"{k}.tot"], V[f"{k}.lo"], V[f"{k}.hi"], V[f"{k}.max"] = f1(mu), f1(lo), f1(hi), f1(mx)
            im = [float(r["interval_mape"]) for r in per_wl if r["model"] == m
                  and r["test_platform"] == p and r["train_platform"] == p and r["component"] == "total"]
            V[f"{k}.imape"] = f1(float(np.mean(im)))
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if m == "energy_model":
            lines.append("\\midrule")
    (a.out / "tab_heldout_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- paired comparisons and workload wins vs each comparator ------------------
    for p in PLAT:
        ours = runs.get("energy_model_hist", p)
        for m in ("energy_model", "util_linear", "tdp_parametric", "fan2007", "constant"):
            other = runs.get(m, p)
            d = paired(ours, other)
            lo, hi = boot(d, mean_of)
            k = f"{PLAT[p]}.vs{MKEY[m]}"
            V[f"{k}.d"], V[f"{k}.lo"], V[f"{k}.hi"] = f1(mean_of(d)), f1(lo), f1(hi)
            wlo, whi = boot_workloads(d)
            V[f"{k}.wlo"], V[f"{k}.whi"] = f1(wlo), f1(whi)
            wins = sum(1 for w in ours if np.mean([abs(e) for e in ours[w].values()])
                       < np.mean([abs(e) for e in other[w].values()]))
            V[f"{k}.wins"] = str(wins)
        V[f"{PLAT[p]}.nwl"] = str(len(ours))

    # --- per-component table (ours, both platforms) -------------------------------
    comp_label = {"package": "CPU package", "dram": "DRAM", "gpu": "GPU", "total": "Total"}
    lines = []
    for c in COMPS:
        cells = []
        for p in PLAT:
            for m in ("energy_model_hist", "energy_model_allsignals"):
                byw = runs.get(m, p, c)
                mu = mean_abs(byw)
                lo, hi = boot(byw, mean_abs)
                V[f"{PLAT[p]}.{MKEY[m]}.{c}"] = f1(mu)
                if m == "energy_model_hist":
                    V[f"{PLAT[p]}.{MKEY[m]}.{c}.lo"], V[f"{PLAT[p]}.{MKEY[m]}.{c}.hi"] = f1(lo), f1(hi)
                    cells.append(f"{f1(mu)} [{f1(lo)}, {f1(hi)}]")
                else:
                    cells.append(f1(mu))
        row = f"{comp_label[c]} & " + " & ".join(cells) + " \\\\"
        lines.append(("\\midrule\n" if c == "total" else "") + row)
    (a.out / "tab_components_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for p in PLAT:
        for c in COMPS:
            V[f"{PLAT[p]}.base.{c}"] = f1(mean_abs(runs.get("energy_model", p, c)))

    # --- per-workload signed total error (ours vs the two best comparators) -------
    lines = []
    for w, label, kind in WORKLOADS:
        cells = []
        for p in PLAT:
            for m in ("energy_model_hist", "util_linear", "tdp_parametric"):
                e = float(np.mean(list(runs.get(m, p)[w].values())))
                cells.append(fs(e))
                V[f"{PLAT[p]}.{MKEY[m]}.{WKEY[w]}"] = fs(e)
            for c in ("package", "dram", "gpu"):
                V[f"{PLAT[p]}.hist.{WKEY[w]}.{c}"] = \
                    fs(float(np.mean(list(runs.get('energy_model_hist', p, c)[w].values()))))
            V[f"{PLAT[p]}.base.{WKEY[w]}.gpu"] = \
                fs(float(np.mean(list(runs.get('energy_model', p, 'gpu')[w].values()))))
            V[f"{PLAT[p]}.all.{WKEY[w]}"] = \
                fs(float(np.mean(list(runs.get('energy_model_allsignals', p)[w].values()))))
        lines.append(f"{label} & {kind} & " + " & ".join(cells) + " \\\\")
    (a.out / "tab_workloads_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # --- real vs synthetic held-out ----------------------------------------------
    for p in PLAT:
        byw = runs.get("energy_model_hist", p)
        for kind in ("synthetic", "real"):
            ws = [w for w, _, k in WORKLOADS if k == kind]
            V[f"{PLAT[p]}.hist.{kind}"] = f1(mean_abs({w: byw[w] for w in ws}))
            ub = runs.get("util_linear", p)
            V[f"{PLAT[p]}.util.{kind}"] = f1(mean_abs({w: ub[w] for w in ws}))

    # --- transfer -----------------------------------------------------------------
    for src, dst in (("a4000", "2080ti"), ("2080ti", "a4000")):
        byw = runs.get("energy_model_transfer", dst, "total", train=src)
        V[f"xfer.{PLAT[src]}{PLAT[dst]}"] = f1(mean_abs(byw))
        for c in ("package", "dram", "gpu"):
            byc = runs.get("energy_model_transfer", dst, c, train=src)
            V[f"xfer.{PLAT[src]}{PLAT[dst]}.{c}"] = f1(mean_abs(byc))
    # Signed energy gap of the transferred model, summed over held-out runs, per component.
    gap: dict[tuple[str, str, str], float] = defaultdict(float)
    for r in read_csv(a.analysis / "per_run.csv"):
        if r["model"] == "energy_model_transfer" and r["component"] in ("package", "dram", "gpu"):
            gap[(r["train_platform"], r["test_platform"], r["component"])] += \
                float(r["predicted_j"]) - float(r["measured_j"])
    for src, dst in (("a4000", "2080ti"), ("2080ti", "a4000")):
        tot = sum(gap[(src, dst, c)] for c in ("package", "dram", "gpu"))
        k = f"xfer.{PLAT[src]}{PLAT[dst]}"
        V[f"{k}.dir"] = "under-predict" if tot < 0 else "over-predict"  # plural subject in the text
        V[f"{k}.gpushare"] = f1(100 * gap[(src, dst, "gpu")] / tot)

    # --- coefficient file loaded by the live replay (Section 8) -------------------
    lc = a.analysis / "live_coeffs.csv"
    if lc.is_file():
        r = read_csv(lc)[0]
        V["kwok.coeffs.err"] = f1(float(r["heldout_total_abs_err"]))
        V["kwok.coeffs.tau"] = f"{float(r['tau_label_s']):g}"
        V["kwok.coeffs.tau.capture"] = f"{float(r['capture_tau_s']):g}"

    # --- model selection ----------------------------------------------------------
    for p, cfg in sel.items():
        for var in ("base", "hist"):
            c = cfg[var]
            V[f"{PLAT[p]}.sel.{var}.alpha"] = f"{c['alpha']:g}"
            V[f"{PLAT[p]}.sel.{var}.lowo"] = f1(c["lowo_total"])
            V[f"{PLAT[p]}.sel.{var}.gmem"] = "yes" if "gpu_mem_utilization" in c["features"] else "no"
        V[f"{PLAT[p]}.sel.tau"] = f"{cfg['hist']['tau_s']:g}"
    ms = read_csv(a.analysis / "model_selection.csv")
    V["sel.nconfigs"] = str(len({(r["signal_set"], r["alpha"], r["tau_s"]) for r in ms}))
    for p in PLAT:
        r = [x for x in ms if x["platform"] == p and x["signal_set"] == "swept"
             and float(x["alpha"]) == 0.01 and x["tau_s"] == ""][0]
        V[f"{PLAT[p]}.sel.swept001.lowo"] = f1(float(r["lowo_total_energy_abs_err"]))

    # --- fitted coefficients of the evaluated model, in physical units ------------
    # Eq. comp rewritten as P_c = max(0, a_c + sum_j b_cj (x_j - mu_j)), b_cj = w_cj / sigma_j.
    coef_files = {p: a.analysis / f"coeffs_{p}.json" for p in PLAT}
    if all(f.is_file() for f in coef_files.values()):
        sig = [  # (feature, short key, row label, printed unit per model unit)
            ("cpu_utilization", "u", "$u$ (0--1)", 1.0),
            ("cpu_frequency_mhz", "f", "$f$ (GHz)", 1e-3),
            ("mem_bandwidth_proxy", "m", "$m$ (GiB/s)", 1 / 1024),
            ("gpu_utilization", "g", "$g$ (0--1)", 1.0),
            ("gpu_util_ewma", "h", "$h$ (0--1)", 1.0),
        ]
        ck = {"package": "pkg", "dram": "dram", "gpu": "gpu"}
        co = {p: json.loads(f.read_text(encoding="utf-8")) for p, f in coef_files.items()}

        def wn(x: float) -> str:
            """Watts: one decimal from 10 W up, two below; minus sign typeset."""
            s = f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"
            return ("0.00" if float(s) == 0 else s).replace("-", "$-$")

        cells = []
        for p in PLAT:
            cells.append("--")
            for c, k in ck.items():
                v = wn(co[p]["components"][c]["intercept"])
                cells.append(v)
                V[f"{PLAT[p]}.coef.{k}.a"] = v
        lines = ["Intercept $a_c$ (W) & " + " & ".join(cells) + " \\\\", "\\midrule"]
        for feat, short, label, unit in sig:
            cells = []
            for p in PLAT:
                t = co[p]["transform"]
                mu, sd = t["means"][feat] * unit, t["stds"][feat] * unit
                cells.append(f"{mu:.2f}")
                V[f"{PLAT[p]}.mu.{short}"], V[f"{PLAT[p]}.sd.{short}"] = f"{mu:.2f}", f"{sd:.2f}"
                for c, k in ck.items():
                    b = co[p]["components"][c]["weights"].get(feat, 0.0) / (t["stds"][feat] * unit)
                    cells.append(wn(b))
                    V[f"{PLAT[p]}.coef.{k}.{short}"] = wn(b)
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        (a.out / "tab_coeffs_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        for p in PLAT:
            # h is a smoothed g: at steady state h = g, so only their sum is well identified.
            t, w = co[p]["transform"], co[p]["components"]["gpu"]["weights"]
            V[f"{PLAT[p]}.coef.gpu.gh"] = wn(sum(w[f] / t["stds"][f]
                                                  for f in ("gpu_utilization", "gpu_util_ewma")))

    # --- whole node (platform meter) ---------------------------------------------
    nl = a.analysis / "node_level.csv"
    if nl.is_file():
        rows = read_csv(nl)
        cov = [float(r["coverage_pct"]) for r in rows]
        err = [float(r["node_energy_err_pct"]) for r in rows]
        V["node.cov.min"], V["node.cov.max"] = f"{min(cov):.0f}", f"{max(cov):.0f}"
        V["node.under.min"], V["node.under.max"] = f"{100 - max(cov):.0f}", f"{100 - min(cov):.0f}"
        V["node.alpha"], V["node.beta"] = f"{float(rows[0]['alpha_w']):.0f}", f"{float(rows[0]['beta']):.2f}"
        V["node.err"] = f1(float(np.mean(np.abs(err))))
        byw: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            byw[r["workload"]].append(float(r["node_energy_err_pct"]))
        V["node.err.maxwl"] = f1(max(abs(float(np.mean(v))) for v in byw.values()))
        V["node.n"] = str(len(rows))

    # --- live KWOK (when present) -------------------------------------------------
    kl = a.kwok_live / "kwok_live.csv"
    if kl.is_file():
        kr = sorted(read_csv(kl), key=lambda r: int(r["nodes"]))
        al = read_csv(a.kwok_live / "closed_loop_aligned.csv")
        CTRL = ("kwok", "apiserver", "etcd")

        def num(n: int) -> str:
            return f"{n:,}".replace(",", "{,}")

        def rate(n: int) -> float:
            """Logical ticks per wall second between the first and last stored sample."""
            r = next(x for x in al if int(x["nodes"]) == n)
            span = (int(r["prom_samples"]) - 1) * float(next(x for x in kr if int(x["nodes"]) == n)["scrape_s"])
            return (int(r["window_end_tick"]) - int(r["window_start_tick"])) / span

        lines = []
        for r in kr:
            n = int(r["nodes"])
            ctrl_cpu = sum(float(r[f"{k}_cpu_pct"]) for k in CTRL)
            ctrl_rss = sum(float(r[f"{k}_rss_mb_max"]) for k in CTRL)
            lines.append(" & ".join([
                num(n), f"{float(r['time_to_ready_s']):.0f}",
                f"{1000 * float(r['tick_cost_mean_s']):,.0f} / {1000 * float(r['tick_cost_max_s']):,.0f}"
                .replace(",", "{,}"),
                r["ticks_late"], f"{rate(n):.2f}", f"{float(r['scrape_duration_mean_s']):.2f}",
                f"{float(r['exporter_cpu_pct']):.0f} / {float(r['exporter_rss_mb_max']):,.0f}".replace(",", "{,}"),
                f"{ctrl_cpu:.0f} / {ctrl_rss:,.0f}".replace(",", "{,}"),
                f"{float(r['host_meter_w']):.0f} / {float(r['host_pkg_w']) + float(r['host_dram_w']):.0f}",
            ]) + " \\\\")
        (a.out / "tab_kwok_live_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
        small, big = kr[0], kr[-1]
        nb = int(big["nodes"])
        V["kwok.maxn"] = num(nb)
        V["kwok.minn"] = num(int(small["nodes"]))
        V["kwok.ready"] = f"{float(big['time_to_ready_s']) / 60:.0f}"
        V["kwok.tick"] = f"{1000 * float(big['tick_cost_mean_s']):.0f}"
        V["kwok.tick.max"] = f"{float(big['tick_cost_max_s']):.1f}"
        V["kwok.tick.pernode"] = f"{1e6 * float(big['tick_cost_mean_s']) / nb:.0f}"
        V["kwok.tick.pernode.min"] = f"{min(1e6 * float(r['tick_cost_mean_s']) / int(r['nodes']) for r in kr):.0f}"
        V["kwok.late"] = big["ticks_late"]
        V["kwok.lag"] = f"{100 * (1 - rate(nb)):.0f}"
        V["kwok.scrape"] = f"{float(big['scrape_duration_mean_s']):.1f}"
        V["kwok.scrape.s"] = f"{float(big['scrape_s']):.0f}"
        V["kwok.window"] = f"{float(big['window_s']) / 60:.0f}"
        V["kwok.samples"] = num(int(float(big["samples_per_scrape"])))
        V["kwok.exp.cpu"] = f"{float(big['exporter_cpu_pct']):.0f}"
        V["kwok.exp.rss"] = f"{float(big['exporter_rss_mb_max']) / 1024:.1f}"
        V["kwok.api.cpu"] = f"{float(big['apiserver_cpu_pct']):.0f}"
        V["kwok.ctrl.cpu"] = f"{sum(float(big[f'{k}_cpu_pct']) for k in CTRL):.0f}"
        V["kwok.prom.cpu"] = f"{float(big['prometheus_cpu_pct']):.0f}"
        V["kwok.prom.rss"] = f"{float(big['prometheus_rss_mb_max']):.0f}"
        V["kwok.meter.min"] = f"{float(small['host_meter_w']):.0f}"
        V["kwok.meter.max"] = f"{float(big['host_meter_w']):.0f}"
        V["kwok.meter.delta"] = f"{float(big['host_meter_w']) - float(small['host_meter_w']):.0f}"
        V["kwok.up"] = f"{100 * min(float(r['scrape_up_ratio']) for r in kr):.0f}"
        # Closed loop: exact up to the largest size; there, explained by a tick offset.
        V["kwok.loop.n"] = str(sum(1 for x in al if int(x["nodes"]) == nb))
        exact = [float(r["closed_loop_max_rel_diff"]) for r in kr[:-1]]
        m, e = f"{max(exact):.1e}".split("e")
        V["kwok.loop.exact"] = f"${m}\\times10^{{{int(e)}}}$"
        V["kwok.loop.raw"] = f"{100 * float(big['closed_loop_max_rel_diff']):.2f}"
        V["kwok.loop.fitsunique"] = "yes" if all(x["n_fits"] == "1" for x in al) else "no"
        shifts = sorted({int(x["shift_start"]) for x in al if int(x["nodes"]) == nb}
                        | {int(x["shift_end"]) for x in al if int(x["nodes"]) == nb})
        V["kwok.shift.min"], V["kwok.shift.max"] = str(shifts[0]), str(shifts[-1])
        res = max(float(x["aligned_rel_diff"]) for x in al)
        m, e = f"{res:.0e}".split("e")
        V["kwok.loop.aligned"] = f"${m}\\times10^{{{int(e)}}}$"

    out = ["% generated by tools/e2e/paper_tables.py -- do not edit",
           "\\makeatletter",
           "\\newcommand{\\V}[1]{\\@ifundefined{v@#1}{\\textbf{??#1??}}{\\@nameuse{v@#1}}}"]
    out += [f"\\@namedef{{v@{k}}}{{{v}}}" for k, v in sorted(V.items())]
    out.append("\\makeatother")
    (a.out / "values.tex").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{len(V)} values -> {a.out / 'values.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
