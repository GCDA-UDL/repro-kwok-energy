"""Explain closed-loop gaps of the live KWOK run as a tick offset, per sampled node.

``kwok_live.py`` compares, for a sample of nodes, the increase of the counter stored by
Prometheus with an offline integration of the model over the ticks that had completed
before the first and the last sample's timestamp. Prometheus stamps a sample with the
start of its scrape, while the exporter serialises the counters while the ticker keeps
running; when a scrape lasts longer than a tick, the stored value can already hold one or
two more ticks than the timestamp implies.

For every cluster size this script rebuilds the offline counter table of each sampled
node from the same replay map and coefficients, recovers the tick window (b, e) that
reproduces every logged offline increase, and searches the start/end tick shifts
(-5..5) that reproduce the Prometheus increase. One row per sampled node is written to
``<out>/closed_loop_aligned.csv``; ``n_fits`` counts the shift pairs within tolerance
(1 means the offset is unique).

Run on the host that ran the experiment (the replay map holds absolute trace paths):

    python tools/kwok_live/closed_loop_align.py --out data/kwok_live/a4000 \
        --coeffs data/coeffs/a4000_v2.json
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from kwok_energy.cli.run_exporter import _load_coeffs  # noqa: E402
from kwok_energy.energy_model import FeatureWindow, predict_power  # noqa: E402
from kwok_energy.replay import load_map  # noqa: E402

SHIFTS = range(-5, 6)
TOL_J = 1e-3


def counter_table(source, node: str, coeffs, n_ticks: int, tick_s: float) -> list[float]:
    """by_done[m] = counter after m completed ticks (tick 0 only initialises), as kwok_live."""
    cum = [0.0]
    for k in range(1, n_ticks):
        p = predict_power(FeatureWindow(current=source.signals_at(node, k * tick_s)), coeffs).total
        cum.append(cum[-1] + p * tick_s)
    return [0.0, *cum]


def fits(tables: dict[str, list[float]], b: int, e: int, key: str, rows: list[dict]) -> bool:
    """Window (b, e) reproduces ``key`` for every row, to rounding."""
    return all(abs(tables[r["node"]][e] - tables[r["node"]][b] - float(r[key])) <= 1e-6
               for r in rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("data/kwok_live/a4000"))
    ap.add_argument("--coeffs", required=True)
    ap.add_argument("--window", type=float, default=300.0)
    ap.add_argument("--tick", type=float, default=1.0)
    a = ap.parse_args()
    coeffs = next(iter(_load_coeffs(a.coeffs).values()))
    rows = list(csv.DictReader((a.out / "closed_loop.csv").open(newline="", encoding="utf-8")))
    n_ticks = int(a.window / a.tick) + 120
    out_rows: list[dict] = []
    for n in sorted({int(r["nodes"]) for r in rows}):
        sub = [r for r in rows if int(r["nodes"]) == n]
        source = load_map(a.out / f"artifacts_{n}" / "node_profile_map.json")
        order = {nd: i for i, nd in enumerate(source.nodes())}
        tables = {r["node"]: counter_table(source, r["node"], coeffs, n_ticks, a.tick) for r in sub}

        windows = [(b, e) for b in range(0, 120) for e in range(b + 1, n_ticks + 1)
                   if fits(tables, b, e, "offline_increase_j", sub)]
        if len(windows) != 1:
            raise SystemExit(f"N={n}: offline window not unique: {windows}")
        b, e = windows[0]
        for r in sub:
            t, prom = tables[r["node"]], float(r["prometheus_increase_j"])
            cands = sorted((abs(t[e + se] - t[b + sb] - prom), sb, se)
                           for sb in SHIFTS for se in SHIFTS
                           if b + sb >= 0 and e + se <= n_ticks)
            res, sb, se = cands[0]
            out_rows.append({
                "nodes": n, "node": r["node"], "node_index": order[r["node"]],
                "profile": r["profile"], "prom_samples": r["prom_samples"],
                "window_start_tick": b, "window_end_tick": e,
                "shift_start": sb, "shift_end": se,
                "n_fits": sum(1 for c in cands if c[0] <= TOL_J),
                "raw_rel_diff": r["rel_diff"],
                "aligned_abs_diff_j": f"{res:.6g}",
                "aligned_rel_diff": f"{res / prom:.6g}" if prom else "",
            })
        worst = max(float(x["aligned_abs_diff_j"]) for x in out_rows if x["nodes"] == n)
        shifts = sorted({(x["shift_start"], x["shift_end"]) for x in out_rows if x["nodes"] == n})
        print(f"N={n}: window ticks {b}..{e}, shifts {shifts}, worst aligned |diff| {worst:.3g} J")
    with (a.out / "closed_loop_aligned.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
