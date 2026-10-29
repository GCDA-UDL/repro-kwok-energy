"""scale-demo: run exporter-only scale accounting and write ScaleRows."""

from __future__ import annotations

import sys

from ..scale import measure_prediction_overhead, run, write_csv


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    node_count: int | None = None
    mode = "exporter-only"
    duration_s = 1.0
    output: str | None = None
    coeffs_path: str | None = None

    i = 0
    while i < len(args):
        if args[i] == "--nodes" and i + 1 < len(args):
            node_count = int(args[i + 1])
            i += 2
        elif args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        elif args[i] == "--duration" and i + 1 < len(args):
            duration_s = float(args[i + 1])
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        elif args[i] == "--coeffs" and i + 1 < len(args):
            coeffs_path = args[i + 1]
            i += 2
        else:
            i += 1

    if node_count is None or output is None:
        sys.stderr.write(
            "usage: scale-demo --nodes <count> --mode exporter-only "
            "--duration <seconds> --output <scale.csv> [--coeffs <coeffs.json>]\n"
        )
        return 2

    # With --coeffs, measure REAL exporter-core overhead (CPU%/RSS) for N nodes;
    # otherwise report the metric-surface accounting only (overhead left at 0).
    cpu_pct, mem_mb = 0.0, 0.0
    if coeffs_path:
        import json
        from pathlib import Path

        from ..calibration import Coeffs  # type: ignore[attr-defined]
        d = json.loads(Path(coeffs_path).read_text())
        coeffs = Coeffs(
            platform_id=d["platform_id"],
            covered_components=frozenset(d["covered_components"]),
            transform=d["transform"],
            components=d["components"],
            provenance=d.get("provenance", {}),
        )
        usage = measure_prediction_overhead(node_count, coeffs, duration_s=min(duration_s, 3.0))
        cpu_pct, mem_mb = usage.cpu_percent, usage.memory_mb

    row = run(node_count=node_count, mode=mode, duration_s=duration_s,
              exporter_cpu_percent=cpu_pct, exporter_memory_mb=mem_mb)
    write_csv([row], output)
    sys.stderr.write(f"scale-demo: wrote 1 row to {output}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
