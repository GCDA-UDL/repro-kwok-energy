"""validate: compare predicted vs measured energy on held-out workloads."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from .. import calibration as cal
from .. import validation as val


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) < 3:
        sys.stderr.write(
            "usage: validate --coeffs <coeffs.json> --heldout <trace.csv,...> --output <results.csv>\n"
        )
        return 2

    coeffs_path: str | None = None
    heldout_paths: list[str] = []
    output_path: str | None = None
    i = 0
    while i < len(args):
        if args[i] == "--coeffs" and i + 1 < len(args):
            coeffs_path = args[i + 1]
            i += 2
        elif args[i] == "--heldout" and i + 1 < len(args):
            heldout_paths = [p.strip() for p in args[i + 1].split(",")]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output_path = args[i + 1]
            i += 2
        else:
            i += 1

    if not coeffs_path or not heldout_paths or not output_path:
        sys.stderr.write("validate: --coeffs, --heldout, and --output are required\n")
        return 2

    import json
    data = json.loads(Path(coeffs_path).read_text())
    coeffs = cal.Coeffs(  # type: ignore[attr-defined]
        platform_id=data["platform_id"],
        covered_components=frozenset(data["covered_components"]),
        transform=data["transform"],
        components=data["components"],
        provenance=data.get("provenance", {}),
    )
    rows = val.validate(coeffs, heldout_paths)

    # Write CSV
    fieldnames = [
        "model_name", "workload", "platform_id", "component", "n_runs",
        "measured_joules", "predicted_joules", "mae", "rmse", "mape",
        "mape_masked", "mape_coverage", "smape", "cancellation_flag",
        "source", "coeffs_git_sha",
    ]
    with Path(output_path).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: getattr(r, k, "") for k in fieldnames})

    sys.stderr.write(f"validate: wrote {len(rows)} rows to {output_path}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
