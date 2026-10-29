"""export-paper-tables: regenerate LaTeX tables from result CSVs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _num(value: str, nd: int = 2) -> str:
    """Round a numeric cell for the paper; pass non-numeric values through."""
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return value


def _latex_table(headers: list[str], rows: list[list[str]], caption: str) -> str:
    cols = "l" * len(headers)
    lines = [
        "\\begin{table}",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\toprule",
        " & ".join(headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def export_validation_table(input_csv: str | Path, output_tex: str | Path) -> None:
    rows = _read_rows(input_csv)
    rendered = _latex_table(
        ["Model", "Workload", "Component", "Measured J", "Predicted J", "RMSE", "MAPE"],
        [
            [
                row.get("model_name", ""),
                row.get("workload", ""),
                row.get("component", ""),
                _num(row.get("measured_joules", ""), 1),
                _num(row.get("predicted_joules", ""), 1),
                _num(row.get("rmse", ""), 2),
                _num(row.get("mape", ""), 1),
            ]
            for row in rows
        ],
        "Validation results",
    )
    Path(output_tex).write_text(rendered, encoding="utf-8")


def export_scale_table(input_csv: str | Path, output_tex: str | Path) -> None:
    rows = _read_rows(input_csv)
    rendered = _latex_table(
        ["Nodes", "Series", "Samples/s", "CPU %", "Memory MB", "Failures", "Mode"],
        [
            [
                row.get("node_count", ""),
                row.get("series_count", ""),
                _num(row.get("samples_per_second", ""), 1),
                _num(row.get("exporter_cpu_percent", ""), 1),
                _num(row.get("exporter_memory_mb", ""), 1),
                _num(row.get("scrape_failure_rate", ""), 3),
                row.get("mode", ""),
            ]
            for row in rows
        ],
        "Scalability results",
    )
    Path(output_tex).write_text(rendered, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    validation: str | None = None
    scale: str | None = None
    validation_output: str | None = None
    scale_output: str | None = None

    i = 0
    while i < len(args):
        if args[i] == "--validation" and i + 1 < len(args):
            validation = args[i + 1]
            i += 2
        elif args[i] == "--scale" and i + 1 < len(args):
            scale = args[i + 1]
            i += 2
        elif args[i] == "--validation-output" and i + 1 < len(args):
            validation_output = args[i + 1]
            i += 2
        elif args[i] == "--scale-output" and i + 1 < len(args):
            scale_output = args[i + 1]
            i += 2
        else:
            i += 1

    if not validation or not scale or not validation_output or not scale_output:
        sys.stderr.write(
            "usage: export-paper-tables --validation <csv> --scale <csv> "
            "--validation-output <tex> --scale-output <tex>\n"
        )
        return 2

    export_validation_table(validation, validation_output)
    export_scale_table(scale, scale_output)
    return 0


__all__ = ["export_scale_table", "export_validation_table", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
