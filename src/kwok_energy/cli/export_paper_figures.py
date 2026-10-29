"""export-paper-figures: regenerate PDF figure artifacts from result CSVs.

This module intentionally keeps the fixture path dependency-free: the generated
PDFs are simple, valid-enough text PDF artifacts whose content is derived from
CSV rows. Publication-quality styling can replace this backend later without
changing the CSV-driven contract.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _pdf_bytes(title: str, lines: list[str]) -> bytes:
    text = title + "\n" + "\n".join(lines)
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 10 Tf 50 750 Td ({escaped}) Tj ET"
    objects = [
        "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        "3 0 obj << /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >> endobj",
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        f"5 0 obj << /Length {len(stream.encode('utf-8'))} >> stream\n{stream}\nendstream endobj",
    ]
    payload = "%PDF-1.4\n% generated from CSV inputs\n" + "\n".join(objects) + "\n%%EOF\n"
    return payload.encode("utf-8")


def export_figures(validation_csv: str | Path, scale_csv: str | Path, output_dir: str | Path) -> None:
    validation_rows = _read_rows(validation_csv)
    scale_rows = _read_rows(scale_csv)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total_lines = [
        f"{r.get('workload', '')}: measured={r.get('measured_joules', '')} predicted={r.get('predicted_joules', '')}"
        for r in validation_rows
        if r.get("component") == "total"
    ]
    component_lines = [
        f"{r.get('workload', '')}/{r.get('component', '')}: rmse={r.get('rmse', '')} mape={r.get('mape', '')}"
        for r in validation_rows
    ]
    scale_lines = [
        f"nodes={r.get('node_count', '')} series={r.get('series_count', '')} samples={r.get('samples_per_second', '')} memory={r.get('exporter_memory_mb', '')}"
        for r in scale_rows
    ]

    (out / "measured_vs_predicted_total.pdf").write_bytes(
        _pdf_bytes("Measured vs predicted total", total_lines)
    )
    (out / "component_error_by_workload.pdf").write_bytes(
        _pdf_bytes("Component error by workload", component_lines)
    )
    (out / "scale_overhead.pdf").write_bytes(_pdf_bytes("Scale overhead", scale_lines))


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    validation: str | None = None
    scale: str | None = None
    output_dir: str | None = None

    i = 0
    while i < len(args):
        if args[i] == "--validation" and i + 1 < len(args):
            validation = args[i + 1]
            i += 2
        elif args[i] == "--scale" and i + 1 < len(args):
            scale = args[i + 1]
            i += 2
        elif args[i] == "--output-dir" and i + 1 < len(args):
            output_dir = args[i + 1]
            i += 2
        else:
            i += 1

    if not validation or not scale or not output_dir:
        sys.stderr.write(
            "usage: export-paper-figures --validation <csv> --scale <csv> --output-dir <dir>\n"
        )
        return 2

    export_figures(validation, scale, output_dir)
    return 0


__all__ = ["export_figures", "main"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
