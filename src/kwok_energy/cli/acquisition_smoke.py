"""CLI entry point for the neutral acquisition smoke gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from kwok_energy.acquisition import (
    append_raw_metrics_csv,
    append_raw_metrics_jsonl,
    check_gpu_backend,
    check_rapl_reader,
    metrics_from_results,
    run_groups,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acquisition-smoke",
        description=(
            "Run vendor-agnostic CPU/GPU acquisition smoke checks through framework "
            "collectors and write neutral raw CSV/JSONL metrics. Use pytest markers "
            "hardware, kwok, and prometheus for the full test gates."
        ),
    )
    parser.add_argument("--csv", type=Path, default=None, help="Append raw metrics to CSV")
    parser.add_argument("--jsonl", type=Path, default=None, help="Append raw metrics to JSONL")
    parser.add_argument(
        "--group",
        choices=("all", "cpu-energy", "gpu-energy"),
        default="all",
        help="Source group to run; KWOK/Prometheus live gates run via pytest markers",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    groups = []
    if args.group in {"all", "cpu-energy"}:
        groups.append(check_rapl_reader)
    if args.group in {"all", "gpu-energy"}:
        groups.append(check_gpu_backend)
    results = run_groups(groups)
    metrics = metrics_from_results(results)
    if args.csv is not None:
        append_raw_metrics_csv(args.csv, metrics)
    if args.jsonl is not None:
        append_raw_metrics_jsonl(args.jsonl, metrics)
    for metric in metrics:
        status = "OK" if metric.ok else "SKIP/FAIL"
        print(f"[{status}] {metric.source}:{metric.check_name} {metric.value} {metric.unit} - {metric.message}")
    return 0 if all(r.ok or r.skipped for r in results) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
