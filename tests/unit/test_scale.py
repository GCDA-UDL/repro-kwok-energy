"""Unit tests for scale demo (Phase 16, Component I)."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import pytest

from kwok_energy.cli.scale_demo import main as scale_demo_main
from kwok_energy.scale import (
    ScaleRow,
    expected_series_count,
    process_resource_usage,
    run,
    scrape_failure_rate,
    write_csv,
)

pytestmark = pytest.mark.unit


def test_scale_row_has_required_columns():
    row = ScaleRow(
        node_count=10,
        series_count=60,
        samples_per_second=120.0,
        exporter_cpu_percent=0.0,
        exporter_memory_mb=0.0,
        scrape_failure_rate=0.0,
        duration_s=1.0,
        mode="exporter-only",
    )

    assert set(row.to_dict()) == {
        "node_count",
        "series_count",
        "samples_per_second",
        "exporter_cpu_percent",
        "exporter_memory_mb",
        "scrape_failure_rate",
        "duration_s",
        "mode",
    }


def test_expected_series_count_for_1000_nodes():
    assert expected_series_count(1000) == 1000 * 6


def test_scrape_failure_rate():
    assert scrape_failure_rate(attempted=10, failed=2) == 0.2
    assert scrape_failure_rate(attempted=0, failed=0) == 0.0


def test_exporter_only_run_does_not_require_kwok():
    row = run(node_count=1000, mode="exporter-only", duration_s=1.0, scrape_attempts=4, scrape_failures=1)

    assert row.node_count == 1000
    assert row.mode == "exporter-only"
    assert row.series_count == expected_series_count(1000)
    assert row.scrape_failure_rate == 0.25


def test_scale_csv_is_deterministic(tmp_path: Path):
    rows = [run(node_count=1, mode="exporter-only", duration_s=1.0)]
    out = tmp_path / "scale.csv"

    write_csv(rows, out)
    first = out.read_bytes()
    write_csv(rows, out)
    second = out.read_bytes()

    assert first == second
    parsed = list(csv.DictReader(out.read_text().splitlines()))
    assert parsed[0]["mode"] == "exporter-only"


def test_scale_demo_cli_writes_csv(tmp_path: Path):
    out = tmp_path / "scale.csv"

    code = scale_demo_main([
        "--nodes", "10",
        "--mode", "exporter-only",
        "--duration", "1.0",
        "--output", str(out),
    ])

    assert code == 0
    parsed = list(csv.DictReader(out.read_text().splitlines()))
    assert parsed[0]["node_count"] == "10"


def test_process_resource_usage_reports_current_process():
    usage = process_resource_usage(os.getpid())

    assert usage.cpu_percent >= 0.0
    assert usage.memory_mb > 0.0
