"""Unit tests for paper figure exports (Phase 17, R4/R5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kwok_energy.cli.export_paper_figures import main as figures_main

pytestmark = pytest.mark.unit


def _write_validation_csv(path: Path) -> None:
    path.write_text(
        "model_name,workload,component,measured_joules,predicted_joules,mae,rmse,mape,source\n"
        "energy_model,cpu,total,100.0,98.0,2.0,2.0,2.0,offline_model\n"
        "energy_model,cpu,package,60.0,59.0,1.0,1.0,1.7,offline_model\n",
        encoding="utf-8",
    )


def _write_scale_csv(path: Path) -> None:
    path.write_text(
        "node_count,series_count,samples_per_second,exporter_cpu_percent,exporter_memory_mb,scrape_failure_rate,duration_s,mode\n"
        "1000,6000,6000.0,1.5,64.0,0.0,1.0,exporter-only\n",
        encoding="utf-8",
    )


def test_export_paper_figures_from_csv(tmp_path: Path):
    validation_csv = tmp_path / "validation.csv"
    scale_csv = tmp_path / "scale.csv"
    out = tmp_path / "figures"
    _write_validation_csv(validation_csv)
    _write_scale_csv(scale_csv)

    code = figures_main([
        "--validation", str(validation_csv),
        "--scale", str(scale_csv),
        "--output-dir", str(out),
    ])

    assert code == 0
    for name in (
        "measured_vs_predicted_total.pdf",
        "component_error_by_workload.pdf",
        "scale_overhead.pdf",
    ):
        payload = (out / name).read_bytes()
        assert payload.startswith(b"%PDF")
        assert b"98.0" in payload or b"6000" in payload or b"2.0" in payload


def test_export_paper_figures_source_has_no_fixture_numeric_literals():
    source = Path("src/kwok_energy/cli/export_paper_figures.py").read_text(encoding="utf-8")

    assert "98.0" not in source
    assert "6000.0" not in source
    assert "64.0" not in source
