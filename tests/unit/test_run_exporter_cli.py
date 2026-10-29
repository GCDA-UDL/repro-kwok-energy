"""Unit tests for run-exporter CLI (cluster-integration change)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kwok_energy.cli.run_exporter import build_exporter
from kwok_energy.exporter import EnergyExporter

pytestmark = pytest.mark.unit

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
MAP = FIX / "node_profile_map.json"
CAL = FIX / "traces" / "cal_normal.csv"


def _write_coeffs(path: Path) -> None:
    from kwok_energy.calibration import fit

    coeffs = fit([str(CAL)], platform_id="xeon-silver+rtx2080ti")
    payload = {
        "platform_id": coeffs.platform_id,
        "covered_components": sorted(coeffs.covered_components),
        "transform": dict(coeffs.transform),
        "components": {k: dict(v) for k, v in coeffs.components.items()},
        "provenance": dict(coeffs.provenance),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_build_exporter_returns_energy_exporter(tmp_path: Path):
    coeffs_path = tmp_path / "coeffs.json"
    _write_coeffs(coeffs_path)

    exp = build_exporter(map_path=MAP, coeffs_path=coeffs_path)

    assert isinstance(exp, EnergyExporter)
    assert len(exp._source.nodes()) == 2


def test_build_exporter_ticks_and_serves_metrics(tmp_path: Path):
    coeffs_path = tmp_path / "coeffs.json"
    _write_coeffs(coeffs_path)

    exp = build_exporter(map_path=MAP, coeffs_path=coeffs_path)
    exp.tick(0.0)
    exp.tick(1.0)

    mt = exp.metrics_text()
    assert b"node_power_watts" in mt
    assert b"node_energy_joules_total" in mt
    assert exp.cluster_power() > 0
    assert exp.cluster_energy() > 0


def test_build_exporter_rejects_unknown_platform(tmp_path: Path):
    coeffs_path = tmp_path / "coeffs.json"
    _write_coeffs(coeffs_path)

    exp = build_exporter(map_path=MAP, coeffs_path=coeffs_path, platform_override="wrong")
    # The exporter rejects unknown platforms at tick time (lazy validation by design)
    from kwok_energy.exporter import UnknownPlatformError

    with pytest.raises(UnknownPlatformError):
        exp.tick(0.0)
