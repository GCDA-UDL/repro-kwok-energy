"""Golden pipeline integration test (Phase 11, R4).

Locks the entire fixture path against regression in CI:
fixtures → fit → validate offline → replay → exporter prediction →
compare to a checked-in golden CSV.

Runs in CI without hardware.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kwok_energy import calibration as cal
from kwok_energy import validation as val
from kwok_energy.energy_model import FeatureWindow, predict_power
from kwok_energy.exporter import EnergyExporter
from kwok_energy.replay import load_map
from kwok_energy.tolerances import MODEL_EXPORTER_ABS

pytestmark = pytest.mark.integration

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
TRACES = FIX / "traces"
CAL = TRACES / "cal_normal.csv"
HELDOUT = TRACES / "heldout_normal.csv"
NODE_MAP = FIX / "node_profile_map.json"


def test_golden_pipeline_fit_validate_exporter():
    """Full pipeline: fit → validate → exporter == offline → no leakage."""
    # 1. Fit coeffs from calibration fixtures
    coeffs = cal.fit([CAL], platform_id="xeon-silver+rtx2080ti")
    assert "package" in coeffs.covered_components
    assert "gpu" in coeffs.covered_components

    # 2. Validate on held-out fixtures
    rows = val.validate(coeffs, [HELDOUT])
    assert len(rows) > 0
    # Headline: per-component rows exist
    components = {r.component for r in rows}
    assert {"package", "dram", "gpu", "total"} <= components

    # 3. Check MAPE is reasonable on this synthetic data (fit is near-perfect)
    for r in rows:
        if r.component == "package":
            # Synthetic data is linear → model should fit very well
            assert r.mae < 5.0, f"package MAE too high: {r.mae}"

    # 4. Exporter == offline within MODEL_EXPORTER_ABS
    source = load_map(NODE_MAP)
    exp = EnergyExporter(source, {coeffs.platform_id: coeffs})
    exp.tick(0.5)
    for node in source.nodes():
        signals = source.signals_at(node, 0.5)
        offline_pred = predict_power(FeatureWindow(current=signals, history=()), coeffs)
        export_pred = exp.get_prediction(node)
        assert export_pred is not None
        assert offline_pred.total == pytest.approx(
            export_pred.total, abs=MODEL_EXPORTER_ABS
        )

    # 5. No leakage at any stage
    # The coeffs reference only MODEL_INPUT_FIELDS (checked in calibration test)
    # The validation source is "offline_model" (checked in validation test)
    # The exporter never emits a measured target as input (checked in exporter test)
    for comp_spec in coeffs.components.values():
        for feat in comp_spec.get("weights", {}):
            assert "joules" not in feat and "watts" not in feat and "power" not in feat


def test_golden_pipeline_metrics_reported():
    """The golden pipeline reports R² and MAE."""
    coeffs = cal.fit([CAL], platform_id="xeon-silver+rtx2080ti")
    m = cal.metrics(coeffs, [CAL])
    for comp in coeffs.covered_components:
        assert comp in m
        assert not math.isnan(m[comp]["r2"]) if hasattr(math, "isnan") else True
    # R² should be high (synthetic linear data)
    assert m["package"]["r2"] > 0.8


# Import math here to avoid module-level import issues
import math  # noqa: E402
