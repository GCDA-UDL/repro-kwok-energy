"""Unit tests for calibration (Phase 9, Component D).

Covers: I1 (no target features), held-out refusal, coeffs validate against schema,
R²+MAE metrics, provenance + transform, DRAM availability, determinism (R1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kwok_energy import calibration as cal
from kwok_energy.energy_model import FeatureWindow, predict_power
from kwok_energy.schema import MODEL_INPUT_FIELDS, CoeffsModel

pytestmark = pytest.mark.unit

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "traces"
CAL = FIX / "cal_normal.csv"
HELDOUT = FIX / "heldout_normal.csv"


@pytest.fixture()
def coeffs():
    return cal.fit([CAL], platform_id="xeon-silver+rtx2080ti")


# --- I1: no target features -------------------------------------------------

def test_fit_excludes_target_fields_from_features(coeffs):
    for comp_spec in coeffs.components.values():
        for feat in comp_spec.get("weights", {}):
            assert feat in MODEL_INPUT_FIELDS, f"feature {feat!r} not a model input (I1)"


# --- held-out refusal -------------------------------------------------------

def test_fit_refuses_heldout_path():
    with pytest.raises(ValueError, match="held.out"):
        cal.fit([HELDOUT])


# --- coeffs validate + load -------------------------------------------------

def test_coeffs_validate_against_schema(coeffs):
    payload = {
        "platform_id": coeffs.platform_id,
        "covered_components": sorted(coeffs.covered_components),
        "transform": dict(coeffs.transform),
        "components": {k: dict(v) for k, v in coeffs.components.items()},
        "provenance": dict(coeffs.provenance),
    }
    model = CoeffsModel.model_validate(payload)
    assert model.platform_id == coeffs.platform_id


def test_model_loads_and_predicts(coeffs):
    signals = {
        "cpu_utilization": 0.5, "cpu_frequency_mhz": 2400.0,
        "mem_bandwidth_proxy": 10.0, "mem_proxy_method": "s",
        "gpu_utilization": 0.3, "gpu_mem_utilization": 0.2,
        "disk_read_bps": 0.0, "disk_write_bps": 0.0,
        "net_rx_bps": 0.0, "net_tx_bps": 0.0, "platform_id": "xeon-silver+rtx2080ti",
    }
    pred = predict_power(FeatureWindow(current=signals, history=()), coeffs)
    assert pred.package is not None and pred.package >= 0
    assert pred.gpu is not None and pred.gpu >= 0


# --- metrics ----------------------------------------------------------------

def test_metrics_returns_r2_and_mae(coeffs):
    m = cal.metrics(coeffs, [CAL])
    for comp in coeffs.covered_components:
        assert comp in m
        assert "r2" in m[comp] and "mae" in m[comp]
        # R² should be high on the training data (linear model fits linear data)
        assert m[comp]["r2"] > 0.5, f"{comp} R² too low: {m[comp]['r2']}"


# --- provenance + transform -------------------------------------------------

def test_provenance_and_transform_written(coeffs):
    assert "means" in coeffs.transform
    assert "stds" in coeffs.transform
    assert "features" in coeffs.transform
    prov = coeffs.provenance
    assert "git_sha" in prov
    assert "created_at" in prov
    assert "solver" in prov
    assert "calibration_trace_sha256" in prov


# --- DRAM availability ------------------------------------------------------

def test_dram_fitted_when_available(coeffs):
    # cal_normal.csv has rapl_dram_available=True
    assert "dram" in coeffs.covered_components


def test_dram_absent_when_unavailable():
    # Use missing_dram fixture (DRAM null + flag false)
    coeffs = cal.fit([FIX / "missing_dram.csv"], platform_id="test")
    assert "dram" not in coeffs.covered_components


# --- determinism (R1) -------------------------------------------------------

def test_deterministic_fitting():
    c1 = cal.fit([CAL], platform_id="test")
    c2 = cal.fit([CAL], platform_id="test")
    # Same coefficients (ignoring provenance timestamps)
    for comp in c1.components:
        assert c1.components[comp]["intercept"] == pytest.approx(
            c2.components[comp]["intercept"]
        )
        for feat in c1.components[comp]["weights"]:
            assert c1.components[comp]["weights"][feat] == pytest.approx(
                c2.components[comp]["weights"][feat]
            )
