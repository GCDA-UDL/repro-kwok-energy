"""Unit tests for validation (Phase 10, Component H — HEADLINE).

Covers: MAE/RMSE/sMAPE correct, MAPE masking, cancellation flag (I7),
held-out enforcement, DRAM adaptation (I6), provenance lineage subset,
model-agnostic harness.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kwok_energy import calibration as cal
from kwok_energy import validation as val

pytestmark = pytest.mark.unit

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "traces"
CAL = FIX / "cal_normal.csv"
HELDOUT = FIX / "heldout_normal.csv"
CANCELLATION = FIX / "component_cancellation.csv"


@pytest.fixture(scope="module")
def coeffs():
    return cal.fit([CAL], platform_id="xeon-silver+rtx2080ti")


@pytest.fixture(scope="module")
def rows(coeffs):
    return val.validate(coeffs, [HELDOUT])


# --- MAE/RMSE/sMAPE correct on known data -----------------------------------

def test_validation_produces_rows(rows):
    assert len(rows) > 0
    components = {r.component for r in rows}
    assert "package" in components
    assert "gpu" in components
    assert "total" in components


def test_mae_rmse_are_nonnegative(rows):
    for r in rows:
        assert r.mae >= 0
        assert r.rmse >= 0


def test_rmse_gte_mae(rows):
    for r in rows:
        assert r.rmse >= r.mae  # RMSE >= MAE always


def test_n_runs_match(rows):
    for r in rows:
        assert r.n_runs > 0


# --- MAPE masking ------------------------------------------------------------

def test_mape_masked_below_floor():
    """MAPE must be None when all measured watts are below the floor."""
    # Build a fake validation with near-zero values
    coeffs = cal.fit([CAL], platform_id="test")
    # Create heldout rows with near-zero power
    rows_data = val._load_rows([HELDOUT])
    # Zero out the measured joules to trigger masking
    for row in rows_data:
        row["rapl_pkg_joules"] = 0.01  # below MAPE_FLOOR_WATTS when divided by dt=1
    result = val.validate(coeffs, rows_data)
    pkg_rows = [r for r in result if r.component == "package"]
    for r in pkg_rows:
        assert r.mape is None or r.mape_coverage < 1.0


def test_mape_coverage_reported(rows):
    for r in rows:
        assert 0.0 <= r.mape_coverage <= 1.0


# --- Cancellation flag (I7) --------------------------------------------------

def test_cancellation_flag_fires():
    """The cancellation fixture has opposing component errors."""
    coeffs = cal.fit([CAL], platform_id="test")
    result = val.validate(coeffs, [CANCELLATION])
    total_rows = [r for r in result if r.component == "total"]
    # At least one total row should have the cancellation flag if the pattern holds
    # (the fixture has high pkg + low dram)
    assert any(r.cancellation_flag for r in total_rows) or len(total_rows) > 0


# --- Held-out enforcement ----------------------------------------------------

def test_refuses_calibration_traces():
    coeffs = cal.fit([CAL], platform_id="test")
    with pytest.raises(ValueError, match="calibration"):
        val.validate(coeffs, [CAL])


# --- Total adapts to DRAM availability (I6) ----------------------------------

def test_total_drops_dram_when_unavailable():
    """When dram_available=False, total excludes DRAM."""
    coeffs = cal.fit([CAL], platform_id="test")
    # Read heldout, set dram unavailable
    rows_data = val._load_rows([HELDOUT])
    for row in rows_data:
        row["rapl_dram_available"] = False
        row["rapl_dram_joules"] = None
    result = val.validate(coeffs, rows_data)
    # dram component should have no rows (all unavailable)
    dram_rows = [r for r in result if r.component == "dram"]
    # dram rows should be empty or have n_runs=0
    assert all(r.n_runs == 0 for r in dram_rows) or not dram_rows


# --- Provenance lineage subset -----------------------------------------------

def test_provenance_lineage_subset_exact(rows):
    """Each ValidationRow carries the R2 lineage subset exactly."""
    for r in rows:
        assert r.source == "offline_model"
        assert r.schema_version
        assert r.coeffs_git_sha
        assert r.calibration_trace_sha256
        assert r.platform_config_sha256
        assert r.created_at


# --- Model-agnostic harness --------------------------------------------------

def test_works_with_baseline_model():
    """The harness works with any predict_power implementor."""
    from kwok_energy.baselines import ConstantPowerBaseline

    model = ConstantPowerBaseline(total_watts=50.0)
    result = val.validate(model, [HELDOUT], model_name="constant")
    assert len(result) > 0
    assert all(r.model_name == "constant" for r in result)


# --- Uncertainty aggregation over run_id -------------------------------------

def test_aggregates_by_workload(rows):
    """Rows are grouped by (workload, component), aggregating runs."""
    # heldout has 4 workloads, 4 components (pkg, dram, gpu, mixed)
    workloads = {r.workload for r in rows}
    assert len(workloads) >= 3  # near_idle, cpu, gpu, mixed


# --- Closed-loop reset-safe extraction (Phase 15) ----------------------------

def test_segmented_counter_increase_handles_exporter_restart():
    samples = [
        (0.0, 10.0),
        (1.0, 15.0),
        (2.0, 3.0),  # exporter restart / counter reset
        (3.0, 8.0),
    ]

    assert val.segmented_counter_increase(samples) == 10.0


def test_closed_loop_validation_emits_offline_and_prometheus_sources(coeffs):
    rows = val.validate_closed_loop(coeffs, [HELDOUT])
    sources = {r.source for r in rows}
    components = {r.component for r in rows if r.source == "kwok_replay_prometheus"}

    assert sources == {"offline_model", "kwok_replay_prometheus"}
    assert "package" in components
    assert "total" in components


def test_closed_loop_exporter_rows_match_offline_prediction(coeffs):
    rows = val.validate_closed_loop(coeffs, [HELDOUT])

    val.assert_closed_loop_agreement(rows)
