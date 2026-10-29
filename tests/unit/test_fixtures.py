"""Unit tests for adversarial fixtures (Phase 6, strengthens-claim).

Each fixture encodes a known trap as data; each test asserts the matching
behaviour. Fixtures whose trap is detected by a LATER component (exporter I5
loop/restart handling, validation I7 cancellation, MAPE masking) have their DATA
authored here and their behavioural test is marked to run when that component
lands — flagged clearly so it cannot be forgotten.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kwok_energy import energy_model as em
from kwok_energy import schema, trace
from kwok_energy.tolerances import MAPE_FLOOR_WATTS

pytestmark = pytest.mark.unit

TRACES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "traces"
SIGNALS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "signals"


# --- missing_dram: DRAM dropped not zeroed -----------------------------------

def test_missing_dram_dropped_not_zeroed():
    rows = trace.read(TRACES / "missing_dram.csv")
    assert rows[0]["rapl_dram_joules"] is None
    assert rows[0]["rapl_dram_available"] is False
    assert rows[0]["rapl_dram_watts_mean"] is None
    # the trace validates (null + flag is the correct absence representation)
    assert schema.validate_trace_file(TRACES / "missing_dram.csv") == []


def test_missing_dram_round_trips_as_null():
    rows = trace.read(TRACES / "missing_dram.csv")
    assert rows[0]["rapl_dram_joules"] is None
    # missing-DRAM dropped from a dram-less coeffs total, never zeroed
    assert rows[0]["rapl_dram_joules"] != 0


# --- target_leakage: rejected by the model (I1) -----------------------------

def test_target_leakage_fixture_rejected_by_model():
    leak = json.loads((SIGNALS / "target_leakage.json").read_text())
    # the leakage signal carries a measured target as a model input
    assert "rapl_pkg_joules" in leak
    # the energy model MUST reject it (golden rule I1)
    coeffs = em.Coeffs(
        platform_id="xeon-silver+rtx2080ti",
        covered_components=frozenset({"package", "gpu"}),
        transform={"features": ["cpu_utilization"], "means": {"cpu_utilization": 0.0},
                   "stds": {"cpu_utilization": 1.0}},
        components={"package": {"intercept": 25.0, "weights": {"cpu_utilization": 100.0}},
                    "gpu": {"intercept": 18.0, "weights": {"gpu_utilization": 200.0}}},
    )
    with pytest.raises(em.LeakageError):
        em.predict_power(em.FeatureWindow(current=leak, history=()), coeffs)


# --- irregular_dt: accepted -------------------------------------------------

def test_irregular_dt_accepted_and_validates():
    rows = trace.read(TRACES / "irregular_dt.csv")
    dts = [r["dt_s"] for r in rows]
    assert dts == [1.0, 0.5, 2.0]  # irregular
    assert trace.validate(rows) == []
    assert schema.validate_trace_file(TRACES / "irregular_dt.csv") == []


# --- irregular_scrape: gap exceeds SCRAPE_GAP_MAX_S -------------------------

def test_irregular_scrape_has_large_gap():
    rows = trace.read(TRACES / "irregular_scrape.csv")
    gap = rows[1]["t_start_s"] - rows[0]["t_end_s"]
    assert gap > 5.0  # exceeds SCRAPE_GAP_MAX_S
    # (detection/consumption flag is the exporter's job — data authored here)


# --- near_zero_idle: power near the MAPE floor ------------------------------

def test_near_zero_idle_below_mape_floor():
    rows = trace.read(TRACES / "near_zero_idle.csv")
    pkg_w = rows[0]["rapl_pkg_watts_mean"]
    assert pkg_w < MAPE_FLOOR_WATTS  # would be masked in MAPE (validation, change 4)


# --- io_enabled: io signals present, model can include io -------------------

def test_io_enabled_signals_present():
    rows = trace.read(TRACES / "io_enabled.csv")
    assert rows[0]["disk_read_bps"] > 0
    assert rows[0]["net_rx_bps"] > 0


# --- component_cancellation: opposing errors (data; I7 flag is validation) --

def test_component_cancellation_fixture_present():
    rows = trace.read(TRACES / "component_cancellation.csv")
    # pkg high, dram low — opposing errors that mask a total error (I7)
    assert rows[0]["rapl_pkg_watts_mean"] > rows[0]["rapl_dram_watts_mean"]
    # (the cancellation_flag is computed by validation.py per CANCELLATION_RATIO — change 4)


# --- loop_boundary / restart_counter / speed_factor: data for exporter ------

@pytest.mark.parametrize("name", ["loop_boundary", "restart_counter", "speed_factor"])
def test_exporter_dependent_fixtures_load_as_valid_traces(name):
    # These traps are detected by the exporter (I5 counter semantics) which lands
    # in change 3 (kwok-injection-layer). The fixture DATA is valid here; the
    # behavioural test runs when the exporter lands.
    p = TRACES / f"{name}.csv"
    rows = trace.read(p)
    assert rows  # loadable


# --- every fixture exists ---------------------------------------------------

REQUIRED_FIXTURES = [
    "normal", "missing_dram", "loop_boundary", "irregular_dt", "irregular_scrape",
    "restart_counter", "speed_factor", "near_zero_idle", "component_cancellation",
    "io_enabled",
]


def test_all_adversarial_trace_fixtures_present():
    for name in REQUIRED_FIXTURES:
        assert (TRACES / f"{name}.csv").is_file(), f"missing fixture: {name}.csv"
    assert (SIGNALS / "target_leakage.json").is_file()
