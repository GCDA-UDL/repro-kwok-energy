"""Unit tests for signal_source + replay (Phase 6a + Phase 7, I4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from kwok_energy.replay import ReplayEngine, load_map
from kwok_energy.schema import DERIVED_TARGET_FIELDS, MEASURED_TARGET_FIELDS
from kwok_energy.signal_source import SignalSource

pytestmark = pytest.mark.unit

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
MAP = FIX / "node_profile_map.json"


@pytest.fixture()
def engine() -> ReplayEngine:
    return load_map(MAP)


# --- SignalSource protocol --------------------------------------------------

def test_replay_engine_is_a_signal_source(engine: ReplayEngine):
    assert isinstance(engine, SignalSource)


def test_non_conforming_object_is_not_signal_source():
    assert not isinstance(42, SignalSource)
    assert not isinstance({"nodes": []}, SignalSource)


def test_signals_at_returns_model_inputs_only(engine: ReplayEngine):
    row = engine.signals_at("n0", 0.5)
    target_fields = MEASURED_TARGET_FIELDS | DERIVED_TARGET_FIELDS
    assert not (set(row) & target_fields), "replay output contains target fields!"


# --- row_at determinism + correctness ---------------------------------------

def test_row_at_deterministic(engine: ReplayEngine):
    a = engine.row_at("n0", 0.5)
    b = engine.row_at("n0", 0.5)
    assert a == b


def test_row_at_returns_correct_interval(engine: ReplayEngine):
    # n0 normal.csv: interval 0 is [0,1), interval 1 is [1,1.5), interval 2 is [1.5,3.5)
    row0 = engine.row_at("n0", 0.0)
    assert float(row0["t_start_s"]) == 0.0
    row1 = engine.row_at("n0", 1.2)  # in [1, 1.5)
    assert float(row1["t_start_s"]) == 1.0
    row2 = engine.row_at("n0", 2.0)  # in [1.5, 3.5)
    assert float(row2["t_start_s"]) == 1.5


def test_irregular_dt_lookup_correct(engine: ReplayEngine):
    # n1 irregular_dt.csv: [0,1) dt=1, [1,1.5) dt=0.5, [1.5,3.5) dt=2
    engine_once = engine  # n1 is "once" mode, speed=2
    row = engine_once.row_at("n1", 0.6)  # logical 0.6 -> t = 0.6*2 = 1.2 -> [1, 1.5)
    assert float(row["t_start_s"]) == 1.0


# --- loop / once modes ------------------------------------------------------

def test_loop_wraps_around(engine: ReplayEngine):
    # n0 normal.csv duration ≈ 3.5, loop mode
    dur = engine.trace_duration("n0")
    row_wrap = engine.row_at("n0", dur + 0.5)  # wraps to 0.5
    row_at_05 = engine.row_at("n0", 0.5)
    assert float(row_wrap["t_start_s"]) == float(row_at_05["t_start_s"])


def test_once_clamps(engine: ReplayEngine):
    dur = engine.trace_duration("n1")
    row_clamped = engine.row_at("n1", dur * 10)  # way past end
    # n1 irregular_dt last interval is [1.5, 3.5); t clamps to min(t, dur) = dur
    # which falls in the last interval (t_start_s=1.5)
    assert float(row_clamped["t_start_s"]) == 1.5
    # clamped (never wraps to the start)
    assert float(row_clamped["t_start_s"]) > 0


# --- speed + cadence independence (I4) --------------------------------------

def test_same_logical_time_two_cadences_same_row(engine: ReplayEngine):
    # Cadence doesn't matter — row_at is pure in logical_time_s
    r1 = engine.row_at("n0", 2.5)
    r2 = engine.row_at("n0", 2.5)
    assert r1 == r2


def test_speed_scales_time(engine: ReplayEngine):
    # n1 speed=2: logical 0.5 -> t = 1.0, which falls in [1, 1.5) for once mode
    row = engine.row_at("n1", 0.5)
    assert float(row["t_start_s"]) == 1.0


# --- trace_duration + errors ------------------------------------------------

def test_trace_duration(engine: ReplayEngine):
    dur = engine.trace_duration("n0")
    assert dur > 0


def test_unknown_node_fails(engine: ReplayEngine):
    with pytest.raises(KeyError):
        engine.row_at("nonexistent", 0.0)
    with pytest.raises(KeyError):
        engine.platform_id("nonexistent")


def test_platform_id(engine: ReplayEngine):
    assert engine.platform_id("n0") == "xeon-silver+rtx2080ti"


def test_nodes_list(engine: ReplayEngine):
    assert sorted(engine.nodes()) == ["n0", "n1"]


# --- parametrized contract test ---------------------------------------------

def test_replay_satisfies_signal_source_contract(engine: ReplayEngine):
    """The contract test: any SignalSource must satisfy these."""
    for node in engine.nodes():
        assert engine.platform_id(node)
        row = engine.signals_at(node, 0.0)
        assert "cpu_utilization" in row  # has model inputs
        # no targets
        assert not (set(row) & (MEASURED_TARGET_FIELDS | DERIVED_TARGET_FIELDS))
