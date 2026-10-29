"""Unit tests for the trace core (Phase 2).

Covers the trace-core spec:
- read -> write -> read preserves irregular dt_s, signals, and null DRAM (I3);
- validate enforces interval ordering / non-overlap / dt_s == t_end - t_start;
- derive_watts yields *_watts_mean == joules / dt_s (I2);
- hash_trace is stable across round-trip and identical for identical logical
  content (R1/R5).
"""

from __future__ import annotations

import pytest

from kwok_energy import trace

pytestmark = pytest.mark.unit

# Canonical column order used by trace.write (must match the module).
TRACE_COLS = trace.TRACE_COLUMNS


def _row(node="n0", t_start=0.0, t_end=1.0, dt=1.0, dram_joules=None, **over):
    row = {
        "schema_version": "1",
        "run_id": "r1",
        "workload": "idle",
        "node": node,
        "timestamp": "2025-01-01T00:00:00Z",
        "t_start_s": t_start,
        "t_end_s": t_end,
        "dt_s": dt,
        "platform_id": "xeon-silver+rtx2080ti",
        "cpu_utilization": 0.5,
        "cpu_frequency_mhz": 2400.0,
        "mem_bandwidth_proxy": 10.0,
        "mem_proxy_method": "stream_gb_per_s",
        "gpu_utilization": 0.3,
        "gpu_mem_utilization": 0.2,
        "disk_read_bps": 1000.0,
        "disk_write_bps": 500.0,
        "net_rx_bps": 200.0,
        "net_tx_bps": 100.0,
        "rapl_pkg_joules": 50.0,
        "rapl_dram_joules": dram_joules,
        "gpu_energy_joules": 20.0,
        "rapl_pkg_watts_mean": 50.0 / dt,
        "rapl_dram_watts_mean": None if dram_joules is None else dram_joules / dt,
        "gpu_watts_mean": 20.0 / dt,
        "rapl_pkg_available": True,
        "rapl_dram_available": dram_joules is not None,
        "gpu_available": True,
        "gpu_energy_method": "nvml",
    }
    row.update(over)
    return row


def _irregular_rows():
    return [
        _row(t_start=0.0, t_end=1.0, dt=1.0),
        _row(t_start=1.0, t_end=1.5, dt=0.5),
        _row(t_start=1.5, t_end=3.5, dt=2.0),
    ]


# --- read / write round-trip ------------------------------------------------

def test_round_trip_preserves_irregular_dt_signals_and_null_dram(tmp_path):
    src = tmp_path / "trace.csv"
    rows = _irregular_rows()
    trace.write(rows, src)
    read1 = trace.read(src)
    assert len(read1) == 3
    # irregular dt preserved
    assert [r["dt_s"] for r in read1] == [1.0, 0.5, 2.0]
    # null DRAM preserved as None with availability False
    assert read1[0]["rapl_dram_joules"] is None
    assert read1[0]["rapl_dram_available"] is False

    # write -> read again, must be identical (I3 round-trip)
    dst = tmp_path / "trace2.csv"
    trace.write(read1, dst)
    read2 = trace.read(dst)
    assert read1 == read2


def test_round_trip_preserves_signals(tmp_path):
    src = tmp_path / "trace.csv"
    trace.write([_row(cpu_utilization=0.73, gpu_utilization=0.91)], src)
    rows = trace.read(src)
    assert rows[0]["cpu_utilization"] == pytest.approx(0.73)
    assert rows[0]["gpu_utilization"] == pytest.approx(0.91)


# --- validate (ordering / non-overlap) -------------------------------------

def test_validate_monotonic_passes():
    assert trace.validate(_irregular_rows()) == []


def test_validate_overlapping_fails():
    rows = _irregular_rows()
    rows[1] = _row(t_start=0.8, t_end=1.5, dt=0.7)  # overlaps row 0 [0,1)
    errs = trace.validate(rows)
    assert errs


def test_validate_unordered_fails():
    rows = _irregular_rows()
    rows[0], rows[1] = rows[1], rows[0]
    errs = trace.validate(rows)
    assert errs


def test_validate_dt_equals_t_end_minus_t_start():
    ok = _row(t_start=0.0, t_end=2.0, dt=2.0)
    assert trace.validate([ok]) == []
    bad = _row(t_start=0.0, t_end=2.0, dt=1.0)
    assert trace.validate([bad])


# --- derive_watts (I2) ------------------------------------------------------

def test_derive_watts_correct():
    row = _row(t_start=0.0, t_end=2.0, dt=2.0,
               dram_joules=8.0, rapl_pkg_joules=60.0, gpu_energy_joules=24.0)
    watts = trace.derive_watts(row)
    assert watts["rapl_pkg_watts_mean"] == pytest.approx(60.0 / 2.0)
    assert watts["rapl_dram_watts_mean"] == pytest.approx(8.0 / 2.0)
    assert watts["gpu_watts_mean"] == pytest.approx(24.0 / 2.0)


def test_derive_watts_drops_missing_component():
    row = _row(dt=1.0, dram_joules=None)
    watts = trace.derive_watts(row)
    assert watts["rapl_dram_watts_mean"] is None
    assert watts["rapl_pkg_watts_mean"] == pytest.approx(50.0)


# --- hash_trace (R1/R5) -----------------------------------------------------

def test_hash_stable_across_round_trip(tmp_path):
    src = tmp_path / "a.csv"
    trace.write(_irregular_rows(), src)
    h1 = trace.hash_trace(src)

    dst = tmp_path / "b.csv"
    trace.write(trace.read(src), dst)
    h2 = trace.hash_trace(dst)
    assert h1 == h2


def test_hash_identical_logical_content(tmp_path):
    # Two files encoding the same logical rows (different float spelling) hash alike.
    rows = _irregular_rows()
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    trace.write(rows, a)
    # rebuild b with int-spelled dt then re-coerce via round trip
    trace.write([_row(t_start=0.0, t_end=1, dt=1),
                 _row(t_start=1, t_end=1.5, dt=0.5),
                 _row(t_start=1.5, t_end=3.5, dt=2)], b)
    assert trace.hash_trace(a) == trace.hash_trace(b)


def test_hash_changes_when_content_changes(tmp_path):
    rows = _irregular_rows()
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    trace.write(rows, a)
    rows2 = [dict(r) for r in rows]
    rows2[0]["cpu_utilization"] = 0.99
    trace.write(rows2, b)
    assert trace.hash_trace(a) != trace.hash_trace(b)


def test_hash_is_hex_sha256(tmp_path):
    src = tmp_path / "t.csv"
    trace.write([_row()], src)
    h = trace.hash_trace(src)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)
