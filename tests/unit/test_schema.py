"""Unit tests for the signal / trace / platform schema (Phase 1, Component A).

Covers the signal-schema spec: SignalRow validation (I1/I3), utilization and
throughput ranges, missing-DRAM-null + flag, joules-canonical watts reconstruction
(I2), field-group disjointness + leakage rejection (I1), platform config
validation, and trace-file validation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from kwok_energy import schema
from kwok_energy.tolerances import WATTS_RECON

pytestmark = pytest.mark.unit


# --- row builders -----------------------------------------------------------

def signal_row(**over):
    row = {
        "schema_version": "1",
        "run_id": "r1",
        "workload": "idle",
        "node": "n0",
        "timestamp": "2025-01-01T00:00:00Z",
        "t_start_s": 0.0,
        "t_end_s": 1.0,
        "dt_s": 1.0,
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
        "platform_id": "xeon-silver+rtx2080ti",
    }
    row.update(over)
    return row


def trace_row(**over):
    row = signal_row()
    row.update(
        {
            "rapl_pkg_joules": 50.0,
            "rapl_dram_joules": None,
            "gpu_energy_joules": 20.0,
            "rapl_pkg_watts_mean": 50.0,
            "rapl_dram_watts_mean": None,
            "gpu_watts_mean": 20.0,
            "rapl_pkg_available": True,
            "rapl_dram_available": False,
            "gpu_available": True,
            "gpu_energy_method": "nvml",
        }
    )
    row.update(over)
    return row


# --- validate_signal_row ----------------------------------------------------

def test_valid_signal_row_passes():
    assert schema.validate_signal_row(signal_row()) == []


def test_missing_schema_version_fails():
    row = signal_row()
    del row["schema_version"]
    assert schema.validate_signal_row(row)


def test_non_positive_dt_fails():
    assert schema.validate_signal_row(signal_row(dt_s=0.0))
    assert schema.validate_signal_row(signal_row(dt_s=-1.0, t_end_s=-1.0))


def test_inverted_interval_fails():
    assert schema.validate_signal_row(signal_row(t_start_s=1.0, t_end_s=0.5, dt_s=-0.5))


def test_dt_mismatch_fails():
    assert schema.validate_signal_row(signal_row(dt_s=2.0))  # 2.0 != 1.0


def test_utilization_out_of_range_fails():
    assert schema.validate_signal_row(signal_row(cpu_utilization=1.5))
    assert schema.validate_signal_row(signal_row(cpu_utilization=-0.1))
    assert schema.validate_signal_row(signal_row(gpu_utilization=1.01))
    assert schema.validate_signal_row(signal_row(gpu_mem_utilization=-0.2))


def test_negative_throughput_fails():
    assert schema.validate_signal_row(signal_row(cpu_frequency_mhz=-1.0))
    assert schema.validate_signal_row(signal_row(disk_read_bps=-1.0))
    assert schema.validate_signal_row(signal_row(mem_bandwidth_proxy=-1.0))
    assert schema.validate_signal_row(signal_row(net_rx_bps=-1.0))


# --- leakage (I1) -----------------------------------------------------------

def test_target_named_field_rejected_in_signal_row():
    row = signal_row()
    row["rapl_pkg_joules"] = 100.0  # leakage
    errs = schema.validate_signal_row(row)
    assert errs and any("rapl_pkg_joules" in e or "leak" in e.lower() for e in errs)


def test_is_leakage_field_detects_targets():
    assert schema.is_leakage_field("rapl_pkg_joules")
    assert schema.is_leakage_field("gpu_power_draw")
    assert schema.is_leakage_field("node_energy_total")
    assert not schema.is_leakage_field("cpu_utilization")
    assert not schema.is_leakage_field("platform_id")


def test_field_groups_are_disjoint():
    inp = schema.MODEL_INPUT_FIELDS
    measured = schema.MEASURED_TARGET_FIELDS
    derived = schema.DERIVED_TARGET_FIELDS
    assert not (inp & measured)
    assert not (inp & derived)
    # availability and identity also disjoint from inputs
    assert not (inp & schema.AVAILABILITY_FIELDS)
    assert not (inp & schema.IDENTITY_FIELDS)


def test_all_field_groups_nonempty_and_named():
    for grp in (
        schema.IDENTITY_FIELDS,
        schema.MODEL_INPUT_FIELDS,
        schema.MEASURED_TARGET_FIELDS,
        schema.DERIVED_TARGET_FIELDS,
        schema.AVAILABILITY_FIELDS,
    ):
        assert grp, "field group is empty"


# --- validate_trace_row (I2 + missing DRAM) --------------------------------

def test_valid_trace_row_passes():
    assert schema.validate_trace_row(trace_row()) == []


def test_missing_dram_stays_null_with_flag():
    # available=False with joules=None is valid
    assert schema.validate_trace_row(trace_row(rapl_dram_joules=None,
                                               rapl_dram_available=False,
                                               rapl_dram_watts_mean=None)) == []
    # using 0 as a sentinel for absence is rejected
    errs = schema.validate_trace_row(trace_row(rapl_dram_joules=0.0,
                                               rapl_dram_available=False,
                                               rapl_dram_watts_mean=0.0))
    assert errs


def test_watts_reconstructed_from_joules_within_tolerance():
    # consistent
    assert schema.validate_trace_row(trace_row(rapl_pkg_joules=50.0,
                                               rapl_pkg_watts_mean=50.0,
                                               dt_s=1.0)) == []
    # inconsistent beyond WATTS_RECON
    errs = schema.validate_trace_row(trace_row(rapl_pkg_joules=50.0,
                                               rapl_pkg_watts_mean=50.0 + WATTS_RECON + 1.0,
                                               dt_s=1.0))
    assert errs


def test_trace_row_missing_schema_version_fails():
    row = trace_row()
    del row["schema_version"]
    assert schema.validate_trace_row(row)


# --- validate_platform_config ----------------------------------------------

def _valid_platform_cfg():
    return {
        "cpu": {"model": "Xeon Silver", "tdp_watts": 120.0,
                "idle_watts_estimate": 25.0, "sockets": 1},
        "gpu": {"present": True, "model": "RTX 2080 Ti", "tdp_watts": 250.0,
                "idle_watts_estimate": 18.0},
        "rapl": {"has_dram": True},
        "sampling_interval_seconds": 0.5,
    }


def test_valid_platform_config_passes():
    assert schema.validate_platform_config(_valid_platform_cfg()) == []


def test_platform_config_missing_tdp_watts_fails():
    cfg = _valid_platform_cfg()
    del cfg["cpu"]["tdp_watts"]
    assert schema.validate_platform_config(cfg)


def test_platform_config_missing_cpu_section_fails():
    cfg = _valid_platform_cfg()
    del cfg["cpu"]
    assert schema.validate_platform_config(cfg)


# --- validate_trace_file ----------------------------------------------------

def _write_trace_csv(path: Path, rows: list[dict]) -> None:
    cols = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_validate_trace_file_returns_per_row_errors(tmp_path):
    p = tmp_path / "bad.csv"
    rows = [trace_row(node="n0"), trace_row(node="n1", dt_s=2.0)]  # row 2 invalid
    _write_trace_csv(p, rows)
    results = schema.validate_trace_file(p)
    assert results  # non-empty
    # identifies the offending row
    assert any(r.index == 1 for r in results)


def test_validate_trace_file_valid_returns_empty(tmp_path):
    p = tmp_path / "good.csv"
    _write_trace_csv(p, [trace_row(node="n0"), trace_row(node="n1",
                                                          t_start_s=1.0,
                                                          t_end_s=2.0,
                                                          dt_s=1.0)])
    assert schema.validate_trace_file(p) == []


def test_json_schemas_generated_and_consistent():
    signals = Path(schema.__file__).parent / "schemas" / "signals.json"
    platform = Path(schema.__file__).parent / "schemas" / "platform.json"
    assert signals.is_file() and platform.is_file()
    import json
    sj = json.loads(signals.read_text())
    # signal schema references the model input fields
    props = sj.get("properties", sj)
    assert "cpu_utilization" in props
