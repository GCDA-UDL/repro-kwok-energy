"""Tests for the acquisition/plumbing smoke gate.

These are hardware-free contract tests for the runner/writer. Live acquisition groups
are separately marked hardware/kwok; this file proves the suite uses framework
boundaries, writes neutral raw metrics, and keeps skip semantics explicit.
"""

from __future__ import annotations

import csv
import json

import pytest

from kwok_energy import schema
from kwok_energy.acquisition import (
    RawAcquisitionMetric,
    SmokeCheck,
    SmokeGroupResult,
    append_raw_metrics_csv,
    append_raw_metrics_jsonl,
    assert_presentation_free,
    build_trace_row,
    check_gpu_backend,
    check_rapl_reader,
    deterministic_metric,
    run_groups,
    validate_raw_metric,
)
from kwok_energy.rapl import DEFAULT_PKG_MAX_UJ

pytestmark = pytest.mark.unit


class FakeRAPLReader:
    source = "fixture-rapl"
    available = True
    dram_available = False
    pkg_max_uj = DEFAULT_PKG_MAX_UJ
    dram_max_uj = None

    def __init__(self, *, pkg=(1000, 4000), dram=(None, None), raises=False):
        self._pkg = list(pkg)
        self._dram = list(dram)
        self.raises = raises
        self.package_reads = 0
        self.dram_reads = 0

    def read_package_uj(self):
        self.package_reads += 1
        if self.raises:
            raise PermissionError("energy_uj unreadable")
        if len(self._pkg) == 1:
            return self._pkg[0]
        return self._pkg.pop(0)

    def read_dram_uj(self):
        self.dram_reads += 1
        if len(self._dram) == 1:
            return self._dram[0]
        return self._dram.pop(0)


def test_rapl_smoke_uses_injected_framework_reader_boundary():
    reader = FakeRAPLReader()
    result = check_rapl_reader(reader, busy_interval=lambda: None)
    assert result.source == "rapl"
    assert result.ok
    assert reader.package_reads >= 2
    assert any(c.name == "counter_advances" and c.ok and c.value > 0 for c in result.checks)


def test_rapl_permission_skip_is_actionable_and_not_success():
    result = check_rapl_reader(FakeRAPLReader(raises=True), busy_interval=lambda: None)
    assert not result.ok
    assert result.skipped
    messages = "\n".join(c.message for c in result.checks)
    assert "grant read access" in messages
    assert "energy_uj" in messages


def test_rapl_trace_row_valid_with_dram_absent_flagged_null():
    reader = FakeRAPLReader(pkg=(1000, 3000), dram=(None, None))
    result = check_rapl_reader(reader, busy_interval=lambda: None)
    row = build_trace_row(result)
    assert row["rapl_pkg_available"] is True
    assert row["rapl_pkg_joules"] > 0
    assert row["rapl_dram_available"] is False
    assert row["rapl_dram_joules"] is None
    assert schema.validate_trace_row(row) == []


def test_rapl_wrap_delta_is_non_negative_sane():
    reader = FakeRAPLReader(pkg=(DEFAULT_PKG_MAX_UJ - 10, 5))
    result = check_rapl_reader(reader, busy_interval=lambda: None)
    counter = {c.name: c for c in result.checks}["counter_advances"]
    assert counter.ok
    assert counter.value == pytest.approx(15 / 1e6)


class FakeGPUReader:
    available = True
    backend = "fixture-gpu"
    vendor = "amd"
    energy_method = "drm_energy_counter"

    def __init__(self, *, energy=(1000.0, 3000.0), power=(40.0, 60.0), util=None):
        self._energy = list(energy) if energy is not None else None
        self._power = list(power) if power is not None else None
        self._util = util or {"gpu_utilization": 0.5, "gpu_mem_utilization": 0.25}
        self.power_reads = 0
        self.energy_reads = 0
        self.util_reads = 0

    def read_energy_mj(self):
        self.energy_reads += 1
        if self._energy is None:
            return None
        if len(self._energy) == 1:
            return self._energy[0]
        return self._energy.pop(0)

    def read_power_w(self):
        self.power_reads += 1
        if self._power is None:
            return None
        if len(self._power) == 1:
            return self._power[0]
        return self._power.pop(0)

    def read_utilization(self):
        self.util_reads += 1
        return self._util

    def shutdown(self):
        return None


def test_gpu_smoke_uses_vendor_agnostic_backend_and_prefers_counter():
    reader = FakeGPUReader()
    result = check_gpu_backend(reader, load_interval=lambda: None)
    assert result.source == "gpu"
    assert result.ok
    assert result.backend == "fixture-gpu"
    assert reader.energy_reads >= 2
    assert any(c.name == "backend_selected" and c.value == "fixture-gpu" for c in result.checks)
    assert any(c.name == "energy_method" and c.value == "drm_energy_counter" for c in result.checks)


def test_gpu_smoke_integrated_power_fallback_records_backend():
    reader = FakeGPUReader(energy=None, power=(40.0, 60.0))
    reader.energy_method = "integrated_power"
    result = check_gpu_backend(reader, load_interval=lambda: None)
    row = build_trace_row(result)
    assert result.ok
    assert row["gpu_available"] is True
    assert row["gpu_energy_method"] == "integrated_power"
    assert row["gpu_energy_joules"] > 0
    assert schema.validate_trace_row(row) == []


def test_gpu_smoke_steady_power_only_is_live_not_failed():
    """A power-only GPU (e.g. amdgpu) at steady state reads the same watts twice.

    Regression: the old 'power_or_energy_changes' check false-failed a working
    steady-state GPU. The integrated energy (>0) proves the sensor is live.
    """
    reader = FakeGPUReader(energy=None, power=(42.0, 42.0))
    reader.energy_method = "integrated_power"
    result = check_gpu_backend(reader, load_interval=lambda: None)
    assert result.ok
    live = {c.name: c for c in result.checks}["power_or_energy_live"]
    assert live.ok


def test_gpu_smoke_zero_power_no_counter_is_not_live():
    """A dead sensor (zero power, no counter) must still fail the liveness check."""
    reader = FakeGPUReader(energy=None, power=(0.0, 0.0))
    reader.energy_method = "integrated_power"
    result = check_gpu_backend(reader, load_interval=lambda: None)
    assert not result.ok
    live = {c.name: c for c in result.checks}["power_or_energy_live"]
    assert not live.ok


def test_gpu_missing_supported_backend_skips_cleanly():
    class MissingGPU:
        available = False
        backend = "none"
        energy_method = "unavailable"

    result = check_gpu_backend(MissingGPU(), load_interval=lambda: None)
    assert result.skipped
    assert not result.ok
    assert "No supported GPU backend" in result.checks[0].message


def test_run_groups_continues_after_skip():
    def skipped():
        return SmokeGroupResult(
            source="gpu", backend="none", ok=False, skipped=True,
            checks=[SmokeCheck("gpu", "skip", False, None, "", "missing")],
        )

    def passed():
        return SmokeGroupResult(
            source="kwok", backend="fixture", ok=True, skipped=False,
            checks=[SmokeCheck("kwok", "node_series", True, 2, "count", "present")],
        )

    results = run_groups([skipped, passed])
    assert [r.source for r in results] == ["gpu", "kwok"]
    assert results[0].skipped
    assert results[1].ok


def test_raw_metric_schema_csv_jsonl_append_and_provenance(tmp_path):
    metric = deterministic_metric(
        source="gpu", check_name="backend_selected", ok=True, value="gpu_drm",
        unit="backend", message="selected", backend="gpu_drm", timestamp="2026-01-01T00:00:00Z",
    )
    assert validate_raw_metric(metric) == []
    csv_path = tmp_path / "raw.csv"
    jsonl_path = tmp_path / "raw.jsonl"
    append_raw_metrics_csv(csv_path, [metric])
    append_raw_metrics_csv(csv_path, [metric])
    append_raw_metrics_jsonl(jsonl_path, [metric])
    append_raw_metrics_jsonl(jsonl_path, [metric])

    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["source"] == "gpu"
    assert rows[0]["selected_backend"] == "gpu_drm"
    data = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert len(data) == 2
    assert data[0]["provenance"]["git_sha"]
    assert data[0]["provenance"]["python"]


def test_raw_metric_validation_rejects_missing_fields():
    bad = RawAcquisitionMetric(
        source="", check_name="", ok=True, value=None, unit="", message="",
        timestamp="", host="", selected_backend="", provenance={},
    )
    errors = validate_raw_metric(bad)
    assert "source" in "\n".join(errors)
    assert "git_sha" in "\n".join(errors)


def test_presentation_free_guard_rejects_latex_and_allows_acquisition_module(tmp_path):
    bad = tmp_path / "bad_export.py"
    bad.write_text("import latex\n", encoding="utf-8")
    assert assert_presentation_free([bad])

    good = tmp_path / "good_export.py"
    good.write_text("from kwok_energy.acquisition import RawAcquisitionMetric\n", encoding="utf-8")
    assert assert_presentation_free([good]) == []


def test_deterministic_metric_fixture_formatting():
    m1 = deterministic_metric(
        source="rapl", check_name="skip", ok=False, value=None, unit="",
        message="missing", backend="none", timestamp="2026-01-01T00:00:00Z",
    )
    m2 = deterministic_metric(
        source="rapl", check_name="skip", ok=False, value=None, unit="",
        message="missing", backend="none", timestamp="2026-01-01T00:00:00Z",
    )
    assert m1.to_json_dict() == m2.to_json_dict()
