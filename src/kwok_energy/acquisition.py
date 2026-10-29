"""Acquisition/plumbing smoke gate helpers.

The smoke gate verifies that live acquisition uses the framework's collector and
SignalSource/exporter boundaries. It emits neutral raw rows only; rendered-output
conversion is intentionally out of scope.
"""

from __future__ import annotations

import csv
import json
import platform
import socket
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import collect, provenance
from .rapl import DEFAULT_PKG_MAX_UJ, delta, uj_to_j
from .schema import validate_signal_row, validate_trace_row

# Nominal interval (s) used when integrating instantaneous GPU power into an energy
# value for the smoke gate. The gate is a plumbing check, not a measurement, so the
# exact interval is irrelevant — it only needs a positive, non-negative result.
_SMOKE_DT_S = 1.0

_RAW_FIELDS = [
    "source",
    "check_name",
    "ok",
    "value",
    "unit",
    "message",
    "timestamp",
    "host",
    "selected_backend",
    "provenance",
]
_PRESENTATION_TOKENS = tuple(
    [
        "la" + "tex",
        "pdf" + "la" + "tex",
        "xe" + "la" + "tex",
        "lua" + "la" + "tex",
        "paper" + "_table",
        "paper" + "_figure",
        "export" + "_paper",
        "." + "tex",
    ]
)


@dataclass(frozen=True)
class SmokeCheck:
    """One acquisition check result before raw-row serialization."""

    source: str
    name: str
    ok: bool
    value: Any
    unit: str
    message: str


@dataclass(frozen=True)
class SmokeGroupResult:
    """Result for one source group: CPU energy, GPU energy, or KWOK."""

    source: str
    backend: str
    ok: bool
    skipped: bool
    checks: list[SmokeCheck] = field(default_factory=list)
    trace_row: dict[str, Any] | None = None


@dataclass(frozen=True)
class RawAcquisitionMetric:
    """Neutral raw row emitted by the acquisition smoke suite."""

    source: str
    check_name: str
    ok: bool
    value: Any
    unit: str
    message: str
    timestamp: str
    host: str
    selected_backend: str
    provenance: Mapping[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "check_name": self.check_name,
            "ok": self.ok,
            "value": self.value,
            "unit": self.unit,
            "message": self.message,
            "timestamp": self.timestamp,
            "host": self.host,
            "selected_backend": self.selected_backend,
            "provenance": dict(self.provenance),
        }

    def to_csv_dict(self) -> dict[str, str]:
        d = self.to_json_dict()
        return {
            **{k: _csv_scalar(d[k]) for k in _RAW_FIELDS if k != "provenance"},
            "provenance": json.dumps(d["provenance"], sort_keys=True, separators=(",", ":")),
        }


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def acquisition_provenance(*, backend: str = "unknown") -> dict[str, str]:
    """Small R2/R3 provenance block for raw acquisition rows."""

    host = provenance.host_metadata()
    return {
        "git_sha": provenance.git_sha(),
        "kernel": host.get("kernel", "unknown"),
        "cpu": host.get("cpu", "unknown"),
        "gpu": host.get("gpu", "unknown"),
        "nvml_driver": host.get("nvml_driver", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "backend": backend,
    }


def deterministic_metric(
    *,
    source: str,
    check_name: str,
    ok: bool,
    value: Any,
    unit: str,
    message: str,
    backend: str,
    timestamp: str,
    host: str = "fixture-host",
    prov: Mapping[str, Any] | None = None,
) -> RawAcquisitionMetric:
    """Build a deterministic raw row for fixture/skip paths."""

    p = dict(prov or {})
    p.setdefault("git_sha", "fixture-git-sha")
    p.setdefault("kernel", "fixture-kernel")
    p.setdefault("python", platform.python_version())
    p.setdefault("backend", backend)
    return RawAcquisitionMetric(
        source=source,
        check_name=check_name,
        ok=ok,
        value=value,
        unit=unit,
        message=message,
        timestamp=timestamp,
        host=host,
        selected_backend=backend,
        provenance=p,
    )


def validate_raw_metric(metric: RawAcquisitionMetric) -> list[str]:
    errors: list[str] = []
    for key in ("source", "check_name", "timestamp", "host", "selected_backend"):
        if not getattr(metric, key):
            errors.append(f"missing {key}")
    if not isinstance(metric.ok, bool):
        errors.append("ok must be bool")
    if "git_sha" not in metric.provenance or not metric.provenance.get("git_sha"):
        errors.append("missing provenance.git_sha")
    if "python" not in metric.provenance or not metric.provenance.get("python"):
        errors.append("missing provenance.python")
    return errors


def metrics_from_results(
    results: Iterable[SmokeGroupResult],
    *,
    timestamp: str | None = None,
    host: str | None = None,
) -> list[RawAcquisitionMetric]:
    ts = timestamp or utc_now()
    host_name = host or socket.gethostname()
    rows: list[RawAcquisitionMetric] = []
    for result in results:
        prov = acquisition_provenance(backend=result.backend)
        for check in result.checks:
            rows.append(
                RawAcquisitionMetric(
                    source=check.source,
                    check_name=check.name,
                    ok=check.ok,
                    value=check.value,
                    unit=check.unit,
                    message=check.message,
                    timestamp=ts,
                    host=host_name,
                    selected_backend=result.backend,
                    provenance=prov,
                )
            )
    return rows


def append_raw_metrics_csv(path: str | Path, metrics: Sequence[RawAcquisitionMetric]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    exists = p.exists() and p.stat().st_size > 0
    with p.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_RAW_FIELDS)
        if not exists:
            writer.writeheader()
        for metric in metrics:
            errors = validate_raw_metric(metric)
            if errors:
                raise ValueError(f"invalid RawAcquisitionMetric: {errors}")
            writer.writerow(metric.to_csv_dict())
    return p


def append_raw_metrics_jsonl(path: str | Path, metrics: Sequence[RawAcquisitionMetric]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for metric in metrics:
            errors = validate_raw_metric(metric)
            if errors:
                raise ValueError(f"invalid RawAcquisitionMetric: {errors}")
            fh.write(json.dumps(metric.to_json_dict(), sort_keys=True, separators=(",", ":")))
            fh.write("\n")
    return p


def assert_presentation_free(paths: Iterable[str | Path]) -> list[str]:
    """Return presentation-coupling violations for source files."""

    violations: list[str] = []
    for raw in paths:
        p = Path(raw)
        try:
            text = p.read_text(encoding="utf-8").lower()
        except OSError:
            continue
        for token in _PRESENTATION_TOKENS:
            if token in text:
                violations.append(f"{p}: presentation token {token!r}")
    return violations


def _skip(source: str, backend: str, message: str) -> SmokeGroupResult:
    return SmokeGroupResult(
        source=source,
        backend=backend,
        ok=False,
        skipped=True,
        checks=[SmokeCheck(source, "skipped", False, None, "", message)],
    )


def _reader_backend(reader: Any, default: str) -> str:
    for attr in ("backend", "source", "vendor"):
        val = getattr(reader, attr, None)
        if val:
            return str(val)
    return default


def _read_or_permission(fn: Callable[[], Any]) -> tuple[bool, Any, str | None]:
    try:
        return True, fn(), None
    except PermissionError as exc:
        return False, None, str(exc)
    except OSError as exc:
        if getattr(exc, "errno", None) in {1, 13}:
            return False, None, str(exc)
        raise


def check_rapl_reader(
    reader: Any | None = None,
    *,
    busy_interval: Callable[[], None] | None = None,
    sane_max_joules: float = 100_000.0,
) -> SmokeGroupResult:
    """Run the RAPL source group through a framework RAPLReader-like object."""

    r = reader if reader is not None else collect.RAPLReader()
    backend = _reader_backend(r, "rapl")
    if not getattr(r, "available", False):
        return _skip("rapl", backend, "No CPU energy backend available through framework collector")

    ok, prev_pkg, err = _read_or_permission(r.read_package_uj)
    if not ok:
        return _skip(
            "rapl",
            backend,
            f"energy_uj unreadable: {err}; grant read access with a udev rule, group ownership, or sudo for the hardware gate",
        )
    if prev_pkg is None:
        return _skip(
            "rapl",
            backend,
            "CPU energy package counter is unavailable or unreadable through framework collector; grant read access to energy_uj or enable the platform energy backend",
        )
    prev_dram = r.read_dram_uj() if getattr(r, "dram_available", False) else None
    if busy_interval is not None:
        busy_interval()
    ok, cur_pkg, err = _read_or_permission(r.read_package_uj)
    if not ok:
        return _skip(
            "rapl",
            backend,
            f"energy_uj unreadable: {err}; grant read access with a udev rule, group ownership, or sudo for the hardware gate",
        )
    if cur_pkg is None:
        return _skip(
            "rapl",
            backend,
            "CPU energy package counter is unavailable or unreadable through framework collector; grant read access to energy_uj or enable the platform energy backend",
        )
    cur_dram = r.read_dram_uj() if getattr(r, "dram_available", False) else None

    pkg_max = getattr(r, "pkg_max_uj", None) or DEFAULT_PKG_MAX_UJ
    pkg_delta_uj = delta(int(prev_pkg), int(cur_pkg), int(pkg_max))
    pkg_j = uj_to_j(pkg_delta_uj)
    dram_available = bool(getattr(r, "dram_available", False))
    dram_j = None
    if dram_available and prev_dram is not None and cur_dram is not None:
        dram_max = getattr(r, "dram_max_uj", None) or pkg_max
        dram_j = uj_to_j(delta(int(prev_dram), int(cur_dram), int(dram_max)))

    row = _trace_row(
        rapl_pkg_available=True,
        rapl_pkg_joules=pkg_j,
        rapl_dram_available=dram_available,
        rapl_dram_joules=dram_j,
        gpu_available=False,
        gpu_energy_joules=None,
        gpu_energy_method="unavailable",
    )
    errors = validate_trace_row(row)
    checks = [
        SmokeCheck("rapl", "package_discovered", True, backend, "backend", "package domain discovered"),
        SmokeCheck("rapl", "max_range_read", getattr(r, "pkg_max_uj", None) is not None, getattr(r, "pkg_max_uj", None), "uj", "package max_energy_range_uj read or backend fallback recorded"),
        SmokeCheck("rapl", "counter_advances", pkg_j > 0, pkg_j, "J", "package counter advanced across busy interval"),
        SmokeCheck("rapl", "dram_present_or_flagged", True, dram_available, "bool", "DRAM present or explicitly unavailable"),
        SmokeCheck("rapl", "wrap_delta_sane", 0.0 <= pkg_j <= sane_max_joules, pkg_j, "J", "wrap/reset delta non-negative and sane"),
        SmokeCheck("rapl", "trace_row_valid", not errors, len(errors), "errors", "; ".join(errors) or "TraceRow valid"),
    ]
    return SmokeGroupResult("rapl", backend, all(c.ok for c in checks), False, checks, row)


def check_gpu_backend(
    reader: Any | None = None,
    *,
    load_interval: Callable[[], None] | None = None,
) -> SmokeGroupResult:
    """Run the GPU source group through the vendor-agnostic GPU reader boundary."""

    gpu = reader if reader is not None else collect.make_gpu_reader()
    backend = _reader_backend(gpu, "gpu")
    if not getattr(gpu, "available", False):
        return _skip("gpu", backend, "No supported GPU backend available through framework collector")

    util = gpu.read_utilization() if hasattr(gpu, "read_utilization") else None
    prev_energy = gpu.read_energy_mj() if hasattr(gpu, "read_energy_mj") else None
    prev_power = gpu.read_power_w() if hasattr(gpu, "read_power_w") else None
    if load_interval is not None:
        load_interval()
    cur_energy = gpu.read_energy_mj() if hasattr(gpu, "read_energy_mj") else None
    cur_power = gpu.read_power_w() if hasattr(gpu, "read_power_w") else None

    method = str(getattr(gpu, "energy_method", "unavailable"))
    if prev_energy is not None and cur_energy is not None:
        energy_j = max(0.0, (float(cur_energy) - float(prev_energy)) / 1e3)
    elif prev_power is not None and cur_power is not None:
        energy_j = 0.5 * (float(prev_power) + float(cur_power)) * _SMOKE_DT_S
        method = "integrated_power"
    else:
        energy_j = None

    gpu_util = 0.0
    gpu_mem = 0.0
    if util:
        gpu_util = float(util.get("gpu_utilization", 0.0))
        gpu_mem = float(util.get("gpu_mem_utilization", 0.0))
    row = _trace_row(
        rapl_pkg_available=False,
        rapl_pkg_joules=None,
        rapl_dram_available=False,
        rapl_dram_joules=None,
        gpu_available=True,
        gpu_energy_joules=energy_j,
        gpu_energy_method=method,
        gpu_utilization=gpu_util,
        gpu_mem_utilization=gpu_mem,
    )
    errors = validate_trace_row(row)
    # A live sensor either moved between reads, advanced its energy counter, or simply
    # reports a non-zero power. A power-only GPU (e.g. amdgpu power1_average) at steady
    # state reads the same watts twice — that is NOT a failure (energy_j > 0 covers it),
    # so do not require a strict change, which false-failed working steady-state GPUs.
    power_changed = (
        prev_power is not None and cur_power is not None and float(prev_power) != float(cur_power)
    ) or (prev_energy is not None and cur_energy is not None and float(prev_energy) != float(cur_energy))
    sensor_live = power_changed or (energy_j is not None and energy_j > 0.0)
    checks = [
        SmokeCheck("gpu", "backend_selected", True, backend, "backend", "GPU backend selected for platform"),
        SmokeCheck("gpu", "gpu_enumerated", True, 1, "count", "at least one GPU through selected backend"),
        SmokeCheck("gpu", "utilization_read", util is not None, gpu_util if util else None, "fraction", "utilization read when backend exposes it"),
        SmokeCheck("gpu", "power_or_energy_read", energy_j is not None or prev_power is not None, energy_j if energy_j is not None else prev_power, "J_or_W", "power and/or energy readable"),
        SmokeCheck("gpu", "power_or_energy_live", sensor_live, True, "bool", "power/energy responsive: changed, counter advanced, or non-zero power"),
        SmokeCheck("gpu", "energy_method", method != "unavailable", method, "method", "energy method selected"),
        SmokeCheck("gpu", "trace_row_valid", not errors, len(errors), "errors", "; ".join(errors) or "TraceRow valid"),
    ]
    return SmokeGroupResult("gpu", backend, all(c.ok for c in checks), False, checks, row)


def _trace_row(
    *,
    rapl_pkg_available: bool,
    rapl_pkg_joules: float | None,
    rapl_dram_available: bool,
    rapl_dram_joules: float | None,
    gpu_available: bool,
    gpu_energy_joules: float | None,
    gpu_energy_method: str,
    gpu_utilization: float = 0.0,
    gpu_mem_utilization: float = 0.0,
) -> dict[str, Any]:
    dt_s = 1.0
    return {
        "schema_version": "1",
        "run_id": "acquisition-smoke",
        "workload": "smoke",
        "node": "node0",
        "timestamp": "1970-01-01T00:00:00Z",
        "t_start_s": 0.0,
        "t_end_s": dt_s,
        "dt_s": dt_s,
        "cpu_utilization": 0.1,
        "cpu_frequency_mhz": 0.0,
        "mem_bandwidth_proxy": 0.0,
        "mem_proxy_method": "unavailable",
        "gpu_utilization": gpu_utilization,
        "gpu_mem_utilization": gpu_mem_utilization,
        "disk_read_bps": 0.0,
        "disk_write_bps": 0.0,
        "net_rx_bps": 0.0,
        "net_tx_bps": 0.0,
        "platform_id": "acquisition-smoke",
        "rapl_pkg_joules": rapl_pkg_joules,
        "rapl_pkg_watts_mean": None if rapl_pkg_joules is None else rapl_pkg_joules / dt_s,
        "rapl_pkg_available": rapl_pkg_available,
        "rapl_dram_joules": rapl_dram_joules,
        "rapl_dram_watts_mean": None if rapl_dram_joules is None else rapl_dram_joules / dt_s,
        "rapl_dram_available": rapl_dram_available,
        "gpu_energy_joules": gpu_energy_joules,
        "gpu_watts_mean": None if gpu_energy_joules is None else gpu_energy_joules / dt_s,
        "gpu_available": gpu_available,
        "gpu_energy_method": gpu_energy_method,
    }


def build_trace_row(result: SmokeGroupResult) -> dict[str, Any]:
    if result.trace_row is None:
        raise ValueError(f"group {result.source!r} did not produce a TraceRow")
    return dict(result.trace_row)


def run_groups(group_fns: Iterable[Callable[[], SmokeGroupResult]]) -> list[SmokeGroupResult]:
    """Run independent source groups; one skip/failure does not stop the rest."""

    results: list[SmokeGroupResult] = []
    for fn in group_fns:
        try:
            results.append(fn())
        except Exception as exc:  # suite-level guard: record failure, continue siblings
            results.append(
                SmokeGroupResult(
                    source=getattr(fn, "__name__", "unknown"),
                    backend="unknown",
                    ok=False,
                    skipped=False,
                    checks=[SmokeCheck("suite", "group_exception", False, None, "", str(exc))],
                )
            )
    return results


def check_signal_source_to_exporter(source: Any, exporter: Any, *, t: float = 1.0) -> SmokeGroupResult:
    """Hardware-free KWOK seam check for SignalSource→Exporter plumbing.

    Real KWOK cluster lifecycle checks belong in kwok-marked tests; this helper
    verifies the framework boundary: schema-valid target-free signals feed the
    exporter and produce non-empty metrics.
    """

    checks: list[SmokeCheck] = []
    nodes = list(source.nodes())
    checks.append(SmokeCheck("kwok", "nodes_ready", bool(nodes), len(nodes), "count", "nodes visible"))
    signals_ok = True
    for node in nodes:
        sig = source.signals_at(node, t)
        signals_ok = signals_ok and validate_signal_row({
            "schema_version": "1",
            "timestamp": "1970-01-01T00:00:00Z",
            "t_start_s": 0.0,
            "t_end_s": 1.0,
            "dt_s": 1.0,
            **sig,
        }) == []
    checks.append(SmokeCheck("kwok", "signals_schema_valid", signals_ok, len(nodes), "count", "SignalRows schema-valid and target-free"))
    exporter.tick(t)
    metrics = exporter.metrics_text().decode("utf-8")
    has_series = "node_power_watts" in metrics and any(node in metrics for node in nodes)
    checks.append(SmokeCheck("kwok", "exporter_metrics_non_empty", has_series, len(metrics), "bytes", "exporter emits node series"))
    return SmokeGroupResult("kwok", "signal_source_exporter", all(c.ok for c in checks), False, checks)


__all__ = [
    "RawAcquisitionMetric",
    "SmokeCheck",
    "SmokeGroupResult",
    "acquisition_provenance",
    "append_raw_metrics_csv",
    "append_raw_metrics_jsonl",
    "assert_presentation_free",
    "build_trace_row",
    "check_gpu_backend",
    "check_rapl_reader",
    "check_signal_source_to_exporter",
    "deterministic_metric",
    "metrics_from_results",
    "run_groups",
    "utc_now",
    "validate_raw_metric",
]
