"""Signal / trace / platform schema (Phase 1, Component A).

Defines and validates every data contract in the project:

- ``SignalRow``: Component-A workload signals (no targets — golden rule I1).
- ``TraceRow``: SignalRow + identity + interval + measured targets + derived
  watts + availability flags.
- Platform config: CPU/GPU/RAPL spec + sampling interval.

The module exposes the disjoint field-group sets that enforce I1, plus the
validators:

- ``validate_signal_row(row) -> list[str]``  (empty == ok)
- ``validate_trace_row(row) -> list[str]``
- ``validate_platform_config(cfg) -> list[str]``
- ``validate_trace_file(path) -> list[RowValidationResult]``

JSON Schemas under ``schemas/`` are generated from Pydantic v2 models (decision
D2) so the contract and the validator share one source and never drift.

Invariants upheld: I1 (no leakage), I2 (joules canonical, watts derived),
I3 (valid intervals).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .tolerances import WATTS_RECON

# --- field groups (the disjoint contract — I1) ------------------------------

IDENTITY_FIELDS: frozenset[str] = frozenset(
    {"schema_version", "run_id", "workload", "node"}
)
INTERVAL_FIELDS: frozenset[str] = frozenset(
    {"timestamp", "t_start_s", "t_end_s", "dt_s"}
)
MODEL_INPUT_FIELDS: frozenset[str] = frozenset(
    {
        "cpu_utilization",
        "cpu_frequency_mhz",
        "mem_bandwidth_proxy",
        "mem_proxy_method",
        "gpu_utilization",
        "gpu_mem_utilization",
        "disk_read_bps",
        "disk_write_bps",
        "net_rx_bps",
        "net_tx_bps",
        "platform_id",
        # v1.1: exponentially-weighted GPU utilisation history (a deterministic
        # function of past gpu_utilization samples, never of measured power). It
        # lets the model represent the driver's post-load power-state tail.
        "gpu_util_ewma",
    }
)
MEASURED_TARGET_FIELDS: frozenset[str] = frozenset(
    {"rapl_pkg_joules", "rapl_dram_joules", "gpu_energy_joules",
     # v1.1: whole-node power from the platform meter (ACPI power_meter /
     # Intel Node Manager), when the host exposes one. Validation only.
     "platform_meter_joules"}
)
DERIVED_TARGET_FIELDS: frozenset[str] = frozenset(
    {"rapl_pkg_watts_mean", "rapl_dram_watts_mean", "gpu_watts_mean",
     "platform_meter_watts_mean"}
)
AVAILABILITY_FIELDS: frozenset[str] = frozenset(
    {"rapl_pkg_available", "rapl_dram_available", "gpu_available", "gpu_energy_method",
     "platform_meter_available"}
)
# v1.1: capture diagnostics. Neither model inputs nor targets: they describe the
# session timeline and the device state (clocks, P-state, temperatures) so the
# analysis can explain errors. The model never reads them.
DIAGNOSTIC_FIELDS: frozenset[str] = frozenset(
    {"session_t_s", "seq_index", "gpu_count", "gpu_sm_clock_mhz", "gpu_pstate",
     "gpu_temp_c", "cpu_temp_c"}
)

# I1: any model-input candidate whose name carries a measured-target substring
# is rejected. This is the substring blocklist backing the MODEL_INPUT allowlist.
TARGET_SUBSTRINGS: tuple[str, ...] = (
    "joules",
    "watts",
    "power",
    "energy",
    "rapl_",
    "gpu_power_",
    "gpu_energy_",
)

# Internal float epsilon for interval consistency (dt_s == t_end - t_start).
_DT_EPS: float = 1e-9


def is_leakage_field(name: str) -> bool:
    """True if ``name`` resembles a measured/derived target (I1 blocklist)."""
    lowered = str(name).lower()
    return any(sub in lowered for sub in TARGET_SUBSTRINGS)


# --- coercion helpers (CSV gives strings; JSON gives typed values) ----------

def _to_float(value: Any) -> float | None:
    """Coerce a value to float. Empty/whitespace/None -> None (absent)."""
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass; never a power number
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return None
    return float(text)  # raises ValueError if not numeric


def _require_float(row: Mapping[str, Any], key: str, errors: list[str]) -> float | None:
    if key not in row:
        errors.append(f"missing required field {key!r}")
        return None
    try:
        val = _to_float(row[key])
    except (TypeError, ValueError):
        errors.append(f"field {key!r} is not numeric: {row[key]!r}")
        return None
    if val is None:
        errors.append(f"required field {key!r} is empty")
    return val


def _to_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "t"):
        return True
    if text in ("false", "0", "no", "f", ""):
        return False
    return None


# --- interval + range checks (shared by signal & trace validation) ----------

def _check_interval(row: Mapping[str, Any], errors: list[str]) -> float | None:
    t_start = _require_float(row, "t_start_s", errors)
    t_end = _require_float(row, "t_end_s", errors)
    dt = _require_float(row, "dt_s", errors)
    if t_start is None or t_end is None or dt is None:
        return None
    if t_start >= t_end:
        errors.append(f"inverted interval: t_start_s={t_start} >= t_end_s={t_end} (I3)")
    if dt <= 0:
        errors.append(f"dt_s must be positive, got {dt} (I3)")
    if abs(dt - (t_end - t_start)) > _DT_EPS:
        errors.append(
            f"dt_s={dt} != t_end_s - t_start_s={t_end - t_start} (I3)"
        )
    return dt


def _check_signal_ranges(row: Mapping[str, Any], errors: list[str]) -> None:
    util_fields = ("cpu_utilization", "gpu_utilization", "gpu_mem_utilization", "gpu_util_ewma")
    thr_fields = (
        "cpu_frequency_mhz",
        "mem_bandwidth_proxy",
        "disk_read_bps",
        "disk_write_bps",
        "net_rx_bps",
        "net_tx_bps",
    )
    for name in util_fields:
        val = _to_float(row.get(name))
        if val is not None and not (0.0 <= val <= 1.0):
            errors.append(f"{name}={val} out of [0, 1]")
    for name in thr_fields:
        val = _to_float(row.get(name))
        if val is not None and val < 0.0:
            errors.append(f"{name}={val} is negative")


# --- public validators ------------------------------------------------------

def _check_signal_contract(row: Mapping[str, Any], errors: list[str]) -> None:
    """Shared identity + interval + range checks (no leakage scan).

    Used by both ``validate_signal_row`` (which additionally rejects every
    target-like field) and ``validate_trace_row`` (which permits the known
    target/derived fields).
    """
    if not row.get("schema_version"):
        errors.append("missing schema_version")
    _check_interval(row, errors)
    _check_signal_ranges(row, errors)


def validate_signal_row(row: Mapping[str, Any]) -> list[str]:
    """Validate a Component-A signal row (identity + interval + signals).

    A signal row MUST carry no measured-target fields (I1). Returns a list of
    human-readable errors (empty == valid).
    """
    errors: list[str] = []
    _check_signal_contract(row, errors)
    # I1: a signal row must be target-free — any target-like field is rejected.
    for key in row:
        if is_leakage_field(key):
            errors.append(
                f"field {key!r} resembles a measured target — rejected (I1 leakage)"
            )
    return errors


def _check_availability_consistency(
    row: Mapping[str, Any], joules_key: str, avail_key: str, errors: list[str]
) -> None:
    """Enforce: availability flag and measured value are consistent.

    When unavailable, the energy MUST be null (never 0 as a sentinel — I1 data
    hygiene). When available, the energy MUST be present.
    """
    avail = _to_bool(row.get(avail_key))
    joules = row.get(joules_key)
    joules_val = _to_float(joules)
    if avail is False:
        if joules_val is not None:
            errors.append(
                f"{joules_key}={joules_val} but {avail_key}=False: missing energy "
                f"must be null, never 0 (sentinel guard)"
            )
    elif avail is True:
        if joules_val is None:
            errors.append(f"{avail_key}=True but {joules_key} is missing")


def _check_watts_reconstruction(
    row: Mapping[str, Any],
    joules_key: str,
    watts_key: str,
    dt: float | None,
    errors: list[str],
) -> None:
    """I2: when both joules and a derived *_watts_mean are present, the watts
    must equal joules / dt_s within WATTS_RECON."""
    if dt is None or dt <= 0:
        return
    joules = _to_float(row.get(joules_key))
    watts = _to_float(row.get(watts_key))
    if joules is None or watts is None:
        return
    expected = joules / dt
    if abs(watts - expected) > WATTS_RECON:
        errors.append(
            f"{watts_key}={watts} != {joules_key}/dt_s={expected:.6g} "
            f"(exceeds WATTS_RECON={WATTS_RECON}, I2)"
        )


def validate_trace_row(row: Mapping[str, Any]) -> list[str]:
    """Validate a full trace row (signal row + targets + derived + availability).

    Adds the I2 (watts reconstructed from joules) and missing-DRAM-null checks
    on top of the signal-row checks. Returns a list of errors (empty == valid).
    """
    errors: list[str] = []
    _check_signal_contract(row, errors)

    # Unknown target-like fields are rejected (sibling-path I1 guard): only the
    # known target/derived sets may carry power/energy names.
    known = (
        IDENTITY_FIELDS
        | INTERVAL_FIELDS
        | MODEL_INPUT_FIELDS
        | MEASURED_TARGET_FIELDS
        | DERIVED_TARGET_FIELDS
        | AVAILABILITY_FIELDS
        | DIAGNOSTIC_FIELDS
    )
    for key in row:
        if key not in known and is_leakage_field(key):
            errors.append(
                f"unknown target-like field {key!r} — rejected (I1 leakage)"
            )

    dt = _to_float(row.get("dt_s"))
    # I2 reconstruction per component.
    _check_watts_reconstruction(row, "rapl_pkg_joules", "rapl_pkg_watts_mean", dt, errors)
    _check_watts_reconstruction(row, "rapl_dram_joules", "rapl_dram_watts_mean", dt, errors)
    _check_watts_reconstruction(row, "gpu_energy_joules", "gpu_watts_mean", dt, errors)
    _check_watts_reconstruction(
        row, "platform_meter_joules", "platform_meter_watts_mean", dt, errors
    )
    # availability <-> value consistency (DRAM is the spec scenario; pkg/gpu siblings)
    _check_availability_consistency(row, "rapl_pkg_joules", "rapl_pkg_available", errors)
    _check_availability_consistency(row, "rapl_dram_joules", "rapl_dram_available", errors)
    _check_availability_consistency(row, "gpu_energy_joules", "gpu_available", errors)
    _check_availability_consistency(
        row, "platform_meter_joules", "platform_meter_available", errors
    )
    return errors


def validate_platform_config(cfg: Mapping[str, Any]) -> list[str]:
    """Validate a platform config (CPU/GPU/RAPL spec + sampling interval)."""
    errors: list[str] = []

    def need(section: str, key: str, *, ge_zero: bool = True) -> float | None:
        if section not in cfg or not isinstance(cfg[section], Mapping):
            errors.append(f"missing section {section!r}")
            return None
        val = _to_float(cfg[section].get(key))
        if val is None:
            errors.append(f"missing required field {section}.{key}")
        elif ge_zero and val < 0:
            errors.append(f"{section}.{key}={val} must be >= 0")
        return val

    need("cpu", "tdp_watts")
    need("cpu", "idle_watts_estimate")
    if "cpu" in cfg and isinstance(cfg["cpu"], Mapping):
        sockets = _to_float(cfg["cpu"].get("sockets"))
        if sockets is None or sockets < 1:
            errors.append("cpu.sockets must be a positive integer")
    # GPU is optional per platform, but if present its spec must be complete.
    gpu = cfg.get("gpu")
    if isinstance(gpu, Mapping) and gpu.get("present"):
        if _to_float(gpu.get("tdp_watts")) is None:
            errors.append("gpu.tdp_watts required when gpu.present is true")
        if _to_float(gpu.get("idle_watts_estimate")) is None:
            errors.append("gpu.idle_watts_estimate required when gpu.present is true")
    rapl = cfg.get("rapl")
    if not (isinstance(rapl, Mapping) and "has_dram" in rapl):
        errors.append("missing rapl.has_dram")
    interval = _to_float(cfg.get("sampling_interval_seconds"))
    if interval is None or interval <= 0:
        errors.append("sampling_interval_seconds must be > 0")
    return errors


@dataclass
class RowValidationResult:
    """Per-row validation outcome for ``validate_trace_file``."""

    index: int
    errors: list[str] = field(default_factory=list)


def validate_trace_file(path: str | Path) -> list[RowValidationResult]:
    """Validate every row of a CSV trace file; return rows that have errors.

    Empty result == all rows valid. Each entry identifies the offending row by
    its 0-based index.
    """
    p = Path(path)
    results: list[RowValidationResult] = []
    with p.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for index, raw in enumerate(reader):
            errs = validate_trace_row(raw)
            if errs:
                results.append(RowValidationResult(index=index, errors=errs))
    return results


# --- Pydantic models (single source for JSON-Schema export — D2) ------------

class SignalModel(BaseModel):
    """Component-A workload signals (no targets)."""

    cpu_utilization: float = Field(ge=0, le=1)
    cpu_frequency_mhz: float = Field(ge=0)
    mem_bandwidth_proxy: float = Field(ge=0)
    mem_proxy_method: str
    gpu_utilization: float = Field(ge=0, le=1)
    gpu_mem_utilization: float = Field(ge=0, le=1)
    disk_read_bps: float = Field(ge=0)
    disk_write_bps: float = Field(ge=0)
    net_rx_bps: float = Field(ge=0)
    net_tx_bps: float = Field(ge=0)
    platform_id: str
    gpu_util_ewma: float | None = Field(default=None, ge=0, le=1)


class CpuSpec(BaseModel):
    model: str | None = None
    tdp_watts: float = Field(ge=0)
    idle_watts_estimate: float = Field(ge=0)
    sockets: int = Field(ge=1)


class GpuSpec(BaseModel):
    present: bool
    model: str | None = None
    tdp_watts: float | None = Field(default=None, ge=0)
    idle_watts_estimate: float | None = Field(default=None, ge=0)


class RaplSpec(BaseModel):
    has_dram: bool


class PlatformConfigModel(BaseModel):
    cpu: CpuSpec
    gpu: GpuSpec
    rapl: RaplSpec
    sampling_interval_seconds: float = Field(gt=0)


class _ComponentCoeffsModel(BaseModel):
    """One per-component linear sub-model (intercept + feature weights)."""

    intercept: float
    weights: dict[str, float] = Field(default_factory=dict)
    solver: str = "ridge"


class CoeffsModel(BaseModel):
    """The fitted-coefficients JSON contract (Phase 4/9).

    Describes the on-disk ``coeffs.json`` shape: ``platform_id``, the covered
    component set, the standardization transform, per-component linear params,
    and the provenance block (R2). The runtime type is
    ``kwok_energy.energy_model.Coeffs``; this model is the JSON-schema authority
    so the two never drift (decision D2/D4).
    """

    platform_id: str
    covered_components: list[str]
    transform: dict[str, object]
    components: dict[str, _ComponentCoeffsModel]
    provenance: dict[str, object] = Field(default_factory=dict)


_SCHEMAS_DIR = Path(__file__).parent / "schemas"


def generate_schemas(out_dir: Path | None = None) -> tuple[Path, Path, Path]:
    """Emit ``signals.json``, ``platform.json``, ``coeffs.json`` from the Pydantic models."""
    target = out_dir or _SCHEMAS_DIR
    target.mkdir(parents=True, exist_ok=True)
    signals_path = target / "signals.json"
    platform_path = target / "platform.json"
    coeffs_path = target / "coeffs.json"
    signals_path.write_text(
        json.dumps(SignalModel.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    platform_path.write_text(
        json.dumps(PlatformConfigModel.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    coeffs_path.write_text(
        json.dumps(CoeffsModel.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return signals_path, platform_path, coeffs_path


__all__ = [
    "IDENTITY_FIELDS",
    "INTERVAL_FIELDS",
    "MODEL_INPUT_FIELDS",
    "MEASURED_TARGET_FIELDS",
    "DERIVED_TARGET_FIELDS",
    "AVAILABILITY_FIELDS",
    "DIAGNOSTIC_FIELDS",
    "TARGET_SUBSTRINGS",
    "is_leakage_field",
    "validate_signal_row",
    "validate_trace_row",
    "validate_platform_config",
    "validate_trace_file",
    "RowValidationResult",
    "SignalModel",
    "PlatformConfigModel",
    "CoeffsModel",
    "generate_schemas",
]
