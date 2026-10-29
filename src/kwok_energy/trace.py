"""Trace core (Phase 2).

Read / write / validate interval traces, derive watts from canonical joules
(I2), and produce a stable content hash for round-trip reproducibility (R1/R5).

On-disk format is CSV (decision D4), matching the measurement-harness output and
human interchange. ``hash_trace`` is computed over the canonical JSON of the
parsed rows (sorted keys, fixed float formatting) — NOT over raw file bytes — so
a CSV round-trip or a float-spelling change (``2400`` vs ``2400.0``) yields the
same digest.

Invariants: I2 (joules canonical, watts derived), I3 (ordered, non-overlapping
intervals).
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .schema import (
    AVAILABILITY_FIELDS,
    DERIVED_TARGET_FIELDS,
    DIAGNOSTIC_FIELDS,
    IDENTITY_FIELDS,
    INTERVAL_FIELDS,
    MEASURED_TARGET_FIELDS,
    MODEL_INPUT_FIELDS,
)

# Canonical CSV column order (identity -> interval -> signals -> targets ->
# derived -> availability). Membership lives in the schema field groups; this
# list only fixes serialization order.
TRACE_COLUMNS: list[str] = [
    # identity
    "schema_version",
    "run_id",
    "workload",
    "node",
    # interval
    "timestamp",
    "t_start_s",
    "t_end_s",
    "dt_s",
    # signals (model inputs)
    "platform_id",
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
    "gpu_util_ewma",
    # measured targets
    "rapl_pkg_joules",
    "rapl_dram_joules",
    "gpu_energy_joules",
    "platform_meter_joules",
    # derived watts
    "rapl_pkg_watts_mean",
    "rapl_dram_watts_mean",
    "gpu_watts_mean",
    "platform_meter_watts_mean",
    # availability
    "rapl_pkg_available",
    "rapl_dram_available",
    "gpu_available",
    "gpu_energy_method",
    "platform_meter_available",
    # diagnostics (v1.1; never model inputs)
    "session_t_s",
    "seq_index",
    "gpu_count",
    "gpu_sm_clock_mhz",
    "gpu_pstate",
    "gpu_temp_c",
    "cpu_temp_c",
]

# Consistency guard: the CSV column contract is exactly the union of the schema
# field groups, so the serialization order can never drift from the disjoint
# contract (I1) defined in schema.py.
assert set(TRACE_COLUMNS) == (
    IDENTITY_FIELDS
    | INTERVAL_FIELDS
    | MODEL_INPUT_FIELDS
    | MEASURED_TARGET_FIELDS
    | DERIVED_TARGET_FIELDS
    | AVAILABILITY_FIELDS
    | DIAGNOSTIC_FIELDS
), "TRACE_COLUMNS must match the schema field groups exactly"

_NUMERIC_COLS: frozenset[str] = frozenset(
    {
        "t_start_s",
        "t_end_s",
        "dt_s",
        "cpu_utilization",
        "cpu_frequency_mhz",
        "mem_bandwidth_proxy",
        "gpu_utilization",
        "gpu_mem_utilization",
        "disk_read_bps",
        "disk_write_bps",
        "net_rx_bps",
        "net_tx_bps",
        "gpu_util_ewma",
        "session_t_s",
        "seq_index",
        "gpu_count",
        "gpu_sm_clock_mhz",
        "gpu_pstate",
        "gpu_temp_c",
        "cpu_temp_c",
    }
    | set(MEASURED_TARGET_FIELDS)
    | set(DERIVED_TARGET_FIELDS)
)
_BOOL_COLS: frozenset[str] = frozenset(
    {"rapl_pkg_available", "rapl_dram_available", "gpu_available", "platform_meter_available"}
)
# Everything else in TRACE_COLUMNS is a string.

_DT_EPS: float = 1e-9

# Re-export the field groups so callers can reach them via the trace module too.
__all__ = [
    "TRACE_COLUMNS",
    "read",
    "write",
    "validate",
    "derive_watts",
    "hash_trace",
]


# --- type coercion ----------------------------------------------------------

def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "yes", "t")


def _coerce(col: str, raw: Any) -> Any:
    """Coerce a raw CSV cell to its typed value (``""`` -> None)."""
    if raw is None or str(raw).strip() == "":
        return None
    if col in _BOOL_COLS:
        return _to_bool(raw)
    if col in _NUMERIC_COLS:
        return float(raw)
    return str(raw)


def _format(col: str, value: Any) -> str:
    """Serialize a typed value to a CSV cell (None -> ``""``)."""
    if value is None:
        return ""
    if col in _BOOL_COLS:
        return "True" if value else "False"
    if col in _NUMERIC_COLS:
        return repr(float(value))  # shortest round-trip float representation
    return str(value)


# --- read / write -----------------------------------------------------------

def read(path: str | Path) -> list[dict[str, Any]]:
    """Read a CSV trace into ordered, typed row dicts (empty cell -> None)."""
    p = Path(path)
    rows: list[dict[str, Any]] = []
    with p.open("r", newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            rows.append({col: _coerce(col, raw.get(col)) for col in TRACE_COLUMNS})
    return rows


def write(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    """Serialize rows to a CSV at ``path`` using the canonical column order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(TRACE_COLUMNS)
        for row in rows:
            writer.writerow([_format(col, row.get(col)) for col in TRACE_COLUMNS])


# --- validate (interval ordering / non-overlap) -----------------------------

def validate(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Check intervals are valid, strictly ordered, and non-overlapping (I3).

    Returns a list of human-readable errors (empty == valid).
    """
    errors: list[str] = []
    prev_end: float | None = None
    for i, row in enumerate(rows):
        try:
            t_start = float(row["t_start_s"])
            t_end = float(row["t_end_s"])
            dt = float(row["dt_s"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"row {i}: interval fields missing or non-numeric")
            continue
        if t_start >= t_end:
            errors.append(f"row {i}: inverted interval t_start={t_start} >= t_end={t_end}")
        if abs(dt - (t_end - t_start)) > _DT_EPS:
            errors.append(f"row {i}: dt_s={dt} != t_end-t_start={t_end - t_start}")
        if prev_end is not None and t_start < prev_end:
            errors.append(
                f"row {i}: overlaps or precedes previous interval "
                f"(t_start={t_start} < prev t_end={prev_end})"
            )
        prev_end = t_end
    return errors


# --- derive_watts (I2) ------------------------------------------------------

_JOULES_TO_WATTS: tuple[tuple[str, str], ...] = (
    ("rapl_pkg_joules", "rapl_pkg_watts_mean"),
    ("rapl_dram_joules", "rapl_dram_watts_mean"),
    ("gpu_energy_joules", "gpu_watts_mean"),
)


def derive_watts(row: Mapping[str, Any]) -> dict[str, float | None]:
    """Compute ``*_watts_mean = joules / dt_s`` for each measured-energy
    component (joules canonical, watts derived — I2). Missing joules -> None."""
    try:
        dt = float(row["dt_s"])
    except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("row has no numeric dt_s") from exc
    out: dict[str, float | None] = {}
    for joules_key, watts_key in _JOULES_TO_WATTS:
        joules = row.get(joules_key)
        if joules is None or (isinstance(joules, str) and joules.strip() == ""):
            out[watts_key] = None
        else:
            out[watts_key] = float(joules) / dt
    return out


# --- hash_trace (R1/R5) -----------------------------------------------------

def _canon_value(value: Any) -> str:
    if value is None:
        return "n:"
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, (int, float)):
        return "f:" + repr(float(value))
    return "s:" + str(value)


# Columns added in schema v1.1. An empty v1.1 cell is left out of the canonical
# payload, so a v1 trace keeps the digest it had before these columns existed.
_V11_COLUMNS: frozenset[str] = frozenset(
    {"gpu_util_ewma", "platform_meter_joules", "platform_meter_watts_mean",
     "platform_meter_available"}
    | DIAGNOSTIC_FIELDS
)


def _canonical_payload(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        [[key, _canon_value(row[key])] for key in sorted(row)
         if not (key in _V11_COLUMNS and row[key] is None)]
        for row in rows
    ]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def hash_trace(path: str | Path) -> str:
    """Stable SHA-256 digest over the canonical parsed trace (R1/R5).

    Invariant under read/write round-trips and under float-spelling changes, so
    the same logical trace always yields the same digest for provenance.
    """
    rows = read(path)
    payload = _canonical_payload(rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
