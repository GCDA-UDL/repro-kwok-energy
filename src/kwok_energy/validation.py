"""Offline validation (Phase 10, Component H — HEADLINE).

Held-out measured-vs-predicted error per component, per model. The harness is
**model-agnostic** — it compares any model with a ``predict_power`` interface
against measured traces. This is the paper's headline deliverable: per-component
MAPE/RMSE (package, dram, gpu) on held-out workloads.

- ``validate(coeffs|model, heldout_traces) → [ValidationRow]``
- ``ValidationRow`` carries the R2 lineage subset exactly.

Guarantees: held-out only; total composed per availability (I6); MAPE masked below
``MAPE_FLOOR_WATTS``; cancellation flagged per ``CANCELLATION_RATIO`` (I7).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .energy_model import Coeffs, FeatureWindow, PowerPrediction, predict_power
from .provenance import SCHEMA_VERSION, git_sha
from .schema import MODEL_INPUT_FIELDS
from .tolerances import CANCELLATION_RATIO, MAPE_FLOOR_WATTS, MODEL_EXPORTER_ABS
from .trace import read as read_trace

# Component → measured-joules field.
_JOULES_KEY: dict[str, str] = {
    "package": "rapl_pkg_joules",
    "dram": "rapl_dram_joules",
    "gpu": "gpu_energy_joules",
}


@dataclass(frozen=True)
class ValidationRow:
    """One ``model × workload × component`` error record."""

    model_name: str
    covered: str  # JSON-encoded set
    workload: str
    platform_id: str
    component: str
    n_runs: int
    measured_joules: float
    predicted_joules: float
    mae: float
    rmse: float
    mape: float | None
    # [strengthens]
    mae_std: float
    mape_masked: float | None
    mape_coverage: float
    smape: float | None
    ci_low: float | None
    ci_high: float | None
    cancellation_flag: bool
    # R2 lineage subset (exactly these — no host/solver/transform per row)
    source: str
    schema_version: str
    coeffs_git_sha: str
    calibration_trace_sha256: str
    platform_config_sha256: str
    created_at: str


def _check_not_calibration(trace_paths: Sequence[str | Path]) -> None:
    """Refuse calibration-tagged paths."""
    for p in trace_paths:
        if "cal_" in str(p).lower() or "calibration" in str(p).lower():
            raise ValueError(
                f"refusing to validate on calibration traces: {p} "
                "(validate uses held-out traces only)"
            )


def _load_rows(traces: Sequence[str | Path] | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load trace rows from paths or pass through pre-read dicts."""
    if traces and isinstance(traces[0], (str, Path)):
        rows: list[dict[str, Any]] = []
        for tp in traces:
            rows.extend(read_trace(tp))
        return rows
    return [dict(r) for r in traces]  # type: ignore[arg-type]


def _compute_errors(
    measured_w: list[float], predicted_w: list[float]
) -> dict[str, Any]:
    """Compute per-component error statistics from watt-level samples."""
    n = len(measured_w)
    if n == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": None, "mae_std": 0.0,
                "mape_masked": None, "mape_coverage": 0.0, "smape": None,
                "ci_low": None, "ci_high": None}

    import numpy as np
    y_true = np.array(measured_w)
    y_pred = np.array(predicted_w)
    errors = np.abs(y_true - y_pred)

    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae_std = float(np.std(errors)) if n > 1 else 0.0

    # MAPE masked below floor
    above_floor = y_true > MAPE_FLOOR_WATTS
    n_above = int(np.sum(above_floor))
    mape_coverage = n_above / n if n > 0 else 0.0
    if n_above > 0:
        mape_masked = float(np.mean(errors[above_floor] / y_true[above_floor]) * 100)
    else:
        mape_masked = None

    # sMAPE (symmetric, more stable near zero)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    nonzero = denom > MAPE_FLOOR_WATTS / 2
    if np.any(nonzero):
        smape = float(np.mean(np.abs(y_true[nonzero] - y_pred[nonzero]) / denom[nonzero]) * 100)
    else:
        smape = None

    # Simple CI (mean ± 1.96 * std/sqrt(n)) on the per-sample absolute error
    if n > 1 and mae > 0:
        ci_half = 1.96 * mae_std / math.sqrt(n)
        ci_low = max(0.0, mae - ci_half)
        ci_high = mae + ci_half
    else:
        ci_low = ci_high = None

    return {
        "mae": mae, "rmse": rmse, "mape": mape_masked, "mae_std": mae_std,
        "mape_masked": mape_masked, "mape_coverage": mape_coverage,
        "smape": smape, "ci_low": ci_low, "ci_high": ci_high,
    }


def _cancellation_fires(
    component_errors: dict[str, list[float]], total_errors: list[float]
) -> bool:
    """I7: a small total error from opposing component errors."""
    if not total_errors or not component_errors:
        return False
    import numpy as np
    total_abs = float(np.mean(np.abs(total_errors)))
    # Sum of mean absolute component errors (each component independently)
    comp_sum = sum(float(np.mean(np.abs(es))) for es in component_errors.values() if es)
    if comp_sum == 0:
        return False
    ratio = total_abs / comp_sum
    return ratio < CANCELLATION_RATIO


def segmented_counter_increase(samples: Sequence[tuple[float, float]]) -> float:
    """Return reset-safe counter increase from raw ``(timestamp, value)`` samples.

    Phase 15 closed-loop validation deliberately avoids PromQL ``increase()`` /
    ``rate()`` interpolation. Counter decreases are treated as exporter restarts:
    the old segment ends and the new segment starts from the reset value.
    """

    if len(samples) < 2:
        return 0.0

    total = 0.0
    previous = float(samples[0][1])
    for _, raw_value in samples[1:]:
        value = float(raw_value)
        if value >= previous:
            total += value - previous
        previous = value
    return total


def validate(
    model: Coeffs | Any,
    heldout_traces: Sequence[str | Path] | Sequence[dict[str, Any]],
    *,
    model_name: str | None = None,
    source: str = "offline_model",
) -> list[ValidationRow]:
    """Validate a model against held-out traces. Returns one row per (workload, component).

    The harness is model-agnostic: accepts a ``Coeffs`` (wrapped in the energy model)
    or any object with ``predict_power(window, coeffs)``. Computes MAE, RMSE, MAPE
    (masked), sMAPE, CIs, and the cancellation flag (I7).
    """
    # Resolve the prediction function + model identity
    if isinstance(model, Coeffs):
        coeffs = model
        name = model_name or "energy_model"
        covered = coeffs.covered_components

        def _predict(signals: dict[str, Any]) -> PowerPrediction:
            return predict_power(FeatureWindow(current=signals, history=()), coeffs)
    else:
        # Baseline or custom model: predict_power takes a FeatureWindow (the
        # PowerModel protocol), so wrap the signal dict before calling it.
        _model_predict = model.predict_power  # type: ignore[union-attr]

        def _predict(signals: dict[str, Any]) -> PowerPrediction:
            return _model_predict(FeatureWindow(current=signals, history=()))
        name = model_name or getattr(model, "__class__", "model").__name__  # type: ignore[union-attr]
        covered = getattr(model, "covered_components", frozenset({"total"}))  # type: ignore[union-attr]

    # Check not calibration
    if heldout_traces and isinstance(heldout_traces[0], (str, Path)):
        _check_not_calibration([Path(p) for p in heldout_traces if isinstance(p, (str, Path))])

    rows = _load_rows(heldout_traces)
    if not rows:
        return []

    # Group by (workload, component)
    # Collect per-group: measured_w, predicted_w, measured_joules, predicted_joules
    groups: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"measured_w": [], "predicted_w": [], "measured_j": [], "predicted_j": []}
    )
    # Also track per-workload component + total errors for cancellation
    wl_component_errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    wl_total_errors: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        dt = float(row.get("dt_s", 1.0))
        workload = str(row.get("workload", "unknown"))
        signals = {k: v for k, v in row.items() if k in MODEL_INPUT_FIELDS}

        pred = _predict(signals)

        # Per-component: compare measured joules vs predicted joules
        dram_available = row.get("rapl_dram_available", True)
        for comp in ("package", "dram", "gpu"):
            if comp == "dram" and not dram_available:
                continue
            joules_key = _JOULES_KEY[comp]
            measured_j = row.get(joules_key)
            if measured_j is None:
                continue
            pred_w = getattr(pred, comp, None)
            if pred_w is None:
                continue
            measured_w = float(measured_j) / dt
            pred_j = pred_w * dt

            key = (workload, comp)
            groups[key]["measured_w"].append(measured_w)
            groups[key]["predicted_w"].append(float(pred_w))
            groups[key]["measured_j"].append(float(measured_j))
            groups[key]["predicted_j"].append(pred_j)

            wl_component_errors[workload][comp].append(measured_w - float(pred_w))

        # Total: sum available component joules
        total_measured_j = 0.0
        for comp_name, j_key in _JOULES_KEY.items():
            if comp_name == "dram" and not dram_available:
                continue
            val = row.get(j_key)
            if val is not None:
                total_measured_j += float(val)
        total_pred_w = pred.total
        groups[(workload, "total")]["measured_w"].append(total_measured_j / dt)
        groups[(workload, "total")]["predicted_w"].append(float(total_pred_w))
        groups[(workload, "total")]["measured_j"].append(total_measured_j)
        groups[(workload, "total")]["predicted_j"].append(total_pred_w * dt)
        wl_total_errors[workload].append(total_measured_j / dt - float(total_pred_w))

    # Build ValidationRows
    result: list[ValidationRow] = []
    now = datetime.now(UTC).isoformat(timespec="seconds")
    sha = git_sha()

    # Provenance lineage subset (from coeffs if available)
    if isinstance(model, Coeffs):
        prov = model.provenance
        cal_sha = str(prov.get("calibration_trace_sha256", "unknown"))
        plat_sha = str(prov.get("platform_config_sha256", "unknown"))
        platform_id = model.platform_id
    else:
        cal_sha = plat_sha = "unknown"
        platform_id = "unknown"

    for (workload, comp), data in sorted(groups.items()):
        errs = _compute_errors(data["measured_w"], data["predicted_w"])
        # Cancellation flag: only for total rows
        if comp == "total":
            canc = _cancellation_fires(
                {c: [a - b for a, b in zip(data_m["measured_w"], data_m["predicted_w"], strict=True)]
                 for (wl, c), data_m in groups.items() if c != "total" and wl == workload},
                wl_total_errors[workload],
            )
        else:
            canc = False

        result.append(ValidationRow(
            model_name=name,
            covered=str(sorted(covered)),
            workload=workload,
            platform_id=platform_id,
            component=comp,
            n_runs=len(data["measured_w"]),
            measured_joules=sum(data["measured_j"]),
            predicted_joules=sum(data["predicted_j"]),
            mae=errs["mae"], rmse=errs["rmse"], mape=errs["mape"],
            mae_std=errs["mae_std"], mape_masked=errs["mape_masked"],
            mape_coverage=errs["mape_coverage"], smape=errs["smape"],
            ci_low=errs["ci_low"], ci_high=errs["ci_high"],
            cancellation_flag=canc,
            source=source, schema_version=SCHEMA_VERSION,
            coeffs_git_sha=sha,
            calibration_trace_sha256=cal_sha,
            platform_config_sha256=plat_sha,
            created_at=now,
        ))

    return result


def validate_closed_loop(
    model: Coeffs | Any,
    heldout_traces: Sequence[str | Path] | Sequence[dict[str, Any]],
    *,
    counter_samples: dict[tuple[str, str], Sequence[tuple[float, float]]] | None = None,
    model_name: str | None = None,
) -> list[ValidationRow]:
    """Return offline rows plus replay-exporter/Prometheus plumbing rows.

    The offline rows are the measured-vs-predicted accuracy claim. The
    ``kwok_replay_prometheus`` rows are a closed-loop plumbing check: they use raw
    counter samples when supplied, extracted via ``segmented_counter_increase``.
    When no samples are supplied (unit fixture seam), exporter rows mirror the
    offline predicted joules so callers can verify source labelling and component
    coverage without a live Prometheus instance.
    """

    offline_rows = validate(model, heldout_traces, model_name=model_name, source="offline_model")
    closed_loop_rows: list[ValidationRow] = []
    for row in offline_rows:
        samples = counter_samples.get((row.workload, row.component)) if counter_samples else None
        predicted_joules = segmented_counter_increase(samples) if samples is not None else row.predicted_joules
        closed_loop_rows.append(
            replace(
                row,
                source="kwok_replay_prometheus",
                predicted_joules=predicted_joules,
            )
        )
    return offline_rows + closed_loop_rows


def assert_closed_loop_agreement(
    rows: Sequence[ValidationRow],
    *,
    abs_tol: float = MODEL_EXPORTER_ABS,
) -> None:
    """Assert replay-exporter rows match offline predictions within tolerance."""

    offline = {
        (row.workload, row.component): row
        for row in rows
        if row.source == "offline_model"
    }
    exporter = [row for row in rows if row.source == "kwok_replay_prometheus"]
    if not exporter:
        raise AssertionError("no kwok_replay_prometheus rows present")
    for row in exporter:
        key = (row.workload, row.component)
        if key not in offline:
            raise AssertionError(f"missing offline_model row for {key!r}")
        delta = abs(row.predicted_joules - offline[key].predicted_joules)
        if delta > abs_tol:
            raise AssertionError(
                f"closed-loop prediction mismatch for {key!r}: {delta} > {abs_tol}"
            )


__all__ = [
    "ValidationRow",
    "assert_closed_loop_agreement",
    "segmented_counter_increase",
    "validate",
    "validate_closed_loop",
]
