"""Calibration (Phase 9, Component D).

Fits per-component coefficients from calibration traces. The fitted ``Coeffs``
drive ``predict_power`` wherever it runs (offline, exporter).

- ``fit(calibration_traces, platform_config) → Coeffs``
- ``metrics(coeffs, traces) → {R², MAE per component}``

Guarantees: calibration traces only (refuses held-out); excludes measured targets
from features (I1); fits on derived interval-mean watts (joules canonical, I2);
writes feature transform + provenance (R2); DRAM only if available; deterministic (R1).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .energy_model import Coeffs, FeatureWindow, predict_power
from .provenance import build_block, hash_file
from .schema import MODEL_INPUT_FIELDS
from .trace import read as read_trace

# Features the model fits on (numeric model-input fields, excluding identifiers).
FIT_FEATURES: list[str] = [
    "cpu_utilization",
    "cpu_frequency_mhz",
    "mem_bandwidth_proxy",
    "gpu_utilization",
    "gpu_mem_utilization",
    "disk_read_bps",
    "disk_write_bps",
    "net_rx_bps",
    "net_tx_bps",
]

# v1.1 feature set: v1 plus the GPU-utilisation history term (``features.py``).
FIT_FEATURES_V2: list[str] = [*FIT_FEATURES, "gpu_util_ewma"]

# Signals the calibration workloads sweep (protocol v2). Disk and network rates stay in
# the traces but are not fitted: no calibration workload exercises them and no measured
# component (RAPL package/DRAM, NVML GPU) contains disk or NIC power; the CPU cost of
# I/O already shows in cpu_utilization. Fitted anyway, their noise-level coefficients
# extrapolate: a 500 MB/s database write drove the package term below zero.
SWEPT_FEATURES: list[str] = [
    "cpu_utilization",
    "cpu_frequency_mhz",
    "mem_bandwidth_proxy",
    "gpu_utilization",
    "gpu_mem_utilization",
]

# Component → the measured-joules field and the derived-watts field.
_COMPONENT_MAP: dict[str, tuple[str, str]] = {
    "package": ("rapl_pkg_joules", "rapl_pkg_watts_mean"),
    "dram": ("rapl_dram_joules", "rapl_dram_watts_mean"),
    "gpu": ("gpu_energy_joules", "gpu_watts_mean"),
}

# Component → which features to use (keep it simple: each component sees all features;
# the ridge fit naturally assigns near-zero weight to irrelevant ones).
# Could be specialized per-component, but linear regression handles it.


def _check_not_heldout(trace_paths: Sequence[str | Path]) -> None:
    """Refuse held-out paths (D2). Tags: 'heldout' in path/name."""
    for p in trace_paths:
        if "heldout" in str(p).lower():
            raise ValueError(
                f"refusing to fit on a held-out trace: {p} "
                "(fit uses calibration traces only)"
            )


def _extract_xy(
    rows: list[dict[str, Any]], component: str, features: Sequence[str] = FIT_FEATURES
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Extract (X, y, available) for one component from trace rows.

    X = feature matrix [n_rows, n_features]; y = interval-mean watts [n_rows].
    ``available`` is False if all rows have null joules for this component.
    """
    joules_key, watts_key = _COMPONENT_MAP[component]
    X_list: list[list[float]] = []
    y_list: list[float] = []
    available = False
    for row in rows:
        joules = row.get(joules_key)
        dt = float(row.get("dt_s", 1.0))
        if joules is None:
            continue
        available = True
        watts = float(joules) / dt  # I2: joules canonical, watts derived
        X_list.append([float(row.get(f) or 0.0) for f in features])
        y_list.append(watts)
    if not X_list:
        return np.empty((0, len(features))), np.empty(0), False
    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64), available


def _fit_component(
    X: np.ndarray, y: np.ndarray, ridge_alpha: float = 0.01,
    features: Sequence[str] = FIT_FEATURES,
) -> dict[str, Any]:
    """Fit a linear model with ridge (L2) regularization via numpy lstsq.

    Returns ``{intercept, weights, solver}``.
    """
    n_features = X.shape[1]
    # Augment with a column of ones for the intercept
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    # Ridge: add L2 penalty (not on the intercept)
    ridge_matrix = np.sqrt(ridge_alpha) * np.eye(n_features + 1)
    ridge_matrix[-1, -1] = 0.0  # no penalty on intercept
    X_ridge = np.vstack([X_aug, ridge_matrix])
    y_ridge = np.concatenate([y, np.zeros(n_features + 1)])
    # Solve least squares
    coeffs, *_ = np.linalg.lstsq(X_ridge, y_ridge, rcond=None)
    weights = coeffs[:-1]
    intercept = float(coeffs[-1])
    return {
        "intercept": intercept,
        "weights": {feat: float(w) for feat, w in zip(features, weights, strict=True)},
        "solver": "ridge",
    }


def _compute_transform(
    X_all: np.ndarray, features: Sequence[str] = FIT_FEATURES
) -> dict[str, Any]:
    """Compute standardization (mean/std) over all calibration rows."""
    if X_all.shape[0] == 0:
        means = {f: 0.0 for f in features}
        stds = {f: 1.0 for f in features}
    else:
        means_arr = X_all.mean(axis=0)
        stds_arr = X_all.std(axis=0)
        # Avoid division by zero: std=0 → 1.0
        stds_arr = np.where(stds_arr == 0, 1.0, stds_arr)
        means = {f: float(m) for f, m in zip(features, means_arr, strict=True)}
        stds = {f: float(s) for f, s in zip(features, stds_arr, strict=True)}
    return {"features": list(features), "means": means, "stds": stds}


def _standardize(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    """Apply standardization to the feature matrix."""
    return (X - means) / stds


def fit(
    calibration_traces: Sequence[str | Path] | Sequence[dict[str, Any]],
    platform_config: dict[str, Any] | None = None,
    *,
    platform_id: str = "unknown",
    ridge_alpha: float = 0.01,
    calibration_trace_sha256: str | None = None,
    features: Sequence[str] | None = None,
) -> Coeffs:
    """Fit per-component Coeffs from calibration traces.

    Args:
        calibration_traces: paths to calibration trace CSVs, OR a list of pre-read
            trace row dicts (for testing).
        platform_config: optional platform config (for idle/TDP context).
        platform_id: the platform these coeffs belong to.
        ridge_alpha: L2 regularization strength.
        calibration_trace_sha256: pre-computed hash of the calibration trace
            (for provenance). If None, hashes the first trace path.
        features: model-input features to fit on (default ``FIT_FEATURES``, the
            v1 set; ``FIT_FEATURES_V2`` adds the GPU-utilisation history term).

    Returns:
        Coeffs with the feature transform, per-component params, and provenance.
    """
    # Accept either paths or pre-read rows
    rows: list[dict[str, Any]]
    if calibration_traces and isinstance(calibration_traces[0], (str, Path)):
        trace_paths = [Path(p) for p in calibration_traces if isinstance(p, (str, Path))]
        _check_not_heldout(trace_paths)
        rows = []
        for tp in trace_paths:
            rows.extend(read_trace(tp))
        if calibration_trace_sha256 is None and trace_paths:
            calibration_trace_sha256 = hash_file(trace_paths[0])
    else:
        rows = [dict(r) for r in calibration_traces]  # type: ignore[arg-type]
        if calibration_trace_sha256 is None:
            calibration_trace_sha256 = "fixture"

    if not rows:
        raise ValueError("no calibration trace rows to fit")

    feats = list(features) if features is not None else list(FIT_FEATURES)

    # Gather all features for transform computation
    X_all_list = []
    for row in rows:
        X_all_list.append([float(row.get(f) or 0.0) for f in feats])
    X_all = np.array(X_all_list, dtype=np.float64)

    # Compute standardization transform
    transform = _compute_transform(X_all, feats)
    means_arr = np.array([transform["means"][f] for f in feats])
    stds_arr = np.array([transform["stds"][f] for f in feats])

    # Fit each available component
    components: dict[str, dict[str, Any]] = {}
    covered: set[str] = set()

    for comp in ("package", "dram", "gpu"):
        X_comp, y_comp, available = _extract_xy(rows, comp, feats)
        if not available:
            continue
        # Standardize this component's features (same transform)
        X_comp_std = _standardize(X_comp, means_arr, stds_arr) if X_comp.shape[0] > 0 else X_comp
        result = _fit_component(X_comp_std, y_comp, ridge_alpha=ridge_alpha, features=feats)
        components[comp] = result
        covered.add(comp)

    # Provenance
    import json as _json
    platform_cfg_sha = "unknown"
    if platform_config:
        platform_cfg_sha = hashlib.sha256(
            _json.dumps(platform_config, sort_keys=True).encode()).hexdigest()
    provenance = build_block(
        solver="ridge",
        transform_kind="standardize",
        calibration_trace_sha256=calibration_trace_sha256 or "fixture",
        platform_config_sha256=platform_cfg_sha,
    )

    return Coeffs(
        platform_id=platform_id,
        covered_components=frozenset(covered),
        transform=transform,
        components=components,
        provenance=provenance,
    )


def combined_trace_sha256(paths: Sequence[str | Path]) -> str:
    """One hash over several calibration traces (sorted by file name)."""
    files = sorted((Path(p) for p in paths), key=lambda p: p.name)
    return hashlib.sha256("".join(f"{f.name}:{hash_file(f)}\n" for f in files).encode()).hexdigest()


def to_payload(coeffs: Coeffs, tau_s: float | None = None) -> dict[str, Any]:
    """JSON payload of fitted Coeffs (the per-platform ``coeffs.json`` contract)."""
    transform = dict(coeffs.transform)
    if tau_s is not None:
        # The history signal must be computed with the tau it was fitted with.
        transform["gpu_util_ewma_tau_s"] = float(tau_s)
    return {
        "platform_id": coeffs.platform_id,
        "covered_components": sorted(coeffs.covered_components),
        "transform": transform,
        "components": {k: dict(v) for k, v in coeffs.components.items()},
        "provenance": dict(coeffs.provenance),
    }


def metrics(coeffs: Coeffs, traces: Sequence[str | Path] | list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Compute per-component R² and MAE of the model on the given traces.

    Returns ``{component: {r2, mae}}``.
    """
    if traces and isinstance(traces[0], (str, Path)):
        rows = []
        for tp in traces:
            rows.extend(read_trace(tp))
    else:
        rows = list(traces)

    result: dict[str, dict[str, float]] = {}
    for comp in coeffs.covered_components:
        joules_key, _ = _COMPONENT_MAP[comp]
        y_true: list[float] = []
        y_pred: list[float] = []
        for row in rows:
            joules = row.get(joules_key)
            if joules is None:
                continue
            dt = float(row.get("dt_s", 1.0))
            measured_w = float(joules) / dt
            signals = {k: v for k, v in row.items() if k in MODEL_INPUT_FIELDS}
            pred = predict_power(FeatureWindow(current=signals, history=()), coeffs)
            pred_w = getattr(pred, comp, None)
            if pred_w is None:
                continue
            y_true.append(measured_w)
            y_pred.append(float(pred_w))
        if not y_true:
            result[comp] = {"r2": float("nan"), "mae": float("nan")}
            continue
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
        ss_tot = float(np.sum((y_true_arr - y_true_arr.mean()) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
        mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
        result[comp] = {"r2": r2, "mae": mae}
    return result


__all__ = ["fit", "metrics", "FIT_FEATURES", "FIT_FEATURES_V2", "SWEPT_FEATURES"]
