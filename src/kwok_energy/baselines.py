"""Baselines (Phase 5 — comparator models, non-blocking).

Each baseline implements the ``PowerModel`` protocol (``predict_power``,
``covered_components``), the same interface the energy model satisfies, so the
validation harness (Phase 10) is model-agnostic and compares each model only on
its covered components + ``total``.

Models:
- ``ConstantPowerBaseline`` (covers ``{total}``) — flat power.
- ``TDPUtilizationBaseline`` (covers ``{total}``) — idle + (tdp-idle)·util.
- ``CPUOnlyLinearBaseline`` (covers ``{package, total}``) — package only.
- ``SPECpowerCurveBaseline`` (covers ``{total}``) — interpolate the platform
  CPU's published SPECpower load→watts curve (whole-node), TDP-parametric GPU, no
  DRAM term. This is Joulie's modelling approach — the **fair comparator**.

The "total = component sum" decomposition check (validation harness) applies
**only** to models claiming a per-component decomposition (``has_decomposition``);
a total-only baseline is exempt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .energy_model import FeatureWindow, PowerPrediction


@runtime_checkable
class PowerModel(Protocol):
    """Uniform interface every model (energy model + baselines) satisfies."""

    covered_components: frozenset[str]
    has_decomposition: bool

    def predict_power(self, window: FeatureWindow) -> PowerPrediction: ...


def _util(window: FeatureWindow) -> float:
    return float(window.current.get("cpu_utilization", 0.0))  # type: ignore[arg-type]


def _interp(x: float, points: list[tuple[float, float]]) -> float:
    """Linear interpolation over a sorted (load_fraction, watts) curve."""
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:], strict=True):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]  # pragma: no cover


@dataclass(frozen=True)
class ConstantPowerBaseline:
    """Flat whole-node power (covers ``{total}``)."""

    total_watts: float
    covered_components: frozenset[str] = field(default_factory=lambda: frozenset({"total"}))
    has_decomposition: bool = False

    def predict_power(self, window: FeatureWindow) -> PowerPrediction:
        return PowerPrediction(total=max(0.0, self.total_watts))


@dataclass(frozen=True)
class TDPUtilizationBaseline:
    """idle + (tdp - idle) · cpu_utilization (covers ``{total}``)."""

    tdp_watts: float
    idle_watts: float
    covered_components: frozenset[str] = field(default_factory=lambda: frozenset({"total"}))
    has_decomposition: bool = False

    def predict_power(self, window: FeatureWindow) -> PowerPrediction:
        util = _util(window)
        total = self.idle_watts + (self.tdp_watts - self.idle_watts) * util
        return PowerPrediction(total=max(0.0, total))


@dataclass(frozen=True)
class CPUOnlyLinearBaseline:
    """Package-only linear: intercept + coef·cpu_util (covers ``{package, total}``)."""

    intercept: float
    cpu_util_coef: float
    covered_components: frozenset[str] = field(
        default_factory=lambda: frozenset({"package", "total"})
    )
    has_decomposition: bool = True

    def predict_power(self, window: FeatureWindow) -> PowerPrediction:
        util = _util(window)
        package = max(0.0, self.intercept + self.cpu_util_coef * util)
        return PowerPrediction(package=package, total=package)


@dataclass(frozen=True)
class SPECpowerCurveBaseline:
    """Interpolate a published SPECpower load→watts curve (the fair Joulie comparator).

    Covers ``{total}`` (whole-node CPU curve; TDP-parametric GPU; no DRAM term).
    The curve is a list of ``(load_percent, watts)`` published by SPECpower
    (load in 0–100); ``cpu_utilization`` (0–1) is scaled to percent before
    interpolation.
    """

    curve: list[tuple[float, float]]
    covered_components: frozenset[str] = field(default_factory=lambda: frozenset({"total"}))
    has_decomposition: bool = False

    def predict_power(self, window: FeatureWindow) -> PowerPrediction:
        load_percent = _util(window) * 100.0
        total = _interp(load_percent, self.curve)
        return PowerPrediction(total=max(0.0, total))


def _gpu_util(window: FeatureWindow) -> float:
    return float(window.current.get("gpu_utilization", 0.0) or 0.0)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TDPParametricBaseline:
    """Nameplate model with a GPU term (covers ``{total}``).

    ``cpu_idle + (cpu_tdp - cpu_idle)·u + n_gpu·(gpu_idle + (gpu_tdp - gpu_idle)·g)``
    with ``u`` the CPU and ``g`` the mean GPU utilisation. Idle values come from one
    idle measurement, TDPs from the datasheets: the TDP-parametric style used by
    estimators that do not calibrate against the host (Joulie's GPU term).
    """

    cpu_idle_watts: float
    cpu_tdp_watts: float
    gpu_idle_watts: float
    gpu_tdp_watts: float
    gpu_count: int
    covered_components: frozenset[str] = field(default_factory=lambda: frozenset({"total"}))
    has_decomposition: bool = False

    def predict_power(self, window: FeatureWindow) -> PowerPrediction:
        u, g = _util(window), _gpu_util(window)
        cpu = self.cpu_idle_watts + (self.cpu_tdp_watts - self.cpu_idle_watts) * u
        gpu = self.gpu_count * (self.gpu_idle_watts + (self.gpu_tdp_watts - self.gpu_idle_watts) * g)
        return PowerPrediction(total=max(0.0, cpu + gpu))


@dataclass(frozen=True)
class Fan2007Baseline:
    """Fan, Weber & Barroso (ISCA 2007) empirical CPU-utilisation curve (``{total}``).

    ``P(u) = P_idle + (P_busy - P_idle)·(2u - u**r)``, calibrated on the host. It is
    the classic whole-server model: CPU utilisation is its only input.
    """

    idle_watts: float
    busy_watts: float
    r: float
    covered_components: frozenset[str] = field(default_factory=lambda: frozenset({"total"}))
    has_decomposition: bool = False

    def predict_power(self, window: FeatureWindow) -> PowerPrediction:
        u = min(1.0, max(0.0, _util(window)))
        total = self.idle_watts + (self.busy_watts - self.idle_watts) * (2.0 * u - u**self.r)
        return PowerPrediction(total=max(0.0, total))


@dataclass(frozen=True)
class UtilLinearTotalBaseline:
    """Calibrated whole-node linear model on CPU and GPU utilisation (``{total}``).

    ``a + b·u + c·g`` fitted by least squares on the calibration traces: the
    strongest utilisation-only comparator (no decomposition, no memory or history
    term).
    """

    intercept: float
    cpu_coef: float
    gpu_coef: float
    covered_components: frozenset[str] = field(default_factory=lambda: frozenset({"total"}))
    has_decomposition: bool = False

    def predict_power(self, window: FeatureWindow) -> PowerPrediction:
        total = self.intercept + self.cpu_coef * _util(window) + self.gpu_coef * _gpu_util(window)
        return PowerPrediction(total=max(0.0, total))


# --- calibration of the comparators (same calibration traces as the model) --

_TOTAL_KEYS = ("rapl_pkg_joules", "rapl_dram_joules", "gpu_energy_joules")


def _total_watts(row: dict) -> float | None:
    """Measured total (package + DRAM + GPU) watts for one interval, or None."""
    dt = float(row.get("dt_s") or 0.0)
    if dt <= 0:
        return None
    parts = [row.get(k) for k in _TOTAL_KEYS]
    if all(p is None for p in parts):
        return None
    return sum(float(p) for p in parts if p is not None) / dt


def fit_util_linear_total(rows: list[dict]) -> UtilLinearTotalBaseline:
    """Least-squares fit of ``a + b·u + c·g`` to measured total watts."""
    import numpy as np

    X, y = [], []
    for r in rows:
        w = _total_watts(r)
        if w is None:
            continue
        X.append([1.0, float(r.get("cpu_utilization") or 0.0),
                  float(r.get("gpu_utilization") or 0.0)])
        y.append(w)
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    return UtilLinearTotalBaseline(float(coef[0]), float(coef[1]), float(coef[2]))


def fit_fan2007(rows: list[dict], r_grid: tuple[float, ...] | None = None) -> Fan2007Baseline:
    """Fit Fan et al.'s curve: grid search over ``r``, least squares for the rest."""
    import numpy as np

    us, ws = [], []
    for row in rows:
        w = _total_watts(row)
        if w is None:
            continue
        us.append(min(1.0, max(0.0, float(row.get("cpu_utilization") or 0.0))))
        ws.append(w)
    u, w = np.array(us), np.array(ws)
    grid = r_grid or tuple(np.round(np.arange(0.5, 3.01, 0.05), 2))
    best: tuple[float, float, float, float] | None = None
    for r in grid:
        basis = 2.0 * u - u**r
        A = np.column_stack([np.ones_like(u), basis])
        (p_idle, span), *_ = np.linalg.lstsq(A, w, rcond=None)
        sse = float(np.sum((A @ np.array([p_idle, span]) - w) ** 2))
        if best is None or sse < best[0]:
            best = (sse, float(p_idle), float(span), float(r))
    assert best is not None
    _, p_idle, span, r = best
    return Fan2007Baseline(idle_watts=p_idle, busy_watts=p_idle + span, r=r)


__all__ = [
    "PowerModel",
    "ConstantPowerBaseline",
    "TDPUtilizationBaseline",
    "CPUOnlyLinearBaseline",
    "SPECpowerCurveBaseline",
    "TDPParametricBaseline",
    "Fan2007Baseline",
    "UtilLinearTotalBaseline",
    "fit_util_linear_total",
    "fit_fan2007",
]
