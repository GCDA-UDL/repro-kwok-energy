"""Energy model (Phase 3, Component E).

Pure power prediction from low-level workload signals, plus energy integration.
This is the mathematical heart of the system: ``signals → watts → joules``.

- ``predict_power(window, coeffs) → PowerPrediction`` — pure; rejects leakage (I1);
  composes the validated total (I6); deterministic (R1).
- ``integrate(prediction, dt_s) → EnergyPrediction`` — exact analytic joules.
- ``covered_components(coeffs) → set``.

The model consumes **only** ``MODEL_INPUT_FIELDS`` and applies the stored feature
transform identically wherever it runs (offline, exporter), which is what makes
offline==exporter (``MODEL_EXPORTER_ABS``). It NEVER takes measured power as input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .schema import MODEL_INPUT_FIELDS, is_leakage_field

# Numeric model inputs (floats). ``platform_id`` and ``mem_proxy_method`` are
# identifiers/metadata, not numeric features the linear predictor consumes.
NUMERIC_INPUT_FIELDS: frozenset[str] = frozenset(
    {
        "cpu_utilization",
        "cpu_frequency_mhz",
        "mem_bandwidth_proxy",
        "gpu_utilization",
        "gpu_mem_utilization",
        "disk_read_bps",
        "disk_write_bps",
        "net_rx_bps",
        "net_tx_bps",
    }
)

# Components the validated total composes (I6). io is off by default.
_PACKAGE_COMPONENTS: frozenset[str] = frozenset({"package", "dram", "gpu"})


class LeakageError(ValueError):
    """A measured/derived target field was supplied as a model input (I1)."""


# --- shared types (PLAN.md §3) ----------------------------------------------

@dataclass(frozen=True)
class FeatureWindow:
    """``{current: SignalRow, history: recent SignalRows}``.

    v1 reads ``current`` only; ``history`` exists so temporal terms can be added
    without changing callers.
    """

    current: Mapping[str, object]
    history: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class PowerPrediction:
    """Predicted watts for ``package, dram, gpu, io, total``.

    Absent components (DRAM not covered, io off) are ``None`` — never ``0``
    (I6: dropped, not zeroed). ``total`` is always the sum of available covered
    components (io excluded by default).
    """

    package: float | None = None
    dram: float | None = None
    gpu: float | None = None
    io: float | None = None
    total: float = 0.0


@dataclass(frozen=True)
class EnergyPrediction:
    """Integrated joules over one interval (``watts × dt_s``). Mirrors PowerPrediction."""

    package: float | None = None
    dram: float | None = None
    gpu: float | None = None
    io: float | None = None
    total: float = 0.0


@dataclass(frozen=True)
class Coeffs:
    """Per-platform fitted params + feature transform + provenance (R2).

    - ``covered_components``: which of package/dram/gpu the coeffs predict (dram
      absent on platforms without the RAPL DRAM domain).
    - ``transform``: ``{features, means, stds}`` — standardization applied to the
      numeric input features identically wherever ``predict_power`` runs.
    - ``components``: per-component ``{intercept, weights, solver}`` (linear).
    - ``provenance``: the full R2 block (see ``provenance.build_block``).
    """

    platform_id: str
    covered_components: frozenset[str]
    transform: Mapping[str, object]
    components: Mapping[str, Mapping[str, object]]
    provenance: Mapping[str, object] = field(default_factory=dict)


# --- prediction -------------------------------------------------------------

def _check_leakage(current: Mapping[str, object]) -> None:
    """Reject any target-named field or any field outside MODEL_INPUT_FIELDS (I1)."""
    for key in current:
        if is_leakage_field(key):
            raise LeakageError(
                f"input field {key!r} resembles a measured target — rejected (I1 leakage)"
            )
    # Defensive: a non-target field outside the allowlist is also rejected.
    unknown = set(current) - MODEL_INPUT_FIELDS
    if unknown:
        raise LeakageError(
            f"input fields not in MODEL_INPUT_FIELDS: {sorted(unknown)} — rejected (I1)"
        )


def _standardize(value: float, mean: float, std: float) -> float:
    """Apply the stored standardization. std==0 -> no scaling (avoid div-by-zero)."""
    if std == 0:
        return value - mean
    return (value - mean) / std


def predict_power(
    window: FeatureWindow,
    coeffs: Coeffs,
    *,
    include_io: bool = False,
) -> PowerPrediction:
    """Predict per-component watts from ``window.current`` (v1: current only).

    Pure and deterministic (R1). Rejects leakage (I1). Components are
    non-negative; ``total`` = sum of available covered components (I6, io off by
    default). DRAM absent from ``covered_components`` => ``None`` (dropped, not 0).
    """
    current = window.current
    _check_leakage(current)

    transform = coeffs.transform
    means: dict[str, float] = dict(transform.get("means") or {})  # type: ignore[arg-type]
    stds: dict[str, float] = dict(transform.get("stds") or {})  # type: ignore[arg-type]

    def _component_watts(name: str) -> float:
        spec = coeffs.components[name]
        intercept: float = float(spec.get("intercept", 0.0))
        weights: dict[str, float] = dict(spec.get("weights") or {})  # type: ignore[arg-type]
        watts = intercept
        for feat, coef in weights.items():
            raw = current.get(feat)
            if raw is None:
                raise KeyError(f"coeffs component {name!r} needs signal {feat!r} which is absent")
            watts += float(coef) * _standardize(float(raw), means.get(feat, 0.0), stds.get(feat, 1.0))
        # Components are non-negative (power floor is the intercept/idle).
        return max(0.0, watts)

    pkg = _component_watts("package") if "package" in coeffs.covered_components else None
    dram = _component_watts("dram") if "dram" in coeffs.covered_components else None
    gpu = _component_watts("gpu") if "gpu" in coeffs.covered_components else None
    io = _component_watts("io") if (include_io and "io" in coeffs.components) else None

    total = sum(w for w in (pkg, dram, gpu, io) if w is not None)
    return PowerPrediction(package=pkg, dram=dram, gpu=gpu, io=io, total=total)


def integrate(prediction: PowerPrediction, dt_s: float) -> EnergyPrediction:
    """Exact analytic energy over the interval: ``joules = watts × dt_s`` (I5/I2).

    Absent components stay ``None``. ``total`` = sum of available components.
    """
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")

    def _j(w: float | None) -> float | None:
        return None if w is None else w * dt_s

    pkg, dram, gpu, io = _j(prediction.package), _j(prediction.dram), _j(prediction.gpu), _j(prediction.io)
    return EnergyPrediction(
        package=pkg,
        dram=dram,
        gpu=gpu,
        io=io,
        total=sum(j for j in (pkg, dram, gpu, io) if j is not None),
    )


def covered_components(coeffs: Coeffs) -> frozenset[str]:
    """The set of components the coeffs predict (matches what predict_power produces)."""
    return coeffs.covered_components


__all__ = [
    "NUMERIC_INPUT_FIELDS",
    "LeakageError",
    "FeatureWindow",
    "PowerPrediction",
    "EnergyPrediction",
    "Coeffs",
    "predict_power",
    "integrate",
    "covered_components",
]
