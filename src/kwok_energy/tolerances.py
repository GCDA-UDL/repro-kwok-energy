"""Single source of named thresholds (I9).

All numerical thresholds used anywhere in the project live here. Every consumer
imports these names from this module; no other module may redefine them as
literals (enforced by a grep-guard unit test). This is invariant **I9**.

The values are defensible defaults chosen at the foundation layer; later changes
that gather real measurements may tune them, but the tuning must happen *here*,
in this one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

# --- I2: joules are canonical, watts are derived. ---------------------------
# Absolute tolerance (watts) for |*_watts_mean - joules/dt_s|. Guards that a
# provided derived-watts field is consistent with the canonical joules counter.
WATTS_RECON: Final[float] = 0.5

# --- Phase 8 (exporter): offline prediction must equal exporter prediction. -
# Absolute tolerance (watts): the exporter runs the *same* model as the offline
# path, so agreement should be near-exact.
MODEL_EXPORTER_ABS: Final[float] = 0.01

# --- R1/R4: golden CI / reproducibility regression. -------------------------
# Relative tolerance for byte-level reproducible comparison of regenerated
# numeric outputs (coefficients, validation CSV, figures).
GOLDEN_REL: Final[float] = 1e-9

# --- I5: analytic energy counter vs offline integrated energy. --------------
# Relative tolerance that the exporter's monotonic analytic counter matches the
# offline integrated energy over equal logical duration.
ENERGY_RECON_REL: Final[float] = 1e-6

# --- Phase 10 (validation): MAPE is undefined near zero measured power. -----
# Measured watts below this floor are masked out of MAPE (relative error blows
# up). MAE/RMSE still apply.
MAPE_FLOOR_WATTS: Final[float] = 0.5

# --- I7: cancellation / counter-wrap guard. ---------------------------------
# If a measured-energy counter's delta-to-elapsed ratio implies a wrap or reset
# beyond this fraction of the expected range, the interval is flagged for
# cancellation rather than trusted as a real energy reading.
CANCELLATION_RATIO: Final[float] = 0.95

# --- I4: time-based replay / Prometheus scrape cadence. --------------------
# Maximum tolerated gap (seconds) between consecutive scrapes; larger gaps flag
# missing data in the replay/exporter timeline.
SCRAPE_GAP_MAX_S: Final[float] = 5.0


# --- Phase 6b (synthetic signal source): never-run plausibility sanity. ------
# SOFT sanity bound (NOT a crisp gate): a LoadProfile built to approximate a
# held-out workload yields predicted energy within this relative tolerance of
# that workload's measured energy. Profile-build fidelity partly determines the
# outcome, so the bound is generous and documented; it is a sanity signal that a
# realistic generator reproduces a known answer. Never a 27-Jun gate.
SYNTH_PLAUSIBILITY_REL: Final[float] = 0.30


@dataclass(frozen=True)
class Tolerances:
    """Frozen bundle of all named thresholds, for structured passing (I9).

    Fields mirror the module-level constants exactly; ``Tolerances()`` always
    reflects the single source of truth defined above.
    """

    WATTS_RECON: float = WATTS_RECON
    MODEL_EXPORTER_ABS: float = MODEL_EXPORTER_ABS
    GOLDEN_REL: float = GOLDEN_REL
    ENERGY_RECON_REL: float = ENERGY_RECON_REL
    MAPE_FLOOR_WATTS: float = MAPE_FLOOR_WATTS
    CANCELLATION_RATIO: float = CANCELLATION_RATIO
    SCRAPE_GAP_MAX_S: float = SCRAPE_GAP_MAX_S
    SYNTH_PLAUSIBILITY_REL: float = SYNTH_PLAUSIBILITY_REL


__all__ = [
    "WATTS_RECON",
    "MODEL_EXPORTER_ABS",
    "GOLDEN_REL",
    "ENERGY_RECON_REL",
    "MAPE_FLOOR_WATTS",
    "CANCELLATION_RATIO",
    "SCRAPE_GAP_MAX_S",
    "SYNTH_PLAUSIBILITY_REL",
    "Tolerances",
]
