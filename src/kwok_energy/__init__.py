"""kwok_energy — hardware-grounded energy estimation for KWOK.

This package calibrates simulated Kubernetes (KWOK) nodes against real RAPL and
NVML power measurements. This is the foundation layer: data contracts, tolerances,
and the trace core. Energy modelling, calibration, and the exporter arrive in
later changes (see ``openspec/``).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
