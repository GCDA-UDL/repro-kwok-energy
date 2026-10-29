"""Unit test: GPU integrated-power fallback in the collector.

When a GPU has no energy counter (e.g. amdgpu ``power1_average`` only), the collector
integrates instantaneous power as P·dt (trapezoidal over boundary samples).
"""

from __future__ import annotations

import pytest

from kwok_energy import collect

pytestmark = pytest.mark.unit

_CPU = [1, 0, 1, 10, 0, 0, 0, 0]
_CPU2 = [2, 0, 2, 18, 0, 0, 0, 0]


def test_gpu_energy_from_integrated_power():
    prev = {"_gpu_power_w": 40.0, "_cpu_stat": _CPU}
    cur = {
        "_gpu_power_w": 60.0, "_cpu_stat": _CPU2,
        "platform_id": "p", "run_id": "r", "workload": "w", "node": "n",
        "timestamp": "t", "cpu_frequency_mhz": 0.0,
        "gpu_utilization": 0.5, "gpu_mem_utilization": 0.2,
        "gpu_available": True, "gpu_energy_method": "integrated_power",
    }
    row = collect.compute_interval(prev, cur, 0.5)
    # trapezoid mean power = (40+60)/2 = 50 W; energy = 50 * 0.5 = 25 J
    assert abs(row["gpu_energy_joules"] - 25.0) < 1e-9
    assert abs(row["gpu_watts_mean"] - 50.0) < 1e-9


def test_energy_counter_preferred_over_power():
    """If both the counter and power exist, the reset-safe counter wins."""
    prev = {"_gpu_energy_mj": 1000.0, "_gpu_power_w": 999.0, "_cpu_stat": _CPU}
    cur = {
        "_gpu_energy_mj": 3000.0, "_gpu_power_w": 999.0, "_cpu_stat": _CPU2,
        "platform_id": "p", "run_id": "r", "workload": "w", "node": "n",
        "timestamp": "t", "cpu_frequency_mhz": 0.0,
        "gpu_utilization": 0.0, "gpu_mem_utilization": 0.0,
        "gpu_available": True, "gpu_energy_method": "drm_energy_counter",
    }
    row = collect.compute_interval(prev, cur, 0.5)
    # counter delta 2000 mJ -> 2 J (not the power integral)
    assert abs(row["gpu_energy_joules"] - 2.0) < 1e-9


def test_no_gpu_data_yields_null():
    prev = {"_cpu_stat": _CPU}
    cur = {
        "_cpu_stat": _CPU2, "platform_id": "p", "run_id": "r", "workload": "w",
        "node": "n", "timestamp": "t", "cpu_frequency_mhz": 0.0,
        "gpu_utilization": 0.0, "gpu_mem_utilization": 0.0,
        "gpu_available": False, "gpu_energy_method": "unavailable",
    }
    row = collect.compute_interval(prev, cur, 0.5)
    assert row["gpu_energy_joules"] is None
    assert row["gpu_watts_mean"] is None
