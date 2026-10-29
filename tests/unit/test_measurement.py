"""Unit tests for NVML measurement + OS signals (Phase 12, Component B).

NVML tests verify graceful skip (no pynvml/device); OS signal tests verify
CPU util delta computation and unit conversions on the real /proc (or fixture).
"""

from __future__ import annotations

import pytest

from kwok_energy.nvml import NVMLReader, mj_to_j, mw_to_w
from kwok_energy.os_signals import bytes_to_bps, cpu_util, sectors_to_bps

pytestmark = pytest.mark.unit


# --- NVML graceful skip -----------------------------------------------------

def test_nvml_reader_skips_gracefully():
    """On a host without NVML, the reader skips (never crashes)."""
    reader = NVMLReader()
    # On this CI host, pynvml is likely absent → unavailable
    if not reader.available:
        assert reader.energy_method == "unavailable"
        assert reader.read_power_mw() is None
        assert reader.read_energy_mj() is None
        assert reader.read_utilization() is None


def test_nvml_unit_conversions():
    assert mj_to_j(1000.0) == 1.0
    assert mw_to_w(1000.0) == 1.0
    assert mj_to_j(0.0) == 0.0


# --- OS signals -------------------------------------------------------------

def test_cpu_util_basic():
    # /proc/stat snapshot: [user, nice, system, idle, iowait, ...]
    prev = [100, 0, 50, 200, 10, 0, 0, 0]  # idle=210, total=360
    cur = [200, 0, 80, 300, 10, 0, 0, 0]   # idle=310, total=590
    util = cpu_util(prev, cur)
    # idle_delta=100, total_delta=230, util=1-100/230≈0.565
    assert 0.0 <= util <= 1.0
    assert util == pytest.approx(1.0 - 100.0 / 230.0)


def test_cpu_util_zero_delta():
    prev = [100, 0, 50, 200]
    util = cpu_util(prev, prev)
    assert util == 0.0


def test_cpu_util_full_load():
    prev = [100, 0, 50, 200, 10]
    cur = [200, 0, 100, 200, 10]  # idle unchanged → full util
    util = cpu_util(prev, cur)
    assert util == pytest.approx(1.0 - 0.0 / 150.0)


def test_sectors_to_bps():
    assert sectors_to_bps(100, 1.0) == 100 * 512
    assert sectors_to_bps(0, 1.0) == 0.0


def test_bytes_to_bps():
    assert bytes_to_bps(1000, 1.0) == 1000.0
    assert bytes_to_bps(0, 1.0) == 0.0
