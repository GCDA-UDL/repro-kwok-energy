"""Unit tests for AMD/Intel GPU measurement via DRM hwmon — vendor-agnostic-measurement."""

from __future__ import annotations

import pytest

from kwok_energy import gpu_drm

pytestmark = pytest.mark.unit


def _make_card(root, idx, driver, *, power_uw=None, energy_uj=None,
               gpu_busy=None, mem_busy=None):
    card = root / f"card{idx}"
    device = card / "device"
    hw = device / "hwmon" / "hwmon0"
    hw.mkdir(parents=True)
    (device / "uevent").write_text(f"DRIVER={driver}\n")  # driver from uevent (no symlink)
    if power_uw is not None:
        (hw / "power1_average").write_text(str(power_uw))
    if energy_uj is not None:
        (hw / "energy1_input").write_text(str(energy_uj))
    if gpu_busy is not None:
        (device / "gpu_busy_percent").write_text(str(gpu_busy))
    if mem_busy is not None:
        (device / "mem_busy_percent").write_text(str(mem_busy))
    return card


def test_amdgpu_power_and_util(tmp_path):
    drm = tmp_path / "drm"
    drm.mkdir()
    _make_card(drm, 0, "amdgpu", power_uw=45_000_000, gpu_busy=80, mem_busy=30)
    r = gpu_drm.GPUDRMReader(drm)
    assert r.available
    assert r.vendor == "amd"
    assert r.energy_method == "integrated_power"  # power only, no counter
    assert abs(r.read_power_w() - 45.0) < 1e-6
    assert r.read_energy_mj() is None
    util = r.read_utilization()
    assert abs(util["gpu_utilization"] - 0.80) < 1e-9
    assert abs(util["gpu_mem_utilization"] - 0.30) < 1e-9


def test_intel_gpu_energy_counter(tmp_path):
    drm = tmp_path / "drm"
    drm.mkdir()
    _make_card(drm, 0, "i915", energy_uj=2_000_000)  # 2 J counter, no busy %
    r = gpu_drm.GPUDRMReader(drm)
    assert r.available
    assert r.vendor == "intel"
    assert r.energy_method == "drm_energy_counter"
    assert abs(r.read_energy_mj() - 2000.0) < 1e-6  # 2_000_000 µJ -> 2000 mJ
    assert r.read_utilization() is None  # Intel hwmon has no busy %


def test_unsupported_or_absent_gpu_skips(tmp_path):
    drm = tmp_path / "drm"
    drm.mkdir()
    _make_card(drm, 0, "nouveau", power_uw=10_000_000)  # not a supported driver
    r = gpu_drm.GPUDRMReader(drm)
    assert not r.available
    assert r.read_power_w() is None
    assert r.read_energy_mj() is None
    assert r.read_utilization() is None
