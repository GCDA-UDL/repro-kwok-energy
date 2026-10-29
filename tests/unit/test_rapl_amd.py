"""Unit tests for vendor-agnostic CPU energy (AMD) — vendor-agnostic-measurement.

Covers: AMD powercap zone prefix (``amd-rapl:*``) and the ``amd_energy`` hwmon
fallback when no powercap RAPL zone exists.
"""

from __future__ import annotations

import pytest

from kwok_energy import rapl

pytestmark = pytest.mark.unit


def test_amd_powercap_prefix_discovered(tmp_path):
    """AMD exposes RAPL under ``amd-rapl:*``; the package-0 zone must be found.

    (Uses a real ``:`` zone dir — runs on Linux CI; Windows can't create the dir.)
    """
    root = tmp_path / "powercap"
    root.mkdir()
    pkg = root / "amd-rapl:0"
    pkg.mkdir()
    (pkg / "name").write_text("package-0")
    (pkg / "energy_uj").write_text("777")
    (pkg / "max_energy_range_uj").write_text("262143328")
    reader = rapl.RAPLReader(root, hwmon_root=tmp_path / "no-hwmon")
    assert reader.available
    assert reader.source == "powercap"
    assert reader.read_package_uj() == 777
    assert reader.pkg_max_uj == 262143328


def test_amd_energy_hwmon_fallback(tmp_path):
    """No powercap RAPL zone, but an ``amd_energy`` hwmon with a socket channel."""
    powercap = tmp_path / "powercap"  # absent
    hwmon = tmp_path / "hwmon"
    hwmon.mkdir()
    h0 = hwmon / "hwmon0"
    h0.mkdir()
    (h0 / "name").write_text("amd_energy")
    (h0 / "energy1_label").write_text("Ecore0")
    (h0 / "energy1_input").write_text("111")
    (h0 / "energy2_label").write_text("Esocket0")
    (h0 / "energy2_input").write_text("123456789")
    reader = rapl.RAPLReader(powercap, hwmon_root=hwmon)
    assert reader.available
    assert reader.source == "amd_energy"
    # the socket-labelled channel is the package energy (not the core channel)
    assert reader.read_package_uj() == 123456789
    assert not reader.dram_available  # AMD RAPL has no DRAM domain
    assert reader.read_dram_uj() is None


def test_no_powercap_no_hwmon_skips(tmp_path):
    reader = rapl.RAPLReader(tmp_path / "none", hwmon_root=tmp_path / "none2")
    assert not reader.available
    assert reader.source == "none"
    assert reader.read_package_uj() is None
