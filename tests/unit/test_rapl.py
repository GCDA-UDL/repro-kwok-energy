"""Unit tests for RAPL measurement (Phase 12, Component B).

Covers: normal delta, wraparound delta, μJ→J, DRAM discovery from a fixture
sysfs tree, graceful skip without powercap.
"""

from __future__ import annotations

import pytest

from kwok_energy import rapl

pytestmark = pytest.mark.unit


# --- delta: normal + wraparound ---------------------------------------------

def test_delta_normal():
    assert rapl.delta(1000, 2000, 100000) == 1000


def test_delta_zero():
    assert rapl.delta(5000, 5000, 100000) == 0


def test_delta_wraparound():
    # counter wraps: prev=99000, cur=500, max=100000 -> 500 + 100000 - 99000 = 1500
    assert rapl.delta(99000, 500, 100000) == 1500


def test_delta_full_wrap():
    # prev near max, cur near 0
    assert rapl.delta(99999, 1, 100000) == 2


# --- μJ → J -----------------------------------------------------------------

def test_uj_to_j_exact():
    assert rapl.uj_to_j(1_000_000) == 1.0
    assert rapl.uj_to_j(500_000) == 0.5
    assert rapl.uj_to_j(0) == 0.0


# --- DRAM discovery from a fixture sysfs tree -------------------------------

@pytest.fixture()
def fixture_sysfs(tmp_path):
    """Build a minimal sysfs powercap tree for testing.

    The package zone is named ``package-0`` — the REAL Intel/AMD zone name
    (regression guard: a fixture named just ``package`` masked a bug where the
    reader never matched the real hardware and reported ``available=False``).
    """
    root = tmp_path / "powercap"
    root.mkdir()
    pkg = root / "intel-rapl:0"
    pkg.mkdir()
    (pkg / "name").write_text("package-0")
    (pkg / "energy_uj").write_text("12345678")
    (pkg / "max_energy_range_uj").write_text("262143328")
    dram = pkg / "intel-rapl:0:0"
    dram.mkdir()
    (dram / "name").write_text("dram")
    (dram / "energy_uj").write_text("9876543")
    (dram / "max_energy_range_uj").write_text("65712999")
    cores = pkg / "intel-rapl:0:1"
    cores.mkdir()
    (cores / "name").write_text("cores")
    (cores / "energy_uj").write_text("1111111")
    return root


def test_discover_domains(fixture_sysfs):
    domains = rapl.discover_domains(fixture_sysfs)
    names = {d["name"] for d in domains}
    assert "package-0" in names
    assert "dram" in names
    assert "cores" in names


def test_find_domain(fixture_sysfs):
    dram = rapl.find_domain("dram", fixture_sysfs)
    assert dram is not None
    assert (dram / "energy_uj").read_text().strip() == "9876543"


def test_read_energy_uj(fixture_sysfs):
    pkg_path = rapl.find_domain("package", fixture_sysfs)
    assert rapl.read_energy_uj(pkg_path) == 12345678


def test_rapl_reader_with_dram(fixture_sysfs):
    reader = rapl.RAPLReader(fixture_sysfs)
    assert reader.available
    assert reader.dram_available
    assert reader.read_package_uj() == 12345678
    assert reader.read_dram_uj() == 9876543


def test_rapl_reader_matches_real_package_zone_name(fixture_sysfs):
    """Regression: the real Intel/AMD zone name is ``package-0``, not ``package``.

    The reader MUST resolve it (else `available=False` on every real bench host,
    silently dropping all package energy — paper-blocking).
    """
    reader = rapl.RAPLReader(fixture_sysfs)
    assert reader.available, "package-0 zone must be discovered as the package domain"
    assert reader.read_package_uj() == 12345678


def test_rapl_reader_reads_max_energy_range(fixture_sysfs):
    """Wrap modulus comes from sysfs per-domain, not a hardcoded default."""
    reader = rapl.RAPLReader(fixture_sysfs)
    assert reader.pkg_max_uj == 262143328
    assert reader.dram_max_uj == 65712999


def test_rapl_reader_without_dram(tmp_path):
    """A sysfs tree with package but no dram domain."""
    root = tmp_path / "powercap"
    root.mkdir()
    pkg = root / "intel-rapl:0"
    pkg.mkdir()
    (pkg / "name").write_text("package-0")
    (pkg / "energy_uj").write_text("5000")
    cores = pkg / "intel-rapl:0:0"
    cores.mkdir()
    (cores / "name").write_text("cores")
    (cores / "energy_uj").write_text("1000")

    reader = rapl.RAPLReader(root)
    assert reader.available
    assert not reader.dram_available  # no dram domain
    assert reader.read_dram_uj() is None  # not a crash
    assert reader.pkg_max_uj is None  # absent max_energy_range_uj => None (uses default)


# --- graceful skip without powercap -----------------------------------------

def test_rapl_reader_absent(tmp_path):
    """No powercap directory and no amd_energy hwmon at all."""
    reader = rapl.RAPLReader(tmp_path / "nonexistent", hwmon_root=tmp_path / "no-hwmon")
    assert not reader.available
    assert reader.read_package_uj() is None


def test_discover_empty(tmp_path):
    assert rapl.discover_domains(tmp_path / "empty") == []
