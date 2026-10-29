"""Shared pytest fixtures and configuration for kwok_energy.

Provides:
- marker auto-selection helpers,
- a fixture loader resolving files under ``tests/fixtures/`` so every later
  adversarial trap fixture is available as data (project-tooling spec).

Markers (declared in pyproject.toml): unit integration hardware kwok prometheus slow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# --- fixture loading --------------------------------------------------------

@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Absolute path to ``tests/fixtures/``."""
    return FIXTURES_DIR


def load_fixture(rel_path: str) -> str:
    """Read the text contents of a fixture under ``tests/fixtures/``.

    ``rel_path`` is relative to the fixtures root, e.g. ``"traces/normal.csv"``.
    """
    path = (FIXTURES_DIR / rel_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"fixture not found: {path}")
    if not str(path).startswith(str(FIXTURES_DIR.resolve())):
        # Defence against path traversal outside the fixtures root.
        raise PermissionError(f"fixture outside fixtures root: {path}")
    return path.read_text(encoding="utf-8")


@pytest.fixture()
def fixture_text():
    """Returns a callable ``load_fixture(rel_path) -> str``."""
    return load_fixture


@pytest.fixture()
def fixture_path():
    """Returns a callable ``(rel_path) -> Path`` resolving under fixtures root."""

    def _resolve(rel_path: str) -> Path:
        path = (FIXTURES_DIR / rel_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"fixture not found: {path}")
        return path

    return _resolve


# --- hardware skip ----------------------------------------------------------

def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    """Auto-skip hardware tests when no CPU/GPU energy backend is available.

    A test marked ``hardware`` that has not already arranged its own skip guard is
    skipped with a clear reason, so CI never fails on a missing device.
    """
    has_cpu_energy = False
    try:
        powercap = Path("/sys/class/powercap")
        has_cpu_energy = powercap.exists() and any(
            p.name.startswith(("intel-rapl", "amd-rapl")) for p in powercap.iterdir()
        )
    except OSError:
        has_cpu_energy = False
    if not has_cpu_energy:
        try:
            hwmon = Path("/sys/class/hwmon")
            has_cpu_energy = hwmon.exists() and any(
                (h / "name").exists() and (h / "name").read_text().strip() == "amd_energy"
                for h in hwmon.glob("hwmon*")
            )
        except OSError:
            has_cpu_energy = False
    try:
        import pynvml  # noqa: F401
        has_nvml = True
    except Exception:  # pragma: no cover - environment dependent
        has_nvml = False
    try:
        drm = Path("/sys/class/drm")
        has_drm_gpu = drm.exists() and any((card / "device" / "hwmon").exists() for card in drm.glob("card[0-9]*"))
    except OSError:
        has_drm_gpu = False

    skip_hardware = pytest.mark.skip(
        reason="hardware marker: no CPU/GPU energy backend or permission available"
    )
    for item in items:
        if "hardware" in item.keywords and not (has_cpu_energy or has_nvml or has_drm_gpu):
            item.add_marker(skip_hardware)
