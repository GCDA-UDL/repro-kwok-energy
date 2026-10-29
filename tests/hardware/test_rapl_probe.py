"""Hardware probe smoke test.

Marked ``hardware``: skipped cleanly (exit 0, clear reason) when no RAPL/NVML
device or permission is available — see ``tests/conftest.py``. CI never runs this
suite; it exists so the skip behaviour is itself testable on a developer bench.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.hardware

_RAPL_ROOT = Path("/sys/class/powercap/intel-rapl:0")


def test_rapl_energy_counter_readable():
    """A real bench host exposes an Intel RAPL energy counter.

    Skips cleanly when the device OR the read permission is absent (spec:
    hardware tests skip, never error, without devices/permissions).
    """
    if not _RAPL_ROOT.is_dir():
        pytest.skip("no Intel RAPL powercap interface available")
    energy = next(_RAPL_ROOT.glob("intel-rapl:*/energy_uj"), None)
    if energy is None:
        pytest.skip("no RAPL energy_uj counter found")
    try:
        value = energy.read_text().strip()
    except PermissionError as exc:
        pytest.skip(f"RAPL counter present but not readable: {exc}")
    assert value.isdigit(), f"RAPL energy_uj not numeric: {value!r}"
