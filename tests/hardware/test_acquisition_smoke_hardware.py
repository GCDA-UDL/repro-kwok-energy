"""Live hardware smoke tests for acquisition helpers.

Marked hardware: on machines without readable CPU/GPU energy backends these tests
skip via conftest or assert clear group-level skips. They never bypass framework
collectors.
"""

from __future__ import annotations

import pytest

from kwok_energy.acquisition import check_gpu_backend, check_rapl_reader

pytestmark = pytest.mark.hardware


def _brief_cpu_busy() -> None:
    total = 0
    for i in range(200_000):
        total += i * i
    assert total >= 0


def test_cpu_energy_group_passes_or_skips_cleanly():
    result = check_rapl_reader(busy_interval=_brief_cpu_busy)
    if result.skipped:
        assert "CPU energy" in result.checks[0].message or "energy_uj" in result.checks[0].message
    else:
        assert result.ok
        assert result.trace_row is not None
        assert result.backend != "none"


def test_gpu_energy_group_passes_or_skips_cleanly():
    result = check_gpu_backend(load_interval=lambda: None)
    if result.skipped:
        assert "No supported GPU backend" in result.checks[0].message
    else:
        assert result.backend != "none"
        assert any(c.name == "backend_selected" and c.ok for c in result.checks)
        assert result.trace_row is not None
