"""KWOK/Prometheus smoke seam tests for acquisition helpers.

These tests use a framework SignalSource/exporter fixture. A real KWOK cluster gate
can reuse the same helper with the concrete KWOK SignalSource implementation.
"""

from __future__ import annotations

import pytest

from kwok_energy.acquisition import check_signal_source_to_exporter
from kwok_energy.energy_model import Coeffs
from kwok_energy.exporter import EnergyExporter

pytestmark = [pytest.mark.integration, pytest.mark.kwok, pytest.mark.prometheus]


class FixtureSignalSource:
    def nodes(self):
        return ["kwok-node-0"]

    def platform_id(self, node):
        assert node == "kwok-node-0"
        return "fixture-platform"

    def signals_at(self, node, logical_time_s):
        assert node == "kwok-node-0"
        return {
            "cpu_utilization": 0.4,
            "cpu_frequency_mhz": 2400.0,
            "mem_bandwidth_proxy": 1.0,
            "mem_proxy_method": "resident-pods",
            "gpu_utilization": 0.0,
            "gpu_mem_utilization": 0.0,
            "disk_read_bps": 0.0,
            "disk_write_bps": 0.0,
            "net_rx_bps": 0.0,
            "net_tx_bps": 0.0,
            "platform_id": "fixture-platform",
        }


def _coeffs():
    return Coeffs(
        platform_id="fixture-platform",
        covered_components=frozenset({"package", "gpu"}),
        transform={"features": ["cpu_utilization", "gpu_utilization"], "means": {}, "stds": {}},
        components={
            "package": {"intercept": 10.0, "weights": {"cpu_utilization": 5.0}},
            "gpu": {"intercept": 0.0, "weights": {"gpu_utilization": 1.0}},
        },
    )


def test_signal_source_to_exporter_prometheus_series_non_empty():
    source = FixtureSignalSource()
    exporter = EnergyExporter(source, {"fixture-platform": _coeffs()})
    result = check_signal_source_to_exporter(source, exporter, t=1.0)
    assert result.ok
    assert any(c.name == "signals_schema_valid" and c.ok for c in result.checks)
    assert any(c.name == "exporter_metrics_non_empty" and c.ok for c in result.checks)
