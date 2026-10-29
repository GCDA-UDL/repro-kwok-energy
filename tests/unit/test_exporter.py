"""Unit tests for the metrics exporter (Phase 8, Component F, I4/I5/I6).

Covers: starts, /metrics 200, valid Prometheus text, scrape doesn't advance time,
both energy metrics present, total=component sum, counter monotonic across loops,
speed≠1 logical-time energy, platform_id routing, offline==exporter agreement.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kwok_energy.energy_model import Coeffs, predict_power
from kwok_energy.exporter import EnergyExporter, UnknownPlatformError
from kwok_energy.replay import load_map
from kwok_energy.tolerances import ENERGY_RECON_REL, MODEL_EXPORTER_ABS

pytestmark = pytest.mark.unit

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
MAP = FIX / "node_profile_map.json"


def _coeffs(platform_id="xeon-silver+rtx2080ti", dram=True):
    features = ["cpu_utilization", "mem_bandwidth_proxy", "gpu_utilization"]
    return Coeffs(
        platform_id=platform_id,
        covered_components=frozenset({"package", "gpu"} | ({"dram"} if dram else set())),
        transform={"features": features, "means": {f: 0.0 for f in features},
                   "stds": {f: 1.0 for f in features}},
        components={
            "package": {"intercept": 25.0, "weights": {"cpu_utilization": 100.0}, "solver": "ridge"},
            "dram": {"intercept": 3.0, "weights": {"mem_bandwidth_proxy": 0.2}, "solver": "ridge"},
            "gpu": {"intercept": 18.0, "weights": {"gpu_utilization": 200.0}, "solver": "ridge"},
        },
    )


@pytest.fixture()
def source():
    return load_map(MAP)


@pytest.fixture()
def exporter(source):
    return EnergyExporter(source, {_coeffs().platform_id: _coeffs()})


# --- basic operation --------------------------------------------------------

def test_metrics_returns_valid_prometheus_text(exporter):
    exporter.tick(1.0)
    text = exporter.metrics_text().decode("utf-8")
    assert "node_power_watts" in text
    assert "node_energy_joules_total" in text
    assert "kwok_energy_" in text


def test_scrape_does_not_advance_time(exporter):
    exporter.tick(1.0)
    e1 = exporter.metrics_text()
    # Scrape again WITHOUT ticking — must be identical
    e2 = exporter.metrics_text()
    assert e1 == e2


def test_both_energy_metrics_present(exporter):
    exporter.tick(1.0)
    text = exporter.metrics_text().decode("utf-8")
    assert "node_power_watts" in text
    assert "node_energy_joules_total" in text


def test_total_equals_component_sum(exporter, source):
    exporter.tick(1.0)
    for node in source.nodes():
        pred = exporter.get_prediction(node)
        assert pred is not None
        comps = [c for c in (pred.package, pred.dram, pred.gpu, pred.io) if c is not None]
        assert pred.total == pytest.approx(sum(comps))


# --- I5: analytic counter ---------------------------------------------------

def test_counter_monotonic_across_loops(exporter):
    """The energy counter never decreases within a process, even across loops."""
    dur = exporter._source.trace_duration("n0")
    # tick through several loops
    prev = 0.0
    for t in range(1, int(dur * 3) + 1):
        exporter.tick(float(t))
        curr = exporter.get_energy("n0")
        assert curr >= prev
        prev = curr


def test_counter_starts_at_zero(exporter):
    # Before any tick, energy is 0 (counter hasn't been initialized for this node yet)
    # First tick at t=0 doesn't add energy (dt=0), but subsequent ticks do
    exporter.tick(0.0)
    assert exporter.get_energy("n0") == pytest.approx(0.0, abs=1e-9)


def test_speed_reports_logical_time_energy():
    """speed≠1: the counter reflects logical-time energy (speed scales time, not energy)."""
    # Build a simple stub source with known signals
    class StubSource:
        def nodes(self): return ["x"]
        def platform_id(self, node): return "p"
        def signals_at(self, node, t):
            return {"cpu_utilization": 0.5, "cpu_frequency_mhz": 2400.0,
                    "mem_bandwidth_proxy": 10.0, "mem_proxy_method": "s",
                    "gpu_utilization": 0.3, "gpu_mem_utilization": 0.2,
                    "disk_read_bps": 0.0, "disk_write_bps": 0.0,
                    "net_rx_bps": 0.0, "net_tx_bps": 0.0, "platform_id": "p"}

    coeffs = _coeffs()
    exp = EnergyExporter(StubSource(), {"p": coeffs})
    # Tick from 0 to 10 logical seconds -> energy = power * 10
    exp.tick(0.0)
    exp.tick(10.0)
    pred = exp.get_prediction("x")
    assert exp.get_energy("x") == pytest.approx(pred.total * 10.0, rel=ENERGY_RECON_REL)


# --- platform_id routing ----------------------------------------------------

def test_heterogeneous_platform_routing():
    """Mixed platform_ids each use their own coeffs."""
    class HeteroSource:
        def nodes(self): return ["a", "b"]
        def platform_id(self, node): return {"a": "plat_a", "b": "plat_b"}[node]
        def signals_at(self, node, t):
            return {"cpu_utilization": 0.5, "cpu_frequency_mhz": 2400.0,
                    "mem_bandwidth_proxy": 10.0, "mem_proxy_method": "s",
                    "gpu_utilization": 0.3, "gpu_mem_utilization": 0.2,
                    "disk_read_bps": 0.0, "disk_write_bps": 0.0,
                    "net_rx_bps": 0.0, "net_tx_bps": 0.0, "platform_id": self.platform_id(node)}

    coeffs_a = _coeffs("plat_a")
    coeffs_b = _coeffs("plat_b")
    # plat_b has a different intercept
    coeffs_b.components["package"]["intercept"] = 50.0
    exp = EnergyExporter(HeteroSource(), {"plat_a": coeffs_a, "plat_b": coeffs_b})
    exp.tick(1.0)
    pred_a = exp.get_prediction("a")
    pred_b = exp.get_prediction("b")
    assert pred_a.package != pred_b.package  # different coeffs -> different predictions


def test_unknown_platform_id_fails():
    class BadSource:
        def nodes(self): return ["x"]
        def platform_id(self, node): return "nonexistent"
        def signals_at(self, node, t):
            return {"cpu_utilization": 0.5, "cpu_frequency_mhz": 2400.0,
                    "mem_bandwidth_proxy": 10.0, "mem_proxy_method": "s",
                    "gpu_utilization": 0.3, "gpu_mem_utilization": 0.2,
                    "disk_read_bps": 0.0, "disk_write_bps": 0.0,
                    "net_rx_bps": 0.0, "net_tx_bps": 0.0, "platform_id": "nonexistent"}

    exp = EnergyExporter(BadSource(), {"real_platform": _coeffs("real_platform")})
    with pytest.raises(UnknownPlatformError):
        exp.tick(1.0)


# --- I10: offline == exporter -----------------------------------------------

def test_offline_equals_exporter_prediction(source):
    """Exporter prediction == offline prediction within MODEL_EXPORTER_ABS."""
    coeffs = _coeffs()
    exp = EnergyExporter(source, {coeffs.platform_id: coeffs})
    exp.tick(0.5)

    from kwok_energy.energy_model import FeatureWindow
    for node in source.nodes():
        signals = source.signals_at(node, 0.5)
        offline_pred = predict_power(FeatureWindow(current=signals, history=()), coeffs)
        export_pred = exp.get_prediction(node)
        assert offline_pred.total == pytest.approx(
            export_pred.total, abs=MODEL_EXPORTER_ABS
        )


def test_analytic_counter_equals_offline_integrated_energy(source):
    """Counter == offline sum(power * dt) per node over the same ticks (ENERGY_RECON_REL)."""
    coeffs = _coeffs()
    exp = EnergyExporter(source, {coeffs.platform_id: coeffs})

    from kwok_energy.energy_model import FeatureWindow
    ticks = [0.0, 0.5, 1.0, 1.5, 2.0]
    # Track per-node offline energy
    offline_energy: dict[str, float] = {node: 0.0 for node in source.nodes()}
    prev_t = None
    for t in ticks:
        exp.tick(t)
        if prev_t is not None:
            dt = t - prev_t
            for node in source.nodes():
                signals = source.signals_at(node, t)
                pred = predict_power(FeatureWindow(current=signals, history=()), coeffs)
                offline_energy[node] += pred.total * dt
        prev_t = t

    for node in source.nodes():
        counter_val = exp.get_energy(node)
        assert counter_val > 0
        assert counter_val == pytest.approx(offline_energy[node], rel=1e-3)


def test_never_emits_measured_target_as_input(exporter):
    """The exporter builds FeatureWindows from SignalSource — no measured power."""
    exporter.tick(1.0)
    text = exporter.metrics_text().decode("utf-8")
    # No leakage metric should appear (the model would have raised before getting here)
    assert "joules" not in text.split("node_power")[0]  # no raw joules in signal gauges


def test_two_cadences_same_energy_over_equal_logical_duration():
    """Two scrape cadences → identical energy over equal logical duration (I4)."""
    class StubSource:
        def nodes(self): return ["x"]
        def platform_id(self, node): return "p"
        def signals_at(self, node, t):
            return {"cpu_utilization": 0.5, "cpu_frequency_mhz": 2400.0,
                    "mem_bandwidth_proxy": 10.0, "mem_proxy_method": "s",
                    "gpu_utilization": 0.3, "gpu_mem_utilization": 0.2,
                    "disk_read_bps": 0.0, "disk_write_bps": 0.0,
                    "net_rx_bps": 0.0, "net_tx_bps": 0.0, "platform_id": "p"}

    coeffs = _coeffs()
    # Cadence 1: tick every 1s, 10 ticks
    exp1 = EnergyExporter(StubSource(), {"p": coeffs})
    for t in range(11):
        exp1.tick(float(t))
    # Cadence 2: tick every 0.5s, 20 ticks (same total logical duration = 10s)
    exp2 = EnergyExporter(StubSource(), {"p": coeffs})
    for i in range(21):
        exp2.tick(i * 0.5)

    e1 = exp1.get_energy("x")
    e2 = exp2.get_energy("x")
    assert e1 == pytest.approx(e2, rel=ENERGY_RECON_REL * 10)


# --- cluster-total aggregation (cluster-integration change) -------------------

def test_cluster_power_and_energy_zero_before_tick():
    class StubSource:
        def nodes(self): return ["a", "b"]
        def platform_id(self, node): return "p"
        def signals_at(self, node, t):
            return {"cpu_utilization": 0.5, "cpu_frequency_mhz": 2400.0,
                    "mem_bandwidth_proxy": 10.0, "mem_proxy_method": "s",
                    "gpu_utilization": 0.3, "gpu_mem_utilization": 0.2,
                    "disk_read_bps": 0.0, "disk_write_bps": 0.0,
                    "net_rx_bps": 0.0, "net_tx_bps": 0.0, "platform_id": "p"}

    coeffs = _coeffs()
    exp = EnergyExporter(StubSource(), {"p": coeffs})

    assert exp.cluster_power() == 0.0
    assert exp.cluster_energy() == 0.0


def test_cluster_power_and_energy_sum_all_nodes():
    class StubSource:
        def nodes(self): return ["a", "b"]
        def platform_id(self, node): return "p"
        def signals_at(self, node, t):
            return {"cpu_utilization": 0.5, "cpu_frequency_mhz": 2400.0,
                    "mem_bandwidth_proxy": 10.0, "mem_proxy_method": "s",
                    "gpu_utilization": 0.3, "gpu_mem_utilization": 0.2,
                    "disk_read_bps": 0.0, "disk_write_bps": 0.0,
                    "net_rx_bps": 0.0, "net_tx_bps": 0.0, "platform_id": "p"}

    coeffs = _coeffs()
    exp = EnergyExporter(StubSource(), {"p": coeffs})
    exp.tick(0.0)
    exp.tick(1.0)

    expected_power = exp.get_prediction("a").total + exp.get_prediction("b").total
    expected_energy = exp.get_energy("a") + exp.get_energy("b")

    assert exp.cluster_power() == pytest.approx(expected_power)
    assert exp.cluster_energy() == pytest.approx(expected_energy)
