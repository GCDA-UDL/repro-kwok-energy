"""Unit tests for baselines (Phase 5, non-blocking comparators).

Covers: each baseline returns a PowerPrediction over its declared covered set;
TDP baseline reads platform config; SPECpower reproduces a known curve point;
total-only baseline does not trip the per-component sum check.
"""

from __future__ import annotations

import pytest

from kwok_energy import baselines as bl
from kwok_energy.energy_model import PowerPrediction

pytestmark = pytest.mark.unit


def _window(**over):
    s = {
        "cpu_utilization": 0.5, "cpu_frequency_mhz": 2400.0, "mem_bandwidth_proxy": 10.0,
        "mem_proxy_method": "stream_gb_per_s", "gpu_utilization": 0.3, "gpu_mem_utilization": 0.2,
        "disk_read_bps": 0.0, "disk_write_bps": 0.0, "net_rx_bps": 0.0, "net_tx_bps": 0.0,
        "platform_id": "xeon-silver+rtx2080ti",
    }
    s.update(over)
    from kwok_energy.energy_model import FeatureWindow
    return FeatureWindow(current=s, history=())


def test_constant_power_baseline():
    model = bl.ConstantPowerBaseline(total_watts=100.0)
    pred = model.predict_power(_window())
    assert isinstance(pred, PowerPrediction)
    assert pred.total == pytest.approx(100.0)
    assert model.covered_components == frozenset({"total"})


def test_tdp_utilization_baseline_reads_platform_config():
    model = bl.TDPUtilizationBaseline(tdp_watts=120.0, idle_watts=25.0)
    pred = model.predict_power(_window(cpu_utilization=0.5))
    # total = idle + (tdp - idle) * util = 25 + 95*0.5 = 72.5
    assert pred.total == pytest.approx(72.5)
    assert model.covered_components == frozenset({"total"})


def test_cpu_only_linear_baseline():
    model = bl.CPUOnlyLinearBaseline(intercept=25.0, cpu_util_coef=100.0)
    pred = model.predict_power(_window(cpu_utilization=0.5))
    assert pred.package == pytest.approx(75.0)
    assert pred.total == pytest.approx(75.0)  # total == package (only component)
    assert model.covered_components == frozenset({"package", "total"})


def test_specpower_curve_reproduces_known_point():
    # A small published SPECpower-style curve (load%, watts)
    curve = [(0, 25.0), (10, 35.0), (50, 72.0), (100, 120.0)]
    model = bl.SPECpowerCurveBaseline(curve=curve)
    # at 50% load, should reproduce 72.0
    pred = model.predict_power(_window(cpu_utilization=0.5))
    assert pred.total == pytest.approx(72.0)
    assert model.covered_components == frozenset({"total"})


def test_specpower_interpolates_between_points():
    curve = [(0, 25.0), (100, 120.0)]
    model = bl.SPECpowerCurveBaseline(curve=curve)
    pred = model.predict_power(_window(cpu_utilization=0.5))
    # linear interp at 50%: 25 + (120-25)*0.5 = 72.5
    assert pred.total == pytest.approx(72.5)


def test_all_baselines_implement_protocol():
    from kwok_energy.baselines import PowerModel
    models = [
        bl.ConstantPowerBaseline(total_watts=50.0),
        bl.TDPUtilizationBaseline(tdp_watts=120.0, idle_watts=25.0),
        bl.CPUOnlyLinearBaseline(intercept=25.0, cpu_util_coef=100.0),
        bl.SPECpowerCurveBaseline(curve=[(0, 25.0), (100, 120.0)]),
    ]
    for m in models:
        assert isinstance(m, PowerModel)
        assert callable(m.predict_power)
        assert isinstance(m.covered_components, frozenset)


def test_total_only_baseline_does_not_trip_component_sum_check():
    # The "total = component sum" check applies only to models claiming a
    # per-component decomposition. A total-only model is exempt.
    model = bl.ConstantPowerBaseline(total_watts=100.0)
    assert not model.has_decomposition  # no per-component claim
    # CPUOnlyLinear claims package decomposition -> check applies
    decomp = bl.CPUOnlyLinearBaseline(intercept=25.0, cpu_util_coef=100.0)
    assert decomp.has_decomposition


def test_decomposed_model_total_equals_component_sum():
    # For a model claiming decomposition, total must equal its components' sum.
    model = bl.CPUOnlyLinearBaseline(intercept=25.0, cpu_util_coef=100.0)
    pred = model.predict_power(_window(cpu_utilization=0.5))
    comps = [c for c in (pred.package, pred.dram, pred.gpu) if c is not None]
    assert pred.total == pytest.approx(sum(comps))
