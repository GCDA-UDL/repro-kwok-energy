"""Unit tests for the energy model (Phase 3, Component E).

Covers: I1 leakage rejection, I6 total composition, determinism (R1),
energy integration, covered_components. The model is a pure predictor over
MODEL_INPUT_FIELDS; it never consumes measured power.
"""

from __future__ import annotations

import pytest

from kwok_energy import energy_model as em
from kwok_energy.energy_model import Coeffs, FeatureWindow

pytestmark = pytest.mark.unit

# --- fixture builders -------------------------------------------------------

def _signals(**over):
    s = {
        "cpu_utilization": 0.5,
        "cpu_frequency_mhz": 2400.0,
        "mem_bandwidth_proxy": 10.0,
        "mem_proxy_method": "stream_gb_per_s",
        "gpu_utilization": 0.3,
        "gpu_mem_utilization": 0.2,
        "disk_read_bps": 0.0,
        "disk_write_bps": 0.0,
        "net_rx_bps": 0.0,
        "net_tx_bps": 0.0,
        "platform_id": "xeon-silver+rtx2080ti",
    }
    s.update(over)
    return s


def _window(**over):
    return FeatureWindow(current=_signals(**over), history=[])


# A minimal coeffs: package depends on cpu_utilization; dram on mem_bandwidth_proxy;
# gpu on gpu_utilization. Transform = identity (mean 0, std 1) for simplicity.
def _coeffs(dram_covered=True):
    features = ["cpu_utilization", "mem_bandwidth_proxy", "gpu_utilization"]
    transform = {
        "features": features,
        "means": {f: 0.0 for f in features},
        "stds": {f: 1.0 for f in features},
    }
    covered = {"package", "gpu"} | ({"dram"} if dram_covered else set())
    components = {
        "package": {"intercept": 25.0, "weights": {"cpu_utilization": 100.0}, "solver": "ridge"},
        "dram": {"intercept": 3.0, "weights": {"mem_bandwidth_proxy": 0.2}, "solver": "ridge"},
        "gpu": {"intercept": 18.0, "weights": {"gpu_utilization": 200.0}, "solver": "ridge"},
    }
    return Coeffs(
        platform_id="xeon-silver+rtx2080ti",
        covered_components=covered,
        transform=transform,
        components=components,
        provenance={"schema_version": "1", "git_sha": "deadbeef"},
    )


# --- I1: leakage rejection --------------------------------------------------

def test_measured_power_field_as_input_is_rejected():
    bad_window = FeatureWindow(current=_signals(rapl_pkg_joules=50.0), history=[])
    with pytest.raises(em.LeakageError):
        em.predict_power(bad_window, _coeffs())


def test_target_named_feature_rejected_any_name():
    for name in ("rapl_pkg_joules", "gpu_power_draw", "node_energy_total", "gpu_watts_mean"):
        bad = FeatureWindow(current=_signals(**{name: 1.0}), history=[])
        with pytest.raises(em.LeakageError):
            em.predict_power(bad, _coeffs())


# --- I6: total composition --------------------------------------------------

def test_components_non_negative():
    pred = em.predict_power(_window(), _coeffs())
    for comp in ("package", "dram", "gpu"):
        val = getattr(pred, comp)
        assert val is not None and val >= 0.0


def test_total_equals_covered_available_sum():
    coeffs = _coeffs()
    pred = em.predict_power(_window(), coeffs)
    assert pred.total == pytest.approx(pred.package + pred.dram + pred.gpu)


def test_io_excluded_from_total_by_default():
    pred = em.predict_power(_window(), _coeffs())
    assert pred.io is None  # off by default
    assert pred.total == pytest.approx(pred.package + pred.dram + pred.gpu)


def test_dram_absent_dropped_not_zeroed():
    coeffs = _coeffs(dram_covered=False)
    pred = em.predict_power(_window(), coeffs)
    assert pred.dram is None  # absent, not 0
    assert pred.total == pytest.approx(pred.package + pred.gpu)


# --- determinism (R1) -------------------------------------------------------

def test_same_input_same_output():
    coeffs = _coeffs()
    w = _window()
    assert em.predict_power(w, coeffs) == em.predict_power(w, coeffs)


def test_predict_value_matches_manual_linear():
    # package = 25 + 100 * cpu_util = 25 + 50 = 75
    coeffs = _coeffs()
    pred = em.predict_power(_window(cpu_utilization=0.5), coeffs)
    assert pred.package == pytest.approx(75.0)
    # gpu = 18 + 200 * 0.3 = 78
    assert pred.gpu == pytest.approx(78.0)
    # dram = 3 + 0.2 * 10 = 5
    assert pred.dram == pytest.approx(5.0)


def test_feature_transform_standardizes():
    # transform with mean=0.5, std=0.25 on cpu_util: standardized = (0.5-0.5)/0.25 = 0
    # so package = 25 + 100*0 = 25
    coeffs = _coeffs()
    coeffs.transform["means"]["cpu_utilization"] = 0.5
    coeffs.transform["stds"]["cpu_utilization"] = 0.25
    pred = em.predict_power(_window(cpu_utilization=0.5), coeffs)
    assert pred.package == pytest.approx(25.0)


# --- energy integration -----------------------------------------------------

def test_integrate_watts_to_joules():
    pred = em.predict_power(_window(), _coeffs())
    joules = em.integrate(pred, dt_s=2.0)
    assert joules.package == pytest.approx(pred.package * 2.0)
    assert joules.gpu == pytest.approx(pred.gpu * 2.0)
    assert joules.dram == pytest.approx(pred.dram * 2.0)
    assert joules.total == pytest.approx(pred.total * 2.0)
    assert joules.io is None


# --- covered_components -----------------------------------------------------

def test_covered_components_matches_prediction():
    coeffs = _coeffs()
    assert em.covered_components(coeffs) == coeffs.covered_components
    # dram-less coeffs -> dram not covered, not in prediction
    coeffs_nodram = _coeffs(dram_covered=False)
    assert "dram" not in em.covered_components(coeffs_nodram)


def test_power_prediction_is_frozen():
    pred = em.predict_power(_window(), _coeffs())
    with pytest.raises(AttributeError):
        pred.package = 999.0  # type: ignore[misc]


# --- coeffs schema (task 2.7) -----------------------------------------------

def test_coeffs_schema_json_present():
    from pathlib import Path
    coeffs_schema = Path(em.__file__).parent / "schemas" / "coeffs.json"
    assert coeffs_schema.is_file()


def test_coeffs_validate_against_schema():
    from kwok_energy.schema import CoeffsModel

    coeffs = _coeffs()
    # Coeffs dataclass -> JSON-serializable dict matching CoeffsModel
    payload = {
        "platform_id": coeffs.platform_id,
        "covered_components": sorted(coeffs.covered_components),
        "transform": dict(coeffs.transform),
        "components": {
            name: {"intercept": spec["intercept"], "weights": dict(spec["weights"]), "solver": spec["solver"]}
            for name, spec in coeffs.components.items()
        },
        "provenance": dict(coeffs.provenance),
    }
    # Round-trips through the pydantic model (the JSON-schema authority)
    model = CoeffsModel.model_validate(payload)
    assert model.platform_id == coeffs.platform_id
    assert set(model.covered_components) == coeffs.covered_components
