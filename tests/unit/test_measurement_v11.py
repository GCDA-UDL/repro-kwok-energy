"""Unit tests for the schema v1.1 measurement additions.

Covers: the GPU-utilisation EWMA (exact update, leakage-safe input), the
multi-GPU NVML reader (aggregation over devices), the platform power meter
reader, the calibrated comparators, the ``features`` argument of ``fit`` and the
baseline path of the validation harness.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from kwok_energy import baselines as bl
from kwok_energy import calibration as cal
from kwok_energy import validation as val
from kwok_energy.energy_model import FeatureWindow, predict_power
from kwok_energy.features import GPUUtilEWMA, add_gpu_util_ewma
from kwok_energy.nvml import NVMLMultiReader
from kwok_energy.platform_meter import PlatformMeterReader
from kwok_energy.schema import MODEL_INPUT_FIELDS, is_leakage_field

pytestmark = pytest.mark.unit


# --- EWMA -------------------------------------------------------------------

def test_ewma_exact_step_response():
    e = GPUUtilEWMA(tau_s=10.0)
    assert e.update(1.0, 10.0) == pytest.approx(1.0 - math.exp(-1.0))
    # Splitting an interval in two gives the same value (exact for held input).
    a, b = GPUUtilEWMA(10.0), GPUUtilEWMA(10.0)
    a.update(1.0, 6.0)
    b.update(1.0, 3.0)
    b.update(1.0, 3.0)
    assert a.value == pytest.approx(b.value)


def test_ewma_decays_after_load():
    rows = [{"gpu_utilization": 1.0, "dt_s": 1.0}] * 60 + [{"gpu_utilization": 0.0, "dt_s": 1.0}] * 20
    out = add_gpu_util_ewma(rows, tau_s=20.0)
    tail = [r["gpu_util_ewma"] for r in out[60:]]
    assert tail[0] > tail[-1] > 0.0
    assert all(x >= y for x, y in zip(tail, tail[1:], strict=False))


def test_ewma_is_an_allowed_signal_not_a_target():
    assert "gpu_util_ewma" in MODEL_INPUT_FIELDS
    assert not is_leakage_field("gpu_util_ewma")
    assert is_leakage_field("platform_meter_joules")


def test_ewma_rejects_bad_tau():
    with pytest.raises(ValueError):
        GPUUtilEWMA(0.0)


# --- multi-GPU NVML reader --------------------------------------------------

class _FakeNVML:
    NVML_CLOCK_SM = 1
    NVML_TEMPERATURE_GPU = 0

    def __init__(self, devs):
        self.devs = devs

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetCount(self):
        return len(self.devs)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        return self.devs[h]["mj"]

    def nvmlDeviceGetPowerUsage(self, h):
        return self.devs[h]["mw"]

    def nvmlDeviceGetUtilizationRates(self, h):
        return SimpleNamespace(gpu=self.devs[h]["u"], memory=self.devs[h]["m"])

    def nvmlDeviceGetClockInfo(self, h, kind):
        return self.devs[h]["clk"]

    def nvmlDeviceGetPerformanceState(self, h):
        return self.devs[h]["p"]

    def nvmlDeviceGetTemperature(self, h, kind):
        return self.devs[h]["t"]


def test_multi_reader_sums_energy_and_averages_util():
    nv = _FakeNVML([
        {"mj": 1000.0, "mw": 15000.0, "u": 100, "m": 40, "clk": 1800, "p": 0, "t": 60},
        {"mj": 500.0, "mw": 16000.0, "u": 0, "m": 0, "clk": 210, "p": 8, "t": 35},
    ])
    r = NVMLMultiReader(pynvml_module=nv)
    assert r.available and r.device_count == 2
    assert r.energy_method == "nvml_energy_counter"
    assert r.read_energy_mj() == pytest.approx(1500.0)
    assert r.read_power_w() == pytest.approx(31.0)
    assert r.read_utilization() == {"gpu_utilization": 0.5, "gpu_mem_utilization": 0.2}
    d = r.read_diagnostics()
    assert d["gpu_pstate"] == 0.0 and d["gpu_temp_c"] == 60.0
    assert d["gpu_sm_clock_mhz"] == pytest.approx(1005.0)


def test_multi_reader_unavailable_without_devices():
    r = NVMLMultiReader(pynvml_module=_FakeNVML([]))
    assert not r.available and r.read_utilization() is None


# --- platform meter ---------------------------------------------------------

def test_platform_meter_reads_power_meter_hwmon(tmp_path):
    other = tmp_path / "hwmon0"
    other.mkdir()
    (other / "name").write_text("coretemp\n")
    pm = tmp_path / "hwmon3"
    pm.mkdir()
    (pm / "name").write_text("power_meter\n")
    (pm / "power1_average").write_text("159000000\n")
    (pm / "power1_average_interval").write_text("1000\n")
    (pm / "power1_accuracy").write_text("97.500%\n")
    r = PlatformMeterReader(root=tmp_path)
    assert r.available
    assert r.read_power_w() == pytest.approx(159.0)
    assert r.average_interval_ms == 1000.0


def test_platform_meter_absent(tmp_path):
    assert not PlatformMeterReader(root=tmp_path).available


# --- comparators + fit(features=...) + baseline validation path ------------

def _row(u, g, ewma, pkg_w, dram_w, gpu_w, workload="w", dt=1.0):
    return {
        "schema_version": "1.1", "run_id": "r1", "workload": workload, "node": "n0",
        "timestamp": "", "t_start_s": 0.0, "t_end_s": dt, "dt_s": dt, "platform_id": "p",
        "cpu_utilization": u, "cpu_frequency_mhz": 2400.0, "mem_bandwidth_proxy": 0.0,
        "mem_proxy_method": "unavailable", "gpu_utilization": g, "gpu_mem_utilization": 0.0,
        "disk_read_bps": 0.0, "disk_write_bps": 0.0, "net_rx_bps": 0.0, "net_tx_bps": 0.0,
        "gpu_util_ewma": ewma,
        "rapl_pkg_joules": pkg_w * dt, "rapl_dram_joules": dram_w * dt,
        "gpu_energy_joules": gpu_w * dt, "rapl_pkg_available": True,
        "rapl_dram_available": True, "gpu_available": True,
    }


def test_fit_util_linear_total_recovers_plane():
    rows = [_row(u, g, 0.0, 50 + 100 * u, 10.0, 20 + 120 * g)
            for u in (0.0, 0.25, 0.5, 1.0) for g in (0.0, 0.5, 1.0)]
    m = bl.fit_util_linear_total(rows)
    assert m.intercept == pytest.approx(80.0)
    assert m.cpu_coef == pytest.approx(100.0)
    assert m.gpu_coef == pytest.approx(120.0)


def test_fit_fan2007_recovers_curve():
    r_true = 1.4
    rows = [_row(u, 0.0, 0.0, 100 + 150 * (2 * u - u**r_true), 0.0, 0.0)
            for u in (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)]
    m = bl.fit_fan2007(rows)
    assert m.r == pytest.approx(r_true, abs=0.051)
    assert m.idle_watts == pytest.approx(100.0, abs=2.0)


def test_tdp_parametric_counts_every_gpu():
    m = bl.TDPParametricBaseline(cpu_idle_watts=60, cpu_tdp_watts=200, gpu_idle_watts=15,
                                 gpu_tdp_watts=140, gpu_count=2)
    w = FeatureWindow(current={"cpu_utilization": 0.5, "gpu_utilization": 0.5})
    assert m.predict_power(w).total == pytest.approx(60 + 70 + 2 * (15 + 62.5))


def test_fit_with_v2_features_uses_ewma_for_gpu_tail():
    # GPU power depends on the history term only: v2 fits it, v1 cannot.
    rows = [_row(0.0, 0.0, e, 50.0, 10.0, 15 + 30 * e) for e in (0.0, 0.2, 0.5, 0.8, 1.0)]
    c2 = cal.fit(rows, platform_id="p", features=cal.FIT_FEATURES_V2, ridge_alpha=1e-9)
    assert "gpu_util_ewma" in c2.components["gpu"]["weights"]
    sig = {k: v for k, v in rows[3].items() if k in MODEL_INPUT_FIELDS}
    assert predict_power(FeatureWindow(current=sig), c2).gpu == pytest.approx(39.0, abs=0.1)
    c1 = cal.fit(rows, platform_id="p")
    assert "gpu_util_ewma" not in c1.components["gpu"]["weights"]


def test_validate_accepts_baseline_models():
    rows = [_row(u, 0.0, 0.0, 50 + 100 * u, 10.0, 15.0, workload="held") for u in (0.2, 0.6)]
    m = bl.UtilLinearTotalBaseline(intercept=75.0, cpu_coef=100.0, gpu_coef=0.0)
    out = val.validate(m, rows, model_name="util_linear")
    total = [r for r in out if r.component == "total"]
    assert len(total) == 1 and total[0].mape == pytest.approx(0.0, abs=1e-9)


# --- multi-socket RAPL ------------------------------------------------------

@pytest.mark.skipif(__import__("sys").platform == "win32",
                    reason="powercap zone names contain ':' (not allowed on NTFS)")
def test_multi_socket_rapl_sums_every_socket_once(tmp_path):
    from kwok_energy.rapl import MultiSocketRAPL

    def zone(path, name, uj, max_uj):
        path.mkdir(parents=True)
        (path / "name").write_text(name + "\n")
        (path / "energy_uj").write_text(f"{uj}\n")
        (path / "max_energy_range_uj").write_text(f"{max_uj}\n")

    for s in (0, 1):
        zone(tmp_path / f"intel-rapl:{s}", f"package-{s}", 1_000_000, 10_000_000)
        zone(tmp_path / f"intel-rapl:{s}" / f"intel-rapl:{s}:0", "dram", 500_000, 10_000_000)
    # sysfs also lists the DRAM sub-zone at the top level (same zone, symlinked)
    (tmp_path / "intel-rapl:0:0").symlink_to(tmp_path / "intel-rapl:0" / "intel-rapl:0:0")

    r = MultiSocketRAPL(root=tmp_path)
    assert r.sockets == 2 and len(r.dram_zones) == 2
    pkg0, dram0 = r.read()
    (tmp_path / "intel-rapl:0" / "energy_uj").write_text("3000000\n")       # +2 J
    (tmp_path / "intel-rapl:1" / "energy_uj").write_text("500000\n")        # wrapped: +9.5 J
    pkg1, dram1 = r.read()
    assert MultiSocketRAPL.joules(pkg0, pkg1, r.package_zones) == pytest.approx(11.5)
    assert MultiSocketRAPL.joules(dram0, dram1, r.dram_zones) == pytest.approx(0.0)


def test_multi_socket_joules_rejects_impossible_wrap():
    from pathlib import Path

    from kwok_energy.rapl import MultiSocketRAPL
    zones = [(Path("z"), 1_000)]  # modulus far below the counter value
    assert MultiSocketRAPL.joules([5_000_000], [10], zones) is None
    assert MultiSocketRAPL.joules([5_000_000], [6_000_000], zones) == 1.0
