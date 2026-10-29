"""NVML GPU energy/util measurement (Phase 12, Component B).

Reads GPU energy and utilization via NVML (pynvml). Energy method priority:
`nvml_energy_counter` (reset-safe, since-driver-reload) → `integrated_power`
(trapezoidal over inner samples). Graceful skip when pynvml/NVML is absent.
"""

from __future__ import annotations

from typing import Any


class NVMLReader:
    """Reads GPU energy/util via NVML. Skips gracefully when unavailable."""

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._handle: Any = None
        self._initialized = False
        self._energy_method = "unavailable"
        self._init()

    def _init(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            # Energy method priority: energy counter > integrated power
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                self._energy_method = "nvml_energy_counter"
            except Exception:
                self._energy_method = "integrated_power"
            self._initialized = True
        except Exception:
            self._energy_method = "unavailable"

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def energy_method(self) -> str:
        return self._energy_method

    def read_power_mw(self) -> float | None:
        """Current GPU power draw (mW), or None if unavailable."""
        if not self._initialized:
            return None
        try:
            import pynvml
            return float(pynvml.nvmlDeviceGetPowerUsage(self._handle))
        except Exception:
            return None

    def read_power_w(self) -> float | None:
        """Current GPU power draw (W), or None. Parity with GPUDRMReader."""
        mw = self.read_power_mw()
        return mw / 1e3 if mw is not None else None

    def read_energy_mj(self) -> float | None:
        """Cumulative GPU energy (mJ) from the counter, or None."""
        if not self._initialized or self._energy_method != "nvml_energy_counter":
            return None
        try:
            import pynvml
            return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception:
            return None

    def read_utilization(self) -> dict[str, float] | None:
        """GPU + memory utilization as {gpu_util, mem_util} in [0,1], or None."""
        if not self._initialized:
            return None
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            return {"gpu_utilization": util.gpu / 100.0, "gpu_mem_utilization": util.memory / 100.0}
        except Exception:
            return None

    def shutdown(self) -> None:
        if self._initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False


class NVMLMultiReader:
    """Node-level reader over every NVIDIA GPU in the host (schema v1.1).

    Same interface as :class:`NVMLReader`, aggregated over devices: energy and
    power are summed, utilisation is the mean over devices (so a node with one of
    two GPUs saturated reports 0.5). ``read_diagnostics`` adds the device state
    (SM clock, P-state, temperature) used only to explain errors, never as model
    input. ``pynvml_module`` is injectable for tests.
    """

    def __init__(self, pynvml_module: Any | None = None) -> None:
        self._nv = pynvml_module
        self._handles: list[Any] = []
        self._initialized = False
        self._energy_method = "unavailable"
        self._init()

    def _init(self) -> None:
        try:
            if self._nv is None:
                import pynvml
                self._nv = pynvml
            nv = self._nv
            nv.nvmlInit()
            count = int(nv.nvmlDeviceGetCount())
            self._handles = [nv.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            if not self._handles:
                return
            try:
                for h in self._handles:
                    nv.nvmlDeviceGetTotalEnergyConsumption(h)
                self._energy_method = "nvml_energy_counter"
            except Exception:
                self._energy_method = "integrated_power"
            self._initialized = True
        except Exception:
            self._energy_method = "unavailable"

    @property
    def available(self) -> bool:
        return self._initialized

    @property
    def energy_method(self) -> str:
        return self._energy_method

    @property
    def device_count(self) -> int:
        return len(self._handles) if self._initialized else 0

    def read_power_w(self) -> float | None:
        """Summed instantaneous power over all GPUs (W), or None."""
        if not self._initialized:
            return None
        try:
            return sum(float(self._nv.nvmlDeviceGetPowerUsage(h)) for h in self._handles) / 1e3
        except Exception:
            return None

    def read_energy_mj(self) -> float | None:
        """Summed cumulative energy counters over all GPUs (mJ), or None."""
        if not self._initialized or self._energy_method != "nvml_energy_counter":
            return None
        try:
            return sum(
                float(self._nv.nvmlDeviceGetTotalEnergyConsumption(h)) for h in self._handles
            )
        except Exception:
            return None

    def read_utilization(self) -> dict[str, float] | None:
        """Mean GPU and memory utilisation over all GPUs, in [0, 1], or None."""
        if not self._initialized:
            return None
        try:
            rates = [self._nv.nvmlDeviceGetUtilizationRates(h) for h in self._handles]
        except Exception:
            return None
        n = len(rates)
        return {
            "gpu_utilization": sum(r.gpu for r in rates) / (100.0 * n),
            "gpu_mem_utilization": sum(r.memory for r in rates) / (100.0 * n),
        }

    def read_diagnostics(self) -> dict[str, float | None]:
        """Device state: mean SM clock (MHz), lowest P-state number (= highest
        performance state of any GPU), hottest temperature (C), device count."""
        out: dict[str, float | None] = {
            "gpu_count": float(self.device_count),
            "gpu_sm_clock_mhz": None,
            "gpu_pstate": None,
            "gpu_temp_c": None,
        }
        if not self._initialized:
            return out
        nv = self._nv
        try:
            clocks = [float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)) for h in self._handles]
            out["gpu_sm_clock_mhz"] = sum(clocks) / len(clocks)
        except Exception:
            pass
        try:
            out["gpu_pstate"] = float(min(int(nv.nvmlDeviceGetPerformanceState(h))
                                          for h in self._handles))
        except Exception:
            pass
        try:
            out["gpu_temp_c"] = float(max(
                nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU) for h in self._handles
            ))
        except Exception:
            pass
        return out

    def shutdown(self) -> None:
        if self._initialized:
            try:
                self._nv.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False


def mj_to_j(energy_mj: float) -> float:
    """Convert millijoules to joules."""
    return energy_mj / 1e3


def mw_to_w(power_mw: float) -> float:
    """Convert milliwatts to watts."""
    return power_mw / 1e3


__all__ = ["NVMLReader", "NVMLMultiReader", "mj_to_j", "mw_to_w"]
