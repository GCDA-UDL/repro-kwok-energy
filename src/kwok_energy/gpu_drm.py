"""AMD/Intel GPU measurement via DRM hwmon sysfs (vendor-agnostic-measurement).

Complements ``nvml.py`` (NVIDIA). Reads GPU power/energy and utilization from the
DRM device's hwmon node, for ``amdgpu`` (AMD) and ``i915``/``xe`` (Intel) GPUs:

  * power:  ``/sys/class/drm/card*/device/hwmon/hwmon*/power1_average``  (μW, instantaneous)
  * energy: ``/sys/class/drm/card*/device/hwmon/hwmon*/energy1_input``   (μJ, counter)
  * util (AMD): ``device/gpu_busy_percent``, ``device/mem_busy_percent``  (0..100)

Energy method priority: ``drm_energy_counter`` (reset-safe) → ``integrated_power``
(the collector integrates ``read_power_w`` as P·dt). Graceful skip when no supported
GPU is present. Mirrors the ``NVMLReader`` shape so ``collect`` can dispatch by vendor.
"""

from __future__ import annotations

from pathlib import Path

DRM_ROOT = Path("/sys/class/drm")
_DRIVER_VENDOR = {"amdgpu": "amd", "i915": "intel", "xe": "intel"}


def _read_text(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def _read_int(p: Path) -> int | None:
    v = _read_text(p)
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        return None


def _driver_of(device: Path) -> str | None:
    """Resolve the bound kernel driver: symlink first, then uevent DRIVER=."""
    link = device / "driver"
    try:
        if link.exists():
            return Path(link.resolve()).name
    except OSError:
        pass
    uevent = _read_text(device / "uevent") or ""
    for line in uevent.splitlines():
        if line.startswith("DRIVER="):
            return line.split("=", 1)[1].strip()
    return None


def discover_drm_gpus(drm_root: Path | None = None) -> list[dict[str, object]]:
    """Discover supported DRM GPUs with a power or energy hwmon node.

    Returns ``{card, vendor, driver, device, hwmon, power_file, energy_file}`` dicts.
    """
    root = drm_root or DRM_ROOT
    found: list[dict[str, object]] = []
    if not root.is_dir():
        return found
    for card in sorted(root.glob("card[0-9]*")):
        device = card / "device"
        drv = _driver_of(device)
        if drv not in _DRIVER_VENDOR:
            continue
        for hw in sorted((device / "hwmon").glob("hwmon*")) if (device / "hwmon").is_dir() else []:
            power = hw / "power1_average"
            energy = hw / "energy1_input"
            has_power = _read_int(power) is not None
            has_energy = _read_int(energy) is not None
            if not (has_power or has_energy):
                continue
            found.append({
                "card": card.name,
                "vendor": _DRIVER_VENDOR[drv],
                "driver": drv,
                "device": device,
                "hwmon": hw,
                "power_file": power if has_power else None,
                "energy_file": energy if has_energy else None,
            })
            break  # one hwmon per card is enough
    return found


class GPUDRMReader:
    """Reads AMD/Intel GPU power/energy/util via DRM hwmon. Skips when unavailable."""

    def __init__(self, drm_root: Path | None = None) -> None:
        self._gpus = discover_drm_gpus(drm_root)
        self._g = self._gpus[0] if self._gpus else None
        if self._g is None:
            self._energy_method = "unavailable"
        elif self._g["energy_file"] is not None:
            self._energy_method = "drm_energy_counter"
        else:
            self._energy_method = "integrated_power"

    @property
    def available(self) -> bool:
        return self._g is not None

    @property
    def vendor(self) -> str | None:
        return self._g["vendor"] if self._g else None  # type: ignore[index]

    @property
    def energy_method(self) -> str:
        return self._energy_method

    def read_power_mw(self) -> float | None:
        """Current GPU power (mW) from ``power1_average`` (μW), or None."""
        if self._g is None or self._g["power_file"] is None:
            return None
        v = _read_int(self._g["power_file"])  # type: ignore[arg-type]
        return v / 1e3 if v is not None else None  # μW → mW

    def read_power_w(self) -> float | None:
        mw = self.read_power_mw()
        return mw / 1e3 if mw is not None else None

    def read_energy_mj(self) -> float | None:
        """Cumulative GPU energy (mJ) from ``energy1_input`` (μJ), or None."""
        if self._g is None or self._energy_method != "drm_energy_counter":
            return None
        v = _read_int(self._g["energy_file"])  # type: ignore[arg-type]
        return v / 1e3 if v is not None else None  # μJ → mJ

    def read_utilization(self) -> dict[str, float] | None:
        """GPU + memory busy fraction in [0,1] (AMD); Intel hwmon has no busy %."""
        if self._g is None:
            return None
        device: Path = self._g["device"]  # type: ignore[assignment]
        gpu = _read_int(device / "gpu_busy_percent")
        mem = _read_int(device / "mem_busy_percent")
        if gpu is None and mem is None:
            return None
        return {
            "gpu_utilization": (gpu or 0) / 100.0,
            "gpu_mem_utilization": (mem or 0) / 100.0,
        }

    def shutdown(self) -> None:  # parity with NVMLReader
        return None


__all__ = ["GPUDRMReader", "discover_drm_gpus"]
