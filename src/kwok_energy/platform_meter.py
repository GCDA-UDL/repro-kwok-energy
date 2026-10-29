"""Whole-node power from the platform meter (schema v1.1).

Some servers expose the board's power meter through the ACPI ``power_meter``
hwmon driver (ACPI000D; on Intel boards backed by Node Manager, "Meter measures
total domain"). ``power1_average`` reports the node's input power in microwatts,
averaged over ``power1_average_interval`` milliseconds, and is world-readable.

This is a *validation-only* target: the energy model never reads it. It sits one
level above RAPL+NVML (it includes PSU losses, fans, board, disks and NICs), so
it measures how much of the node's power the component model covers.
"""

from __future__ import annotations

from pathlib import Path

HWMON_ROOT = Path("/sys/class/hwmon")


def find_power_meter(root: Path = HWMON_ROOT) -> Path | None:
    """Return the hwmon directory whose ``name`` is ``power_meter``, or None."""
    if not root.is_dir():
        return None
    for d in sorted(root.iterdir()):
        try:
            if (d / "name").read_text().strip() == "power_meter" and (
                d / "power1_average"
            ).is_file():
                return d
        except OSError:
            continue
    return None


class PlatformMeterReader:
    """Reads the node power meter; ``available`` is False when the host has none."""

    def __init__(self, root: Path = HWMON_ROOT) -> None:
        self._dir = find_power_meter(root)
        self.average_interval_ms: float | None = None
        self.accuracy: str | None = None
        self.model: str | None = None
        if self._dir is not None:
            self.average_interval_ms = _read_float(self._dir / "power1_average_interval")
            self.accuracy = _read_text(self._dir / "power1_accuracy")
            self.model = _read_text(self._dir / "power1_model_number")
            if self.read_power_w() is None:
                self._dir = None

    @property
    def available(self) -> bool:
        return self._dir is not None

    def read_power_w(self) -> float | None:
        """Latest averaged node power (W), or None."""
        if self._dir is None:
            return None
        uw = _read_float(self._dir / "power1_average")
        return None if uw is None else uw / 1e6


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _read_float(path: Path) -> float | None:
    text = _read_text(path)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


__all__ = ["PlatformMeterReader", "find_power_meter"]
