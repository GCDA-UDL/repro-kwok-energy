"""OS signal measurement (Phase 12).

Reads CPU utilization, CPU frequency, disk/net throughput, and a memory-bandwidth
proxy from /proc and /sys counters. All deltas are counter-based over a time
interval.
"""

from __future__ import annotations

from pathlib import Path

PROC_STAT = Path("/proc/stat")
PROC_DISKSTATS = Path("/proc/diskstats")
PROC_NET_DEV = Path("/proc/net/dev")
CPUFREQ_ROOT = Path("/sys/devices/system/cpu")


def _read_proc_stat() -> dict[str, list[int]]:
    """Parse /proc/stat first line (aggregate CPU)."""
    if not PROC_STAT.is_file():
        return {}
    line = PROC_STAT.read_text().splitlines()[0]
    parts = line.split()
    if parts[0] != "cpu":
        return {}
    vals = [int(x) for x in parts[1:]]
    return {"cpu": vals}


def cpu_util(prev: list[int], cur: list[int]) -> float:
    """Compute CPU utilization in [0, 1] from two /proc/stat snapshots.

    /proc/stat: user, nice, system, idle, iowait, irq, softirq, steal, guest, ...
    idle = idle + iowait; total = sum(all). util = 1 - idle/total.
    """
    prev_idle = prev[3] + (prev[4] if len(prev) > 4 else 0)
    cur_idle = cur[3] + (cur[4] if len(cur) > 4 else 0)
    prev_total = sum(prev)
    cur_total = sum(cur)
    total_delta = cur_total - prev_total
    idle_delta = cur_idle - prev_idle
    if total_delta == 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - idle_delta / total_delta))


def read_cpu_util_snapshot() -> list[int] | None:
    """Read a raw /proc/stat CPU snapshot for delta computation."""
    stat = _read_proc_stat()
    return stat.get("cpu")


def read_cpufreq_mhz() -> float | None:
    """Mean CPU frequency (MHz) from scaling_cur_freq across online cores."""
    freqs: list[float] = []
    if not CPUFREQ_ROOT.is_dir():
        return None
    for cpu_dir in sorted(CPUFREQ_ROOT.glob("cpu[0-9]*/cpufreq")):
        f = cpu_dir / "scaling_cur_freq"
        if f.is_file():
            try:
                freqs.append(int(f.read_text().strip()) / 1000.0)  # kHz → MHz
            except (ValueError, OSError):
                continue
    return sum(freqs) / len(freqs) if freqs else None


def read_disk_stats(device: str = "sda") -> dict[str, int] | None:
    """Read sectors read/written for a disk from /proc/diskstats."""
    if not PROC_DISKSTATS.is_file():
        return None
    for line in PROC_DISKSTATS.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 10 and parts[2] == device:
            # fields 5 (sectors read) and 9 (sectors written); sector = 512B
            return {"sectors_read": int(parts[5]), "sectors_written": int(parts[9])}
    return None


def read_net_stats() -> dict[str, dict[str, int]] | None:
    """Read rx/tx bytes per interface from /proc/net/dev."""
    if not PROC_NET_DEV.is_file():
        return None
    result: dict[str, dict[str, int]] = {}
    lines = PROC_NET_DEV.read_text().splitlines()
    for line in lines[2:]:  # skip headers
        parts = line.split(":")
        if len(parts) < 2:
            continue
        iface = parts[0].strip()
        rx_tx = parts[1].split()
        if len(rx_tx) >= 9:
            result[iface] = {"rx_bytes": int(rx_tx[0]), "tx_bytes": int(rx_tx[8])}
    return result


def sectors_to_bps(sectors_delta: int, dt_s: float, sector_size: int = 512) -> float:
    """Convert a sectors delta to bytes/second."""
    if dt_s <= 0:
        return 0.0
    return (sectors_delta * sector_size) / dt_s


def bytes_to_bps(bytes_delta: int, dt_s: float) -> float:
    """Convert a bytes delta to bytes/second."""
    if dt_s <= 0:
        return 0.0
    return bytes_delta / dt_s


__all__ = [
    "cpu_util",
    "read_cpu_util_snapshot",
    "read_cpufreq_mhz",
    "read_disk_stats",
    "read_net_stats",
    "sectors_to_bps",
    "bytes_to_bps",
]
