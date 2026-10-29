"""Trace collector (Phase 12).

Snapshots all counters (RAPL, NVML, OS) near each interval boundary, records
actual ``t_start_s/t_end_s``, and writes a valid interval trace at a configurable
sampling interval. Graceful skip on missing devices.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .os_signals import (
    cpu_util,
    read_cpu_util_snapshot,
    read_cpufreq_mhz,
    read_disk_stats,
    read_net_stats,
)
from .rapl import DEFAULT_DRAM_MAX_UJ, DEFAULT_PKG_MAX_UJ, RAPLReader, delta, uj_to_j


def collect_snapshot(
    rapl: RAPLReader,
    *,
    nvml_reader: Any | None = None,
    platform_id: str = "unknown",
    workload: str = "unknown",
    run_id: str = "r1",
    node: str = "n0",
) -> dict[str, Any]:
    """Collect one snapshot of all available counters."""
    snap: dict[str, Any] = {
        "platform_id": platform_id,
        "workload": workload,
        "run_id": run_id,
        "node": node,
        "schema_version": "1",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # RAPL
    snap["_pkg_uj"] = rapl.read_package_uj()
    snap["_dram_uj"] = rapl.read_dram_uj()
    snap["_pkg_max_uj"] = rapl.pkg_max_uj
    snap["_dram_max_uj"] = rapl.dram_max_uj
    snap["rapl_pkg_available"] = rapl.available
    snap["rapl_dram_available"] = rapl.dram_available
    # OS
    snap["_cpu_stat"] = read_cpu_util_snapshot()
    snap["cpu_frequency_mhz"] = read_cpufreq_mhz() or 0.0
    snap["_disk"] = read_disk_stats()
    net = read_net_stats()
    snap["_net"] = net
    # GPU (NVML for NVIDIA, DRM-hwmon for AMD/Intel — same reader interface)
    if nvml_reader is not None and nvml_reader.available:
        util = nvml_reader.read_utilization() or {}  # read once (was read twice)
        snap["gpu_utilization"] = util.get("gpu_utilization", 0.0)
        snap["gpu_mem_utilization"] = util.get("gpu_mem_utilization", 0.0)
        snap["_gpu_energy_mj"] = nvml_reader.read_energy_mj()
        # instantaneous power enables the integrated-power fallback (no energy counter)
        snap["_gpu_power_w"] = (
            nvml_reader.read_power_w() if hasattr(nvml_reader, "read_power_w") else None
        )
        snap["gpu_available"] = True
        snap["gpu_energy_method"] = nvml_reader.energy_method
    else:
        snap["gpu_utilization"] = 0.0
        snap["gpu_mem_utilization"] = 0.0
        snap["_gpu_power_w"] = None
        snap["gpu_available"] = False
        snap["gpu_energy_method"] = "unavailable"
    snap["mem_bandwidth_proxy"] = 0.0
    snap["mem_proxy_method"] = "unavailable"
    snap["disk_read_bps"] = 0.0
    snap["disk_write_bps"] = 0.0
    snap["net_rx_bps"] = 0.0
    snap["net_tx_bps"] = 0.0
    return snap


def compute_interval(
    prev: dict[str, Any], cur: dict[str, Any], dt_s: float
) -> dict[str, Any]:
    """Compute interval-level deltas between two snapshots."""
    row: dict[str, Any] = {
        "schema_version": "1",
        "run_id": cur.get("run_id", "r1"),
        "workload": cur.get("workload", "unknown"),
        "node": cur.get("node", "n0"),
        "timestamp": cur.get("timestamp", ""),
        "t_start_s": 0.0,
        "t_end_s": dt_s,
        "dt_s": dt_s,
        "platform_id": cur.get("platform_id", "unknown"),
        "cpu_frequency_mhz": cur.get("cpu_frequency_mhz", 0.0),
        "mem_bandwidth_proxy": cur.get("mem_bandwidth_proxy", 0.0),
        "mem_proxy_method": cur.get("mem_proxy_method", "unavailable"),
        "gpu_utilization": cur.get("gpu_utilization", 0.0),
        "gpu_mem_utilization": cur.get("gpu_mem_utilization", 0.0),
        "disk_read_bps": cur.get("disk_read_bps", 0.0),
        "disk_write_bps": cur.get("disk_write_bps", 0.0),
        "net_rx_bps": cur.get("net_rx_bps", 0.0),
        "net_tx_bps": cur.get("net_tx_bps", 0.0),
    }
    # CPU util
    prev_stat = prev.get("_cpu_stat")
    cur_stat = cur.get("_cpu_stat")
    if prev_stat and cur_stat:
        row["cpu_utilization"] = cpu_util(prev_stat, cur_stat)
    else:
        row["cpu_utilization"] = 0.0
    # RAPL energy deltas (joules). Use the per-domain wrap modulus from sysfs
    # (max_energy_range_uj) when available; fall back to the documented default.
    if prev.get("_pkg_uj") is not None and cur.get("_pkg_uj") is not None:
        pkg_max = cur.get("_pkg_max_uj") or DEFAULT_PKG_MAX_UJ
        pkg_delta = delta(prev["_pkg_uj"], cur["_pkg_uj"], pkg_max)
        row["rapl_pkg_joules"] = uj_to_j(pkg_delta)
        row["rapl_pkg_watts_mean"] = row["rapl_pkg_joules"] / dt_s if dt_s > 0 else 0.0
    else:
        row["rapl_pkg_joules"] = None
        row["rapl_pkg_watts_mean"] = None
    if prev.get("_dram_uj") is not None and cur.get("_dram_uj") is not None:
        dram_max = cur.get("_dram_max_uj") or DEFAULT_DRAM_MAX_UJ
        dram_delta = delta(prev["_dram_uj"], cur["_dram_uj"], dram_max)
        row["rapl_dram_joules"] = uj_to_j(dram_delta)
        row["rapl_dram_watts_mean"] = row["rapl_dram_joules"] / dt_s if dt_s > 0 else 0.0
    else:
        row["rapl_dram_joules"] = None
        row["rapl_dram_watts_mean"] = None
    row["rapl_pkg_available"] = cur.get("rapl_pkg_available", False)
    row["rapl_dram_available"] = cur.get("rapl_dram_available", False)
    # GPU energy: prefer the reset-safe energy counter; else integrate power (P*dt,
    # trapezoidal over the interval's boundary samples) for GPUs without a counter
    # (e.g. amdgpu power1_average). I2: joules canonical, watts derived.
    if prev.get("_gpu_energy_mj") is not None and cur.get("_gpu_energy_mj") is not None:
        from .nvml import mj_to_j
        gpu_delta = cur["_gpu_energy_mj"] - prev["_gpu_energy_mj"]
        if gpu_delta < 0:
            gpu_delta = 0  # reset-safe
        row["gpu_energy_joules"] = mj_to_j(gpu_delta)
        row["gpu_watts_mean"] = row["gpu_energy_joules"] / dt_s if dt_s > 0 else 0.0
    elif prev.get("_gpu_power_w") is not None and cur.get("_gpu_power_w") is not None:
        mean_w = 0.5 * (float(prev["_gpu_power_w"]) + float(cur["_gpu_power_w"]))  # trapezoid
        row["gpu_energy_joules"] = mean_w * dt_s
        row["gpu_watts_mean"] = mean_w
    else:
        row["gpu_energy_joules"] = None
        row["gpu_watts_mean"] = None
    row["gpu_available"] = cur.get("gpu_available", False)
    row["gpu_energy_method"] = cur.get("gpu_energy_method", "unavailable")
    return row


def make_gpu_reader() -> Any:
    """Return a GPU reader by vendor: NVML (NVIDIA) → DRM hwmon (AMD/Intel).

    Both readers share the same interface (``available``, ``energy_method``,
    ``read_energy_mj``, ``read_power_w``, ``read_utilization``, ``shutdown``), so the
    collector treats them uniformly. Returns an unavailable NVML reader if nothing
    is present (graceful skip).
    """
    from .nvml import NVMLReader

    nvml = NVMLReader()
    if nvml.available:
        return nvml
    from .gpu_drm import GPUDRMReader

    drm = GPUDRMReader()
    if drm.available:
        return drm
    return nvml  # unavailable; collector records null GPU energy


def run_collector(
    output_path: str | Path,
    *,
    n_intervals: int = 10,
    interval_s: float = 0.5,
    platform_id: str = "unknown",
    workload: str = "unknown",
    run_id: str = "r1",
    node: str = "n0",
    rapl_root: Path | None = None,
) -> Path:
    """Run the collector for ``n_intervals`` and write a trace.

    Graceful skip: if RAPL is unavailable, the trace records null energy.
    """
    from .trace import write as write_trace

    rapl = RAPLReader(rapl_root)
    gpu = make_gpu_reader()

    rows: list[dict[str, Any]] = []
    prev_snap = None

    for i in range(n_intervals + 1):
        snap = collect_snapshot(
            rapl, nvml_reader=gpu, platform_id=platform_id,
            workload=workload, run_id=run_id, node=node,
        )
        if prev_snap is not None:
            dt = interval_s
            row = compute_interval(prev_snap, snap, dt)
            row["t_start_s"] = (i - 1) * interval_s
            row["t_end_s"] = i * interval_s
            rows.append(row)
        prev_snap = snap
        if i < n_intervals:
            time.sleep(interval_s)

    gpu.shutdown()
    write_trace(rows, output_path)
    return Path(output_path)


__all__ = ["collect_snapshot", "compute_interval", "run_collector", "make_gpu_reader"]
