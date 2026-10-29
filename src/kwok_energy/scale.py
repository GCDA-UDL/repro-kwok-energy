"""Scale demo helpers (Phase 16, Component I).

Exporter-only mode is the required, hardware-free scale path. It reports the
exporter metric surface and scrape accounting without requiring a live KWOK
cluster. Live exporter PID resource accounting can be layered on top by passing
measured values into ``run``.
"""

from __future__ import annotations

import csv
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_SCALE_COLUMNS: tuple[str, ...] = (
    "node_count",
    "series_count",
    "samples_per_second",
    "exporter_cpu_percent",
    "exporter_memory_mb",
    "scrape_failure_rate",
    "duration_s",
    "mode",
)

# Exporter metric surface per node in the current contract:
# node_power_watts package/dram/gpu/total + node_energy_joules_total + one signal
# aggregate series used by the scale seam. Keep this deterministic and explicit.
_SERIES_PER_NODE = 6


@dataclass(frozen=True)
class ScaleRow:
    """One scale-demo result row."""

    node_count: int
    series_count: int
    samples_per_second: float
    exporter_cpu_percent: float
    exporter_memory_mb: float
    scrape_failure_rate: float
    duration_s: float
    mode: str

    def to_dict(self) -> dict[str, str]:
        return {
            "node_count": str(self.node_count),
            "series_count": str(self.series_count),
            "samples_per_second": repr(float(self.samples_per_second)),
            "exporter_cpu_percent": repr(float(self.exporter_cpu_percent)),
            "exporter_memory_mb": repr(float(self.exporter_memory_mb)),
            "scrape_failure_rate": repr(float(self.scrape_failure_rate)),
            "duration_s": repr(float(self.duration_s)),
            "mode": self.mode,
        }


@dataclass(frozen=True)
class ResourceUsage:
    """Exporter process resource usage snapshot."""

    cpu_percent: float
    memory_mb: float


def expected_series_count(node_count: int) -> int:
    """Return expected Prometheus series count for the exporter metric surface."""

    if node_count < 0:
        raise ValueError("node_count must be non-negative")
    return node_count * _SERIES_PER_NODE


def scrape_failure_rate(*, attempted: int, failed: int) -> float:
    """Return failed/attempted, with zero attempts defined as 0.0."""

    if attempted < 0 or failed < 0:
        raise ValueError("attempted and failed scrapes must be non-negative")
    if failed > attempted:
        raise ValueError("failed scrapes cannot exceed attempted scrapes")
    if attempted == 0:
        return 0.0
    return failed / attempted


def run(
    *,
    node_count: int,
    mode: str = "exporter-only",
    duration_s: float = 1.0,
    scrape_attempts: int = 1,
    scrape_failures: int = 0,
    exporter_cpu_percent: float = 0.0,
    exporter_memory_mb: float = 0.0,
) -> ScaleRow:
    """Run the hardware-free scale accounting seam and return one ScaleRow."""

    if mode != "exporter-only":
        raise ValueError("only exporter-only mode is supported without a live KWOK marker")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    series_count = expected_series_count(node_count)
    return ScaleRow(
        node_count=node_count,
        series_count=series_count,
        samples_per_second=series_count * scrape_attempts / duration_s,
        exporter_cpu_percent=exporter_cpu_percent,
        exporter_memory_mb=exporter_memory_mb,
        scrape_failure_rate=scrape_failure_rate(attempted=scrape_attempts, failed=scrape_failures),
        duration_s=duration_s,
        mode=mode,
    )


def measure_prediction_overhead(
    node_count: int, coeffs: object, duration_s: float = 2.0
) -> ResourceUsage:
    """Real exporter-core overhead: CPU% + peak RSS while predicting N nodes' power.

    The exporter's per-tick work is predicting every node's power; this measures that
    work for ``node_count`` nodes over ``duration_s`` (CPU time via getrusage, peak RSS).
    A genuine, monotone-in-N overhead (the previous CLI reported 0). Returns 0 on
    platforms without ``resource`` (e.g. Windows) — the bench hosts are Linux.
    """
    try:
        import resource
    except Exception:
        return ResourceUsage(cpu_percent=0.0, memory_mb=0.0)
    from .energy_model import FeatureWindow, predict_power
    from .schema import MODEL_INPUT_FIELDS

    def _sig(i: int) -> dict:
        s = {f: (getattr(coeffs, "platform_id", "p") if f == "platform_id"
                 else "unavailable" if f == "mem_proxy_method" else 0.0)
             for f in MODEL_INPUT_FIELDS}
        s["cpu_utilization"] = (i % 100) / 100.0
        return s

    windows = [FeatureWindow(current=_sig(i)) for i in range(max(1, node_count))]
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < duration_s:
        for w in windows:
            predict_power(w, coeffs)  # type: ignore[arg-type]
    wall = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu_s = (r1.ru_utime + r1.ru_stime) - (r0.ru_utime + r0.ru_stime)
    cpu_pct = 100.0 * cpu_s / wall if wall > 0 else 0.0
    mem_mb = r1.ru_maxrss / 1024.0  # Linux ru_maxrss is KiB
    return ResourceUsage(cpu_percent=round(cpu_pct, 2), memory_mb=round(mem_mb, 2))


def process_resource_usage(pid: int) -> ResourceUsage:
    """Measure memory for ``pid`` using Linux ``/proc`` without new deps.

    CPU percent is a rate and requires two timed samples; live runners can pass
    the measured rate into ``run``. This snapshot still attributes memory to the
    exporter PID and returns a non-negative CPU placeholder.
    """

    if pid <= 0:
        raise ValueError("pid must be positive")
    statm = Path(f"/proc/{pid}/statm")
    fields = statm.read_text(encoding="utf-8").split()
    resident_pages = int(fields[1]) if len(fields) > 1 else 0
    page_size = os.sysconf("SC_PAGE_SIZE")
    memory_mb = resident_pages * page_size / (1024 * 1024)
    return ResourceUsage(cpu_percent=0.0, memory_mb=memory_mb)


def write_csv(rows: Sequence[ScaleRow], path: str | Path) -> None:
    """Write ScaleRows with stable column order and float formatting (R1)."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_SCALE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


__all__ = [
    "ResourceUsage",
    "ScaleRow",
    "expected_series_count",
    "measure_prediction_overhead",
    "process_resource_usage",
    "run",
    "scrape_failure_rate",
    "write_csv",
]
