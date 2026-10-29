"""Metrics exporter (Phase 8, Component F, I4/I5/I6).

Serves replayed signals + predicted power as Prometheus metrics.

Structure: a **background ticker** reads ``SignalSource``, builds ``FeatureWindow``s,
computes predictions into a cache, and updates the **analytic** energy counter
(exact integral of power over logical time — not a tick-sum, I5). The ``/metrics``
handler only serializes the cache — it MUST NOT advance time or run the model.

Metrics:
- ``node_power_watts{node,component=package|dram|gpu|total,platform_id}`` (Gauge)
- ``node_energy_joules_total{node,platform_id}`` (Counter — monotonic, exact integral)
- ``kwok_energy_*{node,platform_id}`` signal gauges

Coeffs routing: ``platform_id → Coeffs`` map; unknown platform_id fails loudly.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest

from .energy_model import Coeffs, FeatureWindow, PowerPrediction, predict_power
from .signal_source import SignalSource

# Signal fields to expose as kwok_energy_* gauges (numeric only).
_SIGNAL_GAUGE_FIELDS = [
    "cpu_utilization",
    "cpu_frequency_mhz",
    "mem_bandwidth_proxy",
    "gpu_utilization",
    "gpu_mem_utilization",
    "disk_read_bps",
    "disk_write_bps",
    "net_rx_bps",
    "net_tx_bps",
]


class UnknownPlatformError(KeyError):
    """A node declares a platform_id not in the coeffs map."""


class EnergyExporter:
    """Predicts power/energy from a ``SignalSource`` and serves Prometheus metrics.

    The ticker advances logical time, reads signals, predicts power, and updates
    the analytic energy counter + prediction cache. The ``/metrics`` handler reads
    only the cache.
    """

    def __init__(
        self,
        source: SignalSource,
        coeffs_map: dict[str, Coeffs],
        *,
        include_io: bool = False,
    ) -> None:
        self._source = source
        self._coeffs_map = coeffs_map
        self._include_io = include_io

        # Private registry (test isolation)
        self._registry = CollectorRegistry()

        # Metrics
        self._power = Gauge(
            "node_power_watts", "Predicted power (W)", ["node", "component", "platform_id"],
            registry=self._registry,
        )
        self._energy = Counter(
            "node_energy_joules_total", "Cumulative predicted energy (J) — analytic integral",
            ["node", "platform_id"], registry=self._registry,
        )
        self._signal_gauges: dict[str, Gauge] = {
            field: Gauge(
                f"kwok_energy_{field}", f"Replayed signal: {field}", ["node", "platform_id"],
                registry=self._registry,
            )
            for field in _SIGNAL_GAUGE_FIELDS
        }

        # Per-node state
        self._last_t: dict[str, float] = {}
        self._cache: dict[str, PowerPrediction] = {}
        self._lock = threading.Lock()

    # --- the ticker (advances time + runs model) ----------------------------

    def tick(self, current_logical_time_s: float) -> None:
        """Advance logical time: read signals, predict, update counter + cache.

        This is the ONLY method that advances time or runs the model.
        """
        with self._lock:
            for node in self._source.nodes():
                pid = self._source.platform_id(node)
                if pid not in self._coeffs_map:
                    raise UnknownPlatformError(
                        f"node {node!r} declares platform_id {pid!r} not in coeffs map "
                        f"(known: {sorted(self._coeffs_map)})"
                    )
                coeffs = self._coeffs_map[pid]
                signals = self._source.signals_at(node, current_logical_time_s)

                window = FeatureWindow(current=signals, history=())
                pred = predict_power(window, coeffs, include_io=self._include_io)

                # Analytic energy counter: exact integral (I5)
                last = self._last_t.get(node, current_logical_time_s)
                dt = max(0.0, current_logical_time_s - last)
                if dt > 0:
                    self._energy.labels(node=node, platform_id=pid).inc(pred.total * dt)

                self._cache[node] = pred
                self._last_t[node] = current_logical_time_s

                # Update signal gauges from the cache
                for field in _SIGNAL_GAUGE_FIELDS:
                    val = signals.get(field)
                    if val is not None:
                        self._signal_gauges[field].labels(node=node, platform_id=pid).set(float(val))

                # Update power gauges
                self._power.labels(node=node, component="total", platform_id=pid).set(pred.total)
                for comp in ("package", "dram", "gpu", "io"):
                    val = getattr(pred, comp)
                    if val is not None:
                        self._power.labels(node=node, component=comp, platform_id=pid).set(val)

    # --- /metrics handler (serializes cache only — NEVER advances) ----------

    def metrics_text(self) -> bytes:
        """Serialize the current cache + counter as Prometheus exposition text."""
        return generate_latest(self._registry)

    def get_prediction(self, node: str) -> PowerPrediction | None:
        """Read the cached prediction for ``node`` (does NOT advance time)."""
        with self._lock:
            return self._cache.get(node)

    def get_energy(self, node: str) -> float:
        """Read the cumulative energy counter for ``node`` (J)."""
        pid = self._source.platform_id(node)
        return self._energy.labels(node=node, platform_id=pid)._value.get()

    def cluster_power(self) -> float:
        """Sum per-node cached ``total`` predictions (watts). Zero before first tick."""
        with self._lock:
            return sum(
                pred.total for pred in self._cache.values() if pred is not None
            )

    def cluster_energy(self) -> float:
        """Sum per-node analytic energy counters (joules). Zero before first tick."""
        total = 0.0
        for node in self._source.nodes():
            pid = self._source.platform_id(node)
            total += self._energy.labels(node=node, platform_id=pid)._value.get()
        return total


# --- HTTP server wrapper (thin; just serves metrics_text) -------------------

class _MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler that serializes the cache — never advances time."""

    exporter: EnergyExporter  # injected via factory

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/metrics":
            self.send_error(404)
            return
        body = self.exporter.metrics_text()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # silent


def make_handler(exporter: EnergyExporter) -> type[_MetricsHandler]:
    """Create a handler class bound to the given exporter."""
    return type("BoundMetricsHandler", (_MetricsHandler,), {"exporter": exporter})


def start_server(exporter: EnergyExporter, port: int = 9100, **kwargs: Any) -> HTTPServer:
    """Start the ``/metrics`` HTTP server in a background daemon thread."""
    handler = make_handler(exporter)
    server = HTTPServer(("0.0.0.0", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


__all__ = [
    "UnknownPlatformError",
    "EnergyExporter",
    "start_server",
    "make_handler",
]
