"""Replay engine (Phase 7, I4 — time-based replay).

Turns (node, logical time) into a ``SignalRow`` from that node's recorded trace.
Time is **logical**, not wall-clock: ``t = elapsed * speed``. Scrape cadence is
irrelevant to the result (I4).

- ``load_map(path) → ReplayEngine`` — bind nodes to traces + platform + mode/speed.
- ``row_at(node, logical_time_s) → SignalRow`` — the interval containing ``t``.
- ``trace_duration(node) → float`` — the node's total trace duration.

Modes:
- ``loop``: ``t = logical_time_s % duration`` (wraps around).
- ``once``: ``t = min(logical_time_s, duration)`` (clamps to the last interval).

Returns model inputs only (never targets). Never advances per scrape.
"""

from __future__ import annotations

import json
from bisect import bisect_right
from pathlib import Path
from typing import Any

# Signal-only fields (model inputs) — the replay output schema.
# Everything except identity, interval, measured targets, derived, availability.
from .schema import (
    AVAILABILITY_FIELDS,
    DERIVED_TARGET_FIELDS,
    MEASURED_TARGET_FIELDS,
    MODEL_INPUT_FIELDS,
)
from .trace import read as read_trace

# Fields to STRIP from the replay output (targets + derived + availability).
# Identity fields are kept for context (platform_id is a model input).
_STRIP_FIELDS: frozenset[str] = (
    MEASURED_TARGET_FIELDS | DERIVED_TARGET_FIELDS | AVAILABILITY_FIELDS
)

# The model-input fields plus platform_id (which is in MODEL_INPUT_FIELDS).
_SIGNAL_KEEP: frozenset[str] = MODEL_INPUT_FIELDS


class NodeProfile:
    """One node's replay binding: trace rows, platform, mode, speed."""

    __slots__ = ("node", "platform_id", "rows", "mode", "speed", "_t_starts")

    def __init__(
        self,
        node: str,
        platform_id: str,
        rows: list[dict[str, Any]],
        mode: str = "loop",
        speed: float = 1.0,
    ) -> None:
        if mode not in ("loop", "once"):
            raise ValueError(f"mode must be 'loop' or 'once', got {mode!r}")
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed}")
        self.node = node
        self.platform_id = platform_id
        self.rows = rows
        self.mode = mode
        self.speed = speed
        # Precompute start times for binary search
        self._t_starts = [float(r["t_start_s"]) for r in rows]

    @property
    def duration(self) -> float:
        if not self.rows:
            return 0.0
        return float(self.rows[-1]["t_end_s"]) - float(self.rows[0]["t_start_s"])

    def row_at(self, logical_time_s: float) -> dict[str, Any]:
        """Return the interval containing ``logical_time_s`` (I4).

        Returns interval metadata (``t_start_s``, ``t_end_s``, ``dt_s``) plus the
        model-input signals — NO measured/derived targets or availability flags.
        Callers needing a pure ``SignalRow`` (model inputs only) use
        ``signals_at`` instead.
        """
        t = logical_time_s * self.speed
        if self.mode == "loop":
            dur = self.duration
            t = t % dur if dur > 0 else 0.0
        else:  # once
            dur = self.duration
            t = min(t, dur)
        # Binary search: find the interval where t_start_s <= t < t_end_s
        idx = bisect_right(self._t_starts, t) - 1
        if idx < 0:
            idx = 0
        row = self.rows[idx]
        return _to_interval_plus_signals(row)


def _to_interval_plus_signals(row: dict[str, Any]) -> dict[str, Any]:
    """Return interval metadata + model-input signals; strip targets/derived/availability (I1)."""
    keep = _SIGNAL_KEEP | {"t_start_s", "t_end_s", "dt_s"}
    return {k: v for k, v in row.items() if k in keep}


def _to_signal_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return ONLY MODEL_INPUT_FIELDS — a pure SignalRow (I1).

    Used by ``signals_at`` so the model/exporter never see identity, interval,
    or any target fields.
    """
    return {k: v for k, v in row.items() if k in _SIGNAL_KEEP}


class ReplayEngine:
    """Recorded-trace replay implementing ``SignalSource`` (Component G, I10, I4).

    Created by ``load_map(path)``. Provides ``nodes()``, ``platform_id(node)``,
    ``signals_at(node, t)``, plus ``row_at(node, t)`` and ``trace_duration(node)``.
    """

    def __init__(self, profiles: dict[str, NodeProfile]) -> None:
        self._profiles = profiles

    def nodes(self) -> list[str]:
        return list(self._profiles.keys())

    def platform_id(self, node: str) -> str:
        return self._profile(node).platform_id

    def trace_duration(self, node: str) -> float:
        return self._profile(node).duration

    def row_at(self, node: str, logical_time_s: float) -> dict[str, Any]:
        return self._profile(node).row_at(logical_time_s)

    def signals_at(self, node: str, logical_time_s: float) -> dict[str, Any]:
        """Return a pure ``SignalRow`` (MODEL_INPUT_FIELDS only) for the model/exporter (I1)."""
        interval_row = self.row_at(node, logical_time_s)
        return {k: v for k, v in interval_row.items() if k in _SIGNAL_KEEP}

    def _profile(self, node: str) -> NodeProfile:
        if node not in self._profiles:
            raise KeyError(f"unknown node {node!r}; known: {sorted(self._profiles)}")
        return self._profiles[node]


def load_map(path: str | Path) -> ReplayEngine:
    """Load a node-profile map (JSON) and build a ``ReplayEngine``.

    Map format::

        {
          "nodes": {
            "n0": {"trace_path": "...csv", "platform_id": "...", "mode": "loop", "speed": 1.0},
            ...
          }
        }
    """
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if "nodes" not in data or not isinstance(data["nodes"], dict):
        raise ValueError(f"node-profile map missing 'nodes' dict: {path}")
    profiles: dict[str, NodeProfile] = {}
    base_dir = p.parent
    for node, spec in data["nodes"].items():
        if "trace_path" not in spec:
            raise ValueError(f"node {node!r} missing trace_path")
        trace_path = base_dir / spec["trace_path"]
        if not trace_path.is_file():
            raise FileNotFoundError(f"node {node!r}: trace not found: {trace_path}")
        rows = read_trace(trace_path)
        if not rows:
            raise ValueError(f"node {node!r}: trace is empty: {trace_path}")
        platform_id = spec.get("platform_id", "unknown")
        mode = spec.get("mode", "loop")
        speed = float(spec.get("speed", 1.0))
        profiles[node] = NodeProfile(node, platform_id, rows, mode, speed)
    return ReplayEngine(profiles)


__all__ = ["NodeProfile", "ReplayEngine", "load_map"]
