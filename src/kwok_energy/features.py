"""History features derived from the signal stream (schema v1.1).

``gpu_util_ewma`` is an exponentially weighted moving average of
``gpu_utilization`` over the node's continuous timeline. GPU drivers keep a high
power state for tens of seconds after a kernel finishes, so a GPU that has just
been busy draws well above its cold-idle power while reporting 0 % utilisation.
The current-sample model cannot see that tail; the EWMA can.

The feature is a pure function of past *signals* (never of measured power), so it
satisfies the leakage rule (I1) and a simulator can compute it online from the
same utilisation stream it replays.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_TAU_S = 20.0


class GPUUtilEWMA:
    """Online EWMA of GPU utilisation with time constant ``tau_s`` seconds.

    ``update(util, dt_s)`` folds one interval of mean utilisation ``util`` lasting
    ``dt_s`` seconds and returns the EWMA at the end of the interval. Because the
    input is held constant over the interval, the update is exact for any ``dt_s``.
    """

    def __init__(self, tau_s: float = DEFAULT_TAU_S, initial: float = 0.0) -> None:
        if tau_s <= 0:
            raise ValueError("tau_s must be positive")
        self.tau_s = float(tau_s)
        self.value = float(initial)

    def update(self, util: float, dt_s: float) -> float:
        if dt_s < 0:
            raise ValueError("dt_s must be non-negative")
        a = math.exp(-dt_s / self.tau_s)
        self.value = a * self.value + (1.0 - a) * float(util)
        return self.value


def add_gpu_util_ewma(
    rows: Iterable[Mapping[str, Any]], tau_s: float = DEFAULT_TAU_S
) -> list[dict[str, Any]]:
    """Return copies of time-ordered ``rows`` with ``gpu_util_ewma`` recomputed.

    ``rows`` must be one node's continuous timeline in time order (for a capture
    session, every interval including the gaps between workloads).
    """
    ewma = GPUUtilEWMA(tau_s)
    out: list[dict[str, Any]] = []
    for r in rows:
        row = dict(r)
        row["gpu_util_ewma"] = ewma.update(float(row.get("gpu_utilization") or 0.0),
                                           float(row.get("dt_s") or 0.0))
        out.append(row)
    return out


def retau_rows(
    rows: Iterable[Mapping[str, Any]], session_rows: Iterable[Mapping[str, Any]], tau_s: float
) -> list[dict[str, Any]]:
    """Copies of ``rows`` with ``gpu_util_ewma`` recomputed at ``tau_s`` on the session timeline.

    The history signal of an interval depends on everything before it, so it is
    computed over the continuous session and matched back to each row by ``seq_index``.
    """
    by_seq = {int(r["seq_index"]): r["gpu_util_ewma"]
              for r in add_gpu_util_ewma(session_rows, tau_s)}
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("seq_index") is None or int(r["seq_index"]) not in by_seq:
            raise ValueError("row has no seq_index in the session timeline; cannot recompute history")
        row = dict(r)
        row["gpu_util_ewma"] = by_seq[int(r["seq_index"])]
        out.append(row)
    return out


__all__ = ["DEFAULT_TAU_S", "GPUUtilEWMA", "add_gpu_util_ewma", "retau_rows"]
