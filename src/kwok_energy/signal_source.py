"""SignalSource — the simulator-agnostic seam (Component G, Phase 6a, I10).

A runtime-checkable Protocol: any implementation providing `nodes()`,
`platform_id(node)`, and `signals_at(node, logical_time_s)` is a SignalSource.
The energy model and exporter depend ONLY on this contract (I10), never on KWOK
or any simulator internals.

The recorded-trace replay (Phase 7) and the synthetic signal source (Phase 6b)
are concrete implementations. Stubs/notes for OpenSim/SimKube/Kubemark/kcs/k8ssim
reuse the `k8s-sims module.sh` setup/run/cleanup contract (interface only).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SignalSource(Protocol):
    """Simulator-agnostic signal seam (Component G, I10).

    Implementations:
    - ``kwok_energy.replay.ReplayEngine`` (recorded-trace replay, Phase 7).
    - ``kwok_energy.synthetic.SyntheticSource`` (never-run, Phase 6b).
    - Future: OpenSim/SimKube/Kubemark/kcs/k8ssim (interface only for paper #1).
    """

    def nodes(self) -> list[str]: ...
    def platform_id(self, node: str) -> str: ...
    def signals_at(self, node: str, logical_time_s: float) -> dict[str, Any]: ...


__all__ = ["SignalSource"]
