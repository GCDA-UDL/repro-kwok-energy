"""Cluster generation (Phase 14, I8).

One config produces every identity artifact used by replay/exporter, KWOK fake
nodes/pods, and Prometheus file service discovery. Node identity is minted here
exactly once; downstream synthetic/replay sources swap signal providers without
creating their own node names.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ClusterArtifacts:
    """In-memory Phase 14 artifacts generated from one identity source."""

    node_profile_map: dict[str, Any]
    kwok_fake_nodes: list[dict[str, Any]]
    kwok_fake_pods: list[dict[str, Any]]
    prometheus_targets: list[dict[str, Any]]


@dataclass(frozen=True)
class ArtifactPaths:
    """Paths written by :func:`write_artifacts`."""

    node_profile_map: Path
    kwok_fake_nodes: Path
    kwok_fake_pods: Path
    prometheus_targets: Path


def generate(
    config: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
) -> ClusterArtifacts:
    """Generate node-profile, KWOK, and Prometheus artifacts from one config.

    Expected minimal config::

        {
          "node_count": 5,
          "profiles": [
            {"name": "normal", "count": 5, "platform_id": "...", "trace_path": "..."}
          ],
          "exporter_target": "kwok-energy-exporter:9100"
        }

    Profile counts must sum to ``node_count``. Generation is deterministic and
    names nodes ``kwok-node-0000`` ... in profile order.

    ``trace_path`` values in the config are resolved to **absolute paths** relative
    to ``config_path``'s parent directory (or CWD if ``config_path`` is None), so
    the emitted ``node_profile_map.json`` is self-contained and loadable by
    ``replay.load_map`` from any working directory.
    """

    base_dir = Path(config_path).resolve().parent if config_path else Path.cwd()

    node_count = _positive_int(config.get("node_count"), "node_count")
    raw_profiles = config.get("profiles")
    if not isinstance(raw_profiles, Sequence) or isinstance(raw_profiles, (str, bytes)):
        raise ValueError("profiles must be a non-empty list")
    profiles = [_normalize_profile(p) for p in raw_profiles]
    if not profiles:
        raise ValueError("profiles must be a non-empty list")
    total_count = sum(int(p["count"]) for p in profiles)
    if total_count != node_count:
        raise ValueError(f"profile counts ({total_count}) must equal node_count ({node_count})")

    exporter_target = str(config.get("exporter_target", "kwok-energy-exporter:9100"))

    nodes: dict[str, dict[str, Any]] = {}
    kwok_nodes: list[dict[str, Any]] = []
    kwok_pods: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []

    ordinal = 0
    for profile in profiles:
        # Resolve trace_path to absolute so the map is portable (loadable from any CWD)
        abs_trace = str((base_dir / str(profile["trace_path"])).resolve())
        for _ in range(int(profile["count"])):
            node = f"kwok-node-{ordinal:04d}"
            profile_name = str(profile["name"])
            platform_id = str(profile["platform_id"])
            nodes[node] = {
                "trace_path": abs_trace,
                "platform_id": platform_id,
                "mode": str(profile.get("mode", "loop")),
                "speed": float(profile.get("speed", 1.0)),
                "profile": profile_name,
            }
            kwok_nodes.append(_kwok_node(node, profile_name, platform_id))
            kwok_pods.append(_kwok_pod(node, profile_name, platform_id))
            targets.append(_prometheus_target(exporter_target, node, profile_name, platform_id))
            ordinal += 1

    return ClusterArtifacts(
        node_profile_map={"nodes": nodes},
        kwok_fake_nodes=kwok_nodes,
        kwok_fake_pods=kwok_pods,
        prometheus_targets=targets,
    )


def write_artifacts(artifacts: ClusterArtifacts, output_dir: str | Path) -> ArtifactPaths:
    """Write Phase 14 artifacts under ``output_dir`` using stable formatting."""

    out = Path(output_dir)
    kwok_dir = out / "kwok"
    prom_dir = out / "prometheus"
    kwok_dir.mkdir(parents=True, exist_ok=True)
    prom_dir.mkdir(parents=True, exist_ok=True)

    node_profile_map = out / "node_profile_map.json"
    fake_nodes = kwok_dir / "fake_nodes.yaml"
    fake_pods = kwok_dir / "fake_pods.yaml"
    targets = prom_dir / "generated_targets.yml"

    node_profile_map.write_text(
        json.dumps(artifacts.node_profile_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fake_nodes.write_text(_dump_all_yaml(artifacts.kwok_fake_nodes), encoding="utf-8")
    fake_pods.write_text(_dump_all_yaml(artifacts.kwok_fake_pods), encoding="utf-8")
    targets.write_text(yaml.safe_dump(artifacts.prometheus_targets, sort_keys=True), encoding="utf-8")

    return ArtifactPaths(
        node_profile_map=node_profile_map,
        kwok_fake_nodes=fake_nodes,
        kwok_fake_pods=fake_pods,
        prometheus_targets=targets,
    )


def artifact_hashes(paths: ArtifactPaths) -> dict[str, str]:
    """Return stable sha256 hashes for each generated artifact (R1/R4)."""

    ordered_paths = {
        "kwok/fake_nodes.yaml": paths.kwok_fake_nodes,
        "kwok/fake_pods.yaml": paths.kwok_fake_pods,
        "node_profile_map.json": paths.node_profile_map,
        "prometheus/generated_targets.yml": paths.prometheus_targets,
    }
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in ordered_paths.items()
    }


def _normalize_profile(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("each profile must be a mapping")
    profile = dict(raw)
    for required in ("name", "count", "platform_id", "trace_path"):
        if required not in profile or profile[required] in (None, ""):
            raise ValueError(f"profile missing required field {required!r}")
    profile["count"] = _positive_int(profile["count"], "profile count")
    mode = str(profile.get("mode", "loop"))
    if mode not in {"loop", "once"}:
        raise ValueError(f"profile {profile['name']!r}: mode must be 'loop' or 'once'")
    speed = float(profile.get("speed", 1.0))
    if speed <= 0:
        raise ValueError(f"profile {profile['name']!r}: speed must be > 0")
    profile["mode"] = mode
    profile["speed"] = speed
    return profile


def _positive_int(value: Any, name: str) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if ivalue <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return ivalue


def _kwok_node(node: str, profile: str, platform_id: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Node",
        "metadata": {
            "name": node,
            "labels": {
                "kwok.x-k8s.io/node": "fake",
                "kwok-energy/profile": profile,
                "kwok-energy/platform-id": platform_id,
            },
        },
    }


def _kwok_pod(node: str, profile: str, platform_id: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"{node}-placeholder",
            "namespace": "default",
            "labels": {
                "kwok-energy/node": node,
                "kwok-energy/profile": profile,
                "kwok-energy/platform-id": platform_id,
            },
        },
        "spec": {
            "nodeName": node,
            "containers": [
                {
                    "name": "placeholder",
                    "image": "registry.k8s.io/pause:3.9",
                }
            ],
        },
    }


def _prometheus_target(target: str, node: str, profile: str, platform_id: str) -> dict[str, Any]:
    return {
        "targets": [target],
        "labels": {
            "job": "kwok-energy-exporter",
            "node": node,
            "profile": profile,
            "platform_id": platform_id,
        },
    }


def _dump_all_yaml(items: Sequence[Mapping[str, Any]]) -> str:
    return yaml.safe_dump_all(list(items), explicit_start=True, sort_keys=True)


__all__ = [
    "ArtifactPaths",
    "ClusterArtifacts",
    "artifact_hashes",
    "generate",
    "write_artifacts",
]
