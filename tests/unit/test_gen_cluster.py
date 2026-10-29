"""Unit tests for cluster generation (Phase 14, I8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from kwok_energy.gen_cluster import generate, write_artifacts

pytestmark = pytest.mark.unit


def _config() -> dict:
    return {
        "node_count": 5,
        "profiles": [
            {
                "name": "normal",
                "count": 3,
                "platform_id": "xeon-silver+rtx2080ti",
                "trace_path": "traces/normal.csv",
                "mode": "loop",
                "speed": 1.0,
            },
            {
                "name": "irregular",
                "count": 2,
                "platform_id": "xeon-silver+rtx2080ti",
                "trace_path": "traces/irregular_dt.csv",
                "mode": "once",
                "speed": 2.0,
            },
        ],
        "exporter_target": "kwok-energy-exporter:9100",
    }


def test_generate_uses_one_identity_source_across_all_outputs():
    artifacts = generate(_config())

    map_nodes = set(artifacts.node_profile_map["nodes"])
    kwok_nodes = {item["metadata"]["name"] for item in artifacts.kwok_fake_nodes}
    pod_nodes = {item["spec"]["nodeName"] for item in artifacts.kwok_fake_pods}
    target_nodes = {item["labels"]["node"] for item in artifacts.prometheus_targets}

    assert map_nodes == kwok_nodes == pod_nodes == target_nodes
    assert map_nodes == {"kwok-node-0000", "kwok-node-0001", "kwok-node-0002", "kwok-node-0003", "kwok-node-0004"}


def test_generate_respects_profile_distribution():
    artifacts = generate(_config())
    profiles = [spec["profile"] for spec in artifacts.node_profile_map["nodes"].values()]
    assert profiles.count("normal") == 3
    assert profiles.count("irregular") == 2


def test_generate_1000_nodes_and_yaml_is_valid(tmp_path: Path):
    cfg = {
        "node_count": 1000,
        "profiles": [
            {
                "name": "normal",
                "count": 1000,
                "platform_id": "xeon-silver+rtx2080ti",
                "trace_path": "traces/normal.csv",
            }
        ],
        "exporter_target": "kwok-energy-exporter:9100",
    }
    artifacts = generate(cfg)
    paths = write_artifacts(artifacts, tmp_path)

    assert len(artifacts.node_profile_map["nodes"]) == 1000
    assert json.loads(paths.node_profile_map.read_text())["nodes"]["kwok-node-0999"]
    assert len(list(yaml.safe_load_all(paths.kwok_fake_nodes.read_text()))) == 1000
    assert len(list(yaml.safe_load_all(paths.kwok_fake_pods.read_text()))) == 1000
    targets = yaml.safe_load(paths.prometheus_targets.read_text())
    assert len(targets) == 1000
    assert targets[-1]["labels"]["node"] == "kwok-node-0999"


def test_generate_rejects_distribution_that_does_not_match_node_count():
    cfg = _config()
    cfg["node_count"] = 6
    with pytest.raises(ValueError, match="profile counts.*node_count"):
        generate(cfg)


def test_generated_map_trace_path_is_absolute(tmp_path: Path):
    """The emitted node_profile_map must store absolute trace paths (loadable from any CWD)."""
    # Create a fake trace file in a temp dir
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    trace_file = trace_dir / "normal.csv"
    trace_file.write_text(
        "schema_version,run_id,workload,node,timestamp,t_start_s,t_end_s,dt_s,"
        "platform_id,cpu_utilization,cpu_frequency_mhz,mem_bandwidth_proxy,"
        "mem_proxy_method,gpu_utilization,gpu_mem_utilization,disk_read_bps,"
        "disk_write_bps,net_rx_bps,net_tx_bps,rapl_pkg_joules,rapl_dram_joules,"
        "gpu_energy_joules,rapl_pkg_watts_mean,rapl_dram_watts_mean,"
        "gpu_watts_mean,rapl_pkg_available,rapl_dram_available,gpu_available,"
        "gpu_energy_method\n"
        "1,r1,idle,n0,2025-01-01T00:00:00Z,0.0,1.0,1.0,p,0.5,2400.0,10.0,stream,"
        "0.3,0.2,0.0,0.0,0.0,0.0,50.0,4.0,20.0,50.0,4.0,20.0,True,True,True,nvml\n",
        encoding="utf-8",
    )

    cfg = {
        "node_count": 1,
        "profiles": [{
            "name": "normal", "count": 1,
            "platform_id": "p", "trace_path": "traces/normal.csv",
        }],
    }
    artifacts = generate(cfg, config_path=tmp_path / "cluster.yaml")
    paths = write_artifacts(artifacts, tmp_path / "out")

    npm = json.loads(paths.node_profile_map.read_text())
    emitted_path = npm["nodes"]["kwok-node-0000"]["trace_path"]
    assert Path(emitted_path).is_absolute()
    assert Path(emitted_path).exists()


def test_generated_map_is_loadable_by_replay(tmp_path: Path):
    """End-to-end: generate → write → load_map must succeed (the integration bug)."""
    from kwok_energy.replay import load_map

    # Copy fixture trace into tmp_path/traces/
    fix_trace = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "traces" / "normal.csv"
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "normal.csv").write_text(fix_trace.read_text(), encoding="utf-8")

    cfg = {
        "node_count": 3,
        "profiles": [{
            "name": "normal", "count": 3,
            "platform_id": "xeon-silver+rtx2080ti",
            "trace_path": "traces/normal.csv",
        }],
    }
    artifacts = generate(cfg, config_path=tmp_path / "cluster.yaml")
    paths = write_artifacts(artifacts, tmp_path / "out")

    engine = load_map(paths.node_profile_map)  # MUST NOT raise FileNotFoundError
    assert len(engine.nodes()) == 3
