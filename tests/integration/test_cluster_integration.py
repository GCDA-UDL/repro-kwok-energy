"""Integration test: gen-cluster -> load_map -> exporter -> cluster-total.

Cluster-integration change, task 4. Exercises the full path that was previously
broken by the gen_cluster trace_path bug.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kwok_energy.calibration import fit
from kwok_energy.exporter import EnergyExporter
from kwok_energy.gen_cluster import generate, write_artifacts
from kwok_energy.replay import load_map

pytestmark = pytest.mark.integration

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
FIX_TRACE = FIX / "traces" / "normal.csv"
CAL = str(FIX / "traces" / "cal_normal.csv")


def test_gen_cluster_to_exporter_cluster_total(tmp_path: Path):
    """Full path: gen-cluster (with config_path) -> write -> load_map -> exporter -> cluster-total."""

    # Copy the fixture trace into tmp_path/traces/ so the relative path works
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "normal.csv").write_text(FIX_TRACE.read_text(), encoding="utf-8")

    # 1. Generate a 10-node cluster with a relative trace_path + config_path
    cfg = {
        "node_count": 10,
        "profiles": [{
            "name": "normal", "count": 10,
            "platform_id": "xeon-silver+rtx2080ti",
            "trace_path": "traces/normal.csv",
        }],
    }
    artifacts = generate(cfg, config_path=tmp_path / "cluster.yaml")
    paths = write_artifacts(artifacts, tmp_path / "out")

    # 2. load_map on the written node_profile_map.json (the previously broken step)
    source = load_map(paths.node_profile_map)
    assert len(source.nodes()) == 10

    # 3. Fit coeffs on the calibration fixture
    coeffs = fit([CAL], platform_id="xeon-silver+rtx2080ti")

    # 4. Build exporter and tick
    exp = EnergyExporter(source=source, coeffs_map={"xeon-silver+rtx2080ti": coeffs})
    exp.tick(0.0)
    exp.tick(1.0)

    # 5. Cluster-total aggregation works
    assert exp.cluster_power() > 0
    assert exp.cluster_energy() > 0

    # 6. /metrics output contains both energy metrics for all 10 nodes
    mt = exp.metrics_text()
    assert b"node_power_watts" in mt
    assert b"node_energy_joules_total" in mt
    # 10 nodes × (1 total + components) power series + 10 energy counters
    node_count_in_metrics = mt.count(b'node="kwok-node-')
    assert node_count_in_metrics >= 10
