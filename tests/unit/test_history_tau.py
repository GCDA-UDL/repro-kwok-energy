"""Unit tests for the GPU-history recomputation and calibration provenance hashes.

Covers: ``retau_rows`` matches the session-timeline EWMA at the requested tau,
refuses rows outside the session, and ``platform_config_sha256`` is a real hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kwok_energy import calibration as cal
from kwok_energy.features import add_gpu_util_ewma, retau_rows

pytestmark = pytest.mark.unit

CAL = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "traces" / "cal_normal.csv"


def _session(n: int = 30) -> list[dict]:
    util = [1.0 if 5 <= i < 15 else 0.0 for i in range(n)]
    return [{"seq_index": i, "dt_s": 1.0, "gpu_utilization": u, "gpu_util_ewma": 0.0}
            for i, u in enumerate(util)]


def test_retau_rows_matches_session_ewma():
    session = _session()
    stale = [dict(r, gpu_util_ewma=0.123) for r in session[10:20]]
    out = retau_rows(stale, session, 5.0)
    ref = add_gpu_util_ewma(session, 5.0)[10:20]
    assert [r["gpu_util_ewma"] for r in out] == [r["gpu_util_ewma"] for r in ref]
    # a different tau gives a different history, so the tau actually matters
    assert out[-1]["gpu_util_ewma"] != retau_rows(stale, session, 20.0)[-1]["gpu_util_ewma"]


def test_retau_rows_refuses_rows_outside_session():
    with pytest.raises(ValueError):
        retau_rows([{"seq_index": 999, "gpu_utilization": 0.0}], _session(), 5.0)


def test_platform_config_sha256_is_a_hash():
    cfg = {"platform_id": "x", "cpu": {"tdp_w": 100}}
    c = cal.fit([CAL], cfg, platform_id="x")
    want = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    assert c.provenance["platform_config_sha256"] == want
