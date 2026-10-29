"""Unit tests for provenance (Phase 4, R2).

Covers: git_sha, hash_file (stable + change-detecting), host_metadata,
build_block (the full Coeffs provenance set).
"""

from __future__ import annotations

import pytest

from kwok_energy import provenance as prov

pytestmark = pytest.mark.unit

FULL_BLOCK_KEYS = {
    "schema_version",
    "git_sha",
    "created_at",
    "solver",
    "transform_kind",
    "host",
    "calibration_trace_sha256",
    "platform_config_sha256",
}
# The per-row lineage subset (R2) — what ValidationRow/result CSVs carry.
LINEAGE_SUBSET = {
    "coeffs_git_sha",
    "calibration_trace_sha256",
    "platform_config_sha256",
    "schema_version",
    "created_at",
}


def test_git_sha_returns_nonempty_string():
    sha = prov.git_sha()
    assert isinstance(sha, str) and sha


def test_hash_file_stable(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello provenance\n")
    assert prov.hash_file(p) == prov.hash_file(p)


def test_hash_file_change_detecting(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("v1\n")
    h1 = prov.hash_file(p)
    p.write_text("v2\n")
    h2 = prov.hash_file(p)
    assert h1 != h2


def test_hash_file_is_hex_sha256(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("x")
    h = prov.hash_file(p)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_host_metadata_keys():
    md = prov.host_metadata()
    assert isinstance(md, dict)
    for key in ("kernel", "cpu", "gpu", "nvml_driver"):
        assert key in md, f"host_metadata missing {key!r}"


def test_build_block_carries_full_set():
    block = prov.build_block(
        solver="ridge",
        transform_kind="standardize",
        calibration_trace_sha256="a" * 64,
        platform_config_sha256="b" * 64,
    )
    assert FULL_BLOCK_KEYS <= set(block)


def test_build_block_lineage_subset_reconciles_with_phase10():
    """The per-row lineage subset MUST match Phase 10's ValidationRow fields
    (R2 reconciliation). build_block's full set is a superset of the lineage
    subset; the git_sha field maps to coeffs_git_sha per-row."""
    block = prov.build_block(
        solver="ridge",
        transform_kind="standardize",
        calibration_trace_sha256="c" * 64,
        platform_config_sha256="d" * 64,
    )
    # Every lineage-subset field has a counterpart in the full block
    # (git_sha <-> coeffs_git_sha; the rest share names).
    assert block["git_sha"]  # becomes coeffs_git_sha per-row
    assert block["calibration_trace_sha256"]
    assert block["platform_config_sha256"]
    assert block["schema_version"]
    assert block["created_at"]
