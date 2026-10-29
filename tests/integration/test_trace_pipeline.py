"""Integration test: the trace pipeline end-to-end (docker-free, no hardware).

Exercises read -> validate -> derive_watts -> validate_trace_file -> hash on the
committed reference fixture, crossing the schema and trace modules together.
Marked ``integration`` (and not kwok/prometheus) so ``make test-integration``
runs it in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kwok_energy import schema, trace

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "traces" / "normal.csv"


def test_reference_fixture_round_trips_and_validates():
    rows = trace.read(FIXTURE)
    assert len(rows) == 3

    # every row is a valid signal row (Component A)
    for row in rows:
        # strip measured/derived/availability to get the pure signal row
        signal_only = {
            k: v
            for k, v in row.items()
            if k not in (schema.MEASURED_TARGET_FIELDS
                         | schema.DERIVED_TARGET_FIELDS
                         | schema.AVAILABILITY_FIELDS)
        }
        assert schema.validate_signal_row(signal_only) == [], signal_only
        assert schema.validate_trace_row(row) == []

    # interval structure is valid and ordered
    assert trace.validate(rows) == []

    # whole-file validation is clean
    assert schema.validate_trace_file(FIXTURE) == []

    # derive_watts matches the committed derived columns
    for row in rows:
        watts = trace.derive_watts(row)
        for key, expected in watts.items():
            if expected is None:
                assert row[key] is None
            else:
                assert row[key] == pytest.approx(expected)


def test_fixture_hash_stable_after_rewrite(tmp_path):
    original = trace.hash_trace(FIXTURE)
    rewritten = tmp_path / "rewritten.csv"
    trace.write(trace.read(FIXTURE), rewritten)
    assert trace.hash_trace(rewritten) == original


def test_missing_dram_round_trips_as_null(tmp_path):
    rows = trace.read(FIXTURE)
    # row 1 (cpu workload) is the missing-DRAM row
    assert rows[1]["rapl_dram_joules"] is None
    assert rows[1]["rapl_dram_available"] is False
    assert rows[1]["rapl_dram_watts_mean"] is None

    out = tmp_path / "copy.csv"
    trace.write(rows, out)
    again = trace.read(out)
    assert again[1]["rapl_dram_joules"] is None
    assert again[1]["rapl_dram_available"] is False
