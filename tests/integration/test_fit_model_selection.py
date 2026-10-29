"""fit-model --selection must fit the history model at the tau it declares.

Regression for the defect where the selected tau was written into the payload while the
weights were fitted on the history column stored at capture time (tau 20 s). On the
checked-in A4000 v2 bundle, fit-model must reproduce the archived evaluated coefficients
(written by tools/e2e/q1_analysis.py), and it must refuse a tau without a session timeline.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kwok_energy.cli import fit_model

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "dataset" / "a4000_20261027"


def _selection(tmp_path: Path) -> Path:
    sel = json.loads((BUNDLE / "results" / "selection.json").read_text(encoding="utf-8"))
    p = tmp_path / "selection.json"
    p.write_text(json.dumps(sel), encoding="utf-8")
    return p


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="v2 bundle not checked out")
def test_fit_model_selection_reproduces_evaluated_coeffs(tmp_path):
    cal_files = sorted((BUNDLE / "raw").glob("cal_*.csv"))
    out = tmp_path / "a4000.json"
    rc = fit_model.main(["--platform", "a4000", "--input", ",".join(map(str, cal_files)),
                         "--output", str(out), "--selection", str(_selection(tmp_path)),
                         "--variant", "hist"])
    assert rc == 0
    got = json.loads(out.read_text(encoding="utf-8"))
    ref = json.loads((BUNDLE / "coeffs" / "a4000.json").read_text(encoding="utf-8"))
    assert got["transform"]["gpu_util_ewma_tau_s"] == ref["transform"]["gpu_util_ewma_tau_s"]
    for f in ref["transform"]["features"]:
        assert got["transform"]["means"][f] == pytest.approx(ref["transform"]["means"][f], abs=1e-9)
    for comp, v in ref["components"].items():
        assert got["components"][comp]["intercept"] == pytest.approx(v["intercept"], abs=1e-9)
        for f, w in v["weights"].items():
            assert got["components"][comp]["weights"][f] == pytest.approx(w, abs=1e-9)
    assert got["provenance"]["calibration_trace_sha256"] == ref["provenance"]["calibration_trace_sha256"]


@pytest.mark.skipif(not BUNDLE.is_dir(), reason="v2 bundle not checked out")
def test_fit_model_selection_refuses_tau_without_session(tmp_path):
    src = sorted((BUNDLE / "raw").glob("cal_*.csv"))[0]
    lone = tmp_path / "traces"
    lone.mkdir()
    shutil.copy2(src, lone / src.name)
    rc = fit_model.main(["--platform", "a4000", "--input", str(lone / src.name),
                         "--output", str(tmp_path / "x.json"),
                         "--selection", str(_selection(tmp_path)), "--variant", "hist"])
    assert rc == 2
    assert not (tmp_path / "x.json").exists()
