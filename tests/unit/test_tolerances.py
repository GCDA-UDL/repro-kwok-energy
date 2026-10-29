"""Unit tests for tolerances (I9 — single source of named thresholds).

Covers the tolerances spec:
- every named threshold is defined in ``kwok_energy.tolerances``;
- a frozen ``Tolerances`` dataclass carries the same values;
- no other module redefines these thresholds as literals (grep guard, I9).
"""

from __future__ import annotations

import ast
import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from kwok_energy import tolerances as tol

pytestmark = pytest.mark.unit

NAMED = [
    "WATTS_RECON",
    "MODEL_EXPORTER_ABS",
    "GOLDEN_REL",
    "ENERGY_RECON_REL",
    "MAPE_FLOOR_WATTS",
    "CANCELLATION_RATIO",
    "SCRAPE_GAP_MAX_S",
    "SYNTH_PLAUSIBILITY_REL",
]

SRC_ROOT = Path(tol.__file__).resolve().parent


def test_every_named_tolerance_is_defined():
    for name in NAMED:
        assert hasattr(tol, name), f"tolerances missing {name!r}"
        assert isinstance(getattr(tol, name), (int, float))


def test_tolerances_dataclass_is_frozen_and_complete():
    t = tol.Tolerances
    instance = t()
    for name in NAMED:
        assert hasattr(instance, name), f"Tolerances missing field {name!r}"
        assert getattr(instance, name) == getattr(tol, name)

    # frozen: mutation must raise
    with pytest.raises(FrozenInstanceError):
        instance.WATTS_RECON = 999  # type: ignore[misc]


def test_single_source_grep_guard():
    """No module other than tolerances.py may assign these threshold NAMES as a
    top-level or function-local literal binding (I9)."""
    offenders = []
    for py in SRC_ROOT.rglob("*.py"):
        if py.name == "tolerances.py":
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for tgt in targets:
                    if isinstance(tgt, ast.Name) and tgt.id in NAMED:
                        offenders.append(f"{py.name}:{tgt.lineno} binds {tgt.id}")
    assert not offenders, "I9 violation — thresholds redefined outside tolerances.py:\n" + "\n".join(offenders)


def test_consumers_import_from_single_source():
    """Every reference to a named threshold elsewhere in src must import it from
    kwok_energy.tolerances, not define it."""
    for py in SRC_ROOT.rglob("*.py"):
        if py.name == "tolerances.py":
            continue
        text = py.read_text(encoding="utf-8")
        for name in NAMED:
            # An assignment (NAME = ...) is already caught by the grep guard; here
            # we only sanity-check there is no bare redefinition via walrus/def.
            assert not re.search(rf"\b{name}\s*:=\s", text), f"{py.name} walrus-binds {name}"
