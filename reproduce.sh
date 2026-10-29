#!/bin/bash
# Regenerate every number and table of the paper from the dataset (raw RAPL/NVML
# traces of both platforms) and compare them with expected/generated, the files the
# paper inputs (values.tex holds every number quoted in the text).
#
# Usage: bash reproduce.sh [OUT_DIR]        (default: ./out; about 2 minutes)
#        Inside Docker: see README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/out}" && cd "${1:-$ROOT/out}" && pwd)"
PY="${PYTHON:-python3}"
export PYTHONPATH="$ROOT/src:$ROOT/tools/e2e${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
A=dataset/a4000_20261027
B=dataset/2080ti_20261027

echo "== Dataset integrity (sha256 of every file against its bundle manifest)"
"$PY" -m kwok_energy.cli.verify_dataset

echo "== Model selection on the calibration traces only (leave-one-workload-out CV)"
"$PY" tools/e2e/model_selection.py $A $B --out "$OUT/analysis" > "$OUT/model_selection.log"

echo "== Held-out evaluation of the energy layer and the baselines"
"$PY" tools/e2e/q1_analysis.py $A $B --selection "$OUT/analysis/selection.json" \
    --live-coeffs $A/coeffs/a4000_kwok_live.json --out "$OUT/analysis" > "$OUT/q1_analysis.log"

echo "== Paper numbers (values.tex) and tables"
"$PY" tools/e2e/paper_tables.py --analysis "$OUT/analysis" --kwok-live $A/results \
    --out "$OUT/generated" > "$OUT/paper_tables.log"

echo "== Comparison with the paper (expected/generated)"
status=0
for f in expected/generated/*.tex; do
    n=$(basename "$f")
    if diff -q --strip-trailing-cr "$f" "$OUT/generated/$n" >/dev/null 2>&1; then
        echo "IDENTICAL  $n"
    else
        echo "DIFFERENT  $n   (diff $f $OUT/generated/$n)"
        status=1
    fi
done
[[ $status -eq 0 ]] && echo "Every number and table of the paper is reproduced." \
                    || echo "Some outputs differ from the paper."
exit $status
