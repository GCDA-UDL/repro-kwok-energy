# Changelog

## [0.1.1] — 2026-10-30

- Author metadata: Matias Medina's name in its canonical form (`Medina, Matias`) in
  `CITATION.cff`, `.zenodo.json`, the README and the LICENSE. No change to code, data or results.

## [0.1.0] — 2026-10-30

First version of the reproduction package, built from the development repository at the
commit that fixed the paper's data availability statement.

- Energy layer (`src/kwok_energy`), protocol-v2 harness and analysis (`tools/e2e`), live
  KWOK experiment (`tools/kwok_live`), tests.
- Dataset: the two protocol-v2 capture sessions (`a4000_20261027`, `2080ti_20261027`);
  user names and remote addresses in the background snapshots of `session_manifest.json`
  replaced by labels.
- `reproduce.sh` + `Dockerfile`: every number (`values.tex`) and table of the paper
  regenerated from the raw traces, byte-identical to `expected/generated`.
