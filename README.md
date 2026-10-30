# Toward Hardware-Grounded Energy Estimation for KWOK — reproduction package

[![Code: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC_BY_4.0-lightgrey.svg)](LICENSE-data)
![Reproducible with Docker](https://img.shields.io/badge/reproducible-Docker-2496ED.svg)

Code, dataset and analysis environment for the paper

> **Toward Hardware-Grounded Energy Estimation for KWOK: Calibrating Simulated Kubernetes
> Nodes Against RAPL and NVML** (manuscript).

**Authors:** Vitor L. da Silva, Matias Medina, Rosana Tomàs, Joaquim Oliva,
Fernando Cores and Josep Lluís Lérida — SEMBAC, Universitat de Lleida, Spain.

KWOK simulates thousands of Kubernetes nodes without running any workload. This package
adds an energy layer to it: a per-component model (CPU package, DRAM, GPU) that predicts a
node's power from the signals of its simulated workload, calibrated against Intel RAPL and
NVIDIA NVML on real servers, and exposed to Prometheus by an exporter.

## What is inside

| Path | Contents |
|---|---|
| [`src/kwok_energy/`](src/kwok_energy) | The energy layer: signal schema, model fitting, energy model, exporter, trace replay, RAPL/NVML/platform-meter readers and the command-line tools. |
| [`tools/e2e/`](tools/e2e) | Measurement harness (`capture_v2.py`, `mem_bandwidth.py`, `real/` for the real held-out workloads, `archive_dataset.py`) and the analysis (`model_selection.py`, `q1_analysis.py`, `paper_tables.py`). |
| [`tools/kwok_live/`](tools/kwok_live) | The live KWOK experiment (KWOK cluster + exporter + Prometheus, up to 10,000 nodes) and its closed-loop check. |
| [`dataset/`](dataset) | The two capture sessions of the paper (protocol v2): `a4000_20261027` and `2080ti_20261027`, each with raw per-interval traces, the continuous session timeline, the fitted coefficients, the analysis results and a manifest with the sha256 of every file. See [`dataset/README.md`](dataset/README.md). |
| [`expected/generated/`](expected/generated) | The files the paper inputs, as submitted: `values.tex` (every number quoted in the text) and the table rows. |
| [`tests/`](tests) | Unit and integration tests of the layer (hardware tests skip without devices). |
| [`reproduce.sh`](reproduce.sh), [`Dockerfile`](Dockerfile) | Regenerate the paper's numbers and tables from the dataset and compare them with `expected/`. |
| [`REPRODUCE.md`](REPRODUCE.md) | How to re-run the analysis step by step and how to capture new traces. |

## Quick start: regenerate the paper's numbers and tables (about 2 minutes)

```bash
docker build -t repro-kwok-energy .
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repro" repro-kwok-energy
```

The script verifies every dataset file against its manifest, selects the model on the
calibration traces only, evaluates every model on the held-out traces, writes the paper's
numbers and tables to `out/generated/` and prints `IDENTICAL` for each file that matches
`expected/generated/` byte for byte. Without Docker: `pip install -r requirements.txt &&
bash reproduce.sh` (Python 3.11+).

## Paper element → script → input

| Paper element | Script | Input |
|---|---|---|
| Every number in the text (`values.tex`) | `paper_tables.py` | analysis below + live KWOK results |
| Held-out error tables of the validation section (`tab_heldout_rows`, `tab_workloads_rows`, `tab_components_rows`) | `q1_analysis.py` → `paper_tables.py` | `dataset/*_20261027/raw` (calibration and held-out traces), model selection |
| Fitted coefficients in physical units (`tab_coeffs_rows`) | `q1_analysis.py` → `paper_tables.py` | same |
| Model selection (leave-one-workload-out CV) | `model_selection.py` | calibration traces only |
| Live KWOK cluster, 100–10,000 nodes (`tab_kwok_live_rows`) | `kwok_live.py`, `closed_loop_align.py` (captured), `paper_tables.py` | `dataset/a4000_20261027/results/{kwok_live,closed_loop_aligned}.csv` |

The architecture diagram is drawn by hand and contains no data. The live KWOK results are
captured measurements of a running cluster; `REPRODUCE.md` explains how to re-run them.

## Hosts of the captures

| Platform | CPU | Memory | GPUs | OS | Platform meter |
|---|---|---|---|---|---|
| `a4000` | 2× Xeon Silver 4210R (20c/40t) | 128 GB | 2× RTX A4000 16 GB (driver 580.178) | Rocky Linux 9 (5.14) | yes (±2.5 %) |
| `2080ti` | Xeon W-3245 (16c/32t) | 192 GB | 4× RTX 2080 Ti 11 GB (driver 535.247) | Ubuntu 20.04 (5.4) | — |

Each session: 20 workloads (12 for calibration, 8 held out, five of them real applications)
× 5 repetitions × 120 s, in one randomised session per host. The background activity of
the host is recorded in each `session_manifest.json`; user names and remote addresses in
it are replaced by labels.

## How to cite

Use [`CITATION.cff`](CITATION.cff). The Zenodo archive and its DOI will be added on release.

## License

Code: MIT ([`LICENSE`](LICENSE)). Dataset and expected outputs: CC BY 4.0
([`LICENSE-data`](LICENSE-data)).

## Funding

This work has been granted by the Ministerio de Ciencia, Innovación y Universidades
(MICIU) AEI/10.13039/501100011033 under contract PID2023-146193OB-I00.

## Contact

Vitor L. da Silva — vitor.dasilva@udl.cat
