# Dataset — Hardware-grounded energy traces for KWOK calibration

Open-access measurement dataset accompanying paper #1 *"Toward Hardware-Grounded Energy
Estimation for KWOK"*. Each bundle is the real RAPL (CPU package + DRAM) and NVIDIA
NVML (GPU) energy captured on a bench host under calibration and held-out workloads,
plus the fitted per-component model and validation results — enough to reproduce the
paper's numbers and refit/replicate on other hardware.

## Structure

```
dataset/<platform>_<YYYYMMDD>/
  manifest.json          # hardware, software, git sha, per-file sha256, license, funding
  raw/                   # per-interval traces (the measurements)
    cal_<workload>_r<N>.csv        # calibration (fitter input)
    heldout_<workload>_r<N>.csv    # held-out (validation only)
  results/
    <platform>_validation.csv      # held-out per-component error (MAPE/RMSE/sMAPE)
    <platform>_scale_<nodes>.csv   # exporter host overhead at scale
  coeffs/<platform>.json           # fitted per-component model + provenance
```

Protocol-v2 bundles (`*_20261027`) differ as follows:

```
dataset/<platform>_20261027/
  raw/
    cal_<workload>_r<N>.csv, heldout_<workload>_r<N>.csv   # 20 workloads x 5 repetitions
    session.csv              # the whole capture session, 1-s intervals, incl. set-up and gaps
    session_manifest.json    # args, seed, host, background processes (user names and remote
                             # addresses redacted), order/timing of windows
  results/
    model_selection.csv      # leave-one-workload-out CV on the calibration traces
    selection.json           # selected configuration (base and with GPU history)
    per_workload.csv         # held-out error per model x workload x component
    summary.csv, summary_components.csv
    node_level.csv           # whole-node analysis against the platform meter (a4000 only)
  coeffs/<platform>.json     # selected model with GPU history (tau in the transform): the
                             # model behind the reported numbers, written by q1_analysis.py
  coeffs/a4000_kwok_live.json  # (a4000 only) the earlier fit-model file loaded by the live
                             # KWOK replay (history fitted at the capture tau, see below);
                             # the live results do not depend on its values
  results/live_coeffs.csv    # (a4000 only) held-out error of that file as replayed
```

`<platform>` ∈ {`a4000`, `2080ti`, `amd_ryzen_7_3700x`, …}; one folder per host × capture date.
The primary paper bundles are `a4000_20261027` and `2080ti_20261027` (measurement protocol
v2: one continuous randomised session per host, 120-s windows, five repetitions, real held-out
applications, every CPU socket and every GPU summed, whole-node platform meter on `a4000`).
`a4000_20260620` and `2080ti_20260620` are protocol-v1 bundles kept for the record; the
`a4000_20260620` RAPL totals cover CPU socket 0 only and its GPU energy GPU 0 only, so they
must not be used as whole-node figures. `amd_ryzen_7_3700x_20260620`
is a supplemental local CPU-package-only run from an AMD Ryzen 7 3700X; RAPL DRAM, GPU, and live
memory-bandwidth proxy were unavailable for that host, so it is not used for paper headline
tables/figures. Directories ending in `_old` are superseded first-run archives retained for
audit trail; they are not the source of the paper headline tables/figures.

## Column dictionary

**`raw/*.csv` (TraceRow — one interval per row):** identity (`schema_version, run_id,
`workload, node`), interval (`timestamp, t_start_s, t_end_s, dt_s`), model-input signals
(`cpu_utilization` 0..1, `cpu_frequency_mhz`, `mem_bandwidth_proxy`, `mem_proxy_method`,
`gpu_utilization` 0..1, `gpu_mem_utilization` 0..1, `disk_read_bps`, `disk_write_bps`,
`net_rx_bps`, `net_tx_bps`, `platform_id`; v2 adds `gpu_util_ewma`, the GPU-utilisation
history at the capture tau of 20 s, recomputable from `session.csv`), and measured targets (`rapl_pkg_joules`,
`rapl_pkg_watts_mean`, `rapl_pkg_available`, `rapl_dram_joules`, `rapl_dram_watts_mean`,
`rapl_dram_available`, `gpu_energy_joules`, `gpu_watts_mean`, `gpu_available`,
`gpu_energy_method`; v2 adds `platform_meter_joules/_watts_mean/_available`). v2 rows also
carry diagnostics that are not model inputs: `session_t_s, seq_index, gpu_count,
gpu_sm_clock_mhz, gpu_pstate, gpu_temp_c, cpu_temp_c`. Energy is in joules; watts are derived as joules/dt_s. Missing
DRAM/GPU → `*_available=false` and null joules (never 0).

**`results/<platform>_validation.csv`:** `model_name, workload, platform_id, component
(package|dram|gpu|total), n_runs, measured_joules, predicted_joules, mae, rmse, mape,
mape_masked, mape_coverage, smape, cancellation_flag, source, coeffs_git_sha`.

**`results/<platform>_scale_*.csv`:** `node_count, series_count, samples_per_second,
exporter_cpu_percent, exporter_memory_mb, scrape_failure_rate, duration_s, mode`.

**`coeffs/<platform>.json`:** `platform_id, covered_components, transform (means/std),
components (per-component intercept+weights+solver), provenance`.

## How it was produced

`tools/e2e/run_experiment.sh` on each bench host: capture (`capture_traces.py`) → fit
(`fit-model`) → validate held-out (`validate`) → scale (`scale-demo`) → archive
(`archive_dataset.py`). The current non-`_old` bundles include a live
`mem_bandwidth_proxy` from perf IMC traffic where the host exposes the counters: the DRAM
read and write traffic that perf derives from the IMC CAS counters (64 B per CAS), in MiB/s
(the paper reports its slopes per GiB/s).
Superseded `_old` first-run bundles predate the improved capture path.

Protocol v2: `tools/e2e/capture_v2.py` (see `tools/e2e/RUNBOOK_v2.md`) →
`tools/e2e/model_selection.py` (calibration traces only) → `tools/e2e/q1_analysis.py
--selection` (writes the evaluated `coeffs_<platform>.json`) → `archive_dataset.py --capture`.
`fit-model --selection --variant hist` reproduces those coefficients: it recomputes the GPU
history at the selected tau on `session.csv` (next to the traces, or `--session`). Before
2026-10-29 it fitted the history column stored at capture time (tau 20 s) while declaring the
selected tau; `coeffs/a4000_kwok_live.json`, loaded by the live KWOK replay, is such a file
(held-out error as replayed in `results/live_coeffs.csv`).
Software that differs between the hosts: the LLM server is Ollama 0.40.11 on `a4000` and
Ollama 0.6.0 on `2080ti` (driver 535 there; newer images need driver >= 550); the memory
proxy is sampled per interval on `a4000` and by one continuous `perf stat -I` on `2080ti`.

The energy model predicts power from low-level signals and never consumes measured power;
targets are used only to fit and validate. See `tools/e2e/RUNBOOK.md` and
`tools/e2e/EXPERIMENT_PLAN.md`.

## Scope / limitations

Single-node, component-level energy (RAPL package+DRAM, NVML GPU) — not wall-plug or
PUE/facility energy. Coefficients are per-platform (no zero-shot transfer). The current
`mem_bandwidth_proxy` is perf IMC DRAM traffic (MiB/s) when available and its coefficients
are per platform. No real multi-node cluster is measured; scale results are model extrapolations
with the held-out error band. The `a4000` platform meter (ACPI power meter, Intel Node
Manager, +/-2.5 %) gives whole-node input power for that host only.

## Integrity checks

Verify checked-in files against each bundle's manifest:

```bash
make verify-dataset
```

## Cite / license

Data licensed CC-BY-4.0 (see `LICENSE`). Please cite the paper (DOI on acceptance) and
this dataset. Integrity: verify files against `manifest.json` sha256.

## Funding

This work has been granted by the Ministerio de Ciencia, Innovación y Universidades
(MICIU) AEI/10.13039/501100011033 under contract PID2023-146193OB-I00.
