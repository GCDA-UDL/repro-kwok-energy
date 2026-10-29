# Reproducing the paper

Three levels:

1. **Regenerate the numbers and tables** from the dataset (2 minutes, any machine with
   Docker): `reproduce.sh`, see the README.
2. **Re-run the analysis step by step** (same inputs, any machine with Python 3.11+).
3. **Capture new traces** on a bench host with RAPL and NVIDIA GPUs, and re-run the live
   KWOK experiment.

Run every command from the repository root with `PYTHONPATH=src:tools/e2e` (the Docker
image sets it).

## 1. Regenerate

```bash
docker build -t repro-kwok-energy .
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repro" repro-kwok-energy
# tests
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD:/repro" --entrypoint python repro-kwok-energy \
    -m pytest -q -p no:cacheprovider tests/unit tests/integration
```

## 2. The analysis, step by step

```bash
export PYTHONPATH=src:tools/e2e
A=dataset/a4000_20261027; B=dataset/2080ti_20261027
python -m kwok_energy.cli.verify_dataset                   # sha256 of every file
python tools/e2e/model_selection.py $A $B --out out/analysis   # calibration traces only
python tools/e2e/q1_analysis.py $A $B --selection out/analysis/selection.json \
    --live-coeffs $A/coeffs/a4000_kwok_live.json --out out/analysis
python tools/e2e/paper_tables.py --analysis out/analysis --kwok-live $A/results --out out/generated
```

- `model_selection.py` chooses, per platform, the signal set, the ridge penalty and the
  GPU-history time constant by leave-one-workload-out cross-validation on the calibration
  traces; held-out traces are never read.
- `q1_analysis.py` fits the selected models and the baselines (calibrated utilisation-linear
  node model, Fan et al. curve, datasheet TDP, constant) on the calibration traces and
  evaluates all of them on the held-out traces; with `--live-coeffs` it also evaluates the
  coefficient file that the live KWOK replay loaded.
- `paper_tables.py` writes `values.tex` (every number quoted in the paper, with bootstrap
  confidence intervals over the repetitions) and the table rows.

The coefficient files that `q1_analysis.py` writes differ from the archived ones only in
their provenance fields (creation time, commit).

## 3. Capture new traces

**Host.** Linux on x86-64 with readable Intel RAPL package and DRAM domains
(`/sys/class/powercap`), one or more NVIDIA GPUs with NVML, `perf` with uncore IMC counters
for the memory-traffic signal (optional), Docker with the NVIDIA runtime (for the
PostgreSQL and LLM held-out workloads) and PyTorch with CUDA (for the ResNet workload).
**The host must be otherwise quiet**: RAPL measures the whole package, so any other job
pollutes the measurement. Coefficients do not transfer between hosts; every platform is
calibrated on its own traces.

```bash
pip install -r requirements.txt                 # plus torch for the ResNet workload
export PYTHONPATH=src:tools/e2e
bash tools/e2e/real/setup_real.sh               # one-time: PostgreSQL + pgbench data, LLM server and model, sources
python tools/e2e/capture_v2.py --platform myhost --reps 1 --duration 15 --settle 10 \
    --outdir data/smoke                         # smoke test, about 10 minutes
python tools/e2e/capture_v2.py --platform myhost --reps 5 --duration 120 \
    --outdir data/e2e_v2                        # full session, about 3.6 hours
```

Then analyse the new capture like the published ones (pass `data/e2e_v2/myhost` instead of
a dataset bundle), and archive it as a bundle with `tools/e2e/archive_dataset.py --capture`.

**Live KWOK experiment** (after a capture; about 1.5 hours):

```bash
bash tools/kwok_live/setup_kwok_live.sh         # kwokctl, kubectl and Prometheus (no root)
python -m kwok_energy.cli.fit_model --platform myhost \
    --input "$(ls data/e2e_v2/myhost/cal_*.csv | paste -sd, -)" --output data/coeffs/myhost.json --features v2
python tools/kwok_live/kwok_live.py --coeffs data/coeffs/myhost.json \
    --traces data/e2e_v2/myhost --sizes 100,1000,5000,10000 --out data/kwok_live/myhost
python tools/kwok_live/closed_loop_align.py --out data/kwok_live/myhost --coeffs data/coeffs/myhost.json
```

The tests that checked the layout of the development repository (its Makefile, CI and
documentation) are not part of this package.
