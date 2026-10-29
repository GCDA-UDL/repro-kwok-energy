#!/usr/bin/env bash
# One-time preparation of the real held-out workloads used by capture_v2.py.
# Idempotent; nothing here is measured. Run from the repo root on the bench host:
#   bash tools/e2e/real/setup_real.sh
# Needs: docker (with the NVIDIA runtime for the LLM server), network access,
# the repo .venv with torch (CUDA). Disk: ~1 GB in $KWOK_BENCH_HOME, ~4 GB of docker
# volumes (PostgreSQL scale 100 + the LLM weights).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
BENCH="${KWOK_BENCH_HOME:-$HOME/kwok-bench}"
PY=.venv/bin/python
LLM="${KWOK_BENCH_LLM:-qwen2.5:3b}"
CPY_VER=3.12.7
mkdir -p "$BENCH"

echo "== torch with CUDA (the ResNet-50 workload needs torch only, no torchvision) =="
$PY -c 'import torch;assert torch.cuda.is_available();print("torch",torch.__version__,"GPUs",torch.cuda.device_count())'

echo "== PostgreSQL 16 + pgbench database (scale 100) =="
docker pull postgres:16
if ! docker volume inspect kwokbench-pg >/dev/null 2>&1; then
  docker volume create kwokbench-pg
  docker run -d --rm --name kwokbench-pg-init --shm-size=1g -e POSTGRES_PASSWORD=bench \
    -v kwokbench-pg:/var/lib/postgresql/data postgres:16
  until docker exec kwokbench-pg-init pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
  sleep 3
  docker exec kwokbench-pg-init createdb -U postgres bench
  docker exec kwokbench-pg-init pgbench -U postgres -i -s 100 -q bench
  docker stop kwokbench-pg-init
fi

echo "== Ollama + $LLM weights =="
docker pull ollama/ollama:latest
docker volume inspect kwokbench-ollama >/dev/null 2>&1 || docker volume create kwokbench-ollama
docker rm -f kwokbench-ollama-init >/dev/null 2>&1 || true
docker run -d --rm --name kwokbench-ollama-init --gpus device=0 --dns 8.8.8.8 \
  -v kwokbench-ollama:/root/.ollama ollama/ollama:latest
sleep 5
docker exec kwokbench-ollama-init ollama pull "$LLM"
docker stop kwokbench-ollama-init

echo "== CPython $CPY_VER source tree, configured =="
if [ ! -f "$BENCH/cpython/Makefile" ]; then
  cd "$BENCH"
  curl -fsSLO "https://www.python.org/ftp/python/$CPY_VER/Python-$CPY_VER.tgz"
  rm -rf cpython && mkdir cpython
  tar -xzf "Python-$CPY_VER.tgz" -C cpython --strip-components=1
  cd cpython && ./configure --quiet >/dev/null
  cd - >/dev/null
fi
echo "setup_real: done"
