#!/usr/bin/env bash
# Download the user-space binaries for the live KWOK experiment (no root, no docker).
#   bash tools/kwok_live/setup_kwok_live.sh
# Installs into $KWOK_BENCH_HOME/bin (default ~/kwok-bench/bin): kwokctl + kwok,
# kubectl and Prometheus. kwokctl's binary runtime fetches etcd and the Kubernetes
# control-plane binaries itself on the first `create cluster`.
set -euo pipefail
BENCH="${KWOK_BENCH_HOME:-$HOME/kwok-bench}"
BIN="$BENCH/bin"
KWOK_VERSION="${KWOK_VERSION:-v0.6.1}"
KUBECTL_VERSION="${KUBECTL_VERSION:-v1.31.1}"
PROM_VERSION="${PROM_VERSION:-2.53.2}"
mkdir -p "$BIN"
cd "$BIN"
fetch() { [ -x "$2" ] || { curl -fsSL -o "$2" "$1"; chmod +x "$2"; }; }
fetch "https://github.com/kubernetes-sigs/kwok/releases/download/$KWOK_VERSION/kwokctl-linux-amd64" kwokctl
fetch "https://github.com/kubernetes-sigs/kwok/releases/download/$KWOK_VERSION/kwok-linux-amd64" kwok
fetch "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl" kubectl
if [ ! -x prometheus ]; then
  curl -fsSL "https://github.com/prometheus/prometheus/releases/download/v$PROM_VERSION/prometheus-$PROM_VERSION.linux-amd64.tar.gz" \
    | tar -xz --strip-components=1 "prometheus-$PROM_VERSION.linux-amd64/prometheus"
fi
./kwokctl --version
./kubectl version --client | head -1
./prometheus --version | head -1
echo "setup_kwok_live: binaries in $BIN"
