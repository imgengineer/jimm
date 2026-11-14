#!/usr/bin/env bash
# Launch one host in a multi-host JAX training run.
#
# Unlike timm's per-GPU torchrun launcher, JAX uses every local device from a
# single process, so run this script once per host (not once per GPU).
#
# Usage:
#   ./distributed_train.sh NUM_PROCESSES PROCESS_ID COORDINATOR [jimm.train args...]
#
# Example (host 1 of 2):
#   NUM_PROCESSES=2 PROCESS_ID=1 COORDINATOR=192.168.1.10:12345 \
#     ./distributed_train.sh 2 1 192.168.1.10:12345 \
#     --model resnet50 --data-dir /data/imagenet -b 128 --epochs 90
#
# Set PYTHON_BIN to change the interpreter command (e.g. "uv run python").
set -euo pipefail

if [ "$#" -lt 3 ]; then
    sed -n '2,18p' "$0"
    exit 1
fi

NUM_PROCESSES=$1
PROCESS_ID=$2
COORDINATOR=$3
shift 3

PYTHON_BIN=${PYTHON_BIN:-python}

exec $PYTHON_BIN -m jimm.train \
    --dist-num-processes "$NUM_PROCESSES" \
    --dist-process-id "$PROCESS_ID" \
    --dist-coordinator-address "$COORDINATOR" \
    "$@"
