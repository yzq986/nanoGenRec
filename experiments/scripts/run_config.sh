#!/bin/bash
# Generic wrapper: runs a YAML config through run_exp.py
# Usage: bash experiments/scripts/run_config.sh experiments/configs/exp-047.yaml [extra flags]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${REPO_ROOT}"

CONFIG="${1:?Usage: $0 <config.yaml>}"
shift
python experiments/run_exp.py "${CONFIG}" --no-smoke --commit "$@"
