#!/bin/bash
set -euo pipefail

module purge
module load anaconda

if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: WORK is not set. Run this script after logging in to Swan." >&2
    exit 1
fi

ENV_BASE="${NRDSTOR:-$WORK}"
ENV_DIR="${NANONECKLACE_ENV:-${ENV_BASE}/conda-envs/nanonecklace}"

mkdir -p "$(dirname "$ENV_DIR")"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    echo "Creating conda environment at $ENV_DIR"
    mamba create -y -p "$ENV_DIR" python=3.11 pip
fi

conda activate "$ENV_DIR"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Running consistency tests"
python -m pytest -q

echo
echo "Environment ready: $ENV_DIR"
echo "For later shells:"
echo "  module load anaconda"
echo "  conda activate $ENV_DIR"
