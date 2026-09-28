#!/bin/bash
#SBATCH --job-name=nn-parameters
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/parameters-%j.out
#SBATCH --error=logs/parameters-%j.err

set -euo pipefail

module purge
module load anaconda
cd "$SLURM_SUBMIT_DIR"

ENV_BASE="${NRDSTOR:-$WORK}"
ENV_DIR="${NANONECKLACE_ENV:-${ENV_BASE}/conda-envs/nanonecklace}"
conda activate "$ENV_DIR"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NANONECKLACE_WORKERS="$SLURM_CPUS_PER_TASK"

python parameter_analysis/parameter_sweep_cases.py
