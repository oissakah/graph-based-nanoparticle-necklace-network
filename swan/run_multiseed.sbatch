#!/bin/bash
#SBATCH --job-name=nn-multiseed
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/multiseed-%j.out
#SBATCH --error=logs/multiseed-%j.err

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

python run_multiseed_iv.py "${OUTDIR:-Results_multiseed}"
