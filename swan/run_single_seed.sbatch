#!/bin/bash
#SBATCH --job-name=nn-single
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/single-%j.out
#SBATCH --error=logs/single-%j.err

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

SEED="${SEED:-41}"
OUTDIR="${OUTDIR:-Results_seed${SEED}}"

python run_sweep_analysis.py "$SEED" "$OUTDIR"
