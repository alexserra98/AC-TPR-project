#!/bin/bash
#SBATCH --job-name=ac-tpr-tl3-compare
#SBATCH --partition=GPU
#SBATCH --account=lade
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/compare-tl3-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv/bin/python -u scripts/temporary/compare_transformerlens.py \
    --device cuda --dtype float32 --report-only --output-dir ac-tpr-cache/migration-comparison-tl3-gpu
