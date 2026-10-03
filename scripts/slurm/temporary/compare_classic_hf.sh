#!/bin/bash
#SBATCH --job-name=ac-tpr-tl3-hf
#SBATCH --partition=GPU
#SBATCH --account=lade
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/compare-tl3-hf-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv/bin/python -u scripts/temporary/capture_hf_fp32.py \
    --classic-dir ac-tpr-cache/migration-comparison-tl3-gpu \
    --output-dir ac-tpr-cache/migration-comparison-tl3-hf-fp32
