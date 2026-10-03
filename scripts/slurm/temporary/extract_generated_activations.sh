#!/bin/bash
#SBATCH --job-name=ac-tpr-activations
#SBATCH --partition=GPU
#SBATCH --account=lade
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/extract-%j.log

# Submit from the repository root after running uv sync --locked.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv/bin/python -u scripts/extract_activations.py \
    --corpus data/generated/corpus.csv \
    --output-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/activations/pythia-6.9b-step143000-generated-tl3 \
    --batch-size 32 --device cuda --dtype float32
