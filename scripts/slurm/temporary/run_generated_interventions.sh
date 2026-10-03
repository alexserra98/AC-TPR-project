#!/bin/bash
#SBATCH --job-name=ac-tpr-interventions
#SBATCH --partition=GPU
#SBATCH --account=lade
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/intervene-%j.log

# Submit from the repository root after running uv sync --locked.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

experiment_root=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache
experiment_name=pythia-6.9b-step143000-generated-agent-tl3
experiment_args=(
    --corpus data/generated/corpus.csv
    --vectors-dir "${experiment_root}/syntactic_vectors/pythia-6.9b-step143000-generated-train-tl3"
    --splits test gen_test --batch-size 32 --device cuda --dtype float32
)

.venv/bin/python -u scripts/run_interventions.py "${experiment_args[@]}" \
    --limit 8 --output-dir "${experiment_root}/interventions/pythia-6.9b-step143000-generated-agent-smoke-tl3"

.venv/bin/python -u scripts/run_interventions.py "${experiment_args[@]}" \
    --output-dir "${experiment_root}/interventions/${experiment_name}"
