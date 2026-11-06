#!/bin/bash
#SBATCH --job-name=ac-tpr-qwen3-base
#SBATCH --partition=DGX
#SBATCH --account=lade
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-base-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

run_dir=ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight
if [[ -e "$run_dir" ]]; then
    echo "Output directory already exists: $run_dir" >&2
    exit 1
fi
.venv/bin/python -u scripts/extract_activations.py \
    --corpus data/active_pass_long/corpus.csv --output-dir "$run_dir/activations"
.venv/bin/python -u scripts/extract_syntactic_vectors.py \
    --activations-dir "$run_dir/activations" --output-dir "$run_dir/vectors"
.venv/bin/python -u scripts/run_interventions.py \
    --corpus data/active_pass_long/corpus.csv --vectors-dir "$run_dir/vectors" \
    --output-dir "$run_dir/interventions"
