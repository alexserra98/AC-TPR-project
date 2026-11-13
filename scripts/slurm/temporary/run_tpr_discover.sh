#!/bin/bash
#SBATCH --job-name=ac-tpr-discover
#SBATCH --partition=DGX
#SBATCH --account=lade
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
launch_dir="${1:?Pass the launch directory containing source/ and corpus.csv}"
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONPATH="${launch_dir}/source/src"

if [[ -e "${launch_dir}/pipeline" ]]; then
    echo "Output directory already exists: ${launch_dir}/pipeline" >&2
    exit 1
fi
echo "Starting DISCOVER pipeline in ${launch_dir}/pipeline (Slurm ${SLURM_JOB_ID})"
.venv/bin/python -u "${launch_dir}/source/scripts/temporary/run_tpr_pipeline.py" \
    --corpus "${launch_dir}/corpus.csv" \
    --output-dir "${launch_dir}/pipeline" \
    --layers 5 12 15 24 31 \
    --held-out-bindings John:agent doctor:patient \
    --device cuda --dtype bfloat16 --batch-size 8 \
    --train-batch-size 32 --seed 42 --epochs 100 \
    --eval-every 100 --patience 5 --max-lr-reductions 5 \
    --unpacker-lr 0.0001 --tpr-lr 0.001 \
    --decoder-width 1024 --decoder-layers 6 --decoder-heads 16 \
    --decoder-ffn 4096 --dropout 0.1 --max-length 32 \
    --filler-dim 100 --role-dim 100 --alpha 1
echo "DISCOVER pipeline complete (Slurm ${SLURM_JOB_ID})"
