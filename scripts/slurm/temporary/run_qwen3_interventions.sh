#!/bin/bash
# HISTORICAL: requires its original checkout/environment; use run_qwen3_base.sh for Base.
#SBATCH --job-name=ac-tpr-qwen3
#SBATCH --partition=DGX
#SBATCH --account=lade
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-interventions-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PATH="${PWD}/.venv/bin:${PATH}"

.venv/bin/python -u scripts/temporary/run_qwen3_interventions.py --batch-size 8

.venv/bin/python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path('notebook/role_interventions_qwen3.ipynb')
notebook = nbformat.read(path, as_version=4)
NotebookClient(
    notebook, timeout=600, kernel_name='python3',
    resources={'metadata': {'path': str(Path.cwd())}},
).execute()
nbformat.write(notebook, path)
print(f'Executed {path}', flush=True)
PY
