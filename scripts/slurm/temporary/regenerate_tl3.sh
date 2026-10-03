#!/bin/bash
#SBATCH --job-name=ac-tpr-tl3
#SBATCH --partition=GPU
#SBATCH --account=lade
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/regenerate-tl3-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

.venv/bin/python - <<'PY'
import json
from pathlib import Path
from importlib.metadata import version
report = json.loads(Path("ac-tpr-cache/migration-validation-tl3.json").read_text())
assert report["passed"], "The migration validation must pass before regeneration"
assert report["versions"]["transformer-lens"] == version("transformer-lens")
PY

bash scripts/slurm/temporary/extract_generated_activations.sh
.venv/bin/python -u scripts/extract_syntactic_vectors.py \
    --activations-dir ac-tpr-cache/activations/pythia-6.9b-step143000-generated-tl3 \
    --output-dir ac-tpr-cache/syntactic_vectors/pythia-6.9b-step143000-generated-train-tl3
bash scripts/slurm/temporary/run_generated_interventions.sh
