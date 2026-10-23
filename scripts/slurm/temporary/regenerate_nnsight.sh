#!/bin/bash
#SBATCH --job-name=ac-tpr-nnsight
#SBATCH --partition=GPU
#SBATCH --account=lade
#SBATCH --gres=gpu:V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/regenerate-nnsight-%j.log

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
reports = {
    device: json.loads(Path(f"ac-tpr-cache/migration-comparison-nnsight-{device}/comparison.json").read_text())
    for device in ("cpu", "gpu")
}
for report in reports.values():
    assert report["passed"], "The migration validation must pass before regeneration"
    assert report["versions"] == {name: version(name) for name in ("torch", "transformers", "nnsight")}
    assert report["attention_implementation"] == "eager"
    assert report["controls"] == {"zero_vector": "exact", "final_block": "exact"}
assert reports["gpu"]["conditions"] == 128
Path("ac-tpr-cache/migration-validation-nnsight.json").write_text(
    json.dumps({"passed": True, "comparisons": reports}, indent=2) + "\n"
)
PY

bash scripts/slurm/temporary/extract_generated_activations.sh
.venv/bin/python -u scripts/extract_syntactic_vectors.py \
    --activations-dir ac-tpr-cache/activations/pythia-6.9b-step143000-generated-nnsight \
    --output-dir ac-tpr-cache/syntactic_vectors/pythia-6.9b-step143000-generated-train-nnsight
bash scripts/slurm/temporary/run_generated_interventions.sh

PATH="${PWD}/.venv/bin:${PATH}" .venv/bin/python scripts/temporary/execute_nnsight_notebooks.py
