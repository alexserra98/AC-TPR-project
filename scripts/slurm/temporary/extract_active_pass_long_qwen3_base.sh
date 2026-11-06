#!/bin/bash
#SBATCH --job-name=qwen-base-acts
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-base-active-pass-long-extract-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

run_dir=ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight/activations
.venv/bin/python -u scripts/extract_activations.py \
    --corpus data/active_pass_long/corpus.csv --output-dir "$run_dir" \
    --batch-size 8 --device cuda --dtype bfloat16

.venv/bin/python - "$run_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import torch

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION, load_corpus
from ac_tpr.prompts import PROMPT_TEMPLATE

run = Path(sys.argv[1])
corpus = Path('data/active_pass_long/corpus.csv')
metadata = json.loads((run / 'metadata.json').read_text())
data = torch.load(run / 'activations.pt', map_location='cpu', weights_only=True, mmap=True)
assert metadata['corpus_sha256'] == hashlib.sha256(corpus.read_bytes()).hexdigest()
assert metadata['model']['name'] == TOKENIZER_ID
assert metadata['model']['commit'] == TOKENIZER_REVISION
assert metadata['prompt_template'] == PROMPT_TEMPLATE
assert data['rows'] == load_corpus(corpus)
assert list(data['activations'].shape) == metadata['shape'] == [5808, 36, 2, 4096]
assert data['activations'].dtype == torch.bfloat16
assert list(data['role_positions'].shape) == [5808, 2]
for start in range(0, len(data['rows']), 128):
    assert torch.isfinite(data['activations'][start:start + 128]).all()
print('Verified saved activations:', metadata['shape'], metadata['model']['dtype'], flush=True)
PY
