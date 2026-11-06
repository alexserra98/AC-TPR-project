#!/bin/bash
#SBATCH --job-name=qwen-base-intervene
#SBATCH --partition=H100
#SBATCH --account=lade
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-base-active-pass-long-intervene-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

run_dir=ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight
.venv/bin/python -u scripts/extract_syntactic_vectors.py \
    --activations-dir "$run_dir/activations" --output-dir "$run_dir/vectors"
.venv/bin/python -u scripts/run_interventions.py \
    --corpus data/active_pass_long/corpus.csv --vectors-dir "$run_dir/vectors" \
    --output-dir "$run_dir/interventions" --splits test gen_test \
    --batch-size 8 --device cuda --dtype bfloat16

.venv/bin/python - "$run_dir" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

run = Path(sys.argv[1])
vectors = json.loads((run / 'vectors/metadata.json').read_text())
metadata = json.loads((run / 'interventions/metadata.json').read_text())
assert vectors['sentences'] == 3300
assert metadata['baseline_rows'] == 2096
assert metadata['result_rows'] == 150912
assert metadata['summary_rows'] == 288
assert metadata['edited_roles'] == ['both']
assert metadata['sentences_by_split'] == {'test': 416, 'gen_test': 1680}
assert metadata['controls'] == {'zero_vector_first_batch': 'passed', 'final_block_all_rows': 'passed'}
assert metadata['limit'] is None
for filename, expected in [('baseline.csv', 2096), ('results.csv', 150912)]:
    count = 0
    with (run / 'interventions' / filename).open(newline='') as handle:
        for row in csv.DictReader(handle):
            count += 1
            assert all(math.isfinite(float(value)) for key, value in row.items() if 'logit' in key)
            if filename == 'results.csv':
                assert row['edited_role'] == 'both'
            if row.get('layer') == '35':
                assert float(row['logit_diff_change']) == 0
    assert count == expected, (filename, count)
print('Verified full intervention sweep: 2096 baseline rows, 150912 intervention rows', flush=True)
PY
