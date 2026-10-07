#!/usr/bin/env bash
# Evaluate both vector sets with the same runner, then validate and plot the sweeps.
set -euo pipefail
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
experiment_root=${EXPERIMENT_ROOT:-ac-tpr-cache/local-soft-tpr-v1}
run_root=${RUN_ROOT:-$experiment_root/full-layer-curves}
runtime_python=${PYTHON:-python}
figure_root=${FIGURE_ROOT:-notebook/figures/soft_tpr_comparison}
mkdir -p "$run_root"
for method in centroid soft_tpr; do
    vectors=role_means
    if [[ "$method" == soft_tpr ]]; then vectors=soft_tpr_vectors; fi
    "$runtime_python" -u scripts/temporary/resource_limited_run.py \
        --resource-log "$run_root/$method-resources.jsonl" \
        --gpu-fraction "${GPU_FRACTION:-0.32}" \
        scripts/run_interventions.py \
        --corpus data/generated/corpus.csv \
        --vectors-dir "$experiment_root/training-seed42/$vectors" \
        --output-dir "$run_root/$method" \
        --splits test gen_test --batch-size "${BATCH_SIZE:-128}" --device cuda --dtype float32
 done
"$runtime_python" scripts/temporary/plot_soft_tpr_layers.py \
    --centroid-dir "$run_root/centroid" --soft-tpr-dir "$run_root/soft_tpr" \
    --output-dir "$figure_root"
