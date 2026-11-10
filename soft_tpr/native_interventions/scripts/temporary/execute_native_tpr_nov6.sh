#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
run_dir=${1:?Supply a new or resumable native TPR experiment directory}
runtime_python=${TPR_PYTHON:-/tmp/tpr-binding-gpu-venv/bin/python}
export PYTHONPATH="/tmp/ac-tpr-nov6-extra:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [[ ! -f "$run_dir/plan.json" ]]; then
    "$runtime_python" scripts/temporary/run_native_tpr_nov6.py prepare --run-dir "$run_dir"
fi
if [[ ! -f "$run_dir/calibration/metadata.json" ]]; then
    nice -n 10 "$runtime_python" -u scripts/temporary/run_native_tpr_nov6.py calibrate --run-dir "$run_dir" > "$run_dir/calibration.log" 2>&1
fi
if [[ ! -f "$run_dir/evaluation/metadata.json" ]]; then
    resource_log="$run_dir/resources-$(date -u +%Y%m%dT%H%M%S).jsonl"
    "$runtime_python" -u scripts/temporary/resource_limited_run.py \
        --resource-log "$resource_log" --max-rss-gib 42 --min-available-gib 32 --gpu-fraction 0.28 \
        scripts/temporary/run_native_tpr_nov6.py run --run-dir "$run_dir" \
        > "$run_dir/run.log" 2>&1
fi
if [[ ! -f "$run_dir/diagnostics/metadata.json" ]]; then
    nice -n 10 "$runtime_python" scripts/temporary/diagnose_native_tpr_nov6.py --run-dir "$run_dir" > "$run_dir/diagnostics.log" 2>&1
fi
if [[ ! -f "$run_dir/report/metadata.json" ]]; then
    "$runtime_python" scripts/temporary/report_native_tpr_nov6.py --run-dir "$run_dir"
fi
