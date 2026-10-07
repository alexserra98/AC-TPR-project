# Reproduce the Soft TPR experiments

Run from the repository root. Use the environment and checkpoint setup in the
main README, with plotting and test dependencies (`uv sync --locked --group notebook`).
Activate that environment before running the commands below. The original local
GB10 run used Python 3.12.3 and CUDA Torch 2.13.0+cu130 instead of the repository's
Python 3.11/Torch 2.7.1 environment; both used Transformers 5.9.0 and
TransformerLens 3.9.0. Numerical results can differ across hardware and runtimes.
Use a CUDA-compatible Torch build for your GPU. Model evaluation and the AE/PCA
script require a GPU with enough memory for float32 Pythia 6.9B and its activations.

Choose a fresh experiment root; scripts reject existing output directories.
The examples reuse `data/generated/corpus.csv` if it already exists. To create
it, first run the dataset-generation command in the main README (seed 42,
held-out nouns `child teacher soldier artist`). Model/tokenizer files must be
available locally or downloadable; the full-curve wrapper expects them cached.

```bash
export EXPERIMENT_ROOT=ac-tpr-cache/soft-tpr-repeat
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

python scripts/extract_activations.py \
  --corpus data/generated/corpus.csv \
  --output-dir "$EXPERIMENT_ROOT/activations" \
  --batch-size 32 --device cuda --dtype float32

python scripts/temporary/train_soft_tpr.py \
  --activations-dir "$EXPERIMENT_ROOT/activations" \
  --output-dir "$EXPERIMENT_ROOT/training-seed42" \
  --device cuda --seed 42 --epochs 100 --batch-size 128
```

Training uses eight fixed orthonormal latent slots, role dimension 16, filler
dimension 32, and a shared learned codebook of 64 vectors. Each block has its own
autoencoder. Training activations fit the model; validation reconstruction error
selects its checkpoint. Reconstructed training activations are grouped by the
known agent/patient labels to make centroids. These labels do not supervise the
latent slots. Steering adds the difference between the reconstructed centroids
at a noun token, using the original intervention runner.

## Validation-selected steering pilot

This evaluates candidate blocks 16, 20, 24, 25, and 28, selects conditions on
validation data, and evaluates raw, Soft TPR, and norm-matched Soft TPR directions
on test and held-out-noun sentences.

```bash
python scripts/temporary/evaluate_soft_tpr_pilot.py \
  --corpus data/generated/corpus.csv \
  --experiment-dir "$EXPERIMENT_ROOT/training-seed42" \
  --output-dir "$EXPERIMENT_ROOT/steering" \
  --layers 16 20 24 25 28 --batch-size 32 --device cuda
python scripts/temporary/analyze_soft_tpr_pilot.py \
  --run-dir "$EXPERIMENT_ROOT/steering"
```

## Full-layer curves

The following evaluates both centroid methods over all 32 blocks, both edited
roles, and pooled/per-voice means, then produces the comparison figures.
`PYTHON`, `EXPERIMENT_ROOT`, `RUN_ROOT`, `FIGURE_ROOT`, and `BATCH_SIZE` are configurable.
The wrapper uses `resource_limited_run.py`, which needs `psutil` and limits CPU
threads and memory. Its defaults were chosen for the local GB10 machine: 24 GiB
minimum available system RAM, 80 GiB maximum process RSS, and GPU fraction 0.32.
The example permits the full GPU allocation; reduce batch size if necessary.

```bash
PYTHON=python GPU_FRACTION=1 BATCH_SIZE=32 \
  FIGURE_ROOT="$EXPERIMENT_ROOT/figures" \
  bash scripts/temporary/run_soft_tpr_full_curves.sh
```

`compare_soft_tpr.py` also provides a CSV-only comparison; use `--help` for its
input directory arguments. `plot_soft_tpr_layers.py` can regenerate the full
figures from completed sweeps without loading Pythia.

## Plain autoencoder and PCA controls

These controls use the original pilot's fixed conditions: block 24, by-voice
agent edit for active sentences; block 25, pooled patient edit for passive
sentences. They do not select new conditions in this run. The script fits a
256-dimensional plain AE and PCA on training data, plus variants selected to
match Soft TPR's validation reconstruction error. It evaluates all six methods.
The final single-panel plot shows raw centroids, Soft TPR, AE 256, and PCA 256.

```bash
python scripts/temporary/compare_reconstruction_baselines.py \
  --root "$EXPERIMENT_ROOT" \
  --output-dir "$EXPERIMENT_ROOT/ae-pca-controls" --batch-size 32
python scripts/temporary/plot_reconstruction_shift.py \
  --run-dir "$EXPERIMENT_ROOT/ae-pca-controls"
```

The original evaluation batch size was 128 for full curves and AE/PCA controls,
and 32 for the pilot. The original runs used the memory wrapper, which disables
TF32. For that execution mode, prefix any Python script with
`python scripts/temporary/resource_limited_run.py --resource-log NEW_LOG_PATH --gpu-fraction 1`
followed by the script path and its arguments. Keep the resource log path fresh.

## Checks and files

```bash
python -m pytest -q tests/test_soft_tpr.py tests/test_soft_tpr_pilot.py
```

Implementation: `src/ac_tpr/soft_tpr.py` and `soft_tpr_comparison.py`.
Entry points and analysis: `scripts/temporary/*soft_tpr*.py`,
`compare_reconstruction_baselines.py`, and `plot_reconstruction_shift.py`.
The small addition in `src/ac_tpr/interventions.py` records vector provenance.
Existing dataset, activation extraction, model loading, and intervention code
remain dependencies. Model weights, activations, and trained checkpoints are
regenerated by these commands and are not included in Git.

Scores are next-token patient-minus-agent noun logits under the original
agent-question prompt. Preservation of these centroid edits does not establish
that latent slots encode semantic roles or that Pythia uses a tensor-product
representation. This is a token-activation adaptation without the paper's
weakly supervised role-alignment objective.
