# Soft TPR role steering

This experiment asks whether role directions estimated through a quantized,
tensor-structured autoencoder produce stronger counterfactual steering than the
original arithmetic role means. It uses the existing extracted noun activations,
prompt, block-output edit sites, strength 1, and evaluation metrics. Pythia remains
frozen. No experiment is submitted automatically.

## Representation and training

The architecture adapts the unbinding, quantization, rebinding, and form-loss
construction from [Soft TPR and its reference implementation](https://github.com/gomb0c/soft_tpr/)
to individual residual-stream token vectors. It is an unsupervised token-level
adaptation, not a reproduction of the paper's image experiments or weakly
supervised objective.

One autoencoder is trained independently for every transformer block. Both noun
tokens of every training sentence contribute, with equal token weights. The
encoder never receives agent/patient labels, noun IDs, or voice labels.

1. Subtract the train-token coordinate mean and divide by one train-only scalar
   RMS. This preserves the relative scale of residual coordinates.
2. An MLP encoder produces an unconstrained tensor `z[filler_dim, role_dim]`.
3. Unbind fillers with `f_k = z @ r_k`, where the fixed role vectors are columns
   of a seeded orthonormal matrix.
4. Assign each filler to the nearest entry in a shared learned codebook.
5. Construct an explicit TPR by summing outer products: `T = sum_k q_k outer r_k`.
6. A linear decoder reconstructs the normalized input from flattened `T`.

The default objective is:

```text
MSE(reconstruction, input)
+ form_weight * MSE(z, stop_gradient(T))
+ MSE(quantized_fillers, stop_gradient(soft_fillers))
+ commitment_weight * MSE(soft_fillers, stop_gradient(quantized_fillers))
```

Reconstruction uses a straight-through estimator at the unbound fillers. Its
gradient reaches the encoder through the role subspace; the form loss also
constrains components outside that subspace. Codebook initialization samples
encoded training fillers. The role matrix stays fixed; encoder, codebook, and
decoder are learned. No lexical supervision or codebook labels are imposed.

Defaults: 8 latent roles, role dimension 16, filler dimension 32, 64 codes, encoder
hidden width 256, 100 epochs, Adam at 0.001, batch size 128, form weight 1,
commitment weight 0.25. These are starting hyperparameters, not validated choices.
Each layer uses seed `seed + layer`.

The validation split selects the epoch with lowest reconstruction MSE, breaking
ties in favor of the earliest epoch. Normalization, codebook initialization,
gradients, and exported centroids use training data only. Test and generalization
activations are never used by the autoencoder. Validation must contain both voices.
Runs require the repository's schema-3 activation artifacts.

## Directions and interpretation

The latent roles are **unlabeled slots**, not a prespecified agent/patient binding.
After fitting, reconstruct training tokens through the quantized bottleneck and
undo normalization. Group those reconstructions using the known sentence roles:

```text
mu_role_tpr = mean(decoded_quantized_training_tokens_for_role)
agent edit:  h + mu_patient_tpr - mu_agent_tpr
patient edit: h + mu_agent_tpr - mu_patient_tpr
```

Means are computed both pooled across voices and separately for each voice,
exactly as in the arithmetic baseline. Decoding each token before averaging
avoids quantizing an average latent vector. The live Pythia state receives only
the resulting translation; it is not replaced with its reconstruction. This is
a global decoded-centroid experiment, not a sample-specific latent role swap.

`directions.csv` reports raw and Soft TPR direction norms and their cosine
similarity. Norms are not matched: a stronger effect can reflect direction,
magnitude, or both. `training.csv` reports reconstruction, form, codebook, and
commitment losses plus active-code counts and perplexity for every latent slot.
Inspect these to detect collapse or poor reconstruction before interpreting
steering results. A low reconstruction loss alone does not establish semantic
disentanglement.

## Run

Use the locked environment from the README. A completed activation extraction is
required; autoencoder training does not load Pythia or a tokenizer. CPU is the
training default; `--device cuda` is also supported. All output directories must
be new. Interrupted runs retain partial files without root completion metadata.

```bash
uv run --locked scripts/temporary/train_soft_tpr.py \
  --activations-dir ac-tpr-cache/activations/pythia-6.9b-step143000-generated-tl3 \
  --output-dir ac-tpr-cache/soft-tpr/seed42 \
  --device cpu --seed 42
```

The output contains `checkpoints/layer_XX.pt` (architecture, weights,
normalization, selected epoch and seed), `training.csv`, `directions.csv`, and:

- `role_means/`: the original arithmetic vectors, computed by the existing extractor.
- `soft_tpr_vectors/`: decoded quantized role means in the same vector format.
- `metadata.json`: completion marker, extraction provenance, hyperparameters,
  checkpoint hashes, and train/validation use.

Evaluate both vector sets using the existing runner. First use `--splits val`
and new output directories when selecting hyperparameters. Freeze settings before
the final test/generalization comparison below. A smoke check can add `--limit 8`
with separate output directories for both methods.

```bash
uv run --locked scripts/run_interventions.py \
  --corpus data/generated/corpus.csv \
  --vectors-dir ac-tpr-cache/soft-tpr/seed42/role_means \
  --output-dir ac-tpr-cache/soft-tpr/raw-test \
  --splits test gen_test --batch-size 32 --device cuda --dtype float32

uv run --locked scripts/run_interventions.py \
  --corpus data/generated/corpus.csv \
  --vectors-dir ac-tpr-cache/soft-tpr/seed42/soft_tpr_vectors \
  --output-dir ac-tpr-cache/soft-tpr/soft-test-seed42 \
  --splits test gen_test --batch-size 32 --device cuda --dtype float32

uv run --locked scripts/temporary/compare_soft_tpr.py \
  --baseline-dir ac-tpr-cache/soft-tpr/raw-test \
  --soft-tpr-dir ac-tpr-cache/soft-tpr/soft-test-seed42 \
  --output-dir ac-tpr-cache/soft-tpr/comparison-seed42
```

`comparison.csv` aligns split, voice, layer, mean type, and edited role. It reports
both methods and their difference for mean logit change, patient preference,
all-sentence flip rate, and conditional flip rate among baseline agent wins.
A conditional rate is blank when no baseline examples prefer the agent.
Comparison requires completed runs with identical examples, baseline scores,
model execution settings, and sweep conditions. It rejects mixed old/new backend
results. The runner records the vector method and autoencoder training provenance.

Positive differences mean stronger steering toward the counterfactual patient
answer. They do not mean improved factual QA accuracy. Comparisons are descriptive;
there are no significance tests or automatically selected best layers. Repeat
seeds and use validation for settings before making held-out improvement claims.

## Verification

```bash
OMP_NUM_THREADS=2 uv run --locked pytest
```

The added tests check binding/unbinding, nearest-code assignments, gradient
routes, learning on synthetic data, train-only normalization, checkpoint selection,
decoded-centroid export, test-split isolation, and a tiny-model path through both
steering sweeps and the comparison report. They require no Pythia model weights.

## Full-layer answer-score comparison

The full-curve workflow evaluates the original training centroids and decoded
Soft TPR centroids on all 1,940 test/generalization sentences, all 32 blocks,
both edited roles, and both pooled/per-voice means. Both methods use the same
float32 runtime and batch size 128. These are descriptive curves; the full sweep
does not replace the pilot's validation-selected comparison.

On the local GB10 runtime, run:

```bash
bash scripts/temporary/run_soft_tpr_full_curves.sh
```

The script uses the existing intervention runner and memory watchdog. It writes
new runs under `ac-tpr-cache/local-soft-tpr-v1/full-layer-curves/`, validates that
both methods have identical baseline scores and evaluation settings, then writes
these PNG/PDF pairs under `notebook/figures/soft_tpr_comparison/`:

- `answer_scores_centroid`: the original four-panel format for raw centroids.
- `answer_scores_soft_tpr`: the same format for decoded Soft TPR centroids.
- `answer_scores_comparison`: method columns with shared axes across all groups.
- `answer_scores_difference`: Soft TPR minus centroid scores for each condition.

The figure directory also contains per-condition comparison CSVs and provenance.
The workflow requires fresh output directories and does not overwrite past runs.
To plot independently after both evaluations complete, use
`scripts/temporary/plot_soft_tpr_layers.py --help` for the three directory arguments.
