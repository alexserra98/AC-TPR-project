# Plain autoencoder and PCA steering controls: two-layer pilot

Completed 6 October 2026. Simple reconstruction baselines preserve the measured
role-steering effect. At matched validation reconstruction error, the plain AE
has exactly the same conditional answer-flip rates as the Soft TPR adaptation
in all four held-out groups. Both PCA variants have the same rates as raw means.
This supports generic reconstruction/low-rank preservation as an explanation of
the earlier result; it does not establish a TPR-specific steering advantage.

## Protocol

- Frozen Pythia 6.9B, step143000, raw weights, classic TransformerLens 3.9.0,
  float32; same cached activations and Soft TPR checkpoints as the original pilot.
- Only layers 24 and 25. Earlier validation-selected conditions were frozen:
  active: block 24, by-voice means, agent edit; passive: block 25, pooled means,
  patient edit. No settings were selected using test or generalization scores.
- Fit on all 1,688 training sentences (3,376 noun tokens), use all 212 validation
  sentences for checkpoint/rank selection, then evaluate all 212 test sentences
  and all 1,728 sentences containing held-out nouns. Single seed 42 + layer.
- Reuse Soft TPR's train-token coordinate mean and scalar RMS normalization.
- Plain AE: 4096 -> 256 GELU -> 256 linear latent -> 4096 linear decoder;
  MSE objective, Adam 0.001, batch size 128, 100 epochs, gradient clipping at 1.
  Keep the best validation checkpoint and the checkpoint closest to the Soft TPR
  validation MSE. Check each batch during epochs 1–5 and each later epoch end.
  Matched checkpoints: layer 24 epoch 8, layer 25 epoch 7.
- PCA: exact CPU float64 SVD on normalized training tokens, retain up to 256
  components. Compare rank 256 and the rank with closest validation MSE to
  Soft TPR. Matched ranks are 49 and 48. Validate basis orthonormality and the
  analytic reconstruction-error curve against direct reconstruction.
- All centroids use decoded training tokens only. All methods use native vector
  norms and strength 1, the same model, batches, prompt, and residual hooks.
  Zero-vector and final-block controls pass for each split/voice; the final-block
  checks use every method's selected direction.

The 256-dimensional controls match the 8 x 32 effective filler coordinates, not
Soft TPR's discrete information budget: eight indices into 64 codes carry at most
48 bits. Continuous AE/PCA latents, parameter counts, and model families differ.
The quality-matched PCA additionally uses fewer dimensions; this is a separate
quality control, not a claim to match both bit rate and reconstruction exactly.

## Reconstruction

Normalized validation MSE (lower is better). Quality-matched baselines are all
within 1.55% relative error of Soft TPR. At dimension 256, both plain AE and PCA
reconstruct substantially better than Soft TPR.

| Method | Dimensions: layer 24 / 25 | Layer 24 MSE | Layer 25 MSE |
| --- | --- | --- | --- |
| Soft TPR | 256 / 256 | 0.076423 | 0.081158 |
| AE 256, best reconstruction | 256 / 256 | 0.028562 | 0.028741 |
| AE 256, matched reconstruction | 256 / 256 | 0.076396 | 0.082409 |
| PCA 256 | 256 / 256 | 0.020170 | 0.020756 |
| PCA, matched reconstruction | 49 / 48 | 0.077187 | 0.081070 |

## Steering

Mean patient-minus-agent logit change. Larger values mean stronger counterfactual
steering, not improved factual question answering.

| Group | Raw centroids | Soft TPR | AE 256, best reconstruction | AE 256, matched reconstruction | PCA 256 | PCA, matched reconstruction |
| --- | --- | --- | --- | --- | --- | --- |
| Test / active | 0.52570 | 0.51239 | 0.53680 | 0.51530 | 0.52579 | 0.52620 |
| Test / passive | 0.61537 | 0.61838 | 0.61612 | 0.62155 | 0.61527 | 0.61454 |
| Held-out nouns / active | 0.63137 | 0.62522 | 0.63593 | 0.62780 | 0.63141 | 0.63155 |
| Held-out nouns / passive | 0.54611 | 0.55172 | 0.54257 | 0.55219 | 0.54595 | 0.54461 |

Flips among sentences initially preferring the correct agent:

| Group | Raw | Soft TPR | Matched AE | Matched PCA |
| --- | --- | --- | --- | --- |
| Test / active | 13/54 (24.1%) | 13/54 (24.1%) | 13/54 (24.1%) | 13/54 (24.1%) |
| Test / passive | 17/71 (23.9%) | 17/71 (23.9%) | 17/71 (23.9%) | 17/71 (23.9%) |
| Held-out nouns / active | 138/413 (33.4%) | 137/413 (33.2%) | 137/413 (33.2%) | 138/413 (33.4%) |
| Held-out nouns / passive | 189/551 (34.3%) | 191/551 (34.7%) | 191/551 (34.7%) | 189/551 (34.3%) |

## Interpretation

PCA with only 49/48 coordinates preserves the selected raw direction with cosine
similarities 0.999986/0.999886. The corresponding values for the matched AE are
0.998109/0.996407 and for Soft TPR 0.999490/0.999305. A structured TPR bottleneck
is therefore not needed to reproduce these particular centroid interventions.
This does not test latent binding swaps or semantic alignment of learned slots.

These are descriptive results for two fixed conditions, one model/checkpoint,
one training seed, and one prompt. Equal flip rates are not a formal equivalence
test. Reconstruction quality is matched on validation only. Native direction
norms differ, so small steering differences mix direction and magnitude effects.
Per-condition paired 95% cluster bootstrap intervals use the existing analysis
helper (2,000 replicates, seed 42, unordered noun-pair/verb groups); they condition
on the fitted models and chosen settings and do not account for seed variation.

## Artifacts and verification

Successful artifacts are in `ac-tpr-cache/local-soft-tpr-v1/ae-pca-controls-v2/`:

- `comparison.png` and `.pdf`: all six methods on the four held-out groups.
- `reconstruction.csv`, `directions.csv`, `ae_training.csv`: fitting diagnostics.
- `results.csv`, `summary.csv`, `paired_gains.csv`, `gains_over_soft.csv`: full
  per-example results, summaries, and paired intervals.
- `*_layer24.pt`, `*_layer25.pt`, `vectors.pt`: fitted controls and centroids.
- `plan.json`, `metadata.json`, `verification.json`: protocol, completion, coverage
  checks and artifact hashes. All 11,640 rows cover exactly the same 1,940
  held-out sentences across six methods, without duplicates.

The first attempt stopped on a GPU float32 SVD consistency assertion, before
steering. It is marked superseded and none of its results enter this report.
The successful run uses CPU float64 SVD and completed in about 6.6 minutes.

Reproduce from the project root, using fresh output/log paths:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.cache/steering-run/gpu-venv/bin/python -u \
  scripts/temporary/resource_limited_run.py \
  --resource-log ac-tpr-cache/local-soft-tpr-v1/ae-pca-repeat-resources.jsonl \
  scripts/temporary/compare_reconstruction_baselines.py \
  --output-dir ac-tpr-cache/local-soft-tpr-v1/ae-pca-repeat
```
