# Soft TPR steering pilot: measured results

The decoded-centroid Soft TPR method steers Pythia toward the counterfactual
patient answer, but this run shows no consistent advantage over raw role means.
Native Soft TPR directions produce slightly smaller effects on active sentences
and slightly larger effects on passive sentences. Matching their norms to the raw
directions makes mean steering effects slightly smaller in all four held-out groups.

This is a result for the token-level autoencoder and decoded-centroid method in
`src/ac_tpr/soft_tpr.py`. It does not evaluate sample-specific binding swaps or
establish a general limitation of Soft TPR representations.

## Protocol

- Pythia 6.9B, `step143000`, commit `21bfa02e806e253fe453702c29c81d9f83617255`.
- Re-extracted all 3,840 sentences using raw weights, classic TransformerLens 3.9.0,
  and float32 block-output activations. Pythia weights remained frozen.
- Trained 32 autoencoders, 100 epochs each, seed 42 (layer seed `42 + layer`).
  Architecture and loss settings are the defaults in [the experiment note](soft-tpr-experiment.md).
- Training and centroids used only 1,688 training sentences. The 212 validation
  sentences selected reconstruction checkpoints and steering conditions.
- Searched blocks 16, 20, 24, 25, and 28; pooled/per-voice means; agent/patient
  edits. Candidate blocks were informed by earlier repository results.
- Independently selected the largest validation mean logit change for each
  method and voice. All methods selected the same condition for each voice.
- Evaluated frozen conditions on all 212 test sentences and all 1,728 sentences
  containing held-out nouns. No test scores were used for selection in this run.
- Methods: raw means, decoded Soft TPR means, and Soft TPR translations scaled
  to match the corresponding raw direction norm. All use the unchanged question,
  noun candidates, residual hook, and edit strength 1.

Selected conditions:

| Voice | Block (zero-based) | Means | Edited token |
| --- | ---: | --- | --- |
| Active | 24 | Matching voice | Agent |
| Passive | 25 | Pooled | Patient |

The initial batch-8 validation attempt was stopped before any held-out evaluation
and superseded by the batch-32 run after measuring memory headroom. The final
results all come from the batch-32 run.

## Steering effects

Mean change in `patient_logit - agent_logit`; larger positive values indicate
stronger counterfactual steering, not better factual question answering:

| Split / voice | Sentences | Raw means | Soft TPR | Soft TPR, matched norm |
| --- | ---: | ---: | ---: | ---: |
| Test / active | 106 | 0.5257 | 0.5124 | 0.5047 |
| Test / passive | 106 | 0.6154 | 0.6184 | 0.6127 |
| Held-out nouns / active | 864 | 0.6314 | 0.6252 | 0.6214 |
| Held-out nouns / passive | 864 | 0.5461 | 0.5517 | 0.5408 |

Flips to the patient among examples initially preferring the correct agent:

| Split / voice | Initially preferred agent | Raw means | Soft TPR | Soft TPR, matched norm |
| --- | ---: | ---: | ---: | ---: |
| Test / active | 54 | 13 (24.1%) | 13 (24.1%) | 13 (24.1%) |
| Test / passive | 71 | 17 (23.9%) | 17 (23.9%) | 17 (23.9%) |
| Held-out nouns / active | 413 | 138 (33.4%) | 137 (33.2%) | 136 (32.9%) |
| Held-out nouns / passive | 551 | 189 (34.3%) | 191 (34.7%) | 189 (34.3%) |

These are strict preferences between two noun logits, not unrestricted generated
answers. Baseline agent preference was 50.9% / 67.0% for test active/passive and
47.8% / 63.8% for held-out-noun active/passive sentences.

Paired difference in mean logit change relative to raw means, with 95% cluster
bootstrap intervals:

| Split / voice | Soft TPR minus raw | Matched Soft TPR minus raw |
| --- | --- | --- |
| Test / active | -0.01330 [-0.02085, -0.00669] | -0.02097 [-0.03297, -0.01031] |
| Test / passive | +0.00300 [+0.00053, +0.00562] | -0.00272 [-0.00504, -0.00008] |
| Held-out nouns / active | -0.00615 [-0.00770, -0.00476] | -0.00997 [-0.01263, -0.00767] |
| Held-out nouns / passive | +0.00561 [+0.00503, +0.00623] | -0.00530 [-0.00595, -0.00466] |

Intervals use 2,000 paired bootstrap replicates with seed 42. The resampling unit
is the unordered noun-pair/verb group, keeping reversed role orders together.
They are conditional on the selected settings and this fixed corpus. They do not
include variation over training seeds, prompts, or model checkpoints, and are not
adjusted for multiple comparisons.

## Interpretation and checks

Across the candidate layers, decoded directions have cosine similarity around
0.999 to the raw directions and norm ratios approximately 0.98–1.004. This is
consistent with the very similar steering effects: the decoded-centroid method
largely preserves the original role means. Code usage diagnostics show multiple
active codes per latent slot; this similarity is not simply single-code collapse.
The tiny passive advantage of the native vectors disappears under norm matching.

All zero-vector and final-block controls passed. Controls cover both roles on
the first batch of each split/voice, with raw and Soft TPR vectors at the final
block. The test suite passed 112 tests, including norm matching, validation-only
selection, and a tiny-model run through the same pilot evaluator.

The experiment ran on the local GB10 using two CPU threads and a GPU allocation
cap of 32%, with a watchdog stopping only our process below 24 GiB available RAM
or above 80 GiB process RSS. Across extraction, training, and both evaluation
attempts, observed peak GPU reservation was 26.57 GiB, peak process RSS was
51.16 GiB, and available system memory never fell below 62.79 GiB in two-second
samples. Training alone reserved only 0.16 GiB of GPU memory.

This ARM/GB10 machine needed the existing CUDA-enabled PyTorch 2.13.0+cu130 and
Python 3.12.3 in an isolated runtime; the repository's locked PyTorch 2.7.1 build
is CPU-only here. Transformers 5.9.0 and TransformerLens 3.9.0 remained pinned.
All compared methods used that same runtime, and all 112 tests passed in it.
These numbers should not be treated as bitwise comparisons to the older saved
HF/TL4 float16 runs.

## Local artifacts and reproduction

Artifacts are under the ignored directory `ac-tpr-cache/local-soft-tpr-v1/`:

- `activations/`: full float32 activation extraction and provenance.
- `training-seed42/`: 32 checkpoints, training metrics, direction diagnostics,
  and raw/Soft TPR vector artifacts.
- `steering-b32/`: validation and held-out per-example CSVs, selection, completion
  metadata, `selected_summary.csv`, `paired_gains.csv`, and hashed `report.json`.
- `steering-b32/steering_results.png` and `.pdf`: standalone comparison figure.
- `resource_summary.json` and resource JSONL files: measured resource usage.

After extracting and training as described in the experiment note, the pilot can
be repeated in a suitable CUDA environment with new output paths:

```bash
python scripts/temporary/resource_limited_run.py \
  --resource-log ac-tpr-cache/local-soft-tpr-v1/repeat-resources.jsonl \
  scripts/temporary/evaluate_soft_tpr_pilot.py \
  --corpus data/generated/corpus.csv \
  --experiment-dir ac-tpr-cache/local-soft-tpr-v1/training-seed42 \
  --output-dir ac-tpr-cache/local-soft-tpr-v1/repeat \
  --layers 16 20 24 25 28 --batch-size 32 --device cuda

python scripts/temporary/analyze_soft_tpr_pilot.py \
  --run-dir ac-tpr-cache/local-soft-tpr-v1/repeat
```

No files were committed or pushed. This is a single-seed descriptive pilot of the
current decoded-centroid approach, with measured steering but no consistent
improvement over the simpler baseline.
