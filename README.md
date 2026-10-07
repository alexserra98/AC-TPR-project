# AC TPR project

Controlled active/passive sentences for studying agent and patient representations
in Pythia 6.9B. The tools generate and validate a dataset, extract agent/patient
activations, compute training role means, and evaluate role interventions.

**Voice** describes the active or passive construction. **Role** describes the
agent performing the action or the patient receiving it. For example, the boy
is the agent in both “The boy helped the girl.” and “The girl was helped by the boy.”

## Environment

Run commands from the repository root. The project uses Python 3.11.14 and `uv`, with TransformerLens 3.9.0,
Transformers 5.9.0, and Torch 2.7.1 pinned in the lockfile.
Keep dependency and Hugging Face caches on scratch storage:

```bash
export HF_HOME=/orfeo/scratch/dssc/zenocosini/huggingface
export UV_CACHE_DIR=/orfeo/scratch/dssc/zenocosini/uv-cache
export UV_LINK_MODE=copy
uv sync --locked
```

`uv sync` creates `.venv` and installs the package and test dependencies from
`uv.lock`. Copy mode supports the cache and environment living on different
filesystems. Set these variables in each shell or Slurm job that runs the tools.
Dataset generation uses the CPU and downloads only tokenizer files.

### Classic TransformerLens implementation

The model uses TransformerLens 3.9.0's `HookedTransformer.from_pretrained_no_processing`
with the exact Hugging Face checkpoint commit. LayerNorm folding, weight centering,
value-bias folding, and attention-matrix refactoring are disabled. The fast tokenizer,
right padding, explicit attention masks, and absence of BOS/EOS insertion are unchanged.
Extraction and edits use `blocks.{layer}.hook_resid_post`. Scoring uses `unembed.hook_in`
to send only the final non-padding position to the language-model head.

The default dtype is **float32**. Classic attention computes QK explicitly and can
overflow in float16; its softmax implementation can then replace NaNs with zeros.
The loader rejects float16 for this checkpoint. Float32 avoids that failure without
an attention-method override. Q/K/V, score, and pattern hooks remain available.
The GPU comparison used about 26.3 GiB of allocated memory on a 32 GiB V100.

Activation, vector, and intervention metadata use `schema_version: 3`. The pipeline
requires the classic API, its canonical hooks, raw weights, and the installed
TransformerLens version. Regenerate schema-2 Bridge artifacts before consuming them.
New directories end in `-tl3`; previous HF and `-tl4` runs remain available.
Intervention metadata preserves extraction provenance under `model` and records
execution provenance separately under `execution_model`.

### Validation and regeneration

The saved pre-migration reference is in `ac-tpr-cache/migration-reference-tl4`.
Its tiny model uses float32; the real-checkpoint reference uses float16.
The CPU comparison retains `rtol=1e-5, atol=1e-6`. The GPU comparison with the old
float16 run is descriptive because the execution precision has changed. An
independent HF float32 run checks the same checkpoint, token batches, role vectors,
and all 128 layer/mean/role conditions.

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/python scripts/temporary/compare_transformerlens.py \
  --output-dir ac-tpr-cache/migration-comparison-tl3-cpu
sbatch scripts/slurm/temporary/compare_transformerlens.sh
# After the classic GPU comparison finishes:
sbatch scripts/slurm/temporary/compare_classic_hf.sh
# After the independent HF comparison finishes:
.venv/bin/python scripts/temporary/review_tl3_comparison.py
sbatch scripts/slurm/temporary/regenerate_tl3.sh
```

All output directories must be new. The regeneration wrapper requires
`ac-tpr-cache/migration-validation-tl3.json` to pass with the installed version.
It runs extraction, training-role averaging, the smoke test, and the full sweep.
After completion, execute the analysis notebook as described below.

The 8-row GPU check had a maximum candidate-logit difference of 0.00047 against
HF float32. Activation relative RMS error was 0.00066%. The final activation
bounds are `rtol=1e-4, atol=5e-4` plus relative RMS below `1e-5`; score bounds
remain `rtol=1e-4, atol=1e-4`. The initial stricter activation check failed on
0.2% of entries; that report is preserved. Zero-vector and final-block controls
still require exact equality. Against the old float16 run, the maximum change
in the candidate logit difference was 0.256, with no preference changes in the
sample. These are numerical comparisons, not a claim of bitwise equivalence.

The earlier TL4 migration's exact-match report remains at
`ac-tpr-cache/migration-artifact-comparison-tl4.json`.

## Generate the starter corpus

```bash
uv run --locked scripts/generate_dataset.py \
  --fillers data/fillers.csv \
  --output-dir data/generated \
  --seed 42 \
  --held-out-nouns child teacher soldier artist
```

The starter vocabulary contains 16 singular animate nouns and 8 transitive
verbs. Every ordered pair of distinct nouns is combined with every verb in both
voices: `16 * 15 * 8 = 1,920` pairs and **3,840 sentences**.

| Split | Pairs | Sentences |
| --- | ---: | ---: |
| train | 844 | 1,688 |
| val | 106 | 212 |
| test | 106 | 212 |
| gen_test | 864 | 1,728 |

Output consists of `corpus.csv` and `split_metadata.json` in the chosen directory.
The original files directly under `data/` are the reference corpus.
Successful reruns replace the two output files. Invalid input or tokenization
fails before either output is written.

### Filler format

Supply a CSV with these exact column names and order:

```csv
label,lemma,past,participle
noun,boy,,
noun,girl,,
verb,help,helped,helped
verb,see,saw,seen
```

Labels are `noun` or `verb`. Words must use lowercase ASCII letters. Nouns must
be singular and verbs transitive; the vocabulary author supplies these linguistic
properties. All verbs require explicit past and participle forms, including
regular verbs. Unknown labels, duplicate entries, missing forms, and duplicate
generated sentences are rejected. No inflection library or automatic filtering
is used.

The templates are:

- Active: `The {agent} {past} the {patient}.`
- Passive: `The {patient} was {participle} by the {agent}.`

The generator checks each noun and inflected verb inside every complete sentence
with the fast `EleutherAI/pythia-6.9b` tokenizer at `step143000`. Each filler span
must occupy exactly one token. Special tokens are disabled; offsets account for
leading spaces and terminal punctuation. The CSV's verb lemma is an identifier;
its inflected forms are the tokens used in sentences.

### Splits and schema

A pair belongs to `gen_test` whenever either noun is held out. The remaining
unordered noun-pair/verb groups are shuffled with a local seeded random generator
and divided at `floor(0.8 * groups)` and `floor(0.9 * groups)`. Both role orders
and both voices stay together. This gives matching noun frequencies in agent
and patient roles within each split. Omit `--held-out-nouns` for three splits only.

Corpus columns retain the reference format:

| Column | Meaning |
| --- | --- |
| pair_id | Stable ID for an ordered agent/patient/verb combination |
| sentence | Complete sentence, including its final period |
| voice | `active` or `passive` |
| agent | Noun performing the action, independent of voice |
| patient | Noun receiving the action, independent of voice |
| verb | Verb lemma |
| split | `train`, `val`, `test`, or `gen_test` |

IDs follow sorted vocabulary and do not change with input ordering or seed.
Adding or removing vocabulary can change IDs. Metadata records the input path
and hash, vocabulary, templates, tokenizer revision and backend hash, seed,
held-out nouns, splitting rule, and counts.

## Extract activations

Run on a GPU allocation after installing the locked environment:

```bash
uv run --locked scripts/extract_activations.py \
  --corpus data/generated/corpus.csv \
  --output-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/activations/pythia-6.9b-step143000-generated-tl3 \
  --batch-size 32 --device cuda --dtype float32
```

This processes all 3,840 sentences, retaining their original order and splits.
It uses `EleutherAI/pythia-6.9b` at `step143000`, matching dataset generation.
Inputs contain the sentence alone, with no question or special tokens. Batches
use right padding and an attention mask. Each noun is located by character
offsets and must occupy one token; agent/patient order is independent of voice.

Saved states are the residual stream **after each transformer block**, indexed
0–31. All blocks are captured before the model's final layer normalization,
including block 31. These are the corresponding block output sites for later
interventions. Embeddings and final normalized states are excluded. Extraction
uses TransformerLens `HookedTransformer` with raw Hugging Face weights, explicit
float32 attention, and evaluation mode. Gradients and the KV cache are disabled.
Both extraction and intervention use `blocks.{layer}.hook_resid_post`; weight processing is disabled. Inputs are explicit token tensors without BOS insertion.

The new output directory contains:

- `activations.pt`: a dictionary containing `activations`, a CPU tensor with
  shape `[3840, 32, 2, 4096]` (sentence, layer, role, hidden dimension);
  `rows`, the original CSV records in tensor order; `input_ids`, the unpadded
  token IDs per sentence; and `role_positions`, a `[3840, 2]` tensor of token
  indices. Role index 0 is agent and index 1 is patient.
- `metadata.json`: corpus hash, model revision and resolved commit, activation
  site, tokenization settings, split/voice counts, dtype, software versions,
  device, and Slurm job ID. This file is written last to mark a complete run.

The default float32 tensor occupies about 3.8 GiB on disk and in CPU memory.
Model weights and temporary inference tensors also need memory. The model must
fit on one device. `--dtype` also accepts `bfloat16`; a CPU run should
use `--device cpu --dtype float32`. Existing output directories are rejected.
After an interrupted run, choose a new directory or remove the incomplete output
before rerunning.

Load the saved dictionary with:

```python
from pathlib import Path
import torch

run = Path("/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/activations/pythia-6.9b-step143000-generated-tl3")
assert (run / "metadata.json").exists(), "Extraction is incomplete"
data = torch.load(run / "activations.pt", map_location="cpu", weights_only=True, mmap=True)
train_indices = [i for i, row in enumerate(data["rows"]) if row["split"] == "train"]
agent_layer_0 = data["activations"][train_indices, 0, 0, :]
```

Role means use training rows only. Keep voice and token position available for
analysis because causal context differs between the two voices.

### Slurm run for the generated corpus

The one-off wrapper requests one V100, four CPU cores, and 64 GiB of RAM. It uses
the existing `.venv` and cached checkpoint without network access. Before
submission, download the checkpoint to `HF_HOME` from a node with network access:

```bash
uv run --locked hf download EleutherAI/pythia-6.9b \
  --revision step143000 --include '*.json' 'pytorch_model*.bin'
mkdir -p /orfeo/scratch/dssc/zenocosini/ac-tpr-cache
sbatch scripts/slurm/temporary/extract_generated_activations.sh
```

Submit from the repository root. Logs go to
`/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/extract-<job_id>.log`.

## Extract syntactic vectors

Compute agent/patient means from a completed activation extraction on CPU:

```bash
OMP_NUM_THREADS=2 uv run --locked scripts/extract_syntactic_vectors.py \
  --activations-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/activations/pythia-6.9b-step143000-generated-tl3 \
  --output-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/syntactic_vectors/pythia-6.9b-step143000-generated-train-tl3
```

The script uses only rows with `split == "train"`: 1,688 sentences in the generated
corpus, including 844 active and 844 passive sentences. Each sentence contributes
one vector per role and layer. Pooled means weight each training sentence equally;
per-voice means average active and passive sentences separately. Validation, test,
and generalization test rows do not contribute.

The input tensor is memory mapped and processed one layer at a time. Selected
states are converted to float32 before averaging, and means are stored in
float32. No model or tokenizer is loaded. The means retain the original residual
coordinates at block outputs 0–31, before final normalization. They are arithmetic
means without centering, projection, or unit normalization. Agent and patient are
semantic roles; whether these means capture useful syntax remains an experimental
question.

The new output directory contains:

- `syntactic_vectors.pt`: a dictionary with `pooled`, shaped `[32, 2, 4096]`
  (layer, role, hidden dimension), and `by_voice`, shaped `[2, 32, 2, 4096]`
  (voice, layer, role, hidden dimension). Role order is `[agent, patient]` and
  voice order is `[active, passive]`. Together the tensors occupy 3 MiB.
- `metadata.json`: tensor axes, shapes, output dtype, layer/role/voice ordering,
  training sentence counts, averaging rule, source directory, corpus hash, and
  source model, tokenizer, and activation-site details. `model.dtype` describes
  the source extraction; the top-level `dtype` describes the saved means. Metadata
  is written last as the completion marker.

Existing output directories, missing source metadata, incompatible tensor
metadata, missing training voices, and nonfinite training activations cause the
script to fail. The source artifact supplies the rows and provenance, so the
original CSV and checkpoint are not needed for averaging.

Load the means with:

```python
from pathlib import Path
import torch

run = Path("/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/syntactic_vectors/pythia-6.9b-step143000-generated-train-tl3")
assert (run / "metadata.json").exists(), "Vector extraction is incomplete"
vectors = torch.load(run / "syntactic_vectors.pt", map_location="cpu", weights_only=True)
pooled_agent_layer_0 = vectors["pooled"][0, 0]
passive_patient_layer_0 = vectors["by_voice"][1, 0, 1]
```

## Run role interventions

Run on a GPU allocation with the checkpoint already cached:

```bash
uv run --locked scripts/run_interventions.py \
  --corpus data/generated/corpus.csv \
  --vectors-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/syntactic_vectors/pythia-6.9b-step143000-generated-train-tl3 \
  --output-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/interventions/pythia-6.9b-step143000-generated-agent-tl3 \
  --splits test gen_test --batch-size 32 --device cuda --dtype float32
```

The default splits contain 1,940 sentences: 212 in `test` and 1,728 in `gen_test`.
Every selected sentence is evaluated with this fixed prompt, without special
tokens or trailing whitespace:

```text
{sentence}
Question: Who is performing the action?
Answer:
```

The candidate answers are the agent and patient nouns with a leading space,
such as `" boy"`. Each must append exactly one token to the prompt. The sentence
must remain an unchanged token prefix, preserving the role positions used for
activation extraction. The baseline scores both candidates at the final
non-padding prompt token. Pairwise agent preference means its logit exceeds the
patient's; it does not measure unrestricted answer generation.

For each block 0–31, the script evaluates both `pooled` means and `by_voice` means
matching the sentence's voice. Each condition starts from an independent forward
pass and changes one noun token:

- `edited_role=agent`: add `mu_patient - mu_agent` at the agent token.
- `edited_role=patient`: add `mu_agent - mu_patient` at the patient token.

The edit has strength 1, uses float32 arithmetic, and is cast back to the residual
dtype. TransformerLens hooks act on block outputs before final normalization and are
removed after each forward pass, including failures. The model uses classic attention in evaluation mode, with gradients
and KV caching disabled. The `unembed.hook_in` hook selects the final non-padding state before the
language-model head, so it processes one position per sentence. GPU scoring defaults to float32; `--dtype float32 --device
cpu` supports small-model checks. `--limit N` selects the first N matching rows
in corpus order for smoke runs.

Scores use these definitions:

```text
logit_diff = patient_logit - agent_logit
logit_diff_change = intervened_logit_diff - baseline_logit_diff
```

A positive difference favors the counterfactual patient answer. A positive
change indicates movement toward that answer. All selected rows contribute,
including rows where the baseline already favors the patient. Strict preference
rates count ties toward neither answer; their denominator is all rows in the
group. The agent-to-patient flip rate counts baseline agent wins that become
patient wins, also divided by all rows in the group.

The new output directory contains:

- `baseline.csv`: one row per sentence, preserving corpus columns and recording
  candidate token IDs, role positions, both baseline logits, and their difference.
- `results.csv`: one row per sentence, layer, mean type, and edited role. It
  includes baseline fields, both intervened logits, their difference, and the
  change from baseline. The default run produces 248,320 rows.
- `summary.csv`: counts, mean scores/changes, preference rates, and flip rates
  grouped by split, voice, layer, mean type, and edited role. The default run has
  512 groups. Test and held-out-noun results remain separately identifiable.
- `metadata.json`: input hashes, source model/tokenizer and activation site,
  fixed prompt, scoring definitions, intervention settings, controls, row counts,
  software versions, GPU, and Slurm job ID. Written last to mark completion.

Zero-vector edits at either role on the first batch must reproduce baseline
logits exactly. Every edit after the final block must also leave answer logits
unchanged: there is no later attention layer to carry the noun edit to the answer
position. Nonfinite edits or scores, incompatible vector provenance, or failed
controls stop the run without completion metadata. Existing output directories
are rejected; after a failed run, inspect it and choose a new output directory.

### Slurm smoke check and full run

```bash
sbatch scripts/slurm/temporary/run_generated_interventions.sh
```

Submit from the repository root after installing the locked environment and
caching the checkpoint as described above. The wrapper requests one V100, four
CPU cores, 96 GiB RAM, and four hours. It runs eight selected sentences through
every condition and layer, saving to a directory ending in `-agent-smoke-tl3`. Only a
successful smoke run starts the full evaluation. Both runs use the same fixed
prompt and settings. Logs go to
`/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/intervene-<job_id>.log`.

## Plot intervention results

Open [the executed notebook](notebook/role_interventions.ipynb) to inspect the
completed run. It includes four figures: baseline answer preferences, mean logit
changes across blocks, intervened scores relative to baseline, and conditional
agent-to-patient flip rates. Test/generalization splits and active/passive voices
have separate panels. The notebook reads the saved baseline and summary CSVs
through the `ac-tpr-cache` scratch-storage link and runs entirely on CPU.

Install the optional plotting dependencies with `uv sync --locked --group notebook`
and select the repository's `.venv` Python interpreter when opening the notebook.
To execute and save it from the repository root:

```bash
uv run --locked --group notebook python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path("notebook/role_interventions.ipynb")
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=120, kernel_name="python3").execute()
nbformat.write(notebook, path)
PY
```

Each run saves PNG and PDF figures, baseline statistics, descriptive peak
conditions, and source hashes under `notebook/figures/role_interventions_tl3/`.
Start with the [logit-change plot](notebook/figures/role_interventions_tl3/logit_change.png)
or its [vector PDF](notebook/figures/role_interventions_tl3/logit_change.pdf).

The conditional flip plot divides the saved all-sentence flip rate by the
baseline agent preference rate. Its denominator is therefore baseline agent wins;
post-intervention ties do not count as flips. Curves are descriptive means with
no uncertainty intervals. The largest displayed effects are selected across the
sweep and are not independently validated best settings.

## Soft TPR experiment

See [the reproduction guide](soft_tpr/README.md) for training the quantized
TPR autoencoders, comparing reconstructed and raw centroids, running the AE/PCA
controls, and regenerating the figures. Experiment entry points are in
`scripts/temporary/`, with implementation in `src/ac_tpr/`.

## Tests

```bash
uv run --locked pytest
```

Tests cover sentence roles and irregular verbs, split isolation and balance,
reproducibility, input errors, and tokenization using the real Pythia tokenizer.
Activation tests compare a small random GPT-NeoX model and a classic
HookedTransformer with identical weights on CPU to check role positions,
padding, block outputs before final normalization, cleanup of hooks, and saved
artifacts including a short final batch. They do not download model weights.
The tokenizer must be cached or downloadable. With it cached, tests also run with
`HF_HUB_OFFLINE=1 uv run --locked pytest`.

Syntactic-vector tests use synthetic activations to check arithmetic means with
unequal voice counts, exclusion of held-out rows, float32 precision, ordering,
provenance, and rejection of incomplete or incompatible source artifacts.

Intervention tests compare a small random GPT-NeoX causal model with a matching
classic HookedTransformer to check candidate
tokenization, padded scoring against the full language-model output, voice-specific
means, edit direction and position, hook cleanup, zero-vector and final-block
controls, output arithmetic, summaries, and incompatible-input failures.
Loader tests cover checkpoint pinning, raw-weight provenance, float16 rejection, large float32 query/key values,
and working Q/K/V and attention-pattern hooks. Scoring tests verify that
the language-model head receives only one position per sentence.

## Layout

- `src/ac_tpr/`: dataset, activations, role means, and intervention/evaluation code.
- `scripts/`: command-line entry points.
- `data/`: starter fillers, reference data, and generated corpus.
- `docs/`: [paper review and experiment rationale](docs/acevedo-2026-notes.md).
- `tests/`: dataset, activation, role-mean, and intervention checks.
- `notebook/`: executed intervention analysis and exported figures.

One-off experiment scripts belong in `scripts/temporary/` or
`scripts/slurm/temporary/`. Store model weights and activations on scratch storage.
