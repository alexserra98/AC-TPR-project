# AC TPR project

Controlled active/passive sentences for studying agent and patient representations
in Qwen3-8B-Base. The tools generate and validate a dataset, extract agent/patient
activations, compute training role means, and evaluate role interventions.

**Voice** describes the active or passive construction. **Role** describes the
agent performing the action or the patient receiving it. For example, the boy
is the agent in both “The boy helped the girl.” and “The girl was helped by the boy.”

## Environment

Run commands from the repository root. The project uses Python 3.11.14 and `uv`, with NNsight 0.7.0,
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

### Model and prompt contract

The active pipeline uses only `Qwen/Qwen3-8B-Base`, pinned to commit
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4` for both model and tokenizer.
NNsight `LanguageModel` wraps raw Hugging Face weights with eager attention,
evaluation mode, and frozen parameters. GPU defaults are bfloat16 and batch
size 8; CPU tests use tiny Qwen3 models in float32. Float16 is not supported.
Every trace disables KV caching. Inputs use right padding and explicit attention
masks without added BOS/EOS tokens.

`ac_tpr.prompts` defines the five plain-text demonstrations from the baseline
notebook and the target sentence/question. No chat template or thinking tokens
are used. Extraction and intervention use this same full prompt. Role indices
come from character offsets inside the inserted target sentence, even when the
same sentence or nouns appear in demonstrations. Answers are one leading space
plus the noun and must append exactly one token.

Extraction and edits use `model.model.layers[layer].output` before final RMSNorm.
Extraction saves role states on CPU and stops before vocabulary projection.
Scoring reduces `model.lm_head.input` to the last non-padding position per row.
Edits are confined to each trace, including when a forward fails.

Activation, vector, and intervention metadata use `schema_version: 5` and record
the prompt template. Consumers reject incompatible checkpoint, tokenizer, prompt,
hooks, or software versions. Extraction and intervention accept a keyword-only
`prompt_template` argument; callers must supply the same value at both stages.
Changing notebook prompt settings alone does not change the pipeline default.
Source model provenance and scoring execution provenance remain separate.

### Validation and GPU runs

Run the unit tests and the small real-checkpoint smoke pipeline:

```bash
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python -m pytest
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/temporary/validate_qwen3_base.py \
  --output-dir ac-tpr-cache/qwen3-8b-base-active-pass-long-fewshot-smoke
```

The smoke driver validates all 5,808 corpus prompts, then creates a separate
20-sentence corpus: five training pairs, three test pairs, and two generalization
pairs. It runs the shared extraction, vector, and intervention entrypoints with
batch size 2, checks every layer/mean/role condition (1,440 intervention rows),
and compares baseline scores with independent Transformers forwards at
`rtol=0.01, atol=0.125` for BF16. Zero-vector and final-block controls require
exact equality. `validation.json` is written only after every check passes.
These scores are smoke validation, not research results. Existing output
directories are rejected.

A full run can be submitted with `scripts/slurm/temporary/run_qwen3_base.sh`.
It uses one A100, four CPU cores, 96 GB host memory, and a 12-hour limit.
Historical Pythia and instruction-tuned Qwen launchers, notes, notebooks, and
artifacts are retained but are not compatible inputs to the Base pipeline.
Historical scripts require their original checkout and environment.

## Prepare active_pass_long

The default corpus is `data/active_pass_long/corpus.csv`, derived from
`iuliialysova/active_pass_long` at revision
`949033163b2a9d4d2e1d8c7811e0eaa7f27e749c`. It contains all 5,808 source sentences,
including names, common nouns, and inanimate participants.

To reproduce it in a new output directory:

```bash
.venv/bin/python scripts/temporary/prepare_active_pass_long.py --output-dir data/active_pass_long
```

The converter refuses an existing directory and downloads only the pinned source
CSV (the tokenizer can use the existing cache). It writes the enriched 32-column
corpus, `split_metadata.json`, and a Hugging Face dataset README. See
[data/active_pass_long/README.md](data/active_pass_long/README.md) for schema and
upload instructions. Questions and demonstrations live in `src/ac_tpr/prompts.py`;
none are stored in the CSV. Edit its `QUESTION` constant to change the question
in every demonstration and target prompt for future runs, using the
same prompt for activation extraction and evaluation.

The seven core columns come first, followed by the source annotations except its
six question/answer fields. `pair_id` links active/passive versions of the same
roles; `quad_id` also includes the reversed roles. All four sentences of each
quad stay in one split. Mary, teacher, and rock are held out from training;
the remaining quads are split 80/10/10 with seed 42.

| Split | Sentences |
| --- | ---: |
| train | 3,300 |
| val | 412 |
| test | 416 |
| gen_test | 1,680 |

The pipeline accepts both the original seven-column schema and this exact
32-column schema. Annotations survive extraction and baseline/intervention CSVs;
summary grouping remains unchanged. Only agent/patient fillers and their answer
candidates must be single tokens. Multi-token verbs such as tickled are allowed.
The synthetic generator below retains its stricter template validation.

This corpus and the shared “Who or what is performing the action?” prompt require
fresh activations and vectors. Existing experiment artifacts are preserved;
the active launcher uses a separate active-pass-long run directory.

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
with the fast `Qwen/Qwen3-8B-Base` tokenizer at `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. Each filler span
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
  --corpus data/active_pass_long/corpus.csv \
  --output-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight/activations \
  --batch-size 8 --device cuda --dtype bfloat16
```

This processes all 5,808 sentences, retaining their original order and splits.
It uses `Qwen/Qwen3-8B-Base` at `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, matching dataset generation.
Inputs contain the shared five demonstrations, target sentence, question, and
`Answer:` suffix, without special tokens. Batches
use right padding and an attention mask. Each noun is located by character
offsets and must occupy one token; agent/patient order is independent of voice.

Saved states are the residual stream **after each transformer block**, indexed
0–35. All blocks are captured before the model's final layer normalization,
including block 35. These are the corresponding block output sites for later
interventions. Embeddings and final normalized states are excluded. Extraction
uses NNsight `LanguageModel` with raw Hugging Face weights, eager bfloat16
attention, and evaluation mode. Gradients and the KV cache are disabled. Both
extraction and intervention use `model.layers.{layer}.output`. Inputs are
explicit token tensors without BOS insertion.

The new output directory contains:

- `activations.pt`: a dictionary containing `activations`, a CPU tensor with
  shape `[5808, 36, 2, 4096]` (sentence, layer, role, hidden dimension);
  `rows`, the original CSV records in tensor order; `input_ids`, the unpadded
  token IDs per sentence; and `role_positions`, a `[5808, 2]` tensor of token
  indices. Role index 0 is agent and index 1 is patient.
- `metadata.json`: corpus hash, model revision and resolved commit, activation
  site, tokenization settings, split/voice counts, dtype, software versions,
  device, and Slurm job ID. This file is written last to mark a complete run.

The default bfloat16 tensor occupies about 3.2 GiB on disk and in CPU memory.
Model weights and temporary inference tensors also need memory. The model must
fit on one device. `--dtype` also accepts `float32`; a CPU run should
use `--device cpu --dtype float32`. Existing output directories are rejected.
After an interrupted run, choose a new directory or remove the incomplete output
before rerunning.

Load the saved dictionary with:

```python
from pathlib import Path
import torch

run = Path("/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight/activations")
assert (run / "metadata.json").exists(), "Extraction is incomplete"
data = torch.load(run / "activations.pt", map_location="cpu", weights_only=True, mmap=True)
train_indices = [i for i, row in enumerate(data["rows"]) if row["split"] == "train"]
agent_layer_0 = data["activations"][train_indices, 0, 0, :]
```

Role means use training rows only. Keep voice and token position available for
analysis because causal context differs between the two voices.

### Full pipeline on Slurm

Cache the pinned checkpoint before submitting the full extraction, vector, and
intervention workflow:

```bash
uv run --locked hf download Qwen/Qwen3-8B-Base \
  --revision 49e3418fbbbca6ecbdf9608b4d22e5a407081db4
mkdir -p /orfeo/scratch/dssc/zenocosini/ac-tpr-cache
sbatch scripts/slurm/temporary/run_qwen3_base.sh
```

Submit from the repository root. The new run root is
`ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight/`, with
`activations`, `vectors`, and `interventions` subdirectories. Existing roots
are rejected. The launcher uses the shared pipeline, not the historical chat
experiment script.

## Extract syntactic vectors

Compute agent/patient means from a completed activation extraction on CPU:

```bash
OMP_NUM_THREADS=2 uv run --locked scripts/extract_syntactic_vectors.py \
  --activations-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight/activations \
  --output-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight/vectors
```

The script uses only rows with `split == "train"`: 3,300 sentences in active_pass_long,
including 1,650 active and 1,650 passive sentences. Each sentence contributes
one vector per role and layer. Pooled means weight each training sentence equally;
per-voice means average active and passive sentences separately. Validation, test,
and generalization test rows do not contribute.

The input tensor is memory mapped and processed one layer at a time. Selected
states are converted to float32 before averaging, and means are stored in
float32. No model or tokenizer is loaded. The means retain the original residual
coordinates at block outputs 0–35, before final normalization. They are arithmetic
means without centering, projection, or unit normalization. Agent and patient are
semantic roles; whether these means capture useful syntax remains an experimental
question.

The new output directory contains:

- `syntactic_vectors.pt`: a dictionary with `pooled`, shaped `[36, 2, 4096]`
  (layer, role, hidden dimension), and `by_voice`, shaped `[2, 36, 2, 4096]`
  (voice, layer, role, hidden dimension). Role order is `[agent, patient]` and
  voice order is `[active, passive]`. Together the tensors occupy 3.375 MiB.
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

run = Path("/orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight/vectors")
assert (run / "metadata.json").exists(), "Vector extraction is incomplete"
vectors = torch.load(run / "syntactic_vectors.pt", map_location="cpu", weights_only=True)
pooled_agent_layer_0 = vectors["pooled"][0, 0]
passive_patient_layer_0 = vectors["by_voice"][1, 0, 1]
```

## Run role interventions

Run on a GPU allocation with the checkpoint already cached:

```bash
uv run --locked scripts/run_interventions.py \
  --corpus data/active_pass_long/corpus.csv \
  --vectors-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight/vectors \
  --output-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/qwen3-8b-base-active-pass-long-agent-fewshot-nnsight/interventions \
  --splits test gen_test --batch-size 8 --device cuda --dtype bfloat16
```

The default splits contain 2,096 sentences: 416 in `test` and 1,680 in `gen_test`.
Every selected sentence uses the shared few-shot prefix followed by this target
section, without special tokens or trailing whitespace:

```text
{sentence}
Question: Who or what is performing the action?
Answer:
```

The candidate answers are the agent and patient nouns with a leading space,
such as `" boy"`. Each must append exactly one token to the prompt. Role positions are located inside the target sentence in the full prompt,
using exactly the same preparation as activation extraction. The baseline scores both candidates at the final
non-padding prompt token. Pairwise agent preference means its logit exceeds the
patient's; it does not measure unrestricted answer generation.

For each block 0–35, the script evaluates both `pooled` means and `by_voice` means
matching the sentence's voice. Each condition starts from an independent forward
pass and changes both noun tokens simultaneously at the same block output
(`edited_role=both`):

- At the agent token, add `mu_patient - mu_agent`.
- At the patient token, add `mu_agent - mu_patient`.

The edit has strength 1, uses float32 arithmetic, and is cast back to the residual
dtype. NNsight edits act on block outputs before final normalization and are
confined to one trace, including when a forward fails. The model uses eager
attention in evaluation mode, with gradients and KV caching disabled. Assigning
`lm_head.input` selects the final non-padding state before the language-model
head, so it processes one position per sentence. GPU scoring defaults to bfloat16; `--dtype float32 --device
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
- `results.csv`: one row per sentence, layer, and mean type, with `edited_role=both`. It
  includes baseline fields, both intervened logits, their difference, and the
  change from baseline. The default run produces 150,912 rows.
- `summary.csv`: counts, mean scores/changes, preference rates, and flip rates
  grouped by split, voice, layer, mean type, and edited role. The default run has
  288 groups. Test and held-out-noun results remain separately identifiable.
- `metadata.json`: input hashes, source model/tokenizer and activation site,
  fixed prompt, scoring definitions, intervention settings, controls, row counts,
  software versions, GPU, and Slurm job ID. Written last to mark completion.

Simultaneous zero-vector edits at both roles on the first batch must reproduce baseline
logits exactly. Every edit after the final block must also leave answer logits
unchanged: there is no later attention layer to carry the noun edit to the answer
position. Nonfinite edits or scores, incompatible vector provenance, or failed
controls stop the run without completion metadata. Existing output directories
are rejected; after a failed run, inspect it and choose a new output directory.

Use the smoke driver above for the 20-sentence validation, and the Base Slurm
launcher for the full corpus. They write separate directories and never replace
historical experiment results.

## Plot intervention results

Open [the analysis notebook](notebook/role_interventions.ipynb) after completing a full Base run. It includes four figures: baseline answer preferences, mean logit
changes across blocks, intervened scores relative to baseline, and conditional
agent-to-patient flip rates. Test/generalization splits and active/passive voices
have separate panels. The two intervention curves compare pooled and per-voice
means with both role tokens edited together. The notebook requires joint-run
metadata (`edited_roles=["both"]`) and reads the saved baseline and summary CSVs
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
conditions, and source hashes under `notebook/figures/role_interventions_qwen3_base_active_pass_long/`.
Figures are generated when the analysis notebook is executed; historical figures
remain in their original directories.

The conditional flip plot divides the saved all-sentence flip rate by the
baseline agent preference rate. Its denominator is therefore baseline agent wins;
post-intervention ties do not count as flips. Curves are descriptive means with
no uncertainty intervals. The largest displayed effects are selected across the
sweep and are not independently validated best settings.

## TPR decomposition and steering

`src/ac_tpr/tpr/pipeline.py` documents and runs the DISCOVER adaptation. The
stages are implemented separately in `data.py`, `extraction.py`, `unpacking.py`,
`decomposition.py`, `reconstruction.py`, and `steering.py`; `training.py` supplies the shared fitting
loop. The module uses the existing model loader, prompt preparation, extraction
hook, and final-token scorer. It does not import the sibling DISCOVER checkout.
Upstream attribution and license are in `src/ac_tpr/tpr/NOTICE.md`.

The default experiment uses blocks **5, 12, 15, 24, 31** and independently fits
both **sentence_end** and **prompt_end** sites. The former is the token containing
the target sentence's final period (`".\n"` with the current tokenizer); the latter
is the final non-padding `":"` in `Answer:`. Both come from the complete few-shot
prompt and use block outputs before final RMSNorm. The unpacker reconstructs only
the target sentence, preserving capitalization and treating punctuation as a
separate token.

Each input has four symbolic bindings: noun→agent, verb lemma→verb, noun→patient,
and active/passive→voice. Semantic roles and noun embeddings are shared across
voices. Voice is an explicit categorical binding; it is not a copy of the prompt.
This hypothesis makes the noun-swap direction independent of voice at a given
layer/site. The existing voice-specific demonstrations still confound voice with
prompt context, so these experiments do not isolate syntax from demonstrations.

For each layer/site, train-only mean and population standard deviation define
`z = (h - mean) / scale`, with `scale = std + 1e-10`. The unpacker learns to decode
`z` into the sentence. Independently, the TPR learns
`T(bindings) = W vec(sum(filler_embedding ⊗ role_embedding)) + b` to approximate
that same `z`. It never fits unpacker hidden states. Frozen unpackers evaluate
real states, TPR approximations, and role-edited states using free generation.

### TPR splits and novel bindings

TPR preparation preserves the CSV's `split` and adds `tpr_split`. Original
`gen_test` is excluded because Mary, teacher, and rock are unseen fillers.
By default, every **John→agent** and **doctor→patient** example, in either voice,
is withheld from fitting and assigned to `binding_test`. Both nouns remain in
training in their opposite roles. All vocabulary and normalization fitting uses
only `tpr_split=train`; validation selects checkpoints, and binding-test examples
never enter either process.

| TPR split | Sentences |
| --- | ---: |
| train | 2,868 |
| val | 366 |
| test | 366 |
| binding_test | 528 |

Counterfactual donors exchange the nouns while preserving verb and voice. They
can cross TPR partitions and are used only as controls and expected outputs;
the TPR swap itself needs no donor activation. Evaluation uses the original test
rows plus every binding-test row (894 sources). With the defaults, **50** held-out
source sentences test steering *into* a novel binding. The other 528 binding-test
sources test reconstruction of novel bindings and steering away from them.
Results group source and target TPR splits separately; use
`condition=swap,target_binding_held_out=True` for novel-binding steering.

### Run the stages

Use the locked environment and cache variables described above. A full run uses
DISCOVER-sized unpackers (6 layers, width 1024, 16 heads, FFN 4096) and 100×100
TPR embeddings. This trains ten unpackers and ten TPRs sequentially; the frozen
LLM is released during auxiliary training.

```bash
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/temporary/run_tpr_pipeline.py \
  --output-dir /orfeo/scratch/dssc/zenocosini/ac-tpr-cache/tpr-qwen3-base \
  --layers 5 12 15 24 31
```

The default corpus is `data/active_pass_long/corpus.csv`. Defaults are Adam,
batch size 32, learning rates `1e-4` for unpackers and `1e-3` for TPRs, seed 42,
and a 100-epoch cap. Validation runs every 100 updates and at the final update.
After five evaluations without improvement, training restores the best weights,
halves the learning rate, and resets Adam. It permits five reductions and then
stops on the next plateau. Dimensions, limits, seed, and bindings are configurable;
see `--help`. For example, `--held-out-bindings John:patient doctor:agent` changes
the withheld pairs; `--held-out-bindings` alone disables binding holdouts.

Use `--stage extract`, `unpack`, `fit`, `reconstruct`, or `steer` with the same
output root to run stages individually. `fit` depends only on extraction, not
on the unpacker. Later stages read the extraction's layers, splits, and vocabulary;
extraction flags do not modify an existing cache. Existing output directories for
the selected stage are rejected, including incomplete stages. Use a fresh output
directory or explicitly remove an incomplete stage before retrying. This is not
automatic checkpoint resumption.

Outputs are `encodings/`, `unpackers/`, `tprs/`, `reconstruction/`, and `steering/`.
Checkpoints include the fitted parameters and training histories; caches retain
vocabularies, normalization statistics, and original row indices. Completion
metadata verifies source extraction identity when loading auxiliary models.
Steering also verifies current prompt token IDs, tokenizer, checkpoint, and dtype.

### Interpret evaluation

The role edit is `h + alpha * scale * (T(swapped) - T(original))`, with `alpha=1`
by default. It keeps the activation residual that the TPR did not explain.
Each layer/site is edited independently. Controls are zero delta, original TPR
replacement, and real swapped-sentence donor replacement. All conditions score
the original LLM's final-token candidate logits, using the same patient-minus-agent
convention as centroids. Both all-example and baseline-agent-conditional flip rates
are saved. These are pairwise preferences, not unrestricted generation accuracy.

`reconstruction/results.csv` records generated tokens, exact-match results, and
normalized activation MSE; its summary separates real, TPR, and swapped states.
`steering/baseline.csv`, `results.csv`, and `summary.csv` preserve corpus annotations
and identify the site, condition, original corpus indices, and binding direction.
Final-block sentence-end edits must be null; final-block prompt-end donor edits
must reproduce donor logits. These controls are exercised in tests and the smoke
driver, which adds block 35 to the requested experimental layers.

```bash
OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python scripts/temporary/validate_tpr_pipeline.py \
  --output-dir /tmp/tpr-qwen3-smoke
```

The smoke run uses the real frozen Qwen checkpoint with tiny auxiliary models and
one training epoch. It checks execution, artifact loading, controls, and an
independent Hugging Face baseline. It is not a research result. Successful sentence
reconstruction alone does not guarantee steering: other untouched LLM token states
can still influence the answer. Read donor-control performance alongside TPR effects.

## Tests

```bash
uv run --locked pytest
```

Tests cover sentence roles and irregular verbs, split isolation and balance,
reproducibility, input errors, and tokenization using the real Qwen tokenizer.
Activation tests compare independent small Qwen3 models with identical
weights, one wrapped by NNsight, to check role positions, padding, block outputs
before final normalization, recovery after failed traces, and saved
artifacts including a short final batch. They do not download model weights.
The tokenizer must be cached or downloadable. With it cached, tests also run with
`HF_HUB_OFFLINE=1 uv run --locked pytest`.

Syntactic-vector tests use synthetic activations to check arithmetic means with
unequal voice counts, exclusion of held-out rows, float32 precision, ordering,
provenance, and rejection of incomplete or incompatible source artifacts.

Intervention tests compare a small random Qwen3 causal model with a matching
NNsight `LanguageModel` to check candidate
tokenization, padded scoring against the full language-model output, voice-specific
means, edit direction and position, trace isolation, zero-vector and final-block
controls, output arithmetic, summaries, and incompatible-input failures.
Loader tests cover checkpoint pinning, raw-weight and package-version provenance,
unsupported dtype rejection, and large float32 query/key values. Tests also cover custom
prompts, extraction stopping before vocabulary projection, and scoring projecting
only one position per sentence.

## Layout

- `src/ac_tpr/`: dataset, activations, role means, and intervention/evaluation code.
- `src/ac_tpr/tpr/`: sentence unpacking, symbolic approximation, and TPR steering.
- `scripts/`: command-line entry points.
- `data/`: active_pass_long, starter fillers, reference data, and generated corpus.
- `docs/`: [paper review and experiment rationale](docs/acevedo-2026-notes.md).
- `tests/`: dataset, activation, role-mean, and intervention checks.
- `notebook/`: executed intervention analysis and exported figures.

One-off experiment scripts belong in `scripts/temporary/` or
`scripts/slurm/temporary/`. Store model weights and activations on scratch storage.
