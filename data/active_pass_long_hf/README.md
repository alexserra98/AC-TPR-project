---
language:
- en
configs:
- config_name: default
  data_files:
  - split: train
    path: corpus.csv
---
# Active/passive role-swap corpus

Derived from [iuliialysova/active_pass_long](https://huggingface.co/datasets/iuliialysova/active_pass_long/tree/949033163b2a9d4d2e1d8c7811e0eaa7f27e749c),
revision `949033163b2a9d4d2e1d8c7811e0eaa7f27e749c`, file `role_swap_long.csv`. All 5,808 original sentences
and their row order are preserved. There are 1,452 quads and 2,904 pairs.

## Columns

The first seven columns are `pair_id, sentence, voice, agent, patient, verb, split`.
`pair_id` groups the same semantic roles across active/passive voice;
`quad_id` groups both voices and both role orders. `pair_id = 2 * quad_id + swapped`.
`verb` is the lemma. Agent and patient identify semantic roles regardless of voice.

All other original annotations are retained:
`quad_id, condition, subset, verb_class, np_type, tense, swapped, agent_animacy, patient_animacy, animacy_config, animacy_hierarchy, np1, np2, np1_role, np2_role, np1_animacy, np2_animacy, agent_np_form, patient_np_form, np1_form, np2_form, np1_char_span, np2_char_span, verb_char_span, logical_form`.
`condition` identifies voice and swapping; `subset` identifies names_agentive,
nouns_contact, or mixed_contact. Animacy and NP-form fields describe semantic
roles or sentence-order participants (`np1`, `np2`). Character spans use
zero-based, end-exclusive `start:end` offsets in the unchanged sentence.
The NP spans identify the filler word, excluding its determiner.
`logical_form` encodes the verb and its semantic role assignments.

The six source question/answer columns are omitted. Questions and demonstrations
are configured in the consuming experiment, not in this dataset.

## Experimental splits

| split | sentences |
| --- | ---: |
| train | 3300 |
| val | 412 |
| test | 416 |
| gen_test | 1680 |

Every quad containing Mary, teacher, or rock is assigned to gen_test.
Remaining quad IDs are numerically sorted, shuffled with Python random.Random(42),
and divided at floor(0.8 * groups) and floor(0.9 * groups).
All four sentences of a quad remain together.

Hugging Face exposes the single CSV as one physical `train` split. Use the CSV's
`split` column for the experimental partitions; the physical split is not a
training-only selection. For example, filter rows by `row["split"] == "train"`
before fitting or estimating role vectors.

## Validation and reproduction

Agent/patient fillers occupy one token in their sentence and experiment prompt;
space-prefixed answer candidates also occupy one token. Verified for all rows
with `Qwen/Qwen3-8B-Base` at `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. Verbs may span multiple tokens,
including tickled. Other tokenizers require their own validation.

From the AC-TPR-project checkout, with its environment installed, run:

```bash
.venv/bin/python scripts/temporary/prepare_active_pass_long.py --output-dir data/active_pass_long
```

The output directory must not exist. Source and corpus hashes, splitting rules,
and tokenizer verification are recorded in split_metadata.json. The verification
prompt is identified only by its hash; experiment metadata records actual prompts.

## Upload and experiment use

Upload corpus.csv, split_metadata.json, and this README.md together to the target
Hugging Face dataset repository. This config explicitly selects corpus.csv, so
legacy source CSVs do not become additional examples.

Generate fresh activations and role vectors for this corpus and the configured
prompt. Artifacts produced from another corpus or prompt are incompatible.
