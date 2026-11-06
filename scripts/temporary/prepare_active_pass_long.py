"""Convert the pinned active_pass_long source into the project's enriched corpus."""

import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

from transformers import AutoTokenizer

from ac_tpr.dataset import ENRICHED_CORPUS_COLUMNS, TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.prompts import PROMPT_TEMPLATE, prepare_prompts, tokenize_roles

SOURCE_ID = "iuliialysova/active_pass_long"
SOURCE_REVISION = "949033163b2a9d4d2e1d8c7811e0eaa7f27e749c"
SOURCE_FILE = "role_swap_long.csv"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_ID}/resolve/{SOURCE_REVISION}/{SOURCE_FILE}"
QUESTION_COLUMNS = [
    "q_agent", "q_agent_answer", "q_patient", "q_patient_answer",
    "q_agent_verb", "q_patient_verb",
]
HELD_OUT = ["Mary", "teacher", "rock"]
SEED = 42
EXPECTED_COUNTS = {"train": 3300, "val": 412, "test": 416, "gen_test": 1680}


def convert_rows(source_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Validate source quads and split whole groups, preserving source row order."""
    expected = set(ENRICHED_CORPUS_COLUMNS) - {"pair_id", "split"} | set(QUESTION_COLUMNS)
    if not source_rows or any(
        set(row) != expected or any(value is None or value == "" for value in row.values())
        for row in source_rows
    ):
        raise ValueError("Expected complete source rows with all 36 original columns")
    if len({row["sentence"] for row in source_rows}) != len(source_rows):
        raise ValueError("Source sentences must be unique")
    groups = defaultdict(list)
    for row in source_rows:
        groups[row["quad_id"]].append(row)
    for quad_id, group in groups.items():
        if str(int(quad_id)) != quad_id or int(quad_id) < 0:
            raise ValueError("quad_id must be a nonnegative canonical integer")
        conditions = {(row["voice"], row["swapped"]) for row in group}
        if len(group) != 4 or conditions != {
            ("active", "0"), ("passive", "0"), ("active", "1"), ("passive", "1")}:
            raise ValueError(f"Incomplete quad {quad_id}")
        original = next(row for row in group if row["swapped"] == "0")
        for row in group:
            roles = (original["agent"], original["patient"])
            if row["swapped"] == "1":
                roles = roles[::-1]
            if roles[0] == roles[1] or (row["agent"], row["patient"]) != roles:
                raise ValueError(f"Inconsistent semantic roles in quad {quad_id}")
            if any(row[key] != original[key] for key in ("verb", "subset", "verb_class", "tense")):
                raise ValueError(f"Inconsistent group annotations in quad {quad_id}")
            condition = row["voice"] + ("_swap" if row["swapped"] == "1" else "")
            if row["condition"] != condition:
                raise ValueError(f"Inconsistent condition in quad {quad_id}")
    held_out = set(HELD_OUT)
    vocabulary = {row[role] for row in source_rows for role in ("agent", "patient")}
    if not held_out <= vocabulary:
        raise ValueError("Held-out fillers must be present in the source")
    splits, remaining = {}, []
    for quad_id in sorted(groups, key=int):
        row = groups[quad_id][0]
        if {row["agent"], row["patient"]} & held_out:
            splits[quad_id] = "gen_test"
        else:
            remaining.append(quad_id)
    random.Random(SEED).shuffle(remaining)
    for index, quad_id in enumerate(remaining):
        splits[quad_id] = (
            "train" if index < int(0.8 * len(remaining)) else
            "val" if index < int(0.9 * len(remaining)) else "test"
        )
    rows = []
    for source in source_rows:
        row = {**source, "pair_id": str(2 * int(source["quad_id"]) + int(source["swapped"])),
               "split": splits[source["quad_id"]]}
        rows.append({key: row[key] for key in ENRICHED_CORPUS_COLUMNS})
    return rows


def prepare_dataset(output_dir: Path) -> dict:
    """Download and validate the pinned source before creating the output directory."""
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    with urlopen(SOURCE_URL, timeout=60) as response:
        source_bytes = response.read()
    source_rows = list(csv.DictReader(io.StringIO(source_bytes.decode("utf-8"))))
    rows = convert_rows(source_rows)
    counts = dict(Counter(row["split"] for row in rows))
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected split counts: {counts}")
    if len({row["quad_id"] for row in rows}) != 1452 or len({row["pair_id"] for row in rows}) != 2904:
        raise ValueError("Expected 1,452 quads and 2,904 pairs")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
    tokenize_roles(rows, tokenizer)
    prepare_prompts(rows, tokenizer)
    metadata = {
        "source": {"dataset": SOURCE_ID, "revision": SOURCE_REVISION, "file": SOURCE_FILE,
                   "url": SOURCE_URL, "sha256": hashlib.sha256(source_bytes).hexdigest()},
        "seed": SEED,
        "held_out_fillers": HELD_OUT,
        "conversion": {"pair_id": "2 * int(quad_id) + int(swapped)",
                       "removed_columns": QUESTION_COLUMNS, "source_row_order_preserved": True},
        "split_rules": {"gen_test": "Either role filler belongs to held_out_fillers",
                        "random_split_unit": "quad_id, sorted numerically before seeded shuffle",
                        "ratios_on_remaining_groups": {"train": 0.8, "val": 0.1, "test": 0.1},
                        "boundaries": "floor(0.8 * groups), floor(0.9 * groups)"},
        "counts": {"sentences": len(rows), "quads": 1452, "pairs": 2904,
                   "sentences_by_split": counts,
                   "sentences_by_subset": dict(Counter(row["subset"] for row in rows))},
        "tokenizer": {"name": TOKENIZER_ID, "revision": TOKENIZER_REVISION,
                      "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                      "add_special_tokens": False},
        "validation": {"sentence_role_tokens": "passed", "prompt_roles_and_answer_tokens": "passed",
                       "rows_checked": len(rows), "verb_token_length_restricted": False,
                       "prompt_template_sha256": hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()},
    }
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ENRICHED_CORPUS_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    corpus_bytes = buffer.getvalue().encode("utf-8")
    metadata["corpus_sha256"] = hashlib.sha256(corpus_bytes).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "corpus.csv").write_bytes(corpus_bytes)
    (output_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output_dir / "README.md").write_text(dataset_readme(), encoding="utf-8")
    return metadata


def dataset_readme() -> str:
    """Describe the enriched corpus and its explicit experimental split column."""
    return f"""---
language:
- en
configs:
- config_name: default
  data_files:
  - split: train
    path: corpus.csv
---
# Active/passive role-swap corpus

Derived from [{SOURCE_ID}](https://huggingface.co/datasets/{SOURCE_ID}/tree/{SOURCE_REVISION}),
revision `{SOURCE_REVISION}`, file `{SOURCE_FILE}`. All 5,808 original sentences
and their row order are preserved. There are 1,452 quads and 2,904 pairs.

## Columns

The first seven columns are `pair_id, sentence, voice, agent, patient, verb, split`.
`pair_id` groups the same semantic roles across active/passive voice;
`quad_id` groups both voices and both role orders. `pair_id = 2 * quad_id + swapped`.
`verb` is the lemma. Agent and patient identify semantic roles regardless of voice.

All other original annotations are retained:
`{', '.join(ENRICHED_CORPUS_COLUMNS[7:])}`.
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
with `{TOKENIZER_ID}` at `{TOKENIZER_REVISION}`. Verbs may span multiple tokens,
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
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/active_pass_long"))
    args = parser.parse_args()
    metadata = prepare_dataset(args.output_dir)
    print(json.dumps(metadata["counts"], indent=2))


if __name__ == "__main__":
    main()
