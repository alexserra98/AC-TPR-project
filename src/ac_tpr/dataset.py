"""Generate paired active/passive sentences from explicit noun and verb forms."""

import csv
import hashlib
import json
import random
import re
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

TOKENIZER_ID = "EleutherAI/pythia-6.9b"
TOKENIZER_REVISION = "step143000"
FILLER_COLUMNS = ["label", "lemma", "past", "participle"]
CORPUS_COLUMNS = ["pair_id", "sentence", "voice", "agent", "patient", "verb", "split"]
TEMPLATES = {
    "active": "The {agent} {past} the {patient}.",
    "passive": "The {patient} was {participle} by the {agent}.",
}
# Word indices of the two nouns and the inflected verb in each template.
FILLER_WORD_INDICES = {"active": (1, 2, 4), "passive": (1, 3, 6)}


def load_fillers(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read lowercase, single-word fillers with explicit verb inflections.

    Nouns must be singular; verbs must be transitive. These linguistic properties
    are supplied by the vocabulary author, while spelling and fields are checked.
    """
    nouns = []
    verbs = []
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FILLER_COLUMNS:
            raise ValueError(f"Filler CSV columns must be {FILLER_COLUMNS}")
        # loading fillers line by line
        for line_number, row in enumerate(reader, start=2):
            if None in row or None in row.values():
                raise ValueError(f"Line {line_number}: expected four CSV fields")
            label, lemma = row["label"], row["lemma"]
            # several validation checks 
            if label not in {"noun", "verb"}:
                raise ValueError(f"Line {line_number}: unknown label {label!r}")
            fields = ("lemma",) if label == "noun" else ("lemma", "past", "participle")
            for field in fields:
                if re.fullmatch(r"[a-z]+", row[field]) is None:
                    raise ValueError(
                        f"Line {line_number}: {field} must be one lowercase word"
                    )
            if (label, lemma) in seen:
                # duplicate filler check
                raise ValueError(f"Line {line_number}: duplicate {label} {lemma!r}")
            seen.add((label, lemma))
            if label == "noun":
                # check that nouns do not have verb forms
                if row["past"] or row["participle"]:
                    raise ValueError(f"Line {line_number}: nouns cannot have verb forms")
                nouns.append(lemma)
            else:
                verbs.append({field: row[field] for field in fields})
    if len(nouns) < 2 or not verbs:
        raise ValueError("Provide at least two nouns and one verb")
    return sorted(nouns), sorted(verbs, key=lambda verb: verb["lemma"])


def generate_rows(
    nouns: list[str],
    verbs: list[dict[str, str]],
    held_out_nouns: list[str],
    seed: int = 42,
) -> list[dict[str, str]]:
    """Enumerate both voices and role orders, splitting noun-pair/verb groups.

    Inputs follow the contract checked by load_fillers. Pair IDs depend only on
    sorted vocabulary; changing the seed changes splits but never sentence IDs.
    """
    nouns = sorted(nouns)
    verbs = sorted(verbs, key=lambda verb: verb["lemma"])
    held_out = set(held_out_nouns)
    unknown = held_out - set(nouns)
    if unknown:
        raise ValueError(f"Held-out nouns are absent from the vocabulary: {sorted(unknown)}")
    seen_nouns = [noun for noun in nouns if noun not in held_out]
    if len(seen_nouns) < 2:
        raise ValueError("At least two nouns must remain outside gen_test")

    # generate all possible combinations of noun pairs and verbs
    groups = [
        (first, second, verb["lemma"])
        for first, second in combinations(seen_nouns, 2)
        for verb in verbs
    ]

    # split into train/val/test
    random.Random(seed).shuffle(groups)
    train_end = int(0.8 * len(groups))
    val_end = int(0.9 * len(groups))
    splits = {} # dictionary to hold the split assignment for each group
    for index, group in enumerate(groups):
        if index < train_end:
            splits[group] = "train"
        elif index < val_end:
            splits[group] = "val"
        else:
            splits[group] = "test"

    rows = []
    pair_id = 0
    for agent, patient in permutations(nouns, 2):
        for verb in verbs:
            if agent in held_out or patient in held_out:
                split = "gen_test"
            else:
                first, second = sorted((agent, patient))
                split = splits[(first, second, verb["lemma"])]
            for voice, template in TEMPLATES.items():
                rows.append({
                    "pair_id": str(pair_id),
                    "sentence": template.format(agent=agent, patient=patient, **verb),
                    "voice": voice,
                    "agent": agent,
                    "patient": patient,
                    "verb": verb["lemma"],
                    "split": split,
                })
            pair_id += 1
    if len({row["sentence"] for row in rows}) != len(rows):
        raise ValueError("Verb forms produce duplicate sentences")
    return rows


def validate_tokenization(
    rows: list[dict[str, str]], tokenizer: PreTrainedTokenizerFast
) -> None:
    """Require one token for each noun and inflected verb in complete sentences.

    Offset overlap handles tokenizers that include the preceding space in a
    token's character span. A token may contain that space but no adjacent word
    or punctuation. No BOS or EOS tokens are added.
    """
    if not tokenizer.is_fast:
        raise ValueError("Character-offset validation requires a fast tokenizer")
    for row in rows:
        sentence = row["sentence"]
        encoding = tokenizer(
            sentence, add_special_tokens=False, return_offsets_mapping=True
        )
        words = list(re.finditer(r"[A-Za-z]+", sentence))
        for index in FILLER_WORD_INDICES[row["voice"]]:
            word = words[index]
            offsets = [
                (start, end)
                for start, end in encoding["offset_mapping"]
                if start < word.end() and end > word.start()
            ]
            if len(offsets) != 1:
                raise ValueError(
                    f"Filler {word.group()!r} spans {len(offsets)} tokens in {sentence!r}"
                )
            start, end = offsets[0]
            if sentence[start:end].strip() != word.group():
                raise ValueError(
                    f"Filler {word.group()!r} does not occupy one whole token in {sentence!r}"
                )


def generate_dataset(
    fillers_path: Path,
    output_dir: Path,
    held_out_nouns: list[str],
    seed: int = 42,
) -> dict:
    """Validate the vocabulary and corpus, then write CSV and split metadata.

    Existing output files are replaced only after input validation completes.
    Hugging Face uses HF_HOME for its tokenizer cache; no model is loaded.
    """
    nouns, verbs = load_fillers(fillers_path)
    rows = generate_rows(nouns, verbs, held_out_nouns, seed)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True
    )
    validate_tokenization(rows, tokenizer)
    # how many sentences are in each split
    sentences_by_split = Counter(row["split"] for row in rows) 
    # it should be sentences_by_split/2 but codex conviced me it's better to count it explicitly
    pairs_by_split = Counter(row["split"] for row in rows if row["voice"] == "active") 
    split_names = ("train", "val", "test", "gen_test")
    metadata = {
        "source": str(fillers_path),
        "source_sha256": hashlib.sha256(fillers_path.read_bytes()).hexdigest(),
        "seed": seed,
        "vocabulary": {"nouns": nouns, "verbs": verbs},
        "templates": TEMPLATES,
        "held_out_nouns": sorted(set(held_out_nouns)),
        "tokenizer": {
            "name": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "backend_sha256": hashlib.sha256(
                tokenizer.backend_tokenizer.to_str().encode("utf-8")
            ).hexdigest(),
            "add_special_tokens": False,
        },
        "split_rules": {
            "gen_test": "Either noun belongs to held_out_nouns",
            "random_split_unit": "Unordered noun pair and verb lemma",
            "ratios_on_remaining_groups": {"train": 0.8, "val": 0.1, "test": 0.1},
            "boundaries": "floor(0.8 * groups), floor(0.9 * groups)",
        },
        "counts": {
            "included_pairs": len(rows) // 2,
            "sentences": len(rows),
            "pairs_by_split": {name: pairs_by_split[name] for name in split_names},
            "sentences_by_split": {name: sentences_by_split[name] for name in split_names},
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "corpus.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
