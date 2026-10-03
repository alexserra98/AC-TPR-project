"""Check corpus semantics, split isolation, and the actual Pythia tokenization."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from ac_tpr.dataset import (
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    generate_dataset,
    generate_rows,
    load_fillers,
    validate_tokenization,
)

FILLERS = Path(__file__).resolve().parents[1] / "data" / "fillers.csv"
HELD_OUT = ["child", "teacher", "soldier", "artist"]


@pytest.fixture(scope="module")
def starter_rows():
    nouns, verbs = load_fillers(FILLERS)
    return generate_rows(nouns, verbs, HELD_OUT)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True
    )


def test_irregular_verb_preserves_roles_across_voices():
    verbs = [{"lemma": "see", "past": "saw", "participle": "seen"}]
    rows = generate_rows(["girl", "boy"], verbs, [])
    assert [row["sentence"] for row in rows] == [
        "The boy saw the girl.",
        "The girl was seen by the boy.",
        "The girl saw the boy.",
        "The boy was seen by the girl.",
    ]
    assert [row["agent"] for row in rows] == ["boy", "boy", "girl", "girl"]
    assert [row["patient"] for row in rows] == ["girl", "girl", "boy", "boy"]
    assert [row["pair_id"] for row in rows] == ["0", "0", "1", "1"]
    assert {row["verb"] for row in rows} == {"see"}


def test_starter_counts_pairs_and_sentences(starter_rows):
    assert len(starter_rows) == 3840
    assert len({row["sentence"] for row in starter_rows}) == 3840
    pairs = defaultdict(list)
    for row in starter_rows:
        assert row["agent"] != row["patient"]
        pairs[row["pair_id"]].append(row)
    assert len(pairs) == 1920
    assert set(pairs) == {str(index) for index in range(1920)}
    for pair in pairs.values():
        assert len(pair) == 2
        assert {row["voice"] for row in pair} == {"active", "passive"}
        for field in ("agent", "patient", "verb", "split"):
            assert len({row[field] for row in pair}) == 1
    assert Counter(row["split"] for row in starter_rows) == {
        "train": 1688, "val": 212, "test": 212, "gen_test": 1728
    }


def test_held_out_isolation_and_role_balance(starter_rows):
    groups = defaultdict(list)
    for row in starter_rows:
        contains_held_out = bool({row["agent"], row["patient"]} & set(HELD_OUT))
        assert (row["split"] == "gen_test") == contains_held_out
        noun_pair = tuple(sorted((row["agent"], row["patient"])))
        groups[(*noun_pair, row["verb"])].append(row)
    for group in groups.values():
        assert len(group) == 4
        assert len({row["split"] for row in group}) == 1
        assert len({(row["agent"], row["patient"], row["voice"]) for row in group}) == 4
    for split in ("train", "val", "test", "gen_test"):
        rows = [row for row in starter_rows if row["split"] == split]
        assert Counter(row["agent"] for row in rows) == Counter(row["patient"] for row in rows)
        assert Counter(row["voice"] for row in rows) == {
            "active": len(rows) // 2, "passive": len(rows) // 2
        }


def test_seed_changes_only_splits_and_input_order_is_irrelevant(starter_rows):
    nouns, verbs = load_fillers(FILLERS)
    assert generate_rows(nouns[::-1], verbs[::-1], HELD_OUT[::-1]) == starter_rows
    different_seed = generate_rows(nouns, verbs, HELD_OUT, seed=43)
    assert any(left["split"] != right["split"] for left, right in zip(starter_rows, different_seed))
    for left, right in zip(starter_rows, different_seed):
        assert {key: value for key, value in left.items() if key != "split"} == {
            key: value for key, value in right.items() if key != "split"
        }
    three_splits = generate_rows(nouns, verbs, [])
    assert {row["split"] for row in three_splits} == {"train", "val", "test"}


@pytest.mark.parametrize(
    ("extra_row", "error"),
    [
        ("adjective,kind,,\n", "unknown label"),
        ("noun,boy,,\n", "duplicate noun"),
        ("verb,help,helped,helped\n", "duplicate verb"),
        ("verb,see,,seen\n", "past must be"),
        ("verb,see,saw,\n", "participle must be"),
        ("noun,young boy,,\n", "lemma must be"),
        ("noun,Boy,,\n", "lemma must be"),
        ("noun,teacher,taught,taught\n", "nouns cannot"),
        ("noun,teacher,\n", "four CSV fields"),
        ("noun,teacher,,,extra\n", "four CSV fields"),
    ],
)
def test_invalid_filler_entries(tmp_path, extra_row, error):
    path = tmp_path / "fillers.csv"
    path.write_text(
        "label,lemma,past,participle\nnoun,boy,,\nnoun,girl,,\n"
        "verb,help,helped,helped\n" + extra_row
    )
    with pytest.raises(ValueError, match=error):
        load_fillers(path)


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("lemma,label,past,participle\n", "columns must be"),
        ("label,lemma,past,participle\n", "at least two nouns"),
        ("label,lemma,past,participle\nnoun,boy,,\nnoun,girl,,\n", "one verb"),
    ],
)
def test_invalid_vocabulary_files(tmp_path, content, error):
    path = tmp_path / "fillers.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=error):
        load_fillers(path)


@pytest.mark.parametrize("held_out", [["ghost"], ["boy", "girl"]])
def test_invalid_held_out_nouns(held_out):
    with pytest.raises(ValueError, match="Held-out nouns|At least two nouns"):
        generate_rows(
            ["boy", "girl"],
            [{"lemma": "help", "past": "helped", "participle": "helped"}],
            held_out,
        )


def test_duplicate_surface_sentences_are_rejected():
    verbs = [
        {"lemma": "lie", "past": "lay", "participle": "lain"},
        {"lemma": "lay", "past": "lay", "participle": "laid"},
    ]
    with pytest.raises(ValueError, match="duplicate sentences"):
        generate_rows(["boy", "girl"], verbs, [])


def test_complete_corpus_and_irregular_forms_use_single_tokens(starter_rows, tokenizer):
    validate_tokenization(starter_rows, tokenizer)
    irregular = generate_rows(
        ["boy", "girl"],
        [{"lemma": "see", "past": "saw", "participle": "seen"}],
        [],
    )
    validate_tokenization(irregular, tokenizer)


@pytest.mark.parametrize("field", ["noun", "past", "participle"])
def test_multi_token_fillers_are_rejected_in_context(tokenizer, field):
    nouns = ["boy", "girl"]
    verb = {"lemma": "help", "past": "helped", "participle": "helped"}
    if field == "noun":
        nouns[1] = "neuroscientist"
    else:
        verb[field] = "mischaracterized"
    rows = generate_rows(nouns, [verb], [])
    with pytest.raises(ValueError, match=r"Filler .+ spans \d+ tokens"):
        validate_tokenization(rows, tokenizer)


def test_output_metadata_and_bytes_are_reproducible(tmp_path, tokenizer, monkeypatch):
    monkeypatch.setattr("ac_tpr.dataset.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    output = tmp_path / "generated"
    metadata = generate_dataset(FILLERS, output, HELD_OUT)
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    assert generate_dataset(FILLERS, output, HELD_OUT) == metadata
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first
    assert json.loads((output / "split_metadata.json").read_text()) == metadata
    with (output / "corpus.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == [
            "pair_id", "sentence", "voice", "agent", "patient", "verb", "split"
        ]
        rows = list(reader)
    assert metadata["counts"]["sentences"] == len(rows) == 3840
    assert metadata["counts"]["pairs_by_split"] == dict(
        Counter(row["split"] for row in rows if row["voice"] == "active")
    )
    assert metadata["held_out_nouns"] == sorted(HELD_OUT)
    assert metadata["tokenizer"]["revision"] == "step143000"
    assert metadata["tokenizer"]["add_special_tokens"] is False


def test_failed_validation_preserves_outputs(tmp_path, tokenizer, monkeypatch):
    monkeypatch.setattr("ac_tpr.dataset.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    fillers = tmp_path / "fillers.csv"
    fillers.write_text(
        "label,lemma,past,participle\nnoun,boy,,\nnoun,neuroscientist,,\n"
        "verb,help,helped,helped\n"
    )
    output = tmp_path / "generated"
    with pytest.raises(ValueError, match="tokens"):
        generate_dataset(fillers, output, [])
    assert not output.exists()
    output.mkdir()
    for name in ("corpus.csv", "split_metadata.json"):
        (output / name).write_text("existing data\n")
    with pytest.raises(ValueError, match="tokens"):
        generate_dataset(fillers, output, [])
    assert all(path.read_text() == "existing data\n" for path in output.iterdir())
