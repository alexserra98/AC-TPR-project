"""Check the converted corpus and annotation preservation through the pipeline."""

import csv
import hashlib
import json
import runpy
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import torch

from ac_tpr.activations import extract_activations
from ac_tpr.dataset import CORPUS_COLUMNS, ENRICHED_CORPUS_COLUMNS, load_corpus
from ac_tpr.interventions import run_interventions
from ac_tpr.prompts import PROMPT_TEMPLATE, prepare_prompts, tokenize_roles
from ac_tpr.syntactic_vectors import extract_syntactic_vectors

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/active_pass_long"
CONVERTER = runpy.run_path(str(ROOT / "scripts/temporary/prepare_active_pass_long.py"))


@pytest.fixture(scope="module")
def corpus():
    return load_corpus(DATA / "corpus.csv")


def write_corpus(path, rows, columns):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def source_rows(corpus):
    return [
        {**{k: v for k, v in row.items() if k not in {"pair_id", "split"}},
         **dict.fromkeys(CONVERTER["QUESTION_COLUMNS"], "unused question annotation")}
        for row in corpus
    ]


def test_complete_corpus_semantics_and_split_isolation(corpus):
    assert len(corpus) == len({row["sentence"] for row in corpus}) == 5808
    assert list(corpus[0]) == ENRICHED_CORPUS_COLUMNS
    assert len(ENRICHED_CORPUS_COLUMNS) == 32
    assert not any(key.startswith("q_") for key in corpus[0])
    assert Counter(row["split"] for row in corpus) == {
        "train": 3300, "val": 412, "test": 416, "gen_test": 1680,
    }
    pairs, quads = defaultdict(list), defaultdict(list)
    for row in corpus:
        assert (row["split"] == "gen_test") == bool(
            {row["agent"], row["patient"]} & {"Mary", "teacher", "rock"}
        )
        pairs[row["pair_id"]].append(row)
        quads[row["quad_id"]].append(row)
    assert len(pairs) == 2904
    assert len(quads) == 1452
    for pair in pairs.values():
        assert len(pair) == 2
        assert {r["voice"] for r in pair} == {"active", "passive"}
        for field in ("agent", "patient", "verb", "split"):
            assert len({r[field] for r in pair}) == 1
    for quad in quads.values():
        assert len(quad) == 4
        assert len({r["split"] for r in quad}) == 1
        assert {r["condition"] for r in quad} == {"active", "passive", "active_swap", "passive_swap"}
    metadata = json.loads((DATA / "split_metadata.json").read_text())
    assert metadata["corpus_sha256"] == hashlib.sha256((DATA / "corpus.csv").read_bytes()).hexdigest()


def test_conversion_preserves_values_and_is_deterministic(corpus):
    original = source_rows(corpus)
    assert CONVERTER["convert_rows"](original) == corpus
    assert CONVERTER["convert_rows"](original[::-1]) == corpus[::-1]


@pytest.mark.parametrize("problem", ["incomplete", "duplicate_sentence", "wrong_role", "missing_annotation"])
def test_conversion_rejects_invalid_source(corpus, problem):
    original = source_rows(corpus)
    if problem == "incomplete":
        original.pop()
    elif problem == "duplicate_sentence":
        original[1]["sentence"] = original[0]["sentence"]
    elif problem == "wrong_role":
        original[0]["agent"] = original[0]["patient"]
    else:
        del original[0]["subset"]
    with pytest.raises(ValueError):
        CONVERTER["convert_rows"](original)


def test_conversion_refuses_existing_directory(tmp_path):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        CONVERTER["prepare_dataset"](tmp_path)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("columns", [CORPUS_COLUMNS, ENRICHED_CORPUS_COLUMNS])
def test_loader_accepts_both_schemas(tmp_path, corpus, columns):
    rows = [{key: row[key] for key in columns} for row in corpus[:4]]
    path = tmp_path / "corpus.csv"
    write_corpus(path, rows, columns)
    assert load_corpus(path) == rows


@pytest.mark.parametrize("problem", ["header", "duplicate", "empty", "missing_value", "extra_value"])
def test_loader_rejects_invalid_csv(tmp_path, corpus, problem):
    rows = [dict(corpus[0])]
    columns = ENRICHED_CORPUS_COLUMNS
    if problem == "header":
        columns = columns[::-1]
    elif problem == "duplicate":
        rows *= 2
    elif problem == "empty":
        rows = []
    elif problem == "missing_value":
        rows[0]["subset"] = ""
    path = tmp_path / "corpus.csv"
    write_corpus(path, rows, columns)
    if problem == "extra_value":
        path.write_text(path.read_text().rstrip("\n") + ",extra\n")
    with pytest.raises(ValueError):
        load_corpus(path)


def test_all_roles_and_answers_are_single_tokens(corpus, tokenizer):
    assert len(tokenizer(" tickled", add_special_tokens=False)["input_ids"]) == 2
    ids, positions = tokenize_roles(corpus, tokenizer)
    prompt_ids, prompt_positions, answers = prepare_prompts(corpus, tokenizer)
    assert positions.shape == prompt_positions.shape == answers.shape == (5808, 2)
    for row, tokens, pair, full_tokens, full_pair, candidates in zip(
        corpus, ids, positions.tolist(), prompt_ids, prompt_positions.tolist(), answers.tolist(), strict=True
    ):
        for role, position, full_position, candidate in zip(
            ("agent", "patient"), pair, full_pair, candidates, strict=True
        ):
            assert tokenizer.decode([tokens[position]]).strip() == row[role]
            assert tokenizer.decode([full_tokens[full_position]]).strip() == row[role]
            assert tokenizer.decode([candidate]) == " " + row[role]


def test_enriched_small_model_pipeline(tmp_path, corpus, tokenizer, model, model_metadata, monkeypatch):
    chosen = set()
    for split, subset, verb in [
        ("train", "names_agentive", "tickle"),
        ("test", "mixed_contact", "touch"),
        ("gen_test", "nouns_contact", "hit"),
    ]:
        chosen.add(next(r["quad_id"] for r in corpus
                        if r["split"] == split and r["subset"] == subset and r["verb"] == verb))
    rows = [row for row in corpus if row["quad_id"] in chosen]
    path = tmp_path / "corpus.csv"
    write_corpus(path, rows, ENRICHED_CORPUS_COLUMNS)
    monkeypatch.setattr("ac_tpr.activations.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    for module in ("ac_tpr.activations", "ac_tpr.interventions"):
        monkeypatch.setattr(f"{module}.load_model", lambda *a, **kw: (model, model_metadata))
    activations, vectors, scores = (tmp_path / name for name in ("activations", "vectors", "scores"))
    extract_activations(path, activations, batch_size=3, device="cpu", dtype="float32")
    saved = torch.load(activations / "activations.pt", weights_only=True)
    assert saved["rows"] == rows
    extract_syntactic_vectors(activations, vectors)
    metadata = run_interventions(path, vectors, scores, batch_size=3, device="cpu", dtype="float32")
    assert metadata["baseline_rows"] == 8
    assert metadata["result_rows"] == 32  # 8 sentences × 2 layers × 2 joint-edit mean types.
    assert metadata["prompt_template"] == PROMPT_TEMPLATE
    assert metadata["controls"] == {"zero_vector_first_batch": "passed", "final_block_all_rows": "passed"}
    expected = {(row["pair_id"], row["voice"]): row for row in rows}
    for name in ("baseline.csv", "results.csv"):
        with (scores / name).open() as handle:
            for row in csv.DictReader(handle):
                assert {key: row[key] for key in ENRICHED_CORPUS_COLUMNS} == expected[row["pair_id"], row["voice"]]
