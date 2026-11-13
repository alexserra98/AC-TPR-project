"""Check semantic binding holdouts and target-only sentence capture positions."""

from collections import Counter
from pathlib import Path

import pytest
import torch

from ac_tpr.dataset import generate_rows, load_corpus
from ac_tpr.prompts import FEW_SHOT, TASK_TEMPLATE
from ac_tpr.tpr.data import prepare_data, sentence_tokens


@pytest.fixture
def small_rows():
    rows = generate_rows(
        ["boy", "girl", "dog"],
        [{"lemma": "help", "past": "helped", "participle": "helped"}],
        [],
    )
    for row in rows:
        row["split"] = "train"
    return rows


@pytest.fixture(scope="module")
def corpus_data(tokenizer):
    corpus = Path(__file__).resolve().parents[1] / "data/active_pass_long/corpus.csv"
    rows = load_corpus(corpus)
    return rows, prepare_data(rows, tokenizer)


def test_binding_holdouts_preserve_seen_fillers_and_original_partitions(corpus_data):
    original, data = corpus_data
    assert Counter(row["tpr_split"] for row in data["rows"]) == {
        "train": 2868, "val": 366, "test": 366, "binding_test": 528,
    }
    assert all("tpr_split" not in row for row in original)
    assert data["held_out_bindings"] == [["John", "agent"], ["doctor", "patient"]]
    for index, row in zip(data["corpus_indices"].tolist(), data["rows"], strict=True):
        assert row["split"] == original[index]["split"] != "gen_test"
        held_out = row["agent"] == "John" or row["patient"] == "doctor"
        assert (row["tpr_split"] == "binding_test") == held_out
    training = [row for row in data["rows"] if row["tpr_split"] == "train"]
    assert any(row["patient"] == "John" for row in training)
    assert any(row["agent"] == "doctor" for row in training)
    for noun in ("Mary", "teacher", "rock"):
        assert "noun:" + noun not in data["filler_vocab"]
    assert set(data["token_vocab"]) == {
        "<pad>", "<bos>", "<eos>",
        *(token for row in training for token in sentence_tokens(row["sentence"])),
    }
    assert "John" in data["token_vocab"]
    assert "john" not in data["token_vocab"]
    assert data["lexical_forms"]["noun_np_forms"]["John"] == "name"
    assert data["lexical_forms"]["noun_np_forms"]["doctor"] == "the_noun"


def test_counterfactual_donors_can_cross_binding_partitions(corpus_data):
    _, data = corpus_data
    donors = data["donor_indices"]
    assert torch.equal(donors[donors], torch.arange(len(donors)))
    assert torch.equal(data["filler_ids"][donors], data["filler_ids"][:, [2, 1, 0, 3]])
    assert data["role_ids"].eq(torch.tensor([0, 1, 2, 3])).all()
    assert any(
        source["tpr_split"] == "train" and data["rows"][donor]["tpr_split"] == "binding_test"
        for source, donor in zip(data["rows"], donors.tolist(), strict=True)
    )
    for source, donor_index in zip(data["rows"], donors.tolist(), strict=True):
        donor = data["rows"][donor_index]
        assert donor["voice"] == source["voice"]
        assert donor["verb"] == source["verb"]
        assert donor["agent"] == source["patient"]
        assert donor["patient"] == source["agent"]


def test_sentence_end_targets_inserted_sentence_not_other_periods(small_rows, tokenizer):
    template = small_rows[0]["sentence"] + "\n{few_shot}{sentence}\nOne extra sentence.\nQuestion: Who acted?\nAnswer:"
    data = prepare_data(small_rows, tokenizer, prompt_template=template, held_out_bindings=())
    assert data["sites"] == ["sentence_end", "prompt_end"]
    for row, ids, positions in zip(data["rows"], data["input_ids"], data["positions"].tolist(), strict=True):
        prefix = template.partition("{sentence}")[0].format(few_shot=FEW_SHOT[row["voice"]])
        prompt = template.format(few_shot=FEW_SHOT[row["voice"]], sentence=row["sentence"])
        offsets = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        start, end = offsets[positions[0]]
        period = len(prefix) + len(row["sentence"]) - 1
        assert start <= period < end
        assert prompt[start:end].strip() == "."
        assert positions[1] == len(ids) - 1 > positions[0]
        assert positions[0] < next(i for i, (a, _) in enumerate(offsets) if a > period + 5)


def test_sentence_targets_exclude_prompt_and_special_tokens(small_rows, tokenizer):
    data = prepare_data(small_rows, tokenizer, held_out_bindings=())
    vocabulary = {value: key for key, value in data["token_vocab"].items()}
    for row, target in zip(data["rows"], data["target_ids"], strict=True):
        assert [vocabulary[index] for index in target] == sentence_tokens(row["sentence"])
        assert min(target) >= 3
        assert vocabulary[target[0]] == "The"
        assert vocabulary[target[-1]] == "."
    assert "Question" not in data["token_vocab"]
    assert "Answer" not in data["token_vocab"]


def test_training_only_vocabulary_rejects_unseen_fillers(small_rows, tokenizer):
    for row in small_rows:
        if "boy" in (row["agent"], row["patient"]):
            row["split"] = "test"
    with pytest.raises(ValueError, match="Fillers absent from TPR training"):
        prepare_data(small_rows, tokenizer, held_out_bindings=())


def test_unseen_sentence_token_is_not_silently_replaced(small_rows, tokenizer):
    small_rows[0]["split"] = "test"
    small_rows[0]["sentence"] = "Yesterday " + small_rows[0]["sentence"]
    with pytest.raises(ValueError, match="Sentence tokens absent from TPR training"):
        prepare_data(small_rows, tokenizer, held_out_bindings=())


@pytest.mark.parametrize("bindings", [(("boy", "subject"),), (("cat", "agent"),), (("boy", "agent"), ("boy", "patient"))])
def test_invalid_or_unlearnable_binding_holdouts_fail(small_rows, tokenizer, bindings):
    with pytest.raises(ValueError):
        prepare_data(small_rows, tokenizer, held_out_bindings=bindings)


def test_missing_counterfactual_fails(small_rows, tokenizer):
    with pytest.raises(ValueError, match="Missing role-swapped donor"):
        prepare_data(small_rows[1:], tokenizer, held_out_bindings=())


def test_annotated_noun_forms_must_follow_noun_identity(small_rows, tokenizer):
    for row in small_rows:
        row["agent_np_form"] = "the_noun"
        row["patient_np_form"] = "the_noun"
    small_rows[0]["agent_np_form"] = "name"
    with pytest.raises(ValueError, match="Noun form varies"):
        prepare_data(small_rows, tokenizer, held_out_bindings=())


def test_shared_and_tpr_prompt_candidates_agree(small_rows, tokenizer):
    data = prepare_data(small_rows, tokenizer, prompt_template=TASK_TEMPLATE, held_out_bindings=())
    for row, ids, answers in zip(data["rows"], data["input_ids"], data["answer_ids"].tolist(), strict=True):
        prompt = TASK_TEMPLATE.format(sentence=row["sentence"])
        for role, answer in zip(("agent", "patient"), answers, strict=True):
            assert tokenizer(prompt + " " + row[role], add_special_tokens=False)["input_ids"] == ids + [answer]
