"""Check target sentence alignment in plain-text few-shot prompts."""

import pytest
import torch

from ac_tpr.prompts import FEW_SHOT, PROMPT_TEMPLATE, TASK_TEMPLATE, prepare_prompts


@pytest.fixture
def rows():
    return [
        {"sentence": "The dog loved the girl.", "voice": "active", "agent": "dog", "patient": "girl"},
        {"sentence": "The dog was loved by the girl.", "voice": "passive", "agent": "girl", "patient": "dog"},
    ]


@pytest.mark.parametrize("prefix", ["{few_shot}", "", "An example:\n\n"])
@pytest.mark.parametrize("suffix", ["", " "])
def test_roles_are_in_target_sentence_even_when_example_repeats_it(rows, tokenizer, prefix, suffix):
    template = prefix + TASK_TEMPLATE + suffix
    ids, positions, answers = prepare_prompts(rows, tokenizer, template)
    for row, tokens, pair, candidates in zip(rows, ids, positions, answers, strict=True):
        rendered_prefix = prefix.format(few_shot=FEW_SHOT[row["voice"]])
        prompt = template.format(sentence=row["sentence"], few_shot=FEW_SHOT[row["voice"]])
        encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        assert tokens == encoded["input_ids"]
        for role, position, candidate in zip(("agent", "patient"), pair.tolist(), candidates.tolist()):
            a, b = encoded["offset_mapping"][position]
            assert a >= len(rendered_prefix) - 1
            assert b <= len(rendered_prefix) + len(row["sentence"])
            assert prompt[a:b].strip() == row[role]
            assert tokenizer.decode([candidate]) == " " + row[role]
            assert tokenizer(prompt + " " + row[role], add_special_tokens=False)["input_ids"] == tokens + [candidate]
    padded = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    assert torch.equal(padded["input_ids"].gather(1, positions), answers)
    assert padded["attention_mask"].gather(1, positions).all()


@pytest.mark.parametrize("template", ["No sentence", "{sentence} {sentence}"])
def test_invalid_template_is_rejected(rows, tokenizer, template):
    with pytest.raises(ValueError):
        prepare_prompts(rows, tokenizer, template)


def test_multitoken_noun_is_rejected(rows, tokenizer):
    rows[0]["agent"] = "neuroscientist"
    rows[0]["sentence"] = rows[0]["sentence"].replace("dog", "neuroscientist")
    with pytest.raises(ValueError):
        prepare_prompts(rows, tokenizer, PROMPT_TEMPLATE)
