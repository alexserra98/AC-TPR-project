"""Prepare sentence targets, semantic bindings, and two LLM capture positions."""

import re

import torch
from transformers import PreTrainedTokenizerFast

from ac_tpr.prompts import FEW_SHOT, PROMPT_TEMPLATE, prepare_prompts

SITES = ["sentence_end", "prompt_end"]
ROLE_VOCAB = {"agent": 0, "verb": 1, "patient": 2, "voice": 3}
DEFAULT_HELD_OUT_BINDINGS = (("John", "agent"), ("doctor", "patient"))
SPECIAL_TOKENS = {"<pad>": 0, "<bos>": 1, "<eos>": 2}


def sentence_tokens(sentence: str) -> list[str]:
    """Split words and punctuation without changing capitalization."""
    return re.findall(r"\w+|[^\w\s]", sentence)


def _lexical_forms(rows: list[dict[str, str]]) -> dict:
    """Check that annotated noun and verb forms follow the symbolic inputs."""
    nouns = {}
    verbs = {"active": {}, "passive": {}}
    for row in rows:
        for role in ("agent", "patient"):
            if role + "_np_form" in row:
                noun, form = row[role], row[role + "_np_form"]
                if nouns.setdefault(noun, form) != form:
                    raise ValueError(f"Noun form varies for {noun!r}")
        if "verb_char_span" in row:
            start, end = map(int, row["verb_char_span"].split(":"))
            if not 0 <= start < end <= len(row["sentence"]):
                raise ValueError("Invalid annotated verb character span")
            surface = row["sentence"][start:end]
            forms = verbs[row["voice"]]
            if forms.setdefault(row["verb"], surface) != surface:
                raise ValueError("Verb form varies for the same lemma and voice")
    return {"noun_np_forms": nouns, "verb_forms": verbs}


def prepare_data(
    rows: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerFast,
    *,
    prompt_template: str = PROMPT_TEMPLATE,
    held_out_bindings: tuple[tuple[str, str], ...] = DEFAULT_HELD_OUT_BINDINGS,
) -> dict:
    """Prepare closed-vocabulary data while withholding selected noun-role pairs.

    Original gen_test rows are excluded because their fillers are unseen. Among
    the remaining rows, any occurrence of a held-out semantic binding becomes
    binding_test in either voice. The original split is retained for provenance;
    only tpr_split determines fitting and evaluation. Vocabularies are fitted on
    tpr_split=train, with no unknown-token substitution.

    Capture positions are ordered [sentence_end, prompt_end]. The sentence-end
    token contains the target period, possibly merged with following whitespace.
    Donor indices address this returned row list and may cross TPR partitions.
    """
    held_out_bindings = tuple(tuple(binding) for binding in held_out_bindings)
    if any(len(binding) != 2 or binding[1] not in ("agent", "patient")
           for binding in held_out_bindings):
        raise ValueError("Held-out bindings must be (noun, agent/patient) pairs")
    if len(set(held_out_bindings)) != len(held_out_bindings):
        raise ValueError("Held-out bindings must be unique")
    if any(row["split"] not in ("train", "val", "test", "gen_test") for row in rows):
        raise ValueError("Unknown original corpus split")
    corpus_indices = [i for i, row in enumerate(rows) if row["split"] != "gen_test"]
    selected = [dict(rows[i]) for i in corpus_indices]
    if not selected:
        raise ValueError("No closed-vocabulary rows remain")
    for row in selected:
        if row["voice"] not in ("active", "passive"):
            raise ValueError("Voice must be active or passive")
        row["tpr_split"] = (
            "binding_test" if any(row[role] == noun for noun, role in held_out_bindings)
            else row["split"]
        )
    training = [row for row in selected if row["tpr_split"] == "train"]
    if not training:
        raise ValueError("No TPR training rows remain")
    for noun, role in held_out_bindings:
        other = "patient" if role == "agent" else "agent"
        if not any(row[role] == noun for row in selected):
            raise ValueError(f"Held-out binding {(noun, role)!r} does not occur")
        if not any(row[other] == noun for row in training):
            raise ValueError(f"Held-out noun {noun!r} must occur in the other training role")

    lexical_forms = _lexical_forms(selected)
    bindings = [
        ["noun:" + row["agent"], "verb:" + row["verb"],
         "noun:" + row["patient"], "voice:" + row["voice"]]
        for row in selected
    ]
    filler_vocab = {
        filler: i for i, filler in enumerate(sorted({
            filler for row, fillers in zip(selected, bindings, strict=True)
            if row["tpr_split"] == "train" for filler in fillers
        }))
    }
    unknown_fillers = {filler for fillers in bindings for filler in fillers} - filler_vocab.keys()
    if unknown_fillers:
        raise ValueError(f"Fillers absent from TPR training: {sorted(unknown_fillers)}")
    targets = [sentence_tokens(row["sentence"]) for row in selected]
    token_vocab = dict(SPECIAL_TOKENS)
    token_vocab.update({
        token: i + len(SPECIAL_TOKENS) for i, token in enumerate(sorted({
            token for row, tokens in zip(selected, targets, strict=True)
            if row["tpr_split"] == "train" for token in tokens
        }))
    })
    unknown_tokens = {token for tokens in targets for token in tokens} - token_vocab.keys()
    if unknown_tokens:
        raise ValueError(f"Sentence tokens absent from TPR training: {sorted(unknown_tokens)}")

    row_keys = [(row["agent"], row["verb"], row["patient"], row["voice"]) for row in selected]
    if len(set(row_keys)) != len(row_keys):
        raise ValueError("Semantic sentence keys must be unique for donor lookup")
    lookup = {key: i for i, key in enumerate(row_keys)}
    donor_indices = []
    for agent, verb, patient, voice in row_keys:
        donor = (patient, verb, agent, voice)
        if donor not in lookup:
            raise ValueError(f"Missing role-swapped donor for {(agent, verb, patient, voice)!r}")
        donor_indices.append(lookup[donor])

    input_ids, _, answer_ids = prepare_prompts(selected, tokenizer, prompt_template)
    before, _, after = prompt_template.partition("{sentence}")
    prompts, periods = [], []
    for row in selected:
        if not row["sentence"].endswith("."):
            raise ValueError("Target sentences must end with a period")
        few_shot = FEW_SHOT[row["voice"]] if "{few_shot}" in prompt_template else ""
        prefix = before.format(few_shot=few_shot)
        prompts.append(prefix + row["sentence"] + after.format(few_shot=few_shot))
        periods.append(len(prefix) + len(row["sentence"]) - 1)
    encoding = tokenizer(prompts, add_special_tokens=False, return_offsets_mapping=True)
    positions = []
    for prompt, period, ids, actual_ids, offsets in zip(
        prompts, periods, input_ids, encoding["input_ids"], encoding["offset_mapping"], strict=True
    ):
        if ids != actual_ids:
            raise ValueError("Prompt tokenization differs from shared preparation")
        matches = [i for i, (start, end) in enumerate(offsets) if start <= period < end]
        if len(matches) != 1:
            raise ValueError("Target terminal period must occupy one token")
        index = matches[0]
        start, end = offsets[index]
        if prompt[start:end].strip() != "." or tokenizer.decode([ids[index]]).strip() != ".":
            raise ValueError("Sentence-end token must contain only the period and whitespace")
        if index >= len(ids) - 1:
            raise ValueError("Sentence-end and prompt-end capture positions must differ")
        positions.append([index, len(ids) - 1])

    return {
        "rows": selected,
        "corpus_indices": torch.tensor(corpus_indices, dtype=torch.long),
        "input_ids": input_ids,
        "positions": torch.tensor(positions, dtype=torch.long),
        "answer_ids": answer_ids,
        "donor_indices": torch.tensor(donor_indices, dtype=torch.long),
        "filler_ids": torch.tensor([[filler_vocab[f] for f in fillers] for fillers in bindings], dtype=torch.long),
        "role_ids": torch.arange(4, dtype=torch.long).repeat(len(selected), 1),
        "target_ids": [[token_vocab[token] for token in tokens] for tokens in targets],
        "filler_vocab": filler_vocab,
        "token_vocab": token_vocab,
        "role_vocab": dict(ROLE_VOCAB),
        "sites": list(SITES),
        "held_out_bindings": [list(binding) for binding in held_out_bindings],
        "lexical_forms": lexical_forms,
    }
