"""Prepare plain-text Qwen prompts and locate roles in the target sentence."""

import re

import torch
from transformers import PreTrainedTokenizerFast

QUESTION = "Who or what is performing the action?"
FEW_SHOT = {
    "active": (
        "Examples of the task:\n\n"
        f"Lucia is eating a banana. Question: {QUESTION}\nAnswer: Lucia\n\n"
        f"A man is playing a guitar. Question: {QUESTION}\nAnswer: man\n\n"
        f"The girl loved the dog. Question: {QUESTION}\nAnswer: girl\n\n"
        f"Alice kicks a ball. Question: {QUESTION}\nAnswer: Alice\n\n"
        f"A rocket touches an apple. Question: {QUESTION}\nAnswer: rocket\n\n"
    ),
    "passive": (
        "Examples of the task:\n\n"
        f"A banana is being eaten by Lucia. Question: {QUESTION}\nAnswer: Lucia\n\n"
        f"A guitar is being played by a man. Question: {QUESTION}\nAnswer: man\n\n"
        f"The dog was loved by the girl. Question: {QUESTION}\nAnswer: girl\n\n"
        f"A ball is kicked by Alice. Question: {QUESTION}\nAnswer: Alice\n\n"
        f"An apple is touched by a rocket. Question: {QUESTION}\nAnswer: rocket\n\n"
    ),
}
TASK_TEMPLATE = "{sentence}\nQuestion: " + QUESTION + "\nAnswer:"
PROMPT_TEMPLATE = "{few_shot}" + TASK_TEMPLATE

ROLES = ["agent", "patient"]


def tokenize_roles(
    rows: list[dict[str, str]], tokenizer: PreTrainedTokenizerFast
) -> tuple[list[list[int]], torch.Tensor]:
    """Validate filler tokens and locate semantic roles using character offsets.

    Returned positions have shape [sentence, role], with agent before patient,
    regardless of voice. Token indices exclude BOS/EOS and any batch padding.
    Input:

        row 0 (active):  "The cat chased the dog."
        row 1 (passive): "The dog was chased by the cat."

    Output:

        (input_ids, positions) = (
            [[0, 1, 2, 3, 4, 5],
             [0, 1, 2, 3, 4, 5, 6]],
            tensor([[1, 4],   # row 0: [agent token index, patient token index]
                    [6, 1]])  # row 1: [agent token index, patient token index]"""
    if not tokenizer.is_fast:
        raise ValueError("Character-offset validation requires a fast tokenizer")
    encoding = tokenizer(
        [row["sentence"] for row in rows],
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    positions = []
    for row, offsets in zip(rows, encoding["offset_mapping"], strict=True):
        if row["agent"] == row["patient"]:
            raise ValueError("Agent and patient must be distinct nouns")
        role_positions = []
        for role in ROLES:
            matches = list(re.finditer(rf"\b{re.escape(row[role])}\b", row["sentence"]))
            if len(matches) != 1:
                raise ValueError(f"Expected one {role} noun in {row['sentence']!r}")
            word = matches[0]
            indices = [
                index for index, (start, end) in enumerate(offsets)
                if start < word.end() and end > word.start()
            ]
            if len(indices) != 1:
                raise ValueError(f"The {role} noun must occupy exactly one token")
            index = indices[0]
            start, end = offsets[index]
            if row["sentence"][start:end].strip() != row[role]:
                raise ValueError(f"The {role} noun must occupy one whole token")
            role_positions.append(index)
        positions.append(role_positions)
    return encoding["input_ids"], torch.tensor(positions, dtype=torch.long)


def prepare_prompts(
    rows: list[dict[str, str]], tokenizer: PreTrainedTokenizerFast,
    prompt_template: str = PROMPT_TEMPLATE,
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
    """Return full prompt IDs, target role positions, and single-token answer IDs.

    Roles and answers are ordered [agent, patient]. Only the inserted sentence
    supplies role positions; repeated nouns in demonstrations are ignored.
    Templates containing {few_shot} select demonstrations using each row's voice.
    """
    if prompt_template.count("{sentence}") != 1:
        raise ValueError("Prompt template must contain exactly one {sentence}")
    if not tokenizer.is_fast:
        raise ValueError("Character-offset validation requires a fast tokenizer")
    before, _, after = prompt_template.partition("{sentence}")
    prompts, sentence_starts = [], []
    for row in rows:
        few_shot = FEW_SHOT[row["voice"]] if "{few_shot}" in prompt_template else ""
        prefix = before.format(few_shot=few_shot)
        suffix = after.format(few_shot=few_shot)
        prompts.append(prefix + row["sentence"] + suffix)
        sentence_starts.append(len(prefix))
    encoding = tokenizer(prompts, add_special_tokens=False, return_offsets_mapping=True)
    positions, answers = [], []
    for row, prompt, sentence_start, tokens, offsets in zip(
        rows, prompts, sentence_starts, encoding["input_ids"], encoding["offset_mapping"], strict=True
    ):
        if row["agent"] == row["patient"]:
            raise ValueError("Agent and patient must be distinct nouns")
        role_positions, answer_ids = [], []
        for role in ROLES:
            matches = list(re.finditer(rf"\b{re.escape(row[role])}\b", row["sentence"]))
            if len(matches) != 1:
                raise ValueError(f"Expected one {role} noun in {row['sentence']!r}")
            start = sentence_start + matches[0].start()
            end = sentence_start + matches[0].end()
            indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
            if len(indices) != 1:
                raise ValueError(f"The {role} noun must occupy exactly one token")
            position = indices[0]
            a, b = offsets[position]
            if prompt[a:b].strip() != row[role] or tokenizer.decode([tokens[position]]).strip() != row[role]:
                raise ValueError(f"The {role} noun must occupy one whole token")
            if position >= len(tokens) - 1:
                raise ValueError("Role tokens must precede the final prompt token")
            answer = " " + row[role]
            ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"The {role} answer must be one token: {answer!r}")
            if tokenizer(prompt + answer, add_special_tokens=False)["input_ids"] != tokens + ids:
                raise ValueError("Answer must append exactly one token to the prompt")
            role_positions.append(position)
            answer_ids.append(ids[0])
        positions.append(role_positions)
        answers.append(answer_ids)
    return encoding["input_ids"], torch.tensor(positions, dtype=torch.long), torch.tensor(answers, dtype=torch.long)
