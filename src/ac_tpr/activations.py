"""Extract Pythia residual states at agent and patient tokens in corpus order."""

import csv
import hashlib
import json
import os
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ac_tpr.dataset import (
    CORPUS_COLUMNS,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    validate_tokenization,
)
from ac_tpr.model import BLOCK_HOOK, SCHEMA_VERSION, load_model

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
    validate_tokenization(rows, tokenizer)
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


@torch.inference_mode()
def extract_batch(
    model: HookedTransformer, inputs: dict[str, torch.Tensor], positions: torch.Tensor
) -> torch.Tensor:
    """Return CPU states shaped [sentence, block, role, hidden].

    Each hook reads the residual stream after a transformer block. Block indices
    are zero-based; the last block is captured before final_layer_norm. Embedding
    states and the final normalized model output are not included.
    """
    if model.training:
        raise ValueError("Extraction requires model.eval()")
    positions = positions.to(inputs["input_ids"].device)
    batch_indices = torch.arange(len(positions), device=positions.device)[:, None]
    states = []

    def capture(hidden, hook):
        states.append(hidden[batch_indices, positions].cpu())

    model.run_with_hooks(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        prepend_bos=False, past_kv_cache=None, return_type=None,
        fwd_hooks=[(BLOCK_HOOK.format(layer=layer), capture) for layer in range(model.cfg.n_layers)],
    )
    return torch.stack(states, dim=1)


def extract_activations(
    corpus_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    device: str = "cuda",
    dtype: str = "float32",
) -> dict:
    """Extract all corpus rows with the dataset's fixed Pythia checkpoint.

    The output directory must not exist. activations.pt contains the CPU tensor,
    corpus rows, unpadded input IDs, and role positions. metadata.json is written
    only after extraction and saving succeed, and marks a complete run. The full
    role tensor is kept in CPU memory (about 3.8 GiB for 3,840 rows in float32).
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    with corpus_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CORPUS_COLUMNS:
            raise ValueError(f"Corpus CSV columns must be {CORPUS_COLUMNS}")
        rows = list(reader)
    if not rows or any(
        None in row or any(value is None or value == "" for value in row.values())
        for row in rows
    ):
        raise ValueError("Corpus must contain complete, nonempty rows")
    if len({(row["pair_id"], row["voice"]) for row in rows}) != len(rows):
        raise ValueError("Corpus (pair_id, voice) keys must be unique")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True
    )
    input_ids, positions = tokenize_roles(rows, tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Loading {TOKENIZER_ID} at {TOKENIZER_REVISION} ({dtype}, {device})", flush=True)
    model, model_metadata = load_model(tokenizer, TOKENIZER_REVISION, device, dtype)
    if max(map(len, input_ids)) > model.cfg.n_ctx:
        raise ValueError("A sentence exceeds the model context length")
    output_dir.mkdir(parents=True, exist_ok=False)
    shape = (len(rows), model.cfg.n_layers, len(ROLES), model.cfg.d_model)
    activations = torch.empty(shape, dtype=model.cfg.dtype)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        inputs = tokenizer.pad(
            {"input_ids": input_ids[start:end]}, padding=True, return_tensors="pt"
        ).to(device)
        batch = extract_batch(model, inputs, positions[start:end])
        if not torch.isfinite(batch).all():
            raise ValueError(f"Nonfinite activations in rows {start}:{end}")
        activations[start:end] = batch
        print(f"Extracted {end}/{len(rows)} sentences", flush=True)

    torch.save(
        {
            "activations": activations,
            "rows": rows,
            "input_ids": input_ids,
            "role_positions": positions,
        },
        output_dir / "activations.pt",
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(corpus_path.resolve()),
        "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "model": model_metadata,
        "tokenizer": {
            "name": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "backend_sha256": hashlib.sha256(
                tokenizer.backend_tokenizer.to_str().encode("utf-8")
            ).hexdigest(),
            "add_special_tokens": False,
            "padding_side": "right",
            "pad_token_id": tokenizer.pad_token_id,
        },
        "activation_site": "Transformer block output, before final_layer_norm",
        "hook_names": [BLOCK_HOOK.format(layer=layer) for layer in range(model.cfg.n_layers)],
        "layer_indices": list(range(model.cfg.n_layers)),
        "roles": ROLES,
        "axes": ["sentence", "layer", "role", "hidden"],
        "shape": list(shape),
        "sentences_by_split": dict(Counter(row["split"] for row in rows)),
        "sentences_by_voice": dict(Counter(row["voice"] for row in rows)),
        "batch_size": batch_size,
        "device": str(model.cfg.device),
        "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "transformer_lens": version("transformer-lens"),
            "cuda": torch.version.cuda,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
