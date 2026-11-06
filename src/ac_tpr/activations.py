"""Extract Qwen3-8B-Base residual states at agent and patient tokens in corpus order."""

import hashlib
import json
import os
import platform
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from nnsight import LanguageModel
from transformers import AutoTokenizer

from ac_tpr.dataset import (
    load_corpus,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
)
from ac_tpr.model import BLOCK_HOOK, SCHEMA_VERSION, load_model

from ac_tpr.prompts import PROMPT_TEMPLATE, ROLES, prepare_prompts, tokenize_roles


@torch.inference_mode()
def extract_batch(
    model: LanguageModel, inputs: dict[str, torch.Tensor], positions: torch.Tensor
) -> torch.Tensor:
    """Return CPU states shaped [sentence, block, role, hidden].

    Each trace reads the residual stream after a transformer block. Block indices
    are zero-based; the last block is captured before final RMSNorm. Embedding
    states and the final normalized model output are not included.
    """
    if model.training:
        raise ValueError("Extraction requires model.eval()")
    positions = positions.to(inputs["input_ids"].device)
    batch_indices = torch.arange(len(positions), device=positions.device)[:, None]
    with model.trace(inputs, use_cache=False) as tracer:
        states = []
        for layer in range(model.config.num_hidden_layers):
            hidden = model.model.layers[layer].output
            states.append(hidden[batch_indices, positions].cpu())
        result = torch.stack(states, dim=1).save()
        tracer.stop()
    return result


def extract_activations(
    corpus_path: Path,
    output_dir: Path,
    batch_size: int = 8,
    device: str = "cuda",
    dtype: str = "bfloat16",
    *, prompt_template: str = PROMPT_TEMPLATE,
) -> dict:
    """Extract all corpus rows with the dataset's fixed Qwen3-8B-Base checkpoint.

    The output directory must not exist. activations.pt contains the CPU tensor,
    corpus rows, unpadded input IDs, and role positions. metadata.json is written
    only after extraction and saving succeed, and marks a complete run. The full
    role tensor is kept in CPU memory (about 3.2 GiB for 5,808 rows in bfloat16).
    """
    started = time.monotonic()
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("dtype must be bfloat16 or float32")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    rows = load_corpus(corpus_path)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True
    )
    input_ids, positions, _ = prepare_prompts(rows, tokenizer, prompt_template)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Loading {TOKENIZER_ID} at {TOKENIZER_REVISION} ({dtype}, {device})", flush=True)
    model, model_metadata = load_model(tokenizer, TOKENIZER_REVISION, device, dtype)
    if max(map(len, input_ids)) > model.config.max_position_embeddings:
        raise ValueError("A sentence exceeds the model context length")
    output_dir.mkdir(parents=True, exist_ok=False)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    shape = (len(rows), model.config.num_hidden_layers, len(ROLES), model.config.hidden_size)
    activations = torch.empty(shape, dtype=model.dtype)
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
        "prompt_template": prompt_template,
        "elapsed_seconds": time.monotonic() - started,
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
        "activation_site": "Transformer block output, before final RMSNorm",
        "hook_names": [BLOCK_HOOK.format(layer=layer) for layer in range(model.config.num_hidden_layers)],
        "layer_indices": list(range(model.config.num_hidden_layers)),
        "roles": ROLES,
        "axes": ["sentence", "layer", "role", "hidden"],
        "shape": list(shape),
        "sentences_by_split": dict(Counter(row["split"] for row in rows)),
        "sentences_by_voice": dict(Counter(row["voice"] for row in rows)),
        "batch_size": batch_size,
        "device": str(model.device),
        "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if torch.device(device).type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if torch.device(device).type == "cuda" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "nnsight": version("nnsight"),
            "cuda": torch.version.cuda,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
