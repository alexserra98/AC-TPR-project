"""Cache full-prompt activations at sentence and answer boundaries."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ac_tpr.activations import extract_batch
from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION, load_corpus
from ac_tpr.model import BLOCK_HOOK, SCHEMA_VERSION, load_model, validate_activation_provenance
from ac_tpr.prompts import PROMPT_TEMPLATE
from ac_tpr.tpr.data import DEFAULT_HELD_OUT_BINDINGS, SITES as CAPTURE_SITES, prepare_data

DEFAULT_LAYERS = (5, 12, 15, 24, 31)
DEFAULT_BINDINGS = DEFAULT_HELD_OUT_BINDINGS
SITES = tuple(CAPTURE_SITES)


def training_statistics(activations: torch.Tensor, rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Return training-only population mean and std + 1e-10 in float32."""
    indices = [i for i, row in enumerate(rows) if row["tpr_split"] == "train"]
    if not indices:
        raise ValueError("Normalization requires training examples")
    means, scales = [], []
    for layer in range(activations.shape[1]):
        values = activations[indices, layer].float()
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        means.append(mean)
        scales.append(variance.sqrt() + 1e-10)
    return torch.stack(means), torch.stack(scales)


def validate_cache(cache: dict) -> None:
    """Require completed TPR encodings with matching model and tensor axes."""
    metadata = cache["metadata"]
    validate_activation_provenance(metadata)
    if metadata.get("artifact_type") != "tpr_encodings" or metadata.get("tpr_schema_version") != 1:
        raise ValueError("Expected TPR encoding schema 1")
    layers = metadata["layer_indices"]
    if not layers or len(set(layers)) != len(layers) or metadata["sites"] != list(SITES):
        raise ValueError("Invalid TPR layer or site ordering")
    n = len(cache["rows"])
    shape = (n, len(layers), len(SITES), metadata["hidden_size"])
    if tuple(cache["activations"].shape) != shape or metadata["shape"] != list(shape):
        raise ValueError("TPR activation shape does not match metadata")
    for key in ("mean", "scale"):
        value = cache[key]
        if value.shape != shape[1:] or not torch.isfinite(value).all():
            raise ValueError(f"Invalid normalization {key}")
    if not (cache["scale"] > 0).all():
        raise ValueError("Normalization scale must be positive")
    for key, expected in (("positions", (n, 2)), ("answer_ids", (n, 2)),
                          ("donor_indices", (n,)), ("filler_ids", (n, 4)), ("role_ids", (n, 4))):
        if cache[key].shape != expected or cache[key].dtype != torch.long:
            raise ValueError(f"Invalid {key} tensor")
    if len(cache["input_ids"]) != n or len(cache["target_ids"]) != n:
        raise ValueError("Token rows do not match activation rows")
    if ((cache["donor_indices"] < 0) | (cache["donor_indices"] >= n)).any():
        raise ValueError("Counterfactual indices are outside the cache")
    if not torch.equal(cache["donor_indices"][cache["donor_indices"]], torch.arange(n)):
        raise ValueError("Counterfactual pairing must reverse itself")
    for i, (row, tokens) in enumerate(zip(cache["rows"], cache["input_ids"], strict=True)):
        sentence_end, prompt_end = cache["positions"][i].tolist()
        if not 0 <= sentence_end < prompt_end == len(tokens) - 1:
            raise ValueError("Invalid sentence or prompt boundary")
        donor = cache["rows"][cache["donor_indices"][i].item()]
        if (donor["agent"], donor["patient"], donor["verb"], donor["voice"]) != (
            row["patient"], row["agent"], row["verb"], row["voice"]
        ):
            raise ValueError("Counterfactual annotations do not match")


def load_cache(directory: Path) -> dict:
    """Load an extraction only after its completion metadata has been written."""
    metadata = json.loads((directory / "metadata.json").read_text())
    cache = torch.load(directory / "encodings.pt", map_location="cpu", weights_only=True, mmap=True)
    cache["metadata"] = metadata
    validate_cache(cache)
    return cache


def cache_fingerprint(cache: dict) -> str:
    """Identify the extraction used to fit an auxiliary model."""
    return hashlib.sha256(json.dumps(cache["metadata"], sort_keys=True).encode()).hexdigest()


def extract_encodings(
    corpus_path: Path, output_dir: Path, layers: tuple[int, ...] = DEFAULT_LAYERS,
    batch_size: int = 8, device: str = "cuda", dtype: str = "bfloat16", *,
    prompt_template: str = PROMPT_TEMPLATE,
    held_out_bindings: tuple[tuple[str, str], ...] = DEFAULT_BINDINGS,
) -> dict:
    """Extract both sites together; normalize using only the TPR training split.

    Original gen_test rows are excluded because their nouns are unseen. Binding
    holdouts get tpr_split=binding_test without modifying the corpus split field.
    Existing output directories are rejected, including incomplete extractions.
    """
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if batch_size < 1 or not layers or len(set(layers)) != len(layers):
        raise ValueError("Use a positive batch size and distinct layer indices")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    data = prepare_data(load_corpus(corpus_path), tokenizer, prompt_template=prompt_template,
                        held_out_bindings=held_out_bindings)
    model, provenance = load_model(tokenizer, TOKENIZER_REVISION, device, dtype)
    if min(layers) < 0 or max(layers) >= model.config.num_hidden_layers:
        raise ValueError("Requested layer is outside the model")
    if max(map(len, data["input_ids"])) > model.config.max_position_embeddings:
        raise ValueError("A prompt exceeds the model context length")
    shape = (len(data["rows"]), len(layers), 2, model.config.hidden_size)
    activations = torch.empty(shape, dtype=model.dtype)
    output_dir.mkdir(parents=True)
    for start in range(0, shape[0], batch_size):
        end = min(start + batch_size, shape[0])
        inputs = tokenizer.pad({"input_ids": data["input_ids"][start:end]},
                               padding=True, return_tensors="pt").to(device)
        values = extract_batch(model, inputs, data["positions"][start:end])[:, list(layers)]
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite extracted activations")
        activations[start:end] = values
        print(f"TPR extraction: {end}/{shape[0]} sentences", flush=True)
    data["activations"] = activations
    data["mean"], data["scale"] = training_statistics(activations, data["rows"])
    metadata = {
        "schema_version": SCHEMA_VERSION, "tpr_schema_version": 1,
        "artifact_type": "tpr_encodings", "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(corpus_path.resolve()), "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        "model": provenance,
        "tokenizer": {"name": TOKENIZER_ID, "revision": TOKENIZER_REVISION,
                      "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                      "padding_side": "right", "pad_token_id": tokenizer.pad_token_id,
                      "add_special_tokens": False},
        "prompt_template": prompt_template,
        "prompt_ids_sha256": hashlib.sha256(json.dumps(data["input_ids"]).encode()).hexdigest(),
        "activation_site": "Transformer block output, before final RMSNorm",
        "hook_names": [BLOCK_HOOK.format(layer=layer) for layer in layers],
        "layer_indices": list(layers), "sites": list(SITES), "hidden_size": shape[-1],
        "num_hidden_layers": model.config.num_hidden_layers,
        "axes": ["sentence", "layer", "site", "hidden"], "shape": list(shape),
        "held_out_bindings": [list(binding) for binding in held_out_bindings],
        "tpr_split_counts": dict(Counter(row["tpr_split"] for row in data["rows"])),
        "excluded_splits": ["gen_test"], "normalization": "train population std + 1e-10",
        "batch_size": batch_size, "device": device,
    }
    data["metadata"] = metadata
    validate_cache(data)
    torch.save({key: value for key, value in data.items() if key != "metadata"}, output_dir / "encodings.pt")
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def normalized_states(cache: dict, layer_index: int, site_index: int) -> torch.Tensor:
    """Return float32 states using the cached training-only coordinate scale."""
    return (cache["activations"][:, layer_index, site_index].float()
            - cache["mean"][layer_index, site_index]) / cache["scale"][layer_index, site_index]



def load_auxiliary_metadata(directory: Path, cache: dict, artifact_type: str) -> dict:
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata["artifact_type"] != artifact_type or metadata["cache_fingerprint"] != cache_fingerprint(cache):
        raise ValueError("Auxiliary models do not match this extraction")
    return metadata
