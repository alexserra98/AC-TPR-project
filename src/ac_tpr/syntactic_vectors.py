"""Average training activations into pooled and per-voice agent/patient vectors."""

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from ac_tpr.activations import ROLES
from ac_tpr.model import validate_activation_provenance

VOICES = ["active", "passive"]


@torch.inference_mode()
def extract_syntactic_vectors(activations_dir: Path, output_dir: Path) -> dict:
    """Save float32 role means from a completed activation extraction on CPU.

    Only rows whose split is ``train`` contribute. Pooled means weight every
    sentence equally; per-voice means average active and passive rows separately.
    States retain their original block-output coordinates, without normalization.
    Memory mapping and layer-wise processing limit temporary tensor memory.

    The output directory must be new. syntactic_vectors.pt contains:
    - ``pooled`` [layer, role, hidden] 
    - ``by_voice`` [voice, layer, role, hidden] tensors.
    - metadata.json records their ordering and provenance and is written last to
    mark a complete run. 
    Missing source fields or incompatible artifacts fail.
    """
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    source = json.loads((activations_dir / "metadata.json").read_text(encoding="utf-8"))
    validate_activation_provenance(source)
    provenance = {
        key: source[key] for key in (
            "corpus", "corpus_sha256", "model", "tokenizer", "activation_site",
            "layer_indices", "roles", "schema_version", "hook_names",
        )
    }

    if source["roles"] != ROLES:
        raise ValueError(f"Source roles must be ordered as {ROLES}")
    if source["axes"] != ["sentence", "layer", "role", "hidden"]:
        raise ValueError("Source axes must be [sentence, layer, role, hidden]")
    if source["activation_site"] != "Transformer block output, before final_layer_norm":
        raise ValueError("Source activations must be block outputs before final_layer_norm")

    data = torch.load(
        activations_dir / "activations.pt", map_location="cpu", weights_only=True, mmap=True
    )
    activations, rows = data["activations"], data["rows"]
    if activations.ndim != 4 or any(size == 0 for size in activations.shape):
        raise ValueError("Activations must have nonempty [sentence, layer, role, hidden] axes")
    if list(activations.shape) != source["shape"]:
        raise ValueError("Activation shape does not match source metadata")
    if len(rows) != activations.shape[0] or activations.shape[2] != len(ROLES):
        raise ValueError("Activation shape does not match rows or roles")
    if not activations.is_floating_point():
        raise ValueError("Activations must be floating point")
    if str(activations.dtype).removeprefix("torch.") != source["model"]["dtype"]:
        raise ValueError("Activation dtype does not match source metadata")
    if source["layer_indices"] != list(range(activations.shape[1])):
        raise ValueError("Source layer indices must enumerate all blocks from zero")
    if dict(Counter(row["split"] for row in rows)) != source["sentences_by_split"]:
        raise ValueError("Row split counts do not match source metadata")
    if dict(Counter(row["voice"] for row in rows)) != source["sentences_by_voice"]:
        raise ValueError("Row voice counts do not match source metadata")

    train_indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    if not train_indices:
        raise ValueError("No training rows in source activations")
    train_voices = [rows[i]["voice"] for i in train_indices]
    counts = dict(Counter(train_voices))
    if set(counts) != set(VOICES):
        raise ValueError(f"Training rows must contain exactly these voices: {VOICES}")
    # voice_masks[0]:= mask passive rows; voice_masks[1]:= mask active rows; 
    voice_masks = [torch.tensor([value == voice for value in train_voices]) for voice in VOICES]

    pooled = torch.empty(activations.shape[1:], dtype=torch.float32)
    by_voice = torch.empty((len(VOICES), *pooled.shape), dtype=torch.float32)
    for layer in source["layer_indices"]:
        # states[:, 0, :] -> agents across all sentences
        # states[:, 1, :] -> patients across all sentences
        states = activations[train_indices, layer].float()
        if not torch.isfinite(states).all():
            raise ValueError(f"Nonfinite training activations at layer {layer}")
        pooled[layer] = states.mean(dim=0)
        for voice_index, mask in enumerate(voice_masks):
            by_voice[voice_index, layer] = states[mask].mean(dim=0)
    if not torch.isfinite(pooled).all() or not torch.isfinite(by_voice).all():
        raise ValueError("Nonfinite syntactic vectors")

    metadata = {
        **provenance,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "activations_dir": str(activations_dir.resolve()),
        "split": "train",
        "sentences": len(train_indices),
        "sentences_by_voice": counts,
        "voices": VOICES,
        "dtype": "float32",
        "aggregation": "Arithmetic mean over sentences, independently for each layer and role",
        "axes": {
            "pooled": ["layer", "role", "hidden"],
            "by_voice": ["voice", "layer", "role", "hidden"],
        },
        "shape": {"pooled": list(pooled.shape), "by_voice": list(by_voice.shape)},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"pooled": pooled, "by_voice": by_voice}, output_dir / "syntactic_vectors.pt")
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
