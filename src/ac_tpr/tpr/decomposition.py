"""Fit linearly transformed tensor products to standardized LLM activations.

Adapted from tommccoy1/discover, models.py (MIT license; see NOTICE.md).
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from ac_tpr.tpr.extraction import cache_fingerprint, load_cache, load_auxiliary_metadata, normalized_states
from ac_tpr.tpr.training import TrainingConfig, fit_model, model_directory, release_memory


@dataclass(frozen=True)
class TPRConfig:
    filler_dim: int = 100
    role_dim: int = 100


class TensorProductModel(nn.Module):
    """Sum learned filler-role outer products, then project to LLM hidden width."""

    def __init__(
        self, n_fillers: int, hidden_size: int, filler_dim: int = 100,
        role_dim: int = 100, n_roles: int = 4,
    ):
        super().__init__()
        self.filler_embedding = nn.Embedding(n_fillers, filler_dim)
        self.role_embedding = nn.Embedding(n_roles, role_dim)
        self.output = nn.Linear(filler_dim * role_dim, hidden_size)

    def forward(self, filler_ids: torch.Tensor, role_ids: torch.Tensor) -> torch.Tensor:
        """Map matching [batch, bindings] ID tensors to [batch, hidden] encodings."""
        if filler_ids.ndim != 2 or filler_ids.shape != role_ids.shape:
            raise ValueError("Filler and role IDs must have matching [batch, bindings] shapes")
        fillers = self.filler_embedding(filler_ids)
        roles = self.role_embedding(role_ids)
        bindings = torch.bmm(fillers.transpose(1, 2), roles).flatten(start_dim=1)
        return self.output(bindings)


def tpr_loss(model: TensorProductModel, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, int]:
    """Return summed squared error and the number of activation coordinates."""
    fillers, roles, targets = batch
    predictions = model(fillers, roles)
    return F.mse_loss(predictions, targets, reduction="sum"), targets.numel()


def fit_tprs(
    encodings_dir: Path, output_dir: Path, *, config: TPRConfig = TPRConfig(),
    training: TrainingConfig = TrainingConfig(), device: str = "cuda",
) -> dict:
    """Fit symbolic approximations directly to the LLM states, without an unpacker."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    cache = load_cache(encodings_dir)
    train = [i for i, row in enumerate(cache["rows"]) if row["tpr_split"] == "train"]
    val = [i for i, row in enumerate(cache["rows"]) if row["tpr_split"] == "val"]
    manifest = {"artifact_type": "tpr_models", "cache_fingerprint": cache_fingerprint(cache),
                "config": asdict(config), "training": asdict(training), "models": []}
    output_dir.mkdir(parents=True)
    for li, layer in enumerate(cache["metadata"]["layer_indices"]):
        for si, site in enumerate(cache["metadata"]["sites"]):
            print(f"Fitting TPR: block {layer}, {site}", flush=True)
            z = normalized_states(cache, li, si)
            torch.manual_seed(training.seed)
            model = TensorProductModel(len(cache["filler_vocab"]), z.shape[-1], **asdict(config))
            path = model_directory(output_dir, layer, site)
            history = fit_model(
                model, TensorDataset(cache["filler_ids"][train], cache["role_ids"][train], z[train]),
                TensorDataset(cache["filler_ids"][val], cache["role_ids"][val], z[val]),
                tpr_loss, path, training, device,
            )
            manifest["models"].append({"layer": layer, "site": site,
                                       "best_validation_loss": history["best_validation_loss"]})
            del model, z
            release_memory()
    (output_dir / "metadata.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest



def load_tprs(directory: Path, cache: dict) -> dict[tuple[int, str], TensorProductModel]:
    """Load frozen TPR checkpoints on CPU, verifying their source extraction."""
    metadata = load_auxiliary_metadata(directory, cache, "tpr_models")
    models = {}
    for layer in cache["metadata"]["layer_indices"]:
        for site in cache["metadata"]["sites"]:
            model = TensorProductModel(len(cache["filler_vocab"]), cache["metadata"]["hidden_size"],
                                       **metadata["config"])
            path = model_directory(directory, layer, site) / "best.pt"
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            models[layer, site] = model.eval().requires_grad_(False)
    return models
