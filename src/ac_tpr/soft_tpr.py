"""Soft TPR autoencoders and decoded role means for the token-steering experiment."""

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ac_tpr.syntactic_vectors import VOICES, extract_syntactic_vectors


class SoftTPRAutoencoder(nn.Module):
    """Encode a token into a soft tensor; unbind, quantize, rebind, and decode.

    Latent roles are fixed orthonormal slots, not labeled agent/patient roles.
    A shared learned codebook supplies the fillers for every slot. Reconstruction
    gradients pass through unbound fillers, while the form loss constrains the
    encoder's component orthogonal to the role span.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 256, num_roles: int = 8,
        role_dim: int = 16, filler_dim: int = 32, num_codes: int = 64,
    ):
        super().__init__()
        if min(input_dim, hidden_dim, num_roles, role_dim, filler_dim, num_codes) < 1:
            raise ValueError("All architecture dimensions must be positive")
        if role_dim < num_roles:
            raise ValueError("role_dim must be at least num_roles")
        self.filler_dim, self.role_dim = filler_dim, role_dim
        roles, _ = torch.linalg.qr(torch.randn(role_dim, num_roles), mode="reduced")
        self.register_buffer("roles", roles)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, filler_dim * role_dim),
        )
        self.codebook = nn.Embedding(num_codes, filler_dim)
        self.decoder = nn.Linear(filler_dim * role_dim, input_dim)

    def bind(self, fillers: torch.Tensor) -> torch.Tensor:
        """Sum filler/role outer products: [batch, slot, filler] -> a tensor."""
        return torch.einsum("bkf,rk->bfr", fillers, self.roles)

    def unbind(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bfr,rk->bkf", tensor, self.roles)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        soft = self.encoder(inputs).reshape(-1, self.filler_dim, self.role_dim)
        fillers = self.unbind(soft)
        codes = self.codebook.weight
        distances = (
            fillers.square().sum(-1, keepdim=True) + codes.square().sum(-1)
            - 2 * fillers @ codes.T
        )
        indices = distances.argmin(-1)
        quantized = self.codebook(indices)
        tpr = self.bind(quantized)
        straight_through = quantized.detach() + (fillers - fillers.detach())
        reconstruction = self.decoder(self.bind(straight_through).flatten(1))
        return {
            "soft": soft, "fillers": fillers, "quantized": quantized,
            "tpr": tpr, "indices": indices, "reconstruction": reconstruction,
        }


def soft_tpr_loss(
    output: dict[str, torch.Tensor], target: torch.Tensor,
    form_weight: float = 1.0, commitment_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Reconstruction, soft-to-explicit TPR form, and VQ codebook/commitment losses."""
    reconstruction = F.mse_loss(output["reconstruction"], target)
    form = F.mse_loss(output["soft"], output["tpr"].detach())
    codebook = F.mse_loss(output["quantized"], output["fillers"].detach())
    commitment = F.mse_loss(output["fillers"], output["quantized"].detach())
    return {
        "loss": reconstruction + form_weight * form + codebook + commitment_weight * commitment,
        "reconstruction": reconstruction, "form": form,
        "codebook": codebook, "commitment": commitment,
    }


@torch.no_grad()
def evaluate_autoencoder(model, states, batch_size, device, form_weight, commitment_weight):
    """Average losses per token and report shared-codebook usage by latent slot."""
    model.eval()
    totals = dict.fromkeys(("loss", "reconstruction", "form", "codebook", "commitment"), 0.0)
    counts = torch.zeros(model.roles.shape[1], model.codebook.num_embeddings, dtype=torch.long)
    for batch in states.split(batch_size):
        batch = batch.to(device)
        output = model(batch)
        losses = soft_tpr_loss(output, batch, form_weight, commitment_weight)
        for key, value in losses.items():
            totals[key] += value.item() * len(batch)
        indices = output["indices"].cpu()
        for slot in range(len(counts)):
            counts[slot] += torch.bincount(indices[:, slot], minlength=counts.shape[1])
    totals = {key: value / len(states) for key, value in totals.items()}
    probabilities = counts.float() / counts.sum(1, keepdim=True)
    totals["active_codes_by_slot"] = (counts > 0).sum(1).tolist()
    totals["perplexity_by_slot"] = (
        -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
    ).exp().tolist()
    if not all(math.isfinite(totals[key]) for key in ("loss", "reconstruction", "form", "codebook", "commitment")):
        raise ValueError("Nonfinite autoencoder evaluation")
    return totals


def fit_autoencoder(
    train: torch.Tensor, val: torch.Tensor, config: dict, *, epochs: int,
    batch_size: int, learning_rate: float, form_weight: float,
    commitment_weight: float, seed: int, device: str,
) -> tuple[SoftTPRAutoencoder, torch.Tensor, torch.Tensor, list[dict], int]:
    """Fit on train tokens; select the epoch with lowest validation reconstruction MSE.

    One train-only scalar RMS preserves the relative geometry of residual
    coordinates. Validation never contributes gradients, normalization, or
    codebook initialization. Inputs are CPU float32 [token, hidden] tensors.
    """
    if train.ndim != 2 or val.ndim != 2 or train.shape[1] != val.shape[1] or not len(train) or not len(val):
        raise ValueError("Training and validation require nonempty [token, hidden] tensors")
    if not torch.isfinite(train).all() or not torch.isfinite(val).all():
        raise ValueError("Nonfinite training or validation activations")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    center = train.mean(0)
    scale = (train - center).square().mean().sqrt().clamp_min(1e-6)
    train, val = (train - center) / scale, (val - center) / scale
    model = SoftTPRAutoencoder(**config).to(device)
    # Starting from actual train fillers avoids a codebook at the wrong scale.
    with torch.no_grad():
        sample = train[torch.randperm(len(train), generator=generator)[:min(len(train), 1024)]].to(device)
        fillers = model(sample)["fillers"].flatten(0, 1)
        indices = torch.randint(len(fillers), (config["num_codes"],), generator=generator).to(device)
        model.codebook.weight.copy_(fillers[indices])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(train), generator=generator).split(batch_size):
            batch = train[indices].to(device)
            loss = soft_tpr_loss(model(batch), batch, form_weight, commitment_weight)["loss"]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite autoencoder training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        for split, states in (("train", train), ("val", val)):
            metrics = evaluate_autoencoder(model, states, batch_size, device, form_weight, commitment_weight)
            history.append({"epoch": epoch, "split": split, **metrics})
            if split == "val" and metrics["reconstruction"] < best_loss:
                best_loss, best_epoch = metrics["reconstruction"], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"  epoch {epoch}/{epochs}: val reconstruction={metrics['reconstruction']:.6f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    return model, center, scale, history, best_epoch


@torch.no_grad()
def decoded_states(model, states, center, scale, batch_size, device):
    """Decode quantized TPRs back to the original residual coordinates."""
    model.eval()
    return torch.cat([
        model(((batch - center) / scale).to(device))["reconstruction"].cpu() * scale + center
        for batch in states.split(batch_size)
    ])


def train_soft_tpr_vectors(
    activations_dir: Path, output_dir: Path, *, hidden_dim: int = 256,
    num_roles: int = 8, role_dim: int = 16, filler_dim: int = 32, num_codes: int = 64,
    epochs: int = 100, batch_size: int = 128, learning_rate: float = 1e-3,
    form_weight: float = 1.0, commitment_weight: float = 0.25,
    seed: int = 42, device: str = "cpu",
) -> dict:
    """Train one autoencoder per block and export means for the existing runner.

    role_means/ contains the original arithmetic baseline; soft_tpr_vectors/
    contains means of decoded quantized train activations. Both use the existing
    vector schema. Root metadata is written last, after every layer completes.
    """
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if min(hidden_dim, num_roles, role_dim, filler_dim, num_codes, epochs, batch_size) < 1 or role_dim < num_roles:
        raise ValueError("Positive dimensions/epochs/batch_size and role_dim >= num_roles are required")
    if not math.isfinite(learning_rate) or learning_rate <= 0 or any(
        not math.isfinite(weight) or weight < 0 for weight in (form_weight, commitment_weight)
    ):
        raise ValueError("learning_rate must be positive and loss weights finite and nonnegative")
    # Reuse the baseline extractor's strict artifact checks and exact aggregation.
    baseline = extract_syntactic_vectors(activations_dir, output_dir / "role_means")
    data = torch.load(activations_dir / "activations.pt", map_location="cpu", weights_only=True, mmap=True)
    rows, activations = data["rows"], data["activations"]
    train_indices = [i for i, row in enumerate(rows) if row["split"] == "train"]
    val_indices = [i for i, row in enumerate(rows) if row["split"] == "val"]
    if not val_indices:
        raise ValueError("A nonempty val split is required for checkpoint selection")
    if {rows[i]["voice"] for i in val_indices} != set(VOICES):
        raise ValueError("Validation must contain active and passive sentences")
    masks = [torch.tensor([rows[i]["voice"] == voice for i in train_indices]) for voice in VOICES]
    config = dict(input_dim=activations.shape[-1], hidden_dim=hidden_dim, num_roles=num_roles,
                  role_dim=role_dim, filler_dim=filler_dim, num_codes=num_codes)
    settings = dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                    form_weight=form_weight, commitment_weight=commitment_weight, seed=seed, device=device)
    pooled = torch.empty(baseline["shape"]["pooled"], dtype=torch.float32)
    by_voice = torch.empty(baseline["shape"]["by_voice"], dtype=torch.float32)
    raw = torch.load(output_dir / "role_means" / "syntactic_vectors.pt", weights_only=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir()
    diagnostics, checkpoint_hashes = [], {}
    with (output_dir / "training.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for layer in baseline["layer_indices"]:
            print(f"Training Soft TPR for block {layer}", flush=True)
            train = activations[train_indices, layer].float()
            val = activations[val_indices, layer].float().flatten(0, 1)
            model, center, scale, history, best_epoch = fit_autoencoder(
                train.flatten(0, 1), val, config, **{**settings, "seed": seed + layer},
            )
            for record in history:
                record = {"layer": layer, **record}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(record))
                    writer.writeheader()
                writer.writerow(record)
            handle.flush()
            decoded = decoded_states(model, train.flatten(0, 1), center, scale, batch_size, device).reshape_as(train)
            pooled[layer] = decoded.mean(0)
            for voice, mask in enumerate(masks):
                by_voice[voice, layer] = decoded[mask].mean(0)
            checkpoint = checkpoints / f"layer_{layer:02d}.pt"
            torch.save({
                "config": config, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "center": center, "scale": scale, "layer": layer, "seed": seed + layer,
                "best_epoch": best_epoch,
            }, checkpoint)
            checkpoint_hashes[checkpoint.name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            for name, means in (("pooled", pooled[layer]), ("active", by_voice[0, layer]), ("passive", by_voice[1, layer])):
                reference = raw["pooled"][layer] if name == "pooled" else raw["by_voice"][VOICES.index(name), layer]
                direction, original = means[1] - means[0], reference[1] - reference[0]
                diagnostics.append({
                    "layer": layer, "mean_type": name, "best_epoch": best_epoch,
                    "raw_direction_norm": original.norm().item(),
                    "soft_tpr_direction_norm": direction.norm().item(),
                    "direction_cosine": F.cosine_similarity(direction, original, dim=0).item(),
                })
            del model
    if not torch.isfinite(pooled).all() or not torch.isfinite(by_voice).all():
        raise ValueError("Nonfinite decoded role means")
    experiment = {
        "method": "soft_tpr_decoded_role_means", "config": config, "training": settings,
        "normalization": "Train-token coordinate mean and one scalar RMS per layer",
        "selection": "Lowest validation reconstruction MSE; earliest epoch breaks ties",
        "latent_roles": "Fixed orthonormal slots without semantic labels",
        "direction": "Mean decoded quantized patient activation minus mean decoded quantized agent activation",
        "checkpoint_sha256": checkpoint_hashes,
        "activations_metadata_sha256": hashlib.sha256((activations_dir / "metadata.json").read_bytes()).hexdigest(),
        "validation_sentences": len(val_indices),
        "torch_version": str(torch.__version__),
    }
    vectors_dir = output_dir / "soft_tpr_vectors"
    vectors_dir.mkdir()
    metadata = {
        **baseline, "created_at": datetime.now(timezone.utc).isoformat(),
        "aggregation": "Arithmetic mean of decoded quantized TPR train-token reconstructions in residual coordinates",
        "vector_method": "soft_tpr_decoded_role_means", "soft_tpr": experiment,
    }
    torch.save({"pooled": pooled, "by_voice": by_voice}, vectors_dir / "syntactic_vectors.pt")
    (vectors_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    with (output_dir / "directions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata
