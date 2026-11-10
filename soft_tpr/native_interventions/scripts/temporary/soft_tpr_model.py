"""Soft TPR model and training helpers reused unchanged from ac_tpr.soft_tpr."""

import math
import torch
from torch import nn
from torch.nn import functional as F

class SoftTPRAutoencoder(nn.Module):
    """Encode a token into a soft tensor; unbind, quantize, rebind, and decode.

    Latent roles are fixed orthonormal slots, not labeled agent/patient roles.
    A shared learned codebook supplies the fillers for every slot. Reconstruction
    gradients pass through unbound fillers. The form loss penalizes both the
    in-span quantization discrepancy and the component outside the role span.
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


