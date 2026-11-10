"""Bilinear TPR classification and raw-space interventions for a frozen LM."""

import math

import torch
from torch import nn


class BilinearTPRProbe(nn.Module):
    """Score role/filler assignments through a learned tensor-valued linear map.

    M has axes [hidden, role coordinate, filler coordinate]. R and F contain
    learned role and filler vectors; none of the three parameters is normalized.
    """

    def __init__(self, hidden_dim: int, num_roles: int, num_fillers: int,
                 role_dim: int, filler_dim: int):
        super().__init__()
        if min(hidden_dim, num_roles, num_fillers, role_dim, filler_dim) < 1:
            raise ValueError("All dimensions must be positive")
        self.hidden_dim = hidden_dim
        self.num_roles = num_roles
        self.num_fillers = num_fillers
        self.role_dim = role_dim
        self.filler_dim = filler_dim
        self.M = nn.Parameter(torch.randn(hidden_dim, role_dim, filler_dim) / math.sqrt(hidden_dim))
        self.R = nn.Parameter(torch.randn(num_roles, role_dim) / math.sqrt(role_dim))
        self.F = nn.Parameter(torch.randn(num_fillers, filler_dim) / math.sqrt(filler_dim))

    def binding(self, h: torch.Tensor) -> torch.Tensor:
        """Map [batch, hidden] states to [batch, role coordinate, filler coordinate]."""
        if h.ndim != 2 or h.shape[1] != self.hidden_dim:
            raise ValueError("States must have shape [batch, hidden_dim]")
        return torch.einsum("bd,dij->bij", h, self.M)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Return unnormalized [batch, role, filler] classification logits."""
        return torch.einsum("bij,ri,fj->brf", self.binding(h), self.R, self.F)

    def effective_weights(self) -> torch.Tensor:
        """Return the equivalent linear classifier weights [role, filler, hidden]."""
        return torch.einsum("dij,ri,fj->rfd", self.M, self.R, self.F)

    @torch.no_grad()
    def binding_pseudoinverse(self, rtol: float | None = None) -> tuple[torch.Tensor, dict]:
        """Return pinv(M.flatten(1)) and its SVD diagnostics using one factorization.

        The returned [role_dim*filler_dim, hidden_dim] matrix converts row-vector
        binding differences into minimum-norm hidden-state differences. Singular
        values at or below rtol times the largest are discarded, as in torch.pinv
        with zero absolute tolerance. The default is max(matrix.shape)*dtype eps.
        """
        matrix = self.M.flatten(1)
        if not torch.isfinite(matrix).all():
            raise ValueError("The binding map must be finite")
        tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps if rtol is None else float(rtol)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("rtol must be finite and nonnegative")
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        threshold = tolerance * singular[0]
        retained = singular > threshold
        inverse = torch.zeros_like(singular)
        inverse[retained] = singular[retained].reciprocal()
        pseudoinverse = (vh.mT * inverse) @ u.mT
        rank = int(retained.sum().item())
        diagnostics = {
            "rank": rank,
            "max_rank": min(matrix.shape),
            "rtol": tolerance,
            "threshold": threshold.item(),
            "largest_singular_value": singular[0].item(),
            "smallest_singular_value": singular[-1].item(),
            "condition_number": (singular[0] / singular[-1]).item() if singular[-1] > 0 else None,
            "retained_condition_number": (singular[0] / singular[retained][-1]).item() if rank else None,
        }
        return pseudoinverse, diagnostics

    @torch.no_grad()
    def intervention(self, source: torch.Tensor, target: torch.Tensor, *,
                     rtol: float | None = None, return_diagnostics: bool = False):
        """Lift a sum of role/filler changes to raw hidden space without rescaling.

        Source and target are integer filler IDs of shape [batch, num_roles].
        The desired binding difference is sum_r R[r] outer (F[target]-F[source]).
        When the map is rank deficient this returns the minimum-norm least-squares
        solution; diagnostics expose the part of the desired change not realized.
        This operation changes a target binding, not necessarily the argmax label.
        """
        if source.ndim != 2 or source.shape[1] != self.num_roles or target.shape != source.shape:
            raise ValueError("Labels must have matching [batch, num_roles] shapes")
        for labels in (source, target):
            if labels.dtype != torch.long:
                raise ValueError("Labels must use torch.long filler IDs")
            if ((labels < 0) | (labels >= self.num_fillers)).any():
                raise ValueError("Filler IDs are outside the probe vocabulary")
        source, target = source.to(self.F.device), target.to(self.F.device)
        filler_change = self.F[target] - self.F[source]
        desired = torch.einsum("ri,brj->bij", self.R, filler_change).flatten(1)
        pseudoinverse, diagnostics = self.binding_pseudoinverse(rtol)
        delta = desired @ pseudoinverse
        if not return_diagnostics:
            return delta
        residual = delta @ self.M.flatten(1) - desired
        desired_norm = torch.linalg.vector_norm(desired)
        residual_norm = torch.linalg.vector_norm(residual)
        diagnostics.update({
            "binding_residual_norm": residual_norm.item(),
            "binding_relative_residual": (residual_norm / desired_norm).item() if desired_norm > 0 else 0.0,
        })
        return delta, diagnostics


class LinearProbe(nn.Module):
    """Unconstrained bias-free role/filler classifier."""

    def __init__(self, hidden_dim: int, num_roles: int, num_fillers: int):
        super().__init__()
        if min(hidden_dim, num_roles, num_fillers) < 1:
            raise ValueError("All dimensions must be positive")
        self.hidden_dim, self.num_roles, self.num_fillers = hidden_dim, num_roles, num_fillers
        self.W = nn.Parameter(torch.randn(num_roles, num_fillers, hidden_dim) / math.sqrt(hidden_dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 2 or h.shape[1] != self.hidden_dim:
            raise ValueError("States must have shape [batch, hidden_dim]")
        return torch.einsum("bd,rfd->brf", h, self.W)

    def effective_weights(self) -> torch.Tensor:
        return self.W


class LowRankLinearProbe(nn.Module):
    """Bias-free linear classifier with a shared rank-limited hidden projection."""

    def __init__(self, hidden_dim: int, num_roles: int, num_fillers: int, rank: int):
        super().__init__()
        if min(hidden_dim, num_roles, num_fillers, rank) < 1:
            raise ValueError("All dimensions must be positive")
        if rank > min(hidden_dim, num_roles * num_fillers):
            raise ValueError("Rank exceeds the flattened classifier dimensions")
        self.hidden_dim, self.num_roles, self.num_fillers = hidden_dim, num_roles, num_fillers
        self.rank = rank
        self.U = nn.Parameter(torch.randn(hidden_dim, rank) / math.sqrt(hidden_dim))
        self.V = nn.Parameter(torch.randn(num_roles, num_fillers, rank) / math.sqrt(rank))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 2 or h.shape[1] != self.hidden_dim:
            raise ValueError("States must have shape [batch, hidden_dim]")
        return torch.einsum("bq,rfq->brf", h @ self.U, self.V)

    def effective_weights(self) -> torch.Tensor:
        return torch.einsum("dq,rfq->rfd", self.U, self.V)


def normalized_directions(directions: torch.Tensor, norm: float = 1.0) -> torch.Tensor:
    """Rescale finite nonzero vectors along their last axis to the requested norm."""
    if directions.ndim < 1 or not directions.is_floating_point():
        raise ValueError("Directions must be floating-point vectors")
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("The requested norm must be finite and positive")
    lengths = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    if not torch.isfinite(directions).all() or not torch.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("Directions must have finite, nonzero norms")
    return directions / lengths * norm


def centered_effective_weights(weights: torch.Tensor) -> torch.Tensor:
    """Remove the per-role common filler weight that cancels in a softmax.

    Center across the filler axis before comparing classifier geometry: adding
    the same hidden vector to every filler of a role leaves all its probabilities
    unchanged but can otherwise alter raw weight angles and rank.
    """
    if weights.ndim != 3 or min(weights.shape) < 1:
        raise ValueError("Weights must have shape [role, filler, hidden]")
    return weights - weights.mean(dim=1, keepdim=True)


@torch.no_grad()
def effective_weight_geometry(weights: torch.Tensor, rtol: float | None = None) -> dict:
    """Summarize the gauge-centered classifier matrix, flattened over role/filler."""
    centered = centered_effective_weights(weights).flatten(0, 1)
    if not torch.isfinite(centered).all():
        raise ValueError("Classifier weights must be finite")
    tolerance = max(centered.shape) * torch.finfo(centered.dtype).eps if rtol is None else float(rtol)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("rtol must be finite and nonnegative")
    singular = torch.linalg.svdvals(centered)
    threshold = tolerance * singular[0]
    return {
        "rank": int((singular > threshold).sum().item()),
        "rtol": tolerance,
        "threshold": threshold.item(),
        "singular_values": singular.cpu().tolist(),
        "frobenius_norm": torch.linalg.vector_norm(centered).item(),
    }
