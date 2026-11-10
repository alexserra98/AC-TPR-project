"""Train-only empirical role/filler factors in a frozen TPR binding space.

The native model is C[f,r] = B + R[r] outer F[f]. Fit each voice separately;
its factors have an independent gauge and must not be averaged across voices.
Outer-product fitting is copied from the existing experiment
scripts/temporary/analyze_entity_tensor_factors.py in the main checkout.
"""

import numpy as np

ROLES = ("agent", "patient")
VOICES = ("active", "passive")


def training_cell_means(bindings, token_rows):
    """Build equally weighted filler/role cells, separately for each voice.

Only training rows determine the vocabulary or any mean. This function does not
read held-out binding values, including values containing NaNs. Each training
filler must occur with each role in each voice. The subsequent factor fit gives
each cell equal weight; within a cell observations receive equal weight.
"""
    values = np.asarray(bindings)
    if values.ndim != 3 or len(values) != len(token_rows):
        raise ValueError("Require aligned observation-by-native-matrix bindings")
    train = [i for i, row in enumerate(token_rows) if row["split"] == "train"]
    if not train or not np.isfinite(values[train]).all():
        raise ValueError("Require finite training bindings")
    if any(token_rows[i]["role"] not in ROLES or token_rows[i]["voice"] not in VOICES for i in train):
        raise ValueError("Unknown training role or voice")
    fillers = sorted({token_rows[i]["filler"] for i in train})
    means = np.empty((2, len(fillers), 2, *values.shape[1:]), dtype=np.float64)
    counts = np.zeros((2, len(fillers), 2), dtype=np.int64)
    for v, voice in enumerate(VOICES):
        for f, filler in enumerate(fillers):
            for r, role in enumerate(ROLES):
                ids = [i for i in train if (token_rows[i]["voice"], token_rows[i]["filler"],
                                            token_rows[i]["role"]) == (voice, filler, role)]
                if not ids:
                    raise ValueError(f"Missing training cell: {voice}, {filler}, {role}")
                counts[v, f, r] = len(ids)
                means[v, f, r] = values[ids].mean(0, dtype=np.float64)
    return dict(fillers=fillers, roles=list(ROLES), voices=list(VOICES), means=means, counts=counts)


def _table(value, mask=None):
    table = np.asarray(value, dtype=np.float64)
    if table.ndim != 4 or table.shape[1] != 2 or min(table.shape) < 1:
        raise ValueError("Require a filler-by-two-roles-by-native-matrix table")
    if not np.isfinite(table).all():
        raise ValueError("Native centroid tables must be finite")
    mask = np.ones(table.shape[:2], dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != table.shape[:2] or not mask.any(0).all() or not mask.any(1).all():
        raise ValueError("Every filler and role must have an observed training cell")
    return table, mask


def predict_outer(model):
    return model["bias"][None, None] + np.einsum("ri,fj->frij", model["roles"], model["fillers"])


def _spectral_initialization(table, mask):
    """Use complete role pairs to initialize their shared contrast axis."""
    complete = mask.all(1)
    if complete.sum() < 2:
        return None
    contrasts = table[complete, 1] - table[complete, 0]
    left, singular, _ = np.linalg.svd(contrasts.transpose(1, 0, 2).reshape(table.shape[2], -1), full_matrices=False)
    if singular[0] <= np.finfo(float).eps * max(np.linalg.norm(table[mask]), 1):
        return None
    axis = left[:, 0]
    fillers = np.zeros((len(table), table.shape[3]))
    fillers[complete] = np.einsum("i,fij->fj", axis, contrasts)
    midpoints = table[complete].mean(1)
    centered_fillers = fillers[complete] - fillers[complete].mean(0)
    denominator = np.sum(centered_fillers ** 2)
    if denominator <= np.finfo(float).eps * max(np.sum(fillers ** 2), 1):
        return None
    role_midpoint = np.einsum("fij,fj->i", midpoints - midpoints.mean(0), centered_fillers) / denominator
    roles = np.stack((role_midpoint - .5 * axis, role_midpoint + .5 * axis))
    bias = (midpoints - np.einsum("i,fj->fij", role_midpoint, fillers[complete])).mean(0)
    for f in np.flatnonzero(~complete):
        selected_roles = roles[mask[f]]
        denominator = np.sum(selected_roles ** 2)
        if denominator <= np.finfo(float).tiny:
            return None
        fillers[f] = np.einsum("rij,ri->j", table[f, mask[f]] - bias, selected_roles) / denominator
    return bias, roles, fillers


def fit_outer(table, mask=None, *, starts=3, max_iter=500, tolerance=1e-10, seed=42):
    """Fit a shared outer product by alternating least squares on observed cells.

Every update and start selection uses observed cells only. One spectral
initialization is supplemented by random starts; the fit is nonconvex.
"""
    table, mask = _table(table, mask)
    if starts < 1 or max_iter < 1 or tolerance <= 0:
        raise ValueError("Require positive starts, iterations and convergence tolerance")
    rng = np.random.default_rng(seed)
    n_fillers, _, role_dim, filler_dim = table.shape
    scale = float(np.sum((table[mask] - table[mask].mean(0)) ** 2))
    if scale == 0:
        return {"bias": table[mask].mean(0), "roles": np.zeros((2, role_dim)),
                "fillers": np.zeros((n_fillers, filler_dim)), "sse": 0.,
                "iterations": 0, "converged": True, "chosen_start": 0,
                "start_sse": [0.] * starts}
    spectral = _spectral_initialization(table, mask)
    candidates = []
    for start in range(starts):
        if start == 0 and spectral is not None:
            bias, roles, fillers = (x.copy() for x in spectral)
        else:
            bias = table[mask].mean(0)
            roles = rng.normal(size=(2, role_dim)) / np.sqrt(role_dim)
            fillers = rng.normal(size=(n_fillers, filler_dim)) * np.sqrt(scale / mask.sum() / filler_dim)
        previous = float("inf")
        converged = False
        for iteration in range(1, max_iter + 1):
            bias = (table - np.einsum("ri,fj->frij", roles, fillers))[mask].mean(0)
            centered = table - bias
            denominator = np.einsum("fj,fj,fr->r", fillers, fillers, mask)
            numerator = np.einsum("frij,fj,fr->ri", centered, fillers, mask)
            roles = np.divide(numerator, denominator[:, None], out=np.zeros_like(numerator),
                              where=denominator[:, None] > np.finfo(float).tiny)
            denominator = np.einsum("ri,ri,fr->f", roles, roles, mask)
            numerator = np.einsum("frij,ri,fr->fj", centered, roles, mask)
            fillers = np.divide(numerator, denominator[:, None], out=np.zeros_like(numerator),
                                where=denominator[:, None] > np.finfo(float).tiny)
            role_norm = np.linalg.norm(roles)
            if role_norm > 0:
                roles, fillers = roles / role_norm, fillers * role_norm
            prediction = bias + np.einsum("ri,fj->frij", roles, fillers)
            loss = float(np.sum((table - prediction)[mask] ** 2))
            if not np.isfinite(loss):
                raise ValueError("Nonfinite native factor fit")
            if loss <= tolerance ** 2 * scale or (np.isfinite(previous) and
                    abs(previous - loss) <= tolerance * max(previous, scale * 1e-14)):
                converged = True
                break
            previous = loss
        candidates.append({"bias": bias, "roles": roles, "fillers": fillers, "sse": loss,
                           "iterations": iteration, "converged": converged, "chosen_start": start})
    best = min(candidates, key=lambda candidate: candidate["sse"])
    best["start_sse"] = [candidate["sse"] for candidate in candidates]
    return best


def factor_diagnostics(table, model):
    """Quantify centroid reconstruction and the shared role-contrast axis."""
    table, _ = _table(table)
    prediction = predict_outer(model)
    contrast = table[:, 1] - table[:, 0]
    expected = prediction[:, 1] - prediction[:, 0]
    total = float(np.sum((table - table.mean((0, 1))) ** 2))
    contrast_energy = float(np.sum(contrast ** 2))
    singular = np.linalg.svd(contrast.transpose(1, 0, 2).reshape(table.shape[2], -1), compute_uv=False)
    return dict(centroid_sse=float(np.sum((table - prediction) ** 2)),
                centroid_baseline_sse=total,
                centroid_variance_explained=1 - float(np.sum((table - prediction) ** 2))/total if total else None,
                role_contrast_relative_error=float(np.linalg.norm(contrast-expected)/np.sqrt(contrast_energy)) if contrast_energy else None,
                role_axis_singular_values=singular.tolist(),
                shared_role_axis_residual_fraction=float(np.sum(singular[1:] ** 2)/contrast_energy) if contrast_energy else None,
                role_factor_norms=np.linalg.norm(model["roles"], axis=1).tolist())


def _role_ids(source_role_ids, size):
    ids = np.asarray(source_role_ids)
    if ids.shape != (size,) or not np.issubdtype(ids.dtype, np.integer) or not np.isin(ids, [0, 1]).all():
        raise ValueError("Require aligned integer role IDs: agent=0, patient=1")
    return ids


def infer_filler_factors(bindings, source_role_ids, model):
    """Least-squares unbind using a known source role and train-frozen factors.

f = R_source.T @ (Z-B) / ||R_source||^2 does not need a filler label and applies
to unseen fillers. It assumes the fitted outer-product model; residual error is
not erased by steering. A near-zero role cannot identify a filler and fails.
"""
    z = np.asarray(bindings, dtype=np.float64)
    if z.ndim != 3 or z.shape[1:] != np.asarray(model["bias"]).shape or not np.isfinite(z).all():
        raise ValueError("Require finite aligned native binding matrices")
    ids = _role_ids(source_role_ids, len(z))
    roles = np.asarray(model["roles"], dtype=np.float64)[ids]
    denominator = np.sum(roles ** 2, axis=1)
    if (denominator <= 1e-12).any():
        raise ValueError("A near-zero role factor cannot identify a filler")
    return np.einsum("ni,nij->nj", roles, z-model["bias"]) / denominator[:, None]


def role_swap_deltas(filler_vectors, source_role_ids, model):
    """Swap the semantic role, preserving the supplied filler factor."""
    fillers = np.asarray(filler_vectors, dtype=np.float64)
    if fillers.ndim != 2 or fillers.shape[1] != np.asarray(model["bias"]).shape[1] or not np.isfinite(fillers).all():
        raise ValueError("Require finite observation-by-filler-coordinate factors")
    ids = _role_ids(source_role_ids, len(fillers))
    roles = np.asarray(model["roles"], dtype=np.float64)
    return np.einsum("ni,nj->nij", roles[1-ids] - roles[ids], fillers)


def lift_binding_deltas(binding_deltas, binding_map, input_scale, *, rtol=1e-6):
    """Minimum-norm lift to raw hidden space, with realized-change diagnostics.

The map acts as Z = ((h-center)/scale) @ M. The center cancels in the delta.
"""
    deltas = np.asarray(binding_deltas, dtype=np.float64)
    matrix = np.asarray(binding_map, dtype=np.float64)
    if deltas.ndim != 3 or matrix.ndim != 3 or matrix.shape[1:] != deltas.shape[1:]:
        raise ValueError("Require [observation,role_coord,filler_coord] deltas and [hidden,role_coord,filler_coord] map")
    if not np.isfinite(deltas).all() or not np.isfinite(matrix).all() or not np.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("Deltas, map and positive input scale must be finite")
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("SVD tolerance must be finite and nonnegative")
    flat = matrix.reshape(len(matrix), -1)
    u, singular, vh = np.linalg.svd(flat, full_matrices=False)
    retained = singular > rtol * singular[0]
    inverse = np.zeros_like(singular)
    inverse[retained] = 1 / singular[retained]
    pseudoinverse = (vh.T * inverse) @ u.T
    desired = deltas.reshape(len(deltas), -1)
    hidden_delta = float(input_scale) * desired @ pseudoinverse
    residual = hidden_delta @ flat / float(input_scale) - desired
    norm = float(np.linalg.norm(desired))
    diagnostics = dict(rank=int(retained.sum()), max_rank=min(flat.shape), rtol=float(rtol),
                       binding_residual_norm=float(np.linalg.norm(residual)),
                       binding_relative_residual=float(np.linalg.norm(residual)/norm) if norm else 0.,
                       singular_values=singular.tolist())
    return hidden_delta, diagnostics
