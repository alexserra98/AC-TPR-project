"""Permute existing Soft TPR slots; no new head, factors, or latent directions.

The caller supplies paired training tokens to select one involution per layer.
Slots and codebook entries have no built-in agent/patient or lexical labels.
"""

import math
from numbers import Integral

import torch
from torch.nn import functional as F

from soft_tpr_model import SoftTPRAutoencoder


TIE_RTOL = 1e-12
NUM_SLOTS = 8
FILLER_DIM = 32


def _model_and_inputs(payload, h):
    config = payload['config']
    if config['num_roles'] != NUM_SLOTS or config['filler_dim'] != FILLER_DIM:
        raise ValueError('Native Soft experiment requires 8 slots and 32 filler coordinates')
    if not isinstance(h, torch.Tensor) or not h.is_floating_point() or h.ndim != 2:
        raise ValueError('h must be a floating [token, hidden] tensor')
    if len(h) == 0 or h.shape[1] != config['input_dim'] or not torch.isfinite(h).all():
        raise ValueError('h must be nonempty, finite, and match the checkpoint hidden dimension')
    center = torch.as_tensor(payload['center']).detach().to(device='cpu', dtype=torch.float32)
    scale = torch.as_tensor(payload['scale']).detach().to(device='cpu', dtype=torch.float32)
    if center.shape != (config['input_dim'],) or not torch.isfinite(center).all():
        raise ValueError('Checkpoint center must be a finite hidden vector')
    if scale.numel() != 1 or not torch.isfinite(scale).all() or float(scale) <= 0:
        raise ValueError('Checkpoint scale must be one positive finite scalar')
    # Construction initializes random weights; loading a frozen model must not
    # advance the caller's CPU random generator.
    with torch.random.fork_rng(devices=[]):
        model = SoftTPRAutoencoder(**config).cpu().float()
    model.load_state_dict(payload['state_dict'])
    model.eval().requires_grad_(False)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Checkpoint contains nonfinite parameters')
    return model, h.detach().to(device='cpu', dtype=torch.float32), center, scale.reshape(())


def _encode(model, h, center, scale, batch_size):
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    quantized, codes = [], []
    for batch in h.split(int(batch_size)):
        output = model((batch - center) / scale)
        quantized.append(output['quantized'])
        codes.append(output['indices'])
    q, indices = torch.cat(quantized), torch.cat(codes)
    if not torch.isfinite(q).all():
        raise ValueError('Nonfinite quantized representation')
    return q, indices


@torch.no_grad()
def encode_native_soft(payload, h, batch_size=256):
    """Return frozen quantized fillers [N,8,32] and code indices [N,8] on CPU."""
    model, h, center, scale = _model_and_inputs(payload, h)
    return _encode(model, h, center, scale, batch_size)


def _involutions():
    def visit(remaining, permutation):
        if not remaining:
            yield tuple(permutation)
            return
        first, rest = remaining[0], remaining[1:]
        permutation[first] = first
        yield from visit(rest, permutation)
        for partner in rest:
            permutation[first], permutation[partner] = partner, first
            yield from visit([slot for slot in rest if slot != partner], permutation)
            permutation[partner] = partner
        permutation[first] = first

    return list(visit(list(range(NUM_SLOTS)), list(range(NUM_SLOTS))))


def _validate_permutation(permutation):
    if not isinstance(permutation, (list, tuple)) or len(permutation) != NUM_SLOTS:
        raise ValueError('permutation must contain exactly 8 integer slot indices')
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in permutation):
        raise ValueError('permutation must contain integer slot indices')
    p = [int(i) for i in permutation]
    if sorted(p) != list(range(NUM_SLOTS)):
        raise ValueError('permutation must be a bijection of slots 0 through 7')
    if any(p[p[i]] != i for i in range(NUM_SLOTS)):
        raise ValueError('permutation must be an involution')
    return p


def select_slot_permutation(q, donor_q):
    """Minimize paired TRAIN code distance over all 764 eight-slot involutions.

    Compare every candidate with the global minimum using relative tolerance
    1e-12 and no absolute tolerance; ties prefer fewer moved slots, then the
    lexicographically smallest permutation. Output slot k receives q[:,p[k]].
    The caller is responsible for train-only, same-filler counterfactual pairs.
    """
    if not isinstance(q, torch.Tensor) or not isinstance(donor_q, torch.Tensor):
        raise ValueError('q and donor_q must be tensors')
    if q.ndim != 3 or q.shape[1:] != (NUM_SLOTS, FILLER_DIM) or len(q) == 0:
        raise ValueError('q must have nonempty shape [token,8,32]')
    if donor_q.shape != q.shape or not q.is_floating_point() or not donor_q.is_floating_point():
        raise ValueError('q and donor_q must be floating tensors of identical shape')
    if not torch.isfinite(q).all() or not torch.isfinite(donor_q).all():
        raise ValueError('Nonfinite quantized training representation')
    source = q.detach().to(device='cpu', dtype=torch.float64)
    donor = donor_q.detach().to(device='cpu', dtype=torch.float64)
    # Direct squared differences avoid cancellation for identical native codes.
    costs = torch.stack([
        (source[:, i:i + 1] - donor).square().sum((0, 2))
        for i in range(NUM_SLOTS)
    ])
    if not torch.isfinite(costs).all():
        raise ValueError('Nonfinite slot matching costs')
    candidates = _involutions()
    scored = [(math.fsum(float(costs[p[k], k]) for k in range(NUM_SLOTS)), p)
              for p in candidates]

    def choose(values):
        minimum = min(cost for cost, _ in values)
        tied = [(cost, p) for cost, p in values
                if math.isclose(cost, minimum, rel_tol=TIE_RTOL, abs_tol=0.0)]
        return min(tied, key=lambda item: (sum(i != v for i, v in enumerate(item[1])), item[1])), len(tied)

    selected, tie_count = choose(scored)
    second, _ = choose([item for item in scored if item[1] != selected[1]])
    identity = tuple(range(NUM_SLOTS))
    identity_item = next(item for item in scored if item[1] == identity)
    best_nonidentity, _ = choose([item for item in scored if item[1] != identity])

    def record(item):
        cost, p = item
        return dict(permutation=list(p), objective_sse=cost, mse=cost / q.numel(),
                    moved_slots=sum(i != v for i, v in enumerate(p)))

    identity_cost = identity_item[0]
    diagnostics = dict(
        tokens=len(q), slots=NUM_SLOTS, filler_dim=FILLER_DIM,
        candidate_count=len(candidates), permutation=list(selected[1]),
        best=record(selected), identity=record(identity_item), second=record(second),
        best_nonidentity=record(best_nonidentity), gap=second[0] - selected[0],
        minimum_objective_sse=min(cost for cost, _ in scored),
        relative_improvement=(identity_cost - selected[0]) / identity_cost if identity_cost else 0.0,
        cost_matrix=costs.tolist(), tie_rtol=TIE_RTOL, tie_count=tie_count,
        tie_rule='Within relative 1e-12 of minimum; fewer moved slots; lexicographic',
        objective='Sum squared native-code distances to paired train same-filler opposite-role tokens',
        semantics='Calibrated native slot permutation; slots are not identified semantic roles',
    )
    return list(selected[1]), diagnostics


@torch.no_grad()
def native_slot_swap(payload, h, permutation):
    """Decode a native slot permutation at alpha=1, restoring raw residual scale.

    Delta is scale * W_decoder * vec(bind(q[:,p] - q)). The affine decoder bias
    cancels. This preserves the target latent code multiset and the original
    reconstruction residual; re-encoding the edited hidden state need not
    recover that target. There is no source-role sign or norm matching.
    """
    p = _validate_permutation(permutation)
    model, h, center, scale = _model_and_inputs(payload, h)
    q, codes = _encode(model, h, center, scale, 256)
    target_codes = codes[:, p]
    target_q = q[:, p]
    code_multiset_preserved = torch.equal(codes.sort(1).values, target_codes.sort(1).values)
    latent_double_swap_exact = torch.equal(target_q[:, p], q)
    if not code_multiset_preserved or not latent_double_swap_exact:
        raise AssertionError('Native slot permutation failed code preservation or involution')
    latent_delta = target_q - q
    delta = torch.cat([
        scale * F.linear(model.bind(batch).flatten(1), model.decoder.weight)
        for batch in latent_delta.split(256)
    ])
    if not torch.isfinite(delta).all():
        raise ValueError('Nonfinite native Soft intervention')
    norms = delta.double().norm(dim=1)
    diagnostics = dict(
        tokens=len(h), permutation=p, moved_slots=sum(i != v for i, v in enumerate(p)),
        code_multiset_preserved=code_multiset_preserved,
        latent_double_swap_exact=latent_double_swap_exact,
        fraction_changed_codes=float((target_codes != codes).double().mean()),
        zero_native_edit_fraction=float((norms == 0).double().mean()),
        mean_hidden_edit_norm=float(norms.mean()), min_hidden_edit_norm=float(norms.min()),
        max_hidden_edit_norm=float(norms.max()),
        mean_native_latent_change_norm=float(latent_delta.double().flatten(1).norm(dim=1).mean()),
        decoder_bias_cancelled=True, normalization_scale=float(scale),
        precision='CPU float32 edit; norms calculated in float64 before any BF16 cast',
        preservation='Exact code multiset at latent target; no lexical or re-encoding guarantee',
    )
    return delta, diagnostics
