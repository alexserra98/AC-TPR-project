"""Train-only role centroids of frozen TPR and Soft TPR reconstructions."""

import torch

from native_soft_interventions import _model_and_inputs
from soft_tpr_model import decoded_states
from tpr_probe import BilinearTPRProbe


METHODS = ('tpr_reconstructed_centroid', 'soft_tpr_reconstructed_centroid')
VOICES = ('active', 'passive')
ROLES = ('agent', 'patient')


@torch.no_grad()
def reconstruct_tpr(payload, states):
    """Return c + s ((h-c)/s M) M+; retain the probe binding, not its nullspace."""
    with torch.random.fork_rng(devices=[]):
        model = BilinearTPRProbe(**payload['config']).double()
    model.load_state_dict(payload['state_dict'])
    model.eval().requires_grad_(False)
    center, scale = payload['center'].double(), payload['scale'].double()
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
        raise ValueError('Require a finite positive normalization scale')
    inverse, diagnostics = model.binding_pseudoinverse(rtol=1e-10)
    if diagnostics['rank'] != model.M.flatten(1).shape[1]:
        raise ValueError('Require a full-column-rank TPR binding map')
    decoded = torch.cat([
        center + scale * (model.binding((batch.double() - center) / scale).flatten(1) @ inverse)
        for batch in states.split(256)
    ])
    if not torch.isfinite(decoded).all():
        raise ValueError('Nonfinite TPR reconstructions')
    return decoded


@torch.no_grad()
def reconstruct_soft(payload, states):
    """Decode the original quantized slots, then restore residual coordinates."""
    model, states, center, scale = _model_and_inputs(payload, states)
    decoded = decoded_states(model, states, center, scale, 256, 'cpu')
    if not torch.isfinite(decoded).all():
        raise ValueError('Nonfinite Soft TPR reconstructions')
    return decoded


@torch.no_grad()
def fit_centroids(source, layer, tpr, soft):
    """Read train states only; average reconstructed tokens by voice and role."""
    train_ids = [i for i, row in enumerate(source['rows']) if row['split'] == 'train']
    voices = [source['rows'][i]['voice'] for i in train_ids]
    if not train_ids or set(voices) != set(VOICES):
        raise ValueError('Both voices must be present in training')
    states = source['states'][train_ids, source['layers'].index(layer)].float()
    if states.ndim != 3 or states.shape[1] != 2 or not torch.isfinite(states).all():
        raise ValueError('Require finite training states [sentence, agent/patient, hidden]')
    means, directions, diagnostics = {}, {}, []
    for method, payload, reconstruct in zip(METHODS, (tpr, soft), (reconstruct_tpr, reconstruct_soft), strict=True):
        decoded = reconstruct(payload, states.flatten(0, 1)).reshape(states.shape).double()
        means[method], directions[method] = {}, {}
        for voice in VOICES:
            selected = torch.tensor([v == voice for v in voices])
            means[method][voice] = decoded[selected].mean(0)
            direction = means[method][voice][1] - means[method][voice][0]
            directions[method][voice] = direction.float()
            raw = states[selected].double().mean(0)
            raw_direction = raw[1] - raw[0]
            denominator = direction.norm() * raw_direction.norm()
            diagnostics.append(dict(layer=layer, method=method, voice=voice,
                training_sentences=int(selected.sum()), training_tokens_per_role=int(selected.sum()),
                direction_norm=float(direction.norm()), raw_direction_norm=float(raw_direction.norm()),
                cosine_with_raw=float(direction @ raw_direction / denominator) if denominator > 0 else None,
                reconstruction_mse=float((decoded[selected] - states[selected].double()).square().mean())))
    return dict(layer=layer, methods=list(METHODS), train_row_ids=train_ids,
                means=means, directions=directions, diagnostics=diagnostics)


def centroid_edits(centroids, rows):
    """Add patient-agent at the agent token and its negative at the patient token."""
    if any(row['voice'] not in VOICES for row in rows):
        raise ValueError('Unknown voice')
    edits = {}
    for method in METHODS:
        directions = torch.stack([centroids['directions'][method][row['voice']] for row in rows])
        edits[method] = torch.stack((directions, -directions), dim=1).float()
        if not torch.isfinite(edits[method]).all():
            raise ValueError('Nonfinite centroid edits')
    return edits
