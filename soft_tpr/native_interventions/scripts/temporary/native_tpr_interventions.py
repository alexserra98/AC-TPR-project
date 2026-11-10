"""Swap native TPR role readout channels without new heads or fitted factors.

For x=(h-center)/scale and Z=xM, the original filler logits are R Z F^T.
The intervention solves R deltaZ = P R Z - R Z, then lifts deltaZ with M+.
At native alpha=1 this transfers each role's complete filler profile to the
other role. This is a property of the frozen probe, not an LM identity claim.
Role rows in R read coordinates; columns of R^{-1} are their writing duals.
"""

import torch

from tpr_probe import BilinearTPRProbe

INVERSE_RTOL = 1e-10
NEAR_ZERO_NORM = 1e-12


def _relative(error, target):
    return float(error.norm() / target.norm().clamp_min(1e-12))


@torch.no_grad()
def native_role_swap(payload, h):
    """Return additive CPU float32 deltas and JSON-compatible diagnostics.

    Inputs are a frozen Nov6 probe checkpoint and raw [token, hidden] states.
    No role, voice, or filler annotations enter this operation. R must have two
    independent rows and two coordinates; M must have full column rank. One
    float64 SVD of M is reused for every input token in this call. Checkpoint
    tensors and RNG state are unchanged. Exactly zero swaps are valid.

    Reported FP32/BF16 state diagnostics include the rounding from h.float()+delta.
    Only the FP32 native-alpha-one identity is checked tightly. BF16 errors are
    descriptive; arbitrary norm matching or alpha need not perform a full swap.
    """
    config = payload['config']
    if config['num_roles'] != 2 or config['role_dim'] != 2:
        raise ValueError('Native role swap requires two roles and two role coordinates')
    if not isinstance(h, torch.Tensor) or h.device.type != 'cpu':
        raise ValueError('Hidden states must be a CPU tensor')
    if h.ndim != 2 or not len(h) or h.shape[1] != config['hidden_dim'] or not h.is_floating_point():
        raise ValueError('Require nonempty floating [token, hidden] states')
    if not torch.isfinite(h).all():
        raise ValueError('Hidden states must be finite')
    with torch.random.fork_rng(devices=[]):
        probe = BilinearTPRProbe(**config).double()
    probe.load_state_dict(payload['state_dict'])
    probe.eval().requires_grad_(False)
    if not all(torch.isfinite(value).all() for value in probe.parameters()):
        raise ValueError('Probe parameters must be finite')
    center = torch.as_tensor(payload['center'], device='cpu', dtype=torch.float64)
    scale = torch.as_tensor(payload['scale'], device='cpu', dtype=torch.float64)
    if center.shape != (config['hidden_dim'],) or not torch.isfinite(center).all():
        raise ValueError('Checkpoint center must be finite and match the hidden dimension')
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
        raise ValueError('Checkpoint scale must be finite and positive')
    scale = scale.reshape(())
    role_singular = torch.linalg.svdvals(probe.R)
    role_rank = int((role_singular > INVERSE_RTOL * role_singular[0]).sum())
    if role_rank != 2:
        raise ValueError('Native role matrix is singular or numerically rank deficient')
    pinv, inverse_diagnostics = probe.binding_pseudoinverse(rtol=INVERSE_RTOL)
    if inverse_diagnostics['rank'] != config['role_dim'] * config['filler_dim']:
        raise ValueError('Binding map must have full column rank for an exact native swap')

    z = probe.binding((h.double() - center) / scale)
    channels = torch.einsum('ri,bij->brj', probe.R, z)
    # Solving for the difference gives exactly zero when both channels agree.
    delta_z = torch.linalg.solve(probe.R, channels.flip(1) - channels)
    delta = (scale * delta_z.flatten(1) @ pinv).float()
    if not torch.isfinite(delta).all():
        raise ValueError('Native swap cannot be represented by finite float32 edits')
    realized = probe.binding(delta.double() / scale)
    torch.testing.assert_close(realized, delta_z, rtol=2e-5, atol=2e-6,
                               msg='Float32 lift does not realize the requested native binding change')

    logits = torch.einsum('brj,fj->brf', channels, probe.F)
    target_channels, target_logits = channels.flip(1), logits.flip(1)
    after_float32 = h.float() + delta
    after_bfloat16 = after_float32.bfloat16().float()
    if not torch.isfinite(after_bfloat16).all():
        raise ValueError('Native edited states are not finite after BF16 rounding')
    diagnostics = dict(tokens=len(h), role_rank=role_rank,
        role_condition_number=float(role_singular[0]/role_singular[-1]),
        binding_rank=inverse_diagnostics['rank'],
        binding_condition_number=inverse_diagnostics['condition_number'],
        binding_pseudoinverse=inverse_diagnostics,
        centered_filler_rank=int(torch.linalg.matrix_rank(probe.F-probe.F.mean(0))),
        lifting_relative_error=_relative(realized-delta_z, delta_z),
        identity_scope='Native alpha=1 before BF16; positive rescaling generally changes the target profile',
        preservation_scope='Original probe filler profiles, not verified LM lexical identity')
    for name, states in (('float32', after_float32), ('bfloat16', after_bfloat16)):
        after_z = probe.binding((states.double()-center)/scale)
        after_channels = torch.einsum('ri,bij->brj', probe.R, after_z)
        after_logits = torch.einsum('brj,fj->brf', after_channels, probe.F)
        if name == 'float32':
            torch.testing.assert_close(after_channels, target_channels, rtol=2e-5, atol=2e-6,
                                       msg='Float32 edited states do not swap native role channels')
        diagnostics.update({
            name+'_channel_swap_relative_error': _relative(after_channels-target_channels, target_channels),
            name+'_logit_swap_relative_error': _relative(after_logits-target_logits, target_logits),
            name+'_logit_swap_max_abs_error': float((after_logits-target_logits).abs().max()),
            name+'_mean_role_logit_relative_error': _relative(after_logits.mean(1)-logits.mean(1), logits.mean(1)),
        })
    norms = delta.double().norm(dim=-1)
    diagnostics.update(delta_norm_min=float(norms.min()), delta_norm_mean=float(norms.mean()),
                       delta_norm_max=float(norms.max()), zero_delta_tokens=int((norms==0).sum()),
                       near_zero_delta_tokens=int((norms<=NEAR_ZERO_NORM).sum()),
                       near_zero_threshold=NEAR_ZERO_NORM)
    return delta, diagnostics
