"""Verify native role permutation, dual coordinates, lifting, and invariances."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/temporary'))
from native_tpr_interventions import native_role_swap
from tpr_probe import BilinearTPRProbe


def example():
    torch.manual_seed(29)
    config = dict(hidden_dim=11, num_roles=2, num_fillers=7, role_dim=2, filler_dim=3)
    probe = BilinearTPRProbe(**config).double()
    with torch.no_grad():
        probe.R.copy_(torch.tensor([[1., .4], [.2, 1.3]], dtype=torch.float64))
    payload = dict(config=config, state_dict=probe.state_dict(),
                   center=torch.randn(11, dtype=torch.float64), scale=torch.tensor(2.7, dtype=torch.float64))
    return payload, torch.randn(5, 11, dtype=torch.float64)


def binding(payload, h):
    return torch.einsum('bd,dij->bij', (h.double()-payload['center'])/payload['scale'],
                        payload['state_dict']['M'])


def logits(payload, h):
    return torch.einsum('ri,bij,fj->brf', payload['state_dict']['R'], binding(payload,h), payload['state_dict']['F'])


def test_full_filler_profile_is_transferred_to_other_role():
    payload, h = example()
    delta, diagnostics = native_role_swap(payload, h)
    assert delta.shape == h.shape and delta.dtype == torch.float32 and delta.device.type == 'cpu'
    expected = logits(payload, h).flip(1)
    torch.testing.assert_close(logits(payload, h+delta), expected, rtol=2e-6, atol=5e-7)
    torch.testing.assert_close(logits(payload, h+delta).mean(1), logits(payload,h).mean(1), rtol=2e-6, atol=5e-7)
    assert diagnostics['role_rank'] == 2 and diagnostics['binding_rank'] == 6
    assert diagnostics['float32_logit_swap_relative_error'] < 1e-6
    assert diagnostics['bfloat16_logit_swap_relative_error'] > diagnostics['float32_logit_swap_relative_error']
    json.dumps(diagnostics, allow_nan=False)


def test_delta_equals_dual_role_outer_product_not_naive_primal_swap():
    payload, h = example()
    delta, _ = native_role_swap(payload,h)
    r, m = payload['state_dict']['R'], payload['state_dict']['M']
    z = binding(payload,h)
    u = torch.einsum('ri,bij->brj',r,z)
    dual = torch.linalg.inv(r)
    target_delta = torch.einsum('i,bj->bij',dual[:,1]-dual[:,0],u[:,0]-u[:,1])
    realized = torch.einsum('bd,dij->bij',delta.double()/payload['scale'],m)
    torch.testing.assert_close(realized,target_delta,rtol=2e-6,atol=5e-7)
    naive_f = torch.einsum('i,bij->bj',r[0],z)/r[0].square().sum()
    naive_delta = torch.einsum('i,bj->bij',r[1]-r[0],naive_f)
    assert (naive_delta-target_delta).norm()/target_delta.norm() > .1


def test_invertible_nonorthogonal_coordinate_changes_preserve_raw_edit():
    payload,h = example()
    changed = deepcopy(payload)
    a = torch.tensor([[1.1,.6],[-.2,.8]],dtype=torch.float64)
    b = torch.tensor([[1.,.1,.2],[.3,1.2,0.],[0.,-.2,.9]],dtype=torch.float64)
    state = changed['state_dict']
    state['M'] = torch.einsum('ai,dij,bj->dab',a,state['M'],b)
    state['R'] = state['R']@torch.linalg.inv(a)
    state['F'] = state['F']@torch.linalg.inv(b)
    torch.testing.assert_close(logits(changed,h),logits(payload,h),rtol=1e-12,atol=1e-12)
    original,_ = native_role_swap(payload,h)
    transformed,_ = native_role_swap(changed,h)
    torch.testing.assert_close(transformed,original,rtol=1e-6,atol=1e-6)


def test_swap_is_an_involution_and_keeps_the_raw_nullspace_component():
    payload,h = example()
    first,_ = native_role_swap(payload,h)
    second,_ = native_role_swap(payload,h+first)
    torch.testing.assert_close(second,-first,rtol=2e-6,atol=1e-6)
    m = payload['state_dict']['M'].flatten(1)
    noise = torch.randn_like(h)
    null = noise-noise@m@torch.linalg.pinv(m)
    torch.testing.assert_close(first.double()@null.T,torch.zeros(5,5,dtype=torch.float64),rtol=0,atol=1e-6)
    shifted,_ = native_role_swap(payload,h+null)
    torch.testing.assert_close(shifted,first,rtol=2e-6,atol=1e-6)
    assert torch.all((first.double()+null).norm(dim=-1) >= first.double().norm(dim=-1))


def test_normalization_center_cancels_and_raw_scale_is_restored():
    payload,h = example()
    changed = deepcopy(payload)
    changed['center'] = payload['center']+3.
    changed['scale'] = payload['scale']*2.
    h_changed = changed['center']+2.*(h-payload['center'])
    original,_ = native_role_swap(payload,h)
    scaled,_ = native_role_swap(changed,h_changed)
    torch.testing.assert_close(scaled,2.*original,rtol=2e-6,atol=1e-6)


def test_input_checkpoint_and_rng_are_unchanged():
    payload,h = example()
    before = deepcopy(payload)
    h_before = h.clone()
    rng = torch.random.get_rng_state().clone()
    native_role_swap(payload,h)
    torch.testing.assert_close(torch.random.get_rng_state(),rng,rtol=0,atol=0)
    torch.testing.assert_close(h,h_before,rtol=0,atol=0)
    for key,value in before['state_dict'].items():
        torch.testing.assert_close(payload['state_dict'][key],value,rtol=0,atol=0)
    torch.testing.assert_close(payload['center'],before['center'],rtol=0,atol=0)
    torch.testing.assert_close(payload['scale'],before['scale'],rtol=0,atol=0)


@pytest.mark.parametrize('which', ['role','binding','near_role','near_binding'])
def test_singular_or_effectively_singular_maps_are_rejected(which):
    payload,h = example()
    state = payload['state_dict']
    if which == 'role':
        state['R'][1] = state['R'][0]
    elif which == 'binding':
        state['M'][:,:,2] = 0.
    elif which == 'near_role':
        state['R'][1] *= 1e-14
    else:
        state['M'][:,:,2] *= 1e-14
    with pytest.raises(ValueError,match='rank|singular'):
        native_role_swap(payload,h)


def test_exact_and_near_zero_swaps_are_valid_without_normalizing_them():
    config = dict(hidden_dim=4,num_roles=2,num_fillers=2,role_dim=2,filler_dim=2)
    probe = BilinearTPRProbe(**config).double()
    with torch.no_grad():
        probe.M.copy_(torch.eye(4,dtype=torch.float64).reshape(4,2,2))
        probe.R.copy_(torch.eye(2,dtype=torch.float64))
        probe.F.copy_(torch.eye(2,dtype=torch.float64))
    payload = dict(config=config,state_dict=probe.state_dict(),center=torch.zeros(4,dtype=torch.float64),scale=1.)
    h = torch.tensor([[1.,2.,1.,2.],[1.,2.,1.+1e-13,2.]],dtype=torch.float64)
    delta,diagnostics = native_role_swap(payload,h)
    torch.testing.assert_close(delta[0],torch.zeros(4),rtol=0,atol=0)
    assert 0 < delta[1].norm() < 1e-12
    assert diagnostics['zero_delta_tokens'] == 1 and diagnostics['near_zero_delta_tokens'] == 2
    json.dumps(diagnostics,allow_nan=False)
