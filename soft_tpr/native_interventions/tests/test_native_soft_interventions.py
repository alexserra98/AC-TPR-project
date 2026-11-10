"""Native slot matching, code preservation, affine lifting, and input isolation."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/temporary'))
from native_soft_interventions import encode_native_soft, native_slot_swap, select_slot_permutation
from soft_tpr_model import SoftTPRAutoencoder


def example():
    torch.manual_seed(71)
    config = dict(input_dim=11, hidden_dim=13, num_roles=8, role_dim=16, filler_dim=32, num_codes=9)
    model = SoftTPRAutoencoder(**config)
    with torch.no_grad():
        # Similar code scale to the encoder makes more than one code active.
        model.codebook.weight.mul_(0.15)
    payload = dict(config=config, state_dict=deepcopy(model.state_dict()),
                   center=torch.randn(11), scale=torch.tensor(2.7))
    return payload, torch.randn(7, 11)


def model_from(payload):
    model = SoftTPRAutoencoder(**payload['config']).eval()
    model.load_state_dict(payload['state_dict'])
    return model


def test_encode_matches_frozen_forward_and_returns_native_codebook_values():
    payload, h = example()
    q, codes = encode_native_soft(payload, h, batch_size=3)
    model = model_from(payload)
    expected = model((h - payload['center']) / payload['scale'])
    assert q.shape == (7, 8, 32) and q.dtype == torch.float32 and q.device.type == 'cpu'
    assert codes.shape == (7, 8) and codes.dtype == torch.long
    assert torch.equal(codes, expected['indices'])
    assert torch.equal(q, model.codebook(codes))
    assert not q.requires_grad


def test_native_permutation_preserves_codes_and_is_a_latent_involution():
    payload, h = example()
    p = [1, 0, 2, 5, 4, 3, 6, 7]
    q, codes = encode_native_soft(payload, h)
    assert torch.equal(codes[:, p].sort(1).values, codes.sort(1).values)
    assert torch.equal(q[:, p][:, p], q)
    delta, diag = native_slot_swap(payload, h, p)
    model = model_from(payload)
    source = model.decoder(model.bind(q).flatten(1))
    target = model.decoder(model.bind(q[:, p]).flatten(1))
    torch.testing.assert_close(delta, payload['scale'] * (target - source), rtol=1e-5, atol=1e-6)
    assert delta.dtype == torch.float32 and delta.shape == h.shape and delta.device.type == 'cpu'
    assert diag['code_multiset_preserved'] and diag['latent_double_swap_exact']
    assert diag['moved_slots'] == 4
    json.dumps(diag, allow_nan=False)


def test_decoder_bias_cancels_and_raw_scale_is_restored():
    payload, h = example()
    p = [1, 0, 2, 3, 4, 5, 6, 7]
    original, _ = native_slot_swap(payload, h, p)
    assert original.norm() > 0
    changed = deepcopy(payload)
    changed['state_dict']['decoder.bias'].add_(1000)
    changed['center'] = payload['center'] + 5
    changed['scale'] = payload['scale'] * 3
    shifted_h = changed['center'] + 3 * (h - payload['center'])
    actual, _ = native_slot_swap(changed, shifted_h, p)
    torch.testing.assert_close(actual, 3 * original, rtol=1e-6, atol=1e-6)


def test_raw_edit_keeps_the_original_reconstruction_residual():
    payload, h = example()
    p = [0, 1, 3, 2, 4, 5, 6, 7]
    q, _ = encode_native_soft(payload, h)
    model = model_from(payload)
    source = payload['center'] + payload['scale'] * model.decoder(model.bind(q).flatten(1))
    target = payload['center'] + payload['scale'] * model.decoder(model.bind(q[:, p]).flatten(1))
    delta, _ = native_slot_swap(payload, h, p)
    torch.testing.assert_close(h + delta, target + h - source, rtol=1e-5, atol=2e-6)


def test_identity_and_equal_code_swaps_are_exactly_null():
    payload, h = example()
    for p in [list(range(8)), [1, 0, 2, 3, 4, 5, 6, 7]]:
        if p != list(range(8)):
            payload['state_dict']['codebook.weight'].zero_()
        delta, diag = native_slot_swap(payload, h, p)
        assert torch.count_nonzero(delta) == 0
        assert diag['zero_native_edit_fraction'] == 1
        assert diag['mean_hidden_edit_norm'] == 0


def test_selection_recovers_known_involution_and_reports_independent_loss():
    torch.manual_seed(8)
    q = torch.randn(5, 8, 32)
    expected = [1, 0, 2, 3, 5, 4, 7, 6]
    donor = q[:, expected].clone()
    permutation, diag = select_slot_permutation(q, donor)
    assert permutation == expected and diag['candidate_count'] == 764
    assert diag['tokens'] == 5 and diag['best']['objective_sse'] == 0
    assert diag['identity']['objective_sse'] == pytest.approx(float((q.double() - donor.double()).square().sum()))
    assert diag['relative_improvement'] == 1 and diag['gap'] > 0
    assert diag['cost_matrix'][1][0] == 0
    json.dumps(diag, allow_nan=False)


def test_identity_is_not_forced_to_be_a_nonzero_edit_and_ties_prefer_fewer_moves():
    q = torch.zeros(3, 8, 32)
    permutation, diag = select_slot_permutation(q, q)
    assert permutation == list(range(8)) and diag['tie_count'] == 764
    assert diag['best']['moved_slots'] == 0 and diag['relative_improvement'] == 0
    assert diag['second']['permutation'] == [0, 1, 2, 3, 4, 5, 7, 6]
    assert diag['second']['moved_slots'] == 2 and diag['gap'] == 0


def test_equal_nonidentity_objectives_break_ties_lexicographically():
    q = torch.zeros(1, 8, 32, dtype=torch.float64)
    q[0, :, 0] = torch.tensor([0, 1, 1, 10, 20, 30, 40, 50])
    donor = q.clone()
    donor[0, :3, 0] = torch.tensor([1, 0, 0])
    permutation, diag = select_slot_permutation(q, donor)
    assert permutation == [1, 0, 2, 3, 4, 5, 6, 7]
    assert diag['best']['objective_sse'] == 1 and diag['tie_count'] == 2


@pytest.mark.parametrize('epsilon,expected', [(1e-14, list(range(8))), (1e-8, [1, 0, 2, 3, 4, 5, 6, 7])])
def test_relative_tie_tolerance_is_applied_before_number_of_moved_slots(epsilon, expected):
    q = torch.zeros(1, 8, 32, dtype=torch.float64)
    q[0, :, 0] = torch.tensor([0, 1, 10, 20, 30, 40, 50, 60])
    donor = q.clone()
    donor[0, 0, 0], donor[0, 1, 0] = .5 + epsilon, .5 - epsilon
    permutation, diag = select_slot_permutation(q, donor)
    assert permutation == expected and diag['tie_rtol'] == 1e-12


def test_helpers_do_not_mutate_inputs_checkpoint_or_global_rng():
    payload, h = example()
    before, original_h = deepcopy(payload), h.clone()
    rng = torch.random.get_rng_state().clone()
    q, _ = encode_native_soft(payload, h)
    original_q = q.clone()
    donor = q[:, [1, 0, 2, 3, 4, 5, 6, 7]].clone()
    original_donor = donor.clone()
    permutation, _ = select_slot_permutation(q, donor)
    native_slot_swap(payload, h, permutation)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(h, original_h) and torch.equal(q, original_q) and torch.equal(donor, original_donor)
    for key in ['center', 'scale']:
        assert torch.equal(payload[key], before[key])
    for key in payload['state_dict']:
        assert torch.equal(payload['state_dict'][key], before['state_dict'][key])


@pytest.mark.parametrize('permutation', [[0] * 8, list(range(7)), [1, 2, 0, 3, 4, 5, 6, 7],
                                      [0, 1, 2, 3, 4, 5, 6, 8], [True, 0, 2, 3, 4, 5, 6, 7],
                                      [0., 1., 2., 3., 4., 5., 6., 7.]])
def test_invalid_or_noninvolutive_permutations_are_rejected(permutation):
    payload, h = example()
    with pytest.raises(ValueError, match='permutation'):
        native_slot_swap(payload, h, permutation)


@pytest.mark.parametrize('kind', ['empty', 'shape', 'nan', 'integer'])
def test_invalid_selection_inputs_are_rejected(kind):
    q = torch.zeros(2, 8, 32)
    if kind == 'empty':
        q = q[:0]
    elif kind == 'shape':
        q = q[:, :7]
    elif kind == 'nan':
        q[0, 0, 0] = float('nan')
    else:
        q = q.long()
    with pytest.raises(ValueError):
        select_slot_permutation(q, q)


@pytest.mark.parametrize('kind', ['scale', 'center', 'hidden', 'batch'])
def test_invalid_encoding_inputs_are_rejected(kind):
    payload, h = example()
    batch_size = 3
    if kind == 'scale':
        payload['scale'] = torch.tensor(0.)
    elif kind == 'center':
        payload['center'][0] = float('nan')
    elif kind == 'hidden':
        h = h[:, :-1]
    else:
        batch_size = 0
    with pytest.raises(ValueError):
        encode_native_soft(payload, h, batch_size)
