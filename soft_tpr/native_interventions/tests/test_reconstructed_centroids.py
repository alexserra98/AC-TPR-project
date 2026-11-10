"""Check projection semantics, quantized decoding, leakage, and edit direction."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/temporary'))
from reconstructed_centroids import METHODS, centroid_edits, fit_centroids, reconstruct_soft, reconstruct_tpr
from soft_tpr_model import SoftTPRAutoencoder
from tpr_probe import BilinearTPRProbe


def payloads():
    torch.manual_seed(9)
    config = dict(hidden_dim=11, num_roles=2, num_fillers=3, role_dim=2, filler_dim=3)
    probe = BilinearTPRProbe(**config)
    tpr = dict(config=config, state_dict=deepcopy(probe.state_dict()),
               center=torch.randn(11), scale=torch.tensor(2.3))
    config = dict(input_dim=11, hidden_dim=13, num_roles=8, role_dim=16, filler_dim=32, num_codes=9)
    model = SoftTPRAutoencoder(**config)
    with torch.no_grad():
        model.codebook.weight.mul_(.15)
    soft = dict(config=config, state_dict=deepcopy(model.state_dict()),
                center=torch.randn(11), scale=torch.tensor(3.2))
    return tpr, soft


def test_tpr_reconstruction_retains_binding_and_discards_only_nullspace():
    tpr, _ = payloads()
    h = torch.randn(12, 11)
    decoded = reconstruct_tpr(tpr, h)
    matrix = tpr['state_dict']['M'].double().flatten(1)
    torch.testing.assert_close((decoded - h.double()) @ matrix, torch.zeros(12, 6, dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(reconstruct_tpr(tpr, decoded), decoded, atol=1e-12, rtol=0)
    expected = (h[6:].double().mean(0) - h[:6].double().mean(0)) @ matrix @ torch.linalg.pinv(matrix)
    torch.testing.assert_close(decoded[6:].mean(0)-decoded[:6].mean(0), expected, atol=1e-12, rtol=0)


def test_soft_reconstruction_uses_quantized_decoder_and_raw_scale():
    _, soft = payloads()
    model = SoftTPRAutoencoder(**soft['config']).eval()
    model.load_state_dict(soft['state_dict'])
    h = torch.randn(9, 11)
    output = model((h-soft['center'])/soft['scale'])
    quantized = model.codebook(output['indices'])
    expected = soft['center'] + soft['scale'] * model.decoder(model.bind(quantized).flatten(1))
    torch.testing.assert_close(reconstruct_soft(soft, h), expected)


def test_centroids_ignore_heldout_states_and_use_voice_role_groups():
    tpr, soft = payloads()
    rows = [dict(split=split, voice=voice) for split in ('train', 'train', 'test', 'gen_test')
            for voice in ('active', 'passive')]
    source = dict(rows=rows, layers=[7], states=torch.randn(8, 1, 2, 11))
    fitted = fit_centroids(source, 7, tpr, soft)
    assert fitted['train_row_ids'] == [0, 1, 2, 3]
    poisoned = deepcopy(source)
    poisoned['states'][4:] = float('nan')
    other = fit_centroids(poisoned, 7, tpr, soft)
    for method, payload, reconstruct in zip(METHODS, (tpr, soft), (reconstruct_tpr, reconstruct_soft)):
        for voice, ids in [('active', [0, 2]), ('passive', [1, 3])]:
            decoded = reconstruct(payload, source['states'][ids, 0].flatten(0, 1)).reshape(2, 2, 11).double()
            torch.testing.assert_close(fitted['means'][method][voice], decoded.mean(0))
            torch.testing.assert_close(other['directions'][method][voice], fitted['directions'][method][voice], atol=0, rtol=0)
    edits = centroid_edits(fitted, [dict(voice='active'), dict(voice='passive'), dict(voice='active')])
    for method in METHODS:
        torch.testing.assert_close(edits[method][:, 0], -edits[method][:, 1], atol=0, rtol=0)
        torch.testing.assert_close(edits[method][0], edits[method][2], atol=0, rtol=0)
        torch.testing.assert_close(edits[method][0, 0], fitted['directions'][method]['active'])


def test_rank_deficient_probe_is_rejected():
    tpr, _ = payloads()
    tpr['state_dict']['M'].zero_()
    with pytest.raises(ValueError, match='rank'):
        reconstruct_tpr(tpr, torch.randn(2, 11))


def test_result_validation_rejects_missing_duplicates_and_final_block_changes():
    from evaluate_reconstructed_centroids import validate_results
    rows = [dict(split='test', voice='active', quad_id='a', agent='cat', patient='dog')]
    source = dict(rows=rows)
    baseline = torch.tensor([[2., 1.]])
    records = [dict(row_id=0, split='test', voice='active', quad_id='a', layer=35,
        method=method, alpha=1., known_tokens=2, sentence_has_oov=False,
        baseline_margin=-1., edited_margin=-1., margin_change=0., baseline_agent_win=True,
        patient_win=False, flip=False, agent_edit_norm=2., patient_edit_norm=2.,
        agent_edit_is_zero=False, patient_edit_is_zero=False, result_origin='scored') for method in METHODS]
    validate_results(records, source, [0], baseline, 35, {'cat', 'dog'})
    for malformed in (records[:1], records + records[:1]):
        with pytest.raises(ValueError, match='coverage'):
            validate_results(malformed, source, [0], baseline, 35, {'cat', 'dog'})
    changed = deepcopy(records)
    changed[0].update(edited_margin=-.5, margin_change=.5)
    with pytest.raises(ValueError, match='Final-block'):
        validate_results(changed, source, [0], baseline, 35, {'cat', 'dog'})
