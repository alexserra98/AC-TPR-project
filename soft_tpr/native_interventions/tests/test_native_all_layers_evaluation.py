"""Exercise all-layer scoring, final-block controls, and resumable provenance."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/temporary'))
import native_all_layers_evaluation as evaluation


class Inputs(dict):
    def to(self, device):
        return self


class Tokenizer:
    def pad(self, values, **kwargs):
        ids = torch.tensor(values['input_ids'])
        return Inputs(input_ids=ids, attention_mask=torch.ones_like(ids))


def fixture_run(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, 'LAYERS', [0, 1])
    monkeypatch.setattr(evaluation, 'HIDDEN_SIZE', 4)
    counts = dict(train=2, val=2, test=2, gen_test=2)
    monkeypatch.setattr(evaluation, 'EXPECTED_COUNTS', counts)
    for name in ('extraction', 'training/tpr', 'training/soft', 'calibration'):
        (tmp_path / name).mkdir(parents=True)
    rows = [dict(split=split, voice=voice, quad_id=f'{split}-{voice}', agent='cat',
                 patient='dog' if split != 'gen_test' else 'horse')
            for split in counts for voice in evaluation.VOICES]
    source = dict(rows=rows, layers=[0, 1], states=torch.ones(8, 2, 2, 4, dtype=torch.bfloat16),
                  baseline=torch.tensor([[i + .125, .25] for i in range(8)]),
                  input_ids=[[i, 1, 2, 3] for i in range(8)],
                  positions=torch.tensor([[0, 1]] * 8), answers=torch.tensor([[1, 2]] * 8))
    torch.save(source, tmp_path / 'extraction/states.pt')
    corpus = tmp_path / 'corpus.csv'
    corpus.write_text('test corpus; mocked loader\n')
    monkeypatch.setattr(evaluation, 'load_corpus', lambda path: rows)
    plan = dict(layers=[0, 1], methods=list(evaluation.METHODS), alphas=[1.], fixed_alpha=1.,
                norm_matching=False, batch_size=32, hidden_size=4, expected_counts=counts,
                model=evaluation.TOKENIZER_ID, revision=evaluation.TOKENIZER_REVISION,
                source_sha256={}, corpus=str(corpus), corpus_sha256=evaluation.sha256(corpus),
                prompt_template='{sentence}', demonstrations=[])
    evaluation.atomic_json(tmp_path / 'plan.json', plan)
    provenance = dict(name=plan['model'], revision=plan['revision'], commit=plan['revision'])
    evaluation.atomic_json(tmp_path / 'extraction/metadata.json', dict(status='complete', layers=[0, 1],
        corpus_sha256=plan['corpus_sha256'], batch_size=32, prompt_template='{sentence}', demonstrations=[],
        states_sha256=evaluation.sha256(tmp_path / 'extraction/states.pt'), model=provenance))
    hashes = {}
    for kind in ('tpr', 'soft'):
        for layer in (0, 1):
            payload = dict(fillers=['cat', 'dog'], raw_means={voice: torch.tensor([[0.] * 4, [1.] * 4])
                for voice in evaluation.VOICES})
            path = tmp_path / f'training/{kind}/layer_{layer}.pt'
            torch.save(payload, path)
            hashes[f'{kind}/layer_{layer}.pt'] = evaluation.sha256(path)
    evaluation.atomic_json(tmp_path / 'training/metadata.json', dict(status='complete', layers=[0, 1],
        checkpoint_sha256=hashes, plan_sha256=evaluation.sha256(tmp_path / 'plan.json'),
        extraction_sha256=evaluation.sha256(tmp_path / 'extraction/states.pt'), fillers=['cat', 'dog']))
    calibration_hashes = {}
    for layer in (0, 1):
        path = tmp_path / f'calibration/layer_{layer}.json'
        evaluation.atomic_json(path, dict(layer=layer, permutation=list(range(8)), training_tokens=4,
                                         diagnostics={'tokens': 4}))
        calibration_hashes[path.name] = evaluation.sha256(path)
    pairing_path = tmp_path / 'calibration/train_pairing.json'
    evaluation.atomic_json(pairing_path, dict(split='train', row_ids=[0, 1], donor_token_local_ids=[1, 0, 3, 2]))
    evaluation.atomic_json(tmp_path / 'calibration/metadata.json', dict(status='complete', layers=[0, 1],
        split='train', training_sentences=2, training_tokens=4, shared_across_voices=True,
        vocabulary=['cat', 'dog'], calibration_sha256=calibration_hashes,
        train_pairing_sha256=evaluation.sha256(pairing_path),
        plan_sha256=evaluation.sha256(tmp_path / 'plan.json'),
        extraction_sha256=evaluation.sha256(tmp_path / 'extraction/states.pt'),
        training_metadata_sha256=evaluation.sha256(tmp_path / 'training/metadata.json'),
        soft_checkpoint_sha256={name: value for name, value in hashes.items() if name.startswith('soft/')}))
    calls, loads = [], []
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2, hidden_size=4), device='cpu')

    def load_model():
        loads.append(True)
        return Tokenizer(), model, provenance

    def score(model, inputs, answers, layer=None, positions=None, deltas=None):
        indices = inputs['input_ids'][:, 0]
        values = torch.tensor([[float(i) + .125, .25] for i in indices])
        calls.append(dict(layer=layer, row_ids=indices.tolist(), deltas=None if deltas is None else deltas.clone()))
        if layer == 0:
            values[:, 1] += deltas[:, 0].sum(-1)
        return values

    monkeypatch.setattr(evaluation, 'load_lm', load_model)
    monkeypatch.setattr(evaluation, 'score_batch', score)
    monkeypatch.setattr(evaluation, 'native_role_swap', lambda checkpoint, h: (torch.full_like(h, .125), {'tokens': len(h)}))
    monkeypatch.setattr(evaluation, 'native_slot_swap', lambda checkpoint, h, permutation: (torch.zeros_like(h), {'tokens': len(h)}))
    return SimpleNamespace(run_dir=tmp_path, batch_size=32), plan, source, calls, loads


def test_scores_all_methods_final_block_and_identity_condition_then_resumes(tmp_path, monkeypatch):
    args, plan, source, calls, loads = fixture_run(tmp_path, monkeypatch)
    metadata = evaluation.evaluate(args, plan)
    assert metadata['status'] == 'complete' and metadata['result_rows'] == 24
    assert metadata['rows'] == 4 and metadata['conditions'] == 6
    assert metadata['layers'] == [0, 1] and len(loads) == 1
    assert all(call['row_ids'] == [4, 5, 6, 7] for call in calls)
    final_calls = [call for call in calls if call['layer'] == 1]
    assert len(final_calls) == 3
    assert sum(torch.count_nonzero(call['deltas']).item() == 0 for call in final_calls) == 1
    results = evaluation.read_csv(tmp_path / 'evaluation/results.csv')
    assert {r['result_origin'] for r in results} == {'scored', 'exact_identity'}
    identity = [r for r in results if r['result_origin'] == 'exact_identity']
    assert len(identity) == 4 and {r['method'] for r in identity} == {'soft_native_slot_swap'}
    assert {r['layer'] for r in identity} == {'0'}
    assert all(float(r['margin_change']) == 0 for r in results if r['layer'] == '1')
    assert all(r['known_tokens'] == '1' and r['sentence_has_oov'] == 'True'
               for r in results if r['split'] == 'gen_test')
    original_hash = metadata['results_sha256']
    calls.clear()
    resumed = evaluation.evaluate(args, plan)
    assert len(loads) == 2 and [call['layer'] for call in calls] == [None, 0]
    assert resumed['results_sha256'] == original_hash


def test_final_block_must_be_scored_and_preserve_both_logits(tmp_path, monkeypatch):
    args, plan, source, calls, loads = fixture_run(tmp_path, monkeypatch)
    original = evaluation.score_batch

    def wrong_final(*arguments, **kwargs):
        scores = original(*arguments, **kwargs)
        if len(arguments) > 3 and arguments[3] == 1:
            # Preserve the margin while corrupting both logits: still fail.
            scores += 1.
        return scores

    monkeypatch.setattr(evaluation, 'score_batch', wrong_final)
    with pytest.raises(AssertionError, match='Final-block actual'):
        evaluation.evaluate(args, plan)
    assert not (tmp_path / 'evaluation/metadata.json').exists()
    assert not (tmp_path / 'evaluation/layer_1_metadata.json').exists()
    assert (tmp_path / 'evaluation/layer_0_metadata.json').exists()


def test_resume_rejects_changed_canonical_baseline_sentinel(tmp_path, monkeypatch):
    args, plan, source, calls, loads = fixture_run(tmp_path, monkeypatch)
    evaluation.evaluate(args, plan)
    original = evaluation.score_batch
    monkeypatch.setattr(evaluation, 'score_batch', lambda *args, **kwargs: original(*args, **kwargs) + .5)
    with pytest.raises(AssertionError, match='sentinel differs'):
        evaluation.evaluate(args, plan)


def test_resume_rejects_modified_layer_file(tmp_path, monkeypatch):
    args, plan, source, calls, loads = fixture_run(tmp_path, monkeypatch)
    evaluation.evaluate(args, plan)
    path = tmp_path / 'evaluation/layer_0.csv'
    path.write_text(path.read_text().replace('scored', 'reference_raw'))
    with pytest.raises(ValueError, match='Artifact hash changed'):
        evaluation.evaluate(args, plan)


def test_incomplete_layer_is_recomputed_without_reusing_partial_results(tmp_path, monkeypatch):
    args, plan, source, calls, loads = fixture_run(tmp_path, monkeypatch)
    evaluation.evaluate(args, plan)
    (tmp_path / 'evaluation/layer_0_metadata.json').unlink()
    (tmp_path / 'evaluation/layer_0.csv').write_text('interrupted partial result\n')
    calls.clear()
    evaluation.evaluate(args, plan)
    assert [call['layer'] for call in calls] == [None, 0, 0, 0]
    assert len(evaluation.read_csv(tmp_path / 'evaluation/results.csv')) == 24


@pytest.mark.parametrize('mutation,match', [
    ('nonfinite', 'Nonfinite extraction states'),
    ('split_overlap', 'Quad overlap'),
    ('train_pairing', 'outside the training split'),
    ('missing_checkpoint', 'Training does not cover'),
    ('wrong_layers', 'Evaluation settings differ'),
])
def test_rejects_invalid_lineage_before_loading_model(tmp_path, monkeypatch, mutation, match):
    args, plan, source, calls, loads = fixture_run(tmp_path, monkeypatch)
    if mutation in ('nonfinite', 'split_overlap'):
        if mutation == 'nonfinite':
            source['states'][0, 0, 0, 0] = float('nan')
        else:
            source['rows'][4]['quad_id'] = source['rows'][0]['quad_id']
        torch.save(source, tmp_path / 'extraction/states.pt')
        new_hash = evaluation.sha256(tmp_path / 'extraction/states.pt')
        for relative, key in (('extraction/metadata.json', 'states_sha256'),
                              ('training/metadata.json', 'extraction_sha256')):
            path = tmp_path / relative
            metadata = evaluation.read_json(path)
            metadata[key] = new_hash
            evaluation.atomic_json(path, metadata)
        path = tmp_path / 'calibration/metadata.json'
        metadata = evaluation.read_json(path)
        metadata['extraction_sha256'] = new_hash
        metadata['training_metadata_sha256'] = evaluation.sha256(tmp_path / 'training/metadata.json')
        evaluation.atomic_json(path, metadata)
    elif mutation == 'train_pairing':
        path = tmp_path / 'calibration/train_pairing.json'
        pairing = evaluation.read_json(path)
        pairing['row_ids'] = [4, 5]
        evaluation.atomic_json(path, pairing)
        metadata_path = tmp_path / 'calibration/metadata.json'
        metadata = evaluation.read_json(metadata_path)
        metadata['train_pairing_sha256'] = evaluation.sha256(path)
        evaluation.atomic_json(metadata_path, metadata)
    elif mutation == 'missing_checkpoint':
        path = tmp_path / 'training/metadata.json'
        metadata = evaluation.read_json(path)
        del metadata['checkpoint_sha256']['soft/layer_1.pt']
        evaluation.atomic_json(path, metadata)
    else:
        plan['layers'] = [0]
        evaluation.atomic_json(tmp_path / 'plan.json', plan)
    with pytest.raises(ValueError, match=match):
        evaluation.evaluate(args, plan)
    assert not loads


@pytest.mark.parametrize('mutation,match', [
    ('duplicate', 'coverage'), ('missing', 'coverage'), ('nonfinite', 'Nonfinite'),
    ('old_reference', 'Older-run'), ('false_identity', 'Exact identity'), ('oov', 'vocabulary'),
])
def test_result_validation_rejects_bad_rows(tmp_path, monkeypatch, mutation, match):
    args, plan, source, calls, loads = fixture_run(tmp_path, monkeypatch)
    evaluation.evaluate(args, plan)
    records = evaluation.read_csv(tmp_path / 'evaluation/layer_0.csv')
    if mutation == 'duplicate':
        records.append(deepcopy(records[0]))
    elif mutation == 'missing':
        records.pop()
    elif mutation == 'nonfinite':
        records[0]['edited_margin'] = 'nan'
    elif mutation == 'old_reference':
        records[0]['result_origin'] = 'reference_raw'
    elif mutation == 'false_identity':
        records[0]['result_origin'] = 'exact_identity'
    else:
        records[0]['known_tokens'] = '0'
    baseline = evaluation.validate_baseline(evaluation.read_csv(tmp_path / 'evaluation/baseline.csv'), source, [4, 5, 6, 7])
    with pytest.raises(ValueError, match=match):
        evaluation.validate_results(records, source, [4, 5, 6, 7], baseline, 0, {'cat', 'dog'})


def test_nonfinite_native_edits_rejected(tmp_path, monkeypatch):
    args, plan, source, calls, loads = fixture_run(tmp_path, monkeypatch)
    monkeypatch.setattr(evaluation, 'native_role_swap',
                        lambda checkpoint, h: (torch.full_like(h, float('inf')), {}))
    with pytest.raises(ValueError, match='nonfinite'):
        evaluation.evaluate(args, plan)
    assert not (tmp_path / 'evaluation/layer_0_metadata.json').exists()


def test_canonical_baseline_and_final_controls_with_real_tiny_qwen(tmp_path, monkeypatch, model, tokenizer):
    monkeypatch.setattr(evaluation, 'LAYERS', [0, 1])
    rows = [dict(split='test', voice=voice, agent='cat', patient='dog')
            for voice in ('active', 'passive')]
    source = dict(rows=rows, input_ids=[[10, 11, 12, 13], [14, 15, 16, 17]],
                  positions=torch.tensor([[0, 1], [0, 1]]), answers=torch.tensor([[10, 11], [14, 15]]))
    inputs, answers, positions = evaluation._batch(source, [0, 1], tokenizer, model)
    assert inputs['input_ids'].device == model.device
    source['baseline'] = evaluation.score_batch(model, inputs, answers)
    actual, first_hash = evaluation.canonical_baseline(tmp_path, source, [0, 1], tokenizer, model, {}, {}, 32)
    torch.testing.assert_close(actual, source['baseline'], rtol=0, atol=0)
    deltas = torch.randn(2, 2, model.config.hidden_size, generator=torch.Generator().manual_seed(17))
    final = evaluation.score_batch(model, inputs, answers, 1, positions, deltas)
    torch.testing.assert_close(final, actual, rtol=0, atol=0)
    early = evaluation.score_batch(model, inputs, answers, 0, positions, deltas)
    assert not torch.equal(early, actual)
    resumed, second_hash = evaluation.canonical_baseline(tmp_path, source, [0, 1], tokenizer, model, {}, {}, 32)
    torch.testing.assert_close(resumed, actual, rtol=0, atol=0)
    assert first_hash == second_hash
