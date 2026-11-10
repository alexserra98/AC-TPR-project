"""Exercise split isolation, atomic checkpoint reuse, and train-only calibration."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
import native_all_layers_training as training


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def experiment(path, heldout_offset=0.):
    path.mkdir()
    (path/'extraction').mkdir()
    rows = [dict(split=split, quad_id=f'q{index}', voice=voice, swapped=str(swapped),
                 agent='Anna' if swapped == 0 else 'David',
                 patient='David' if swapped == 0 else 'Anna')
            for index, split in enumerate(('train', 'val', 'test', 'gen_test'))
            for swapped in (0, 1) for voice in ('active', 'passive')]
    # Held-out vocabulary is intentionally absent from training.
    for row in rows[8:]:
        row['agent'] = 'UnseenAgent'
        row['patient'] = 'UnseenPatient'
    generator = torch.Generator().manual_seed(123)
    states = torch.randn(len(rows), 1, 2, 8, generator=generator)
    states[8:] += heldout_offset
    source = dict(states=states, rows=rows, layers=[0])
    torch.save(source, path/'extraction/states.pt')
    plan = dict(layers=[0], hidden_size=8, corpus_sha256='toy-corpus',
                expected_counts=dict(train=4, val=4, test=4, gen_test=4),
                probe=dict(seeds=[42, 43], epochs=3, patience=2, learning_rate=.01, batch_size=4),
                soft=dict(architecture=dict(input_dim=8, hidden_dim=12, num_roles=8,
                                             role_dim=16, filler_dim=32, num_codes=8),
                          training=dict(epochs=2, batch_size=4, learning_rate=.001,
                                        form_weight=1., commitment_weight=.25, seeds=[42, 43])))
    training.atomic_json(path/'plan.json', plan)
    training.atomic_json(path/'extraction/metadata.json', dict(
        status='complete', layers=[0], corpus_sha256='toy-corpus',
        states_sha256=training.digest(path/'extraction/states.pt')))
    return SimpleNamespace(run_dir=path, training_device='cpu'), plan, source


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def test_interruption_resume_normalization_centroids_and_tamper(tmp_path, monkeypatch):
    args, plan, source = experiment(tmp_path/'run')
    original_soft = training._train_soft

    def interrupt(*args, **kwargs):
        raise RuntimeError('simulated interruption')

    monkeypatch.setattr(training, '_train_soft', interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        training.train(args, plan)
    directory = args.run_dir/'training'
    assert (directory/'tpr/layer_0.complete.json').is_file()
    assert not (directory/'soft/layer_0.complete.json').exists()
    assert not (directory/'soft/layer_0.pt').exists()
    assert json.loads((directory/'metadata.json').read_text())['status'] == 'running'

    def no_retrain(*args, **kwargs):
        raise AssertionError('Completed TPR layer was fitted again')

    monkeypatch.setattr(training, '_train_tpr', no_retrain)
    monkeypatch.setattr(training, '_train_soft', original_soft)
    training.train(args, plan)
    tpr = load(directory/'tpr/layer_0.pt')
    soft = load(directory/'soft/layer_0.pt')
    h = source['states'][:4, 0].flatten(0, 1)
    center, scale = training.normalization(h)
    for payload in (tpr, soft):
        torch.testing.assert_close(payload['center'], center, rtol=0, atol=0)
        torch.testing.assert_close(payload['scale'], scale, rtol=0, atol=0)
    assert tpr['fillers'] == ['Anna', 'David']
    for voice in ('active', 'passive'):
        ids = [i for i, row in enumerate(source['rows'][:4]) if row['voice'] == voice]
        torch.testing.assert_close(tpr['raw_means'][voice], source['states'][ids, 0].mean(0))
    for method in ('tpr', 'soft'):
        with (directory/method/'layer_0_metrics.csv').open() as handle:
            assert {row['split'] for row in csv.DictReader(handle)} == {'train', 'val'}
    assert json.loads((directory/'metadata.json').read_text())['status'] == 'complete'
    monkeypatch.setattr(training, '_train_soft', no_retrain)
    before = training.digest(directory/'metadata.json')
    training.train(args, plan)
    assert training.digest(directory/'metadata.json') == before
    with (directory/'tpr/layer_0.pt').open('ab') as handle:
        handle.write(b'tampered')
    with pytest.raises(ValueError, match='artifact changed'):
        training.train(args, plan)


def test_heldout_values_cannot_change_fitted_checkpoints(tmp_path):
    first, plan, _ = experiment(tmp_path/'first')
    second, second_plan, _ = experiment(tmp_path/'second', heldout_offset=10000.)
    training.train(first, plan)
    training.train(second, second_plan)
    for method in ('tpr', 'soft'):
        a = load(first.run_dir/f'training/{method}/layer_0.pt')
        b = load(second.run_dir/f'training/{method}/layer_0.pt')
        assert a['seed'] == b['seed'] and a['best_epoch'] == b['best_epoch']
        for key in ('center', 'scale'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for key in a['state_dict']:
            torch.testing.assert_close(a['state_dict'][key], b['state_dict'][key], rtol=0, atol=0)


def test_calibration_uses_only_train_pairs_and_resumes(tmp_path, monkeypatch):
    args, plan, source = experiment(tmp_path/'run')
    training.train(args, plan)
    original = training.encode_native_soft
    seen = []

    def encode(payload, h):
        seen.append(h.clone())
        return original(payload, h)

    monkeypatch.setattr(training, 'encode_native_soft', encode)
    training.calibrate(args, plan)
    assert len(seen) == 1
    torch.testing.assert_close(seen[0], source['states'][:4, 0].flatten(0, 1))
    out = args.run_dir/'calibration'
    metadata = json.loads((out/'metadata.json').read_text())
    assert metadata['status'] == 'complete' and metadata['split'] == 'train'
    assert metadata['training_tokens'] == 8 and metadata['vocabulary'] == ['Anna', 'David']
    pairing = json.loads((out/'train_pairing.json').read_text())
    assert pairing['row_ids'] == [0, 1, 2, 3]
    assert pairing['donor_token_local_ids'] == training.pair_rows(source['rows'], range(4)).tolist()
    permutation = json.loads((out/'layer_0.json').read_text())
    assert permutation['diagnostics']['candidate_count'] == 764
    before = training.digest(out/'metadata.json')
    training.calibrate(args, plan)
    assert len(seen) == 1
    assert training.digest(out/'metadata.json') == before


def test_changed_extraction_and_cross_split_quads_are_rejected(tmp_path):
    args, plan, source = experiment(tmp_path/'run')
    changed = deepcopy(source)
    changed['rows'][4]['quad_id'] = changed['rows'][0]['quad_id']
    torch.save(changed, args.run_dir/'extraction/states.pt')
    with pytest.raises(ValueError, match='states changed'):
        training.train(args, plan)
    metadata_path = args.run_dir/'extraction/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['states_sha256'] = training.digest(args.run_dir/'extraction/states.pt')
    training.atomic_json(metadata_path, metadata)
    with pytest.raises(ValueError, match='Quad overlap'):
        training.train(args, plan)
