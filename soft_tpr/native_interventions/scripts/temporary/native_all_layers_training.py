"""Train independent native probes and calibrate Soft slots across saved layers.

Each completed layer has a manifest binding its artifacts to the plan, extraction,
training settings, and sources. Incomplete layers are rebuilt; completed layers
are reused only after their inputs and every published artifact are verified.
"""

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace

import torch

from native_soft_interventions import encode_native_soft, select_slot_permutation
from run_native_tpr_nov6 import pair_rows
from soft_tpr_model import evaluate_autoencoder, fit_autoencoder
from tpr_nov6_probe import fit_probe, normalization, token_metrics
from tpr_probe import BilinearTPRProbe


ROOT = Path(__file__).resolve().parents[2]
ROLES = ('agent', 'patient')
VOICES = ('active', 'passive')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='.'+path.name,
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _csv(path, records):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in record.items()})


def _source(args, plan):
    run = Path(args.run_dir)
    if json.loads((run/'plan.json').read_text()) != plan:
        raise ValueError('Supplied plan differs from the saved plan')
    metadata_path = run/'extraction/metadata.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata['status'] != 'complete' or metadata['layers'] != plan['layers']:
        raise ValueError('Require a complete extraction of the planned layers')
    extraction_sha = digest(run/'extraction/states.pt')
    if extraction_sha != metadata['states_sha256']:
        raise ValueError('Extracted states changed after extraction')
    if metadata['corpus_sha256'] != plan['corpus_sha256']:
        raise ValueError('Extraction corpus differs from plan')
    sources = {str(path.relative_to(ROOT)): digest(path) for path in (
        Path(__file__).resolve(),
        *[Path(__file__).with_name(name) for name in (
            'tpr_probe.py', 'tpr_nov6_probe.py', 'soft_tpr_model.py',
            'native_soft_interventions.py', 'run_native_tpr_nov6.py')])}
    for name, sha in sources.items():
        if name in plan.get('source_sha256', {}) and plan['source_sha256'][name] != sha:
            raise ValueError(f'Training source differs from frozen plan: {name}')
    source = torch.load(run/'extraction/states.pt', map_location='cpu', weights_only=True, mmap=True)
    if source['layers'] != plan['layers']:
        raise ValueError('Extraction layer order differs from the plan')
    states, rows = source['states'], source['rows']
    if list(states.shape) != [len(rows), len(plan['layers']), 2, plan['hidden_size']]:
        raise ValueError('Unexpected extraction shape')
    if dict(Counter(row['split'] for row in rows)) != plan['expected_counts']:
        raise ValueError('Extraction split counts differ from the plan')
    quad_splits = {}
    for row in rows:
        if row['voice'] not in VOICES:
            raise ValueError('Unknown sentence voice')
        if quad_splits.setdefault(row['quad_id'], row['split']) != row['split']:
            raise ValueError('Quad overlap between experimental splits')
    inputs = dict(plan_sha256=digest(run/'plan.json'), extraction_sha256=extraction_sha,
                  extraction_metadata_sha256=digest(metadata_path), source_sha256=sources)
    return source, inputs


def _tokens(rows):
    tokens = [dict(filler=row[role], role=role, voice=row['voice'], split=row['split'])
              for row in rows for role in ROLES]
    fillers = sorted({row['filler'] for row in tokens if row['split'] == 'train'})
    lookup = {filler: i for i, filler in enumerate(fillers)}
    labels = torch.tensor([lookup.get(row['filler'], -1) for row in tokens])
    roles = torch.arange(2).repeat(len(rows))
    splits = {split: torch.tensor([i for i, row in enumerate(tokens) if row['split'] == split])
              for split in ('train', 'val')}
    if any(not len(ids) or (labels[ids] < 0).any() for ids in splits.values()):
        raise ValueError('Train/validation tokens require supported training fillers')
    return tokens, fillers, labels, roles, splits


def _completed(out, layer, inputs):
    marker = out/f'layer_{layer}.complete.json'
    if not marker.exists():
        return None
    value = json.loads(marker.read_text())
    if value.get('status') != 'complete' or value.get('layer') != layer or value.get('inputs') != inputs:
        raise ValueError(f'Completed layer inputs changed: {marker}')
    if not value.get('artifacts_sha256'):
        raise ValueError(f'Empty completed layer manifest: {marker}')
    for filename, sha in value['artifacts_sha256'].items():
        path = out/filename
        if path.parent != out or not path.is_file() or digest(path) != sha:
            raise ValueError(f'Completed layer artifact changed: {path}')
    return value


def _publish(work, out, layer, inputs, selection):
    files = sorted(work.iterdir())
    hashes = {path.name: digest(path) for path in files}
    for path in files:
        os.replace(path, out/path.name)
    completion = dict(status='complete', layer=layer, inputs=inputs,
                      selection=selection, artifacts_sha256=hashes)
    atomic_json(out/f'layer_{layer}.complete.json', completion)
    return completion


def _train_tpr(work, layer, h, tokens, fillers, labels, roles, splits, settings):
    train_ids, val_ids = splits['train'], splits['val']
    center, scale = normalization(h[train_ids])
    x = (h-center)/scale
    config = dict(hidden_dim=h.shape[-1], num_roles=2, num_fillers=len(fillers), role_dim=2, filler_dim=8)
    candidates = []
    for seed in settings['seeds']:
        print(f'TPR layer {layer}, seed {seed}', flush=True)
        torch.manual_seed(seed)
        model = BilinearTPRProbe(**config).cpu()
        fit = fit_probe(model, x[train_ids], labels[train_ids], roles[train_ids],
                        x[val_ids], labels[val_ids], roles[val_ids],
                        SimpleNamespace(**settings), seed, work/f'layer_{layer}_seed_{seed}.jsonl')
        candidates.append((fit['validation']['loss'], seed, model, fit))
    _, seed, model, fit = min(candidates, key=lambda value: (value[0], value[1]))
    raw = {}
    for voice in VOICES:
        means = []
        for role in ROLES:
            ids = [i for i in train_ids.tolist() if tokens[i]['voice'] == voice and tokens[i]['role'] == role]
            if not ids:
                raise ValueError('Require train tokens for each voice/role centroid')
            means.append(h[ids].mean(0))
        raw[voice] = torch.stack(means)
    payload = dict(config=config, state_dict=model.state_dict(), center=center, scale=scale,
                   raw_means=raw, fillers=fillers, seed=seed, best_epoch=fit['best_epoch'])
    torch.save(payload, work/f'layer_{layer}.pt')
    records = []
    model.eval()
    with torch.no_grad():
        for split, split_ids in splits.items():
            for voice in VOICES:
                ids = torch.tensor([i for i in split_ids.tolist() if tokens[i]['voice'] == voice])
                if len(ids):
                    records.append(dict(layer=layer, split=split, voice=voice,
                                        **token_metrics(model(x[ids]), roles[ids], labels[ids])))
    _csv(work/f'layer_{layer}_metrics.csv', records)
    selection = dict(layer=layer, seed=seed, best_epoch=fit['best_epoch'],
                     validation_loss=fit['validation']['loss'])
    atomic_json(work/f'layer_{layer}_selection.json', dict(
        **selection, criterion='Minimum validation observed-role filler cross entropy; ties smaller seed',
        candidates=[dict(seed=s, **result) for _, s, _, result in candidates]))
    return selection


def _train_soft(work, layer, h, tokens, splits, config, settings, device):
    candidates = []
    for seed in settings['seeds']:
        print(f'Soft TPR layer {layer}, seed {seed}', flush=True)
        started = time.monotonic()
        model, center, scale, history, epoch = fit_autoencoder(
            h[splits['train']], h[splits['val']], config,
            **{key: settings[key] for key in ('epochs', 'batch_size', 'learning_rate',
                                             'form_weight', 'commitment_weight')},
            seed=seed, device=device)
        validation = next(row for row in history if row['split'] == 'val' and row['epoch'] == epoch)
        model.cpu()
        _csv(work/f'layer_{layer}_seed_{seed}_history.csv', history)
        candidates.append((validation['reconstruction'], seed, model, center, scale, epoch,
                           validation, time.monotonic()-started))
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    _, seed, model, center, scale, epoch, validation, elapsed = min(
        candidates, key=lambda value: (value[0], value[1]))
    torch.save(dict(config=config, state_dict=model.state_dict(), center=center, scale=scale,
                    seed=seed, best_epoch=epoch), work/f'layer_{layer}.pt')
    records = []
    for split, split_ids in splits.items():
        for voice in VOICES:
            ids = torch.tensor([i for i in split_ids.tolist() if tokens[i]['voice'] == voice])
            if len(ids):
                metrics = evaluate_autoencoder(model, (h[ids]-center)/scale, 256, 'cpu',
                                               settings['form_weight'], settings['commitment_weight'])
                records.append(dict(layer=layer, split=split, voice=voice, tokens=len(ids), **metrics))
    _csv(work/f'layer_{layer}_metrics.csv', records)
    selection = dict(layer=layer, seed=seed, best_epoch=epoch,
                     validation_reconstruction=validation['reconstruction'])
    atomic_json(work/f'layer_{layer}_selection.json', dict(
        **selection, criterion='Minimum validation reconstruction MSE; ties smaller seed',
        candidates=[dict(seed=s, best_epoch=e, validation=v, elapsed_seconds=t)
                    for _, s, _, _, _, e, v, t in candidates]))
    return selection


def train(args, plan):
    source, inputs = _source(args, plan)
    tokens, fillers, labels, roles, splits = _tokens(source['rows'])
    out = Path(args.run_dir)/'training'
    out.mkdir(parents=True, exist_ok=True)
    device = str(args.training_device)
    if plan['soft']['architecture']['input_dim'] != plan['hidden_size']:
        raise ValueError('Soft architecture differs from extraction dimension')
    conditions = {
        'tpr': dict(**inputs, settings=plan['probe'], training_device='cpu'),
        'soft': dict(**inputs, settings=plan['soft'], training_device=device),
    }
    completed = {'tpr': [], 'soft': []}
    # Resume validation precedes metadata updates or any fitting.
    for method in completed:
        (out/method).mkdir(exist_ok=True)
        for layer in plan['layers']:
            _completed(out/method, layer, conditions[method])
    atomic_json(out/'metadata.json', dict(status='running', layers=plan['layers'], **inputs))
    for li, layer in enumerate(plan['layers']):
        h = None
        for method in completed:
            directory = out/method
            item = _completed(directory, layer, conditions[method])
            if item is None:
                if h is None:
                    h = source['states'][:, li].float().flatten(0, 1)
                    if not torch.isfinite(h[splits['train']]).all() or not torch.isfinite(h[splits['val']]).all():
                        raise ValueError('Nonfinite train/validation states')
                with tempfile.TemporaryDirectory(prefix=f'.layer_{layer}.', dir=directory) as tmp:
                    work = Path(tmp)
                    if method == 'tpr':
                        selection = _train_tpr(work, layer, h, tokens, fillers, labels, roles,
                                               splits, plan['probe'])
                    else:
                        selection = _train_soft(work, layer, h, tokens, splits,
                                                plan['soft']['architecture'], plan['soft']['training'], device)
                    item = _publish(work, directory, layer, conditions[method], selection)
            else:
                print(f'Verified completed {method} layer {layer}', flush=True)
            completed[method].append(item)
    checkpoints = {f'{method}/layer_{item["layer"]}.pt': item['artifacts_sha256'][f'layer_{item["layer"]}.pt']
                   for method, items in completed.items() for item in items}
    atomic_json(out/'metadata.json', dict(
        status='complete', layers=plan['layers'], fillers=fillers, **inputs,
        training_device={'tpr': 'cpu', 'soft': device}, checkpoint_sha256=checkpoints,
        selected={method: [item['selection'] for item in items] for method, items in completed.items()},
        fitting_split='train', checkpoint_selection_split='val', metrics_splits=['train', 'val']))


def calibrate(args, plan):
    source, inputs = _source(args, plan)
    run = Path(args.run_dir)
    training = json.loads((run/'training/metadata.json').read_text())
    if (training['status'] != 'complete' or training['layers'] != plan['layers']
            or training['extraction_sha256'] != inputs['extraction_sha256']
            or training['plan_sha256'] != inputs['plan_sha256']):
        raise ValueError('Require complete matching probe training before calibration')
    ids = [i for i, row in enumerate(source['rows']) if row['split'] == 'train']
    donors = pair_rows(source['rows'], ids)
    if len(ids) != plan['expected_counts']['train']:
        raise ValueError('Unexpected calibration training size')
    out = run/'calibration'
    out.mkdir(parents=True, exist_ok=True)
    pairing = dict(split='train', row_ids=ids, donor_token_local_ids=donors.tolist())
    pairing_path = out/'train_pairing.json'
    if pairing_path.exists() and json.loads(pairing_path.read_text()) != pairing:
        raise ValueError('Calibration train pairing changed')
    atomic_json(pairing_path, pairing)
    inputs = dict(**inputs, train_pairing_sha256=digest(pairing_path),
                  training_metadata_sha256=digest(run/'training/metadata.json'))
    layer_inputs = {}
    for layer in plan['layers']:
        name = f'soft/layer_{layer}.pt'
        sha = digest(run/'training'/name)
        if sha != training['checkpoint_sha256'][name]:
            raise ValueError(f'Soft checkpoint changed before calibration: {name}')
        layer_inputs[layer] = dict(**inputs, checkpoint_sha256=sha)
        _completed(out, layer, layer_inputs[layer])
    atomic_json(out/'metadata.json', dict(status='running', layers=plan['layers'], **inputs))
    hashes = {}
    for li, layer in enumerate(plan['layers']):
        item = _completed(out, layer, layer_inputs[layer])
        if item is None:
            payload = torch.load(run/f'training/soft/layer_{layer}.pt', map_location='cpu', weights_only=True)
            h = source['states'][ids, li].float().flatten(0, 1)
            q, codes = encode_native_soft(payload, h)
            if q.shape[:2] != (2*len(ids), 8) or codes.shape != q.shape[:2]:
                raise ValueError('Require eight aligned native slots for every training token')
            permutation, diagnostics = select_slot_permutation(q, q[donors])
            with tempfile.TemporaryDirectory(prefix=f'.layer_{layer}.', dir=out) as tmp:
                work = Path(tmp)
                atomic_json(work/f'layer_{layer}.json', dict(
                    layer=layer, permutation=list(map(int, permutation)),
                    training_tokens=len(h), diagnostics=diagnostics))
                item = _publish(work, out, layer, layer_inputs[layer], dict(permutation=list(map(int, permutation))))
            print(f'Calibrated Soft layer {layer}: {list(map(int, permutation))}', flush=True)
        filename = f'layer_{layer}.json'
        hashes[filename] = item['artifacts_sha256'][filename]
    vocabulary = sorted({source['rows'][i][role] for i in ids for role in ROLES})
    atomic_json(out/'metadata.json', dict(
        status='complete', layers=plan['layers'], split='train', training_sentences=len(ids),
        training_tokens=2*len(ids), shared_across_voices=True, vocabulary=vocabulary,
        calibration_sha256=hashes,
        soft_checkpoint_sha256={f'soft/layer_{layer}.pt': layer_inputs[layer]['checkpoint_sha256']
                                for layer in plan['layers']}, **inputs))
