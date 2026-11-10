"""Score the frozen native intervention protocol at every Qwen transformer block."""

from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION, load_corpus
from ac_tpr.interventions import score_batch
from native_soft_interventions import native_slot_swap, _validate_permutation
from native_tpr_interventions import native_role_swap
from run_native_tpr_nov6 import METHODS, ROLES, VOICES, summarize
from run_tpr_nov6 import load_lm, torch_load


ROOT = Path(__file__).resolve().parents[2]
LAYERS = list(range(36))
HIDDEN_SIZE = 4096
EXPECTED_COUNTS = dict(train=3300, val=412, test=416, gen_test=1680)
BASELINE_COLUMNS = ['row_id', 'split', 'voice', 'agent', 'patient', 'agent_logit',
                    'patient_logit', 'extraction_margin']
RESULT_COLUMNS = ['row_id', 'split', 'voice', 'quad_id', 'layer', 'method', 'alpha',
                  'known_tokens', 'sentence_has_oov', 'baseline_margin', 'edited_margin',
                  'margin_change', 'baseline_agent_win', 'patient_win', 'flip',
                  'agent_edit_norm', 'patient_edit_norm', 'agent_edit_is_zero',
                  'patient_edit_is_zero', 'result_origin']


def sha256(path):
    """Hash tensors on disk without allocating another copy of their contents."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path, columns=None):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        if columns is not None and reader.fieldnames != columns:
            raise ValueError(f'Unexpected CSV schema: {path}')
        return list(reader)


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def atomic_csv(path, records, columns=None):
    path = Path(path)
    if not records:
        raise ValueError('Cannot write an empty results table')
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                     suffix='.tmp', newline='', delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=columns or list(records[0]))
            writer.writeheader()
            writer.writerows(records)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def _require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'Artifact hash changed: {path}')
    return actual


def validate_inputs(args, plan):
    """Check the complete all-layer lineage before loading the language model."""
    run = Path(args.run_dir)
    if (plan != read_json(run / 'plan.json') or plan['layers'] != LAYERS
            or list(plan['methods']) != list(METHODS) or plan['alphas'] != [1.]
            or plan['fixed_alpha'] != 1. or plan['norm_matching'] is not False
            or args.batch_size != 32 or plan['batch_size'] != args.batch_size
            or plan['hidden_size'] != HIDDEN_SIZE or plan['expected_counts'] != EXPECTED_COUNTS
            or plan['model'] != TOKENIZER_ID or plan['revision'] != TOKENIZER_REVISION):
        raise ValueError('Evaluation settings differ from the frozen all-layer protocol')
    for relative, expected in plan['source_sha256'].items():
        _require_hash(ROOT / relative, expected)
    _require_hash(plan['corpus'], plan['corpus_sha256'])
    lineage = dict(plan_sha256=sha256(run / 'plan.json'), source_sha256=plan['source_sha256'])
    extraction = read_json(run / 'extraction/metadata.json')
    if (extraction['status'] != 'complete' or extraction['layers'] != LAYERS
            or extraction['corpus_sha256'] != plan['corpus_sha256']
            or extraction['batch_size'] != args.batch_size
            or extraction['prompt_template'] != plan['prompt_template']
            or extraction['demonstrations'] != plan['demonstrations']):
        raise ValueError('Extraction is incomplete or uses a different protocol')
    lineage['extraction_sha256'] = _require_hash(run / 'extraction/states.pt', extraction['states_sha256'])
    lineage['extraction_metadata_sha256'] = sha256(run / 'extraction/metadata.json')
    training = read_json(run / 'training/metadata.json')
    checkpoints = {f'{kind}/layer_{layer}.pt' for kind in ('tpr', 'soft') for layer in LAYERS}
    if (training['status'] != 'complete' or training['layers'] != LAYERS
            or set(training['checkpoint_sha256']) != checkpoints
            or training['plan_sha256'] != lineage['plan_sha256']
            or training['extraction_sha256'] != lineage['extraction_sha256']):
        raise ValueError('Training does not cover the frozen extraction and all layers')
    for relative, expected in training['checkpoint_sha256'].items():
        _require_hash(run / 'training' / relative, expected)
    lineage['training_metadata_sha256'] = sha256(run / 'training/metadata.json')
    calibration = read_json(run / 'calibration/metadata.json')
    if (calibration['status'] != 'complete' or calibration['layers'] != LAYERS
            or calibration['split'] != 'train' or calibration['training_sentences'] != EXPECTED_COUNTS['train']
            or calibration['training_tokens'] != 2 * EXPECTED_COUNTS['train']
            or calibration['shared_across_voices'] is not True
            or set(calibration['calibration_sha256']) != {f'layer_{layer}.json' for layer in LAYERS}
            or calibration['plan_sha256'] != lineage['plan_sha256']
            or calibration['extraction_sha256'] != lineage['extraction_sha256']
            or calibration['training_metadata_sha256'] != lineage['training_metadata_sha256']
            or calibration['soft_checkpoint_sha256'] != {name: value for name, value in
                training['checkpoint_sha256'].items() if name.startswith('soft/')}):
        raise ValueError('Calibration must cover every layer using training rows only')
    for relative, expected in calibration['calibration_sha256'].items():
        _require_hash(run / 'calibration' / relative, expected)
        item = read_json(run / 'calibration' / relative)
        if item['layer'] not in LAYERS or relative != f"layer_{item['layer']}.json":
            raise ValueError('Calibration layer index differs from its filename')
        _validate_permutation(item['permutation'])
    _require_hash(run / 'calibration/train_pairing.json', calibration['train_pairing_sha256'])
    lineage['calibration_metadata_sha256'] = sha256(run / 'calibration/metadata.json')
    source = torch_load(run / 'extraction/states.pt')
    rows = source['rows']
    if rows != load_corpus(Path(plan['corpus'])) or dict(Counter(r['split'] for r in rows)) != EXPECTED_COUNTS:
        raise ValueError('Extraction rows or split coverage differ from the frozen corpus')
    if source['layers'] != LAYERS or source['states'].shape != (len(rows), len(LAYERS), 2, HIDDEN_SIZE):
        raise ValueError('Extraction must cover both noun tokens and every block')
    if source['states'].dtype != torch.bfloat16 or source['baseline'].shape != (len(rows), 2):
        raise ValueError('Extraction states must be BF16 with two baseline scores per row')
    if len(source['input_ids']) != len(rows) or any(r['voice'] not in VOICES for r in rows):
        raise ValueError('Extraction prompt or voice coverage differs')
    for name in ('positions', 'answers'):
        if source[name].shape != (len(rows), 2) or source[name].dtype != torch.long:
            raise ValueError(f'Extraction {name} must contain two integer indices per row')
    for index, tokens in enumerate(source['input_ids']):
        if (not tokens or any(type(token) is not int or token < 0 for token in tokens)
                or not (0 <= source['positions'][index]).all()
                or not (source['positions'][index] < len(tokens) - 1).all()
                or source['positions'][index, 0] == source['positions'][index, 1]
                or (source['answers'][index] < 0).any()):
            raise ValueError('Invalid token IDs or joint intervention positions')
    if not torch.isfinite(source['baseline']).all():
        raise ValueError('Nonfinite extraction scores')
    for batch in source['states'].split(64):
        if not torch.isfinite(batch).all():
            raise ValueError('Nonfinite extraction states')
    quads = {split: {r['quad_id'] for r in rows if r['split'] == split} for split in EXPECTED_COUNTS}
    if any(quads[a] & quads[b] for a in quads for b in quads if a != b):
        raise ValueError('Quad overlap between training, validation, and held-out splits')
    vocabulary = sorted({r[role] for r in rows if r['split'] == 'train' for role in ROLES})
    if calibration['vocabulary'] != vocabulary or training['fillers'] != vocabulary:
        raise ValueError('Vocabulary must come exclusively from training rows')
    pairing = read_json(run / 'calibration/train_pairing.json')
    train_ids = [i for i, row in enumerate(rows) if row['split'] == 'train']
    if pairing['split'] != 'train' or pairing['row_ids'] != train_ids:
        raise ValueError('Calibration pairing includes rows outside the training split')
    ids = [i for i, row in enumerate(rows) if row['split'] in ('test', 'gen_test')]
    return source, ids, set(vocabulary), lineage, training, calibration, extraction


def build_layer_edits(source, ids, layer, tpr, soft, permutation):
    """Keep only one block's three joint noun edits in memory."""
    rows = [source['rows'][i] for i in ids]
    h = source['states'][ids, source['layers'].index(layer)].float()
    raw = torch.empty_like(h)
    for voice in VOICES:
        means = tpr['raw_means'][voice]
        if means.shape != (2, h.shape[-1]) or not torch.isfinite(means).all():
            raise ValueError('Centroid means must be finite, with one vector per role')
        selected = [i for i, row in enumerate(rows) if row['voice'] == voice]
        delta = means[1].float() - means[0].float()
        raw[selected] = torch.stack((delta, -delta)).expand(len(selected), -1, -1)
    tpr_delta, tpr_checks = native_role_swap(tpr, h.flatten(0, 1))
    soft_delta, soft_checks = native_slot_swap(soft, h.flatten(0, 1), permutation)
    edits = dict(raw_centroid=raw, tpr_native_role_swap=tpr_delta.reshape(h.shape),
                 soft_native_slot_swap=soft_delta.reshape(h.shape))
    checks = []
    for method, delta in edits.items():
        if delta.shape != h.shape or not torch.isfinite(delta).all():
            raise ValueError('Invalid native edit shape or nonfinite values')
        norms = delta.double().norm(dim=-1)
        checks.append(dict(layer=layer, method=method, tokens=norms.numel(),
            native_norm_min=float(norms.min()), native_norm_mean=float(norms.mean()), native_norm_max=float(norms.max()),
            exact_zero_tokens=int((norms == 0).sum()), exact_zero_sentences=int((norms == 0).all(dim=1).sum()),
            diagnostics=tpr_checks if method == 'tpr_native_role_swap' else soft_checks if method == 'soft_native_slot_swap' else {}))
    return edits, checks


def _batch(source, ids, tokenizer, model):
    inputs = tokenizer.pad({'input_ids': [source['input_ids'][i] for i in ids]},
                           padding=True, return_tensors='pt').to(model.device)
    return inputs, source['answers'][ids], source['positions'][ids]


def _valid_scores(scores, size):
    if scores.shape != (size, 2) or scores.dtype != torch.float32 or not torch.isfinite(scores).all():
        raise ValueError('Expected finite float32 answer scores for every row')
    return scores


def validate_baseline(records, source, ids):
    if [int(r['row_id']) for r in records] != ids:
        raise ValueError('Baseline row IDs differ from ordered held-out rows')
    for record, index in zip(records, ids, strict=True):
        if any(record[key] != source['rows'][index][key] for key in ('split', 'voice', 'agent', 'patient')):
            raise ValueError('Baseline row labels differ from the extraction')
        if any(not math.isfinite(float(record[key])) for key in ('agent_logit', 'patient_logit', 'extraction_margin')):
            raise ValueError('Nonfinite baseline result')
        if float(record['extraction_margin']) != float(source['baseline'][index, 1] - source['baseline'][index, 0]):
            raise ValueError('Baseline extraction margins differ')
    return torch.tensor([[float(r['agent_logit']), float(r['patient_logit'])] for r in records], dtype=torch.float32)


def canonical_baseline(out, source, ids, tokenizer, model, provenance, lineage, batch_size):
    path, metadata_path = out / 'baseline.csv', out / 'baseline_metadata.json'
    expected = dict(**lineage, row_ids=ids, batch_size=batch_size, model=provenance)
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        if metadata['status'] != 'complete' or any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError('Cached baseline lineage differs from this run')
        _require_hash(path, metadata['baseline_sha256'])
        baseline = validate_baseline(read_csv(path, BASELINE_COLUMNS), source, ids)
        first = ids[:batch_size]
        inputs, answers, positions = _batch(source, first, tokenizer, model)
        sentinel = _valid_scores(score_batch(model, inputs, answers), len(first))
        torch.testing.assert_close(sentinel, baseline[:len(first)], rtol=0, atol=0,
                                   msg='Cached baseline first-batch sentinel differs exactly')
    else:
        records = []
        for start in range(0, len(ids), batch_size):
            chosen = ids[start:start + batch_size]
            inputs, answers, positions = _batch(source, chosen, tokenizer, model)
            scores = _valid_scores(score_batch(model, inputs, answers), len(chosen))
            for j, index in enumerate(chosen):
                row = source['rows'][index]
                records.append(dict(row_id=index, split=row['split'], voice=row['voice'], agent=row['agent'],
                    patient=row['patient'], agent_logit=float(scores[j, 0]), patient_logit=float(scores[j, 1]),
                    extraction_margin=float(source['baseline'][index, 1] - source['baseline'][index, 0])))
            if start % (16 * batch_size) == 0 or start + len(chosen) == len(ids):
                print(f'Canonical baseline: {start + len(chosen)}/{len(ids)} held-out rows', flush=True)
        baseline = validate_baseline(records, source, ids)
        atomic_csv(path, records, BASELINE_COLUMNS)
        atomic_json(metadata_path, dict(status='complete', **expected, baseline_sha256=sha256(path)))
    first = ids[:batch_size]
    inputs, answers, positions = _batch(source, first, tokenizer, model)
    zero = torch.zeros(len(first), 2, model.config.hidden_size)
    control = _valid_scores(score_batch(model, inputs, answers, LAYERS[0], positions, zero), len(first))
    torch.testing.assert_close(control, baseline[:len(first)], rtol=0, atol=0,
                               msg='Zero edits changed the canonical baseline')
    return baseline, sha256(path)


def _boolean(value):
    if value in (True, 'True'):
        return True
    if value in (False, 'False'):
        return False
    raise ValueError(f'Invalid result boolean: {value}')


def validate_results(records, source, ids, baseline, layer, vocabulary):
    """Reject partial, duplicate, nonfinite, or mislabeled resumed conditions."""
    expected = {(i, method) for i in ids for method in METHODS}
    seen = set()
    baseline_by_id = {i: float(baseline[j, 1] - baseline[j, 0]) for j, i in enumerate(ids)}
    for record in records:
        key = (int(record['row_id']), record['method'])
        if key not in expected or key in seen or int(record['layer']) != layer or float(record['alpha']) != 1.:
            raise ValueError('Missing, duplicate, or unexpected layer/method/row coverage')
        seen.add(key)
        row = source['rows'][key[0]]
        if any(record[name] != row[name] for name in ('split', 'voice', 'quad_id')):
            raise ValueError('Result row labels differ from the corpus')
        known = sum(row[role] in vocabulary for role in ROLES)
        if int(record['known_tokens']) != known or _boolean(record['sentence_has_oov']) != (known < 2):
            raise ValueError('Result vocabulary annotations differ')
        values = {name: float(record[name]) for name in ('baseline_margin', 'edited_margin', 'margin_change',
                                                        'agent_edit_norm', 'patient_edit_norm')}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError('Nonfinite intervention result')
        before, after = values['baseline_margin'], values['edited_margin']
        if before != baseline_by_id[key[0]] or values['margin_change'] != after - before:
            raise ValueError('Result baseline or margin change differs')
        flags = dict(baseline_agent_win=before < 0, patient_win=after > 0, flip=before < 0 and after > 0)
        for role in ROLES:
            norm = values[f'{role}_edit_norm']
            if norm < 0:
                raise ValueError('Edit norms must be nonnegative')
            flags[f'{role}_edit_is_zero'] = norm == 0
        if any(_boolean(record[name]) != value for name, value in flags.items()):
            raise ValueError('Result flags do not match scores and edit norms')
        origin = record['result_origin']
        if origin not in ('scored', 'exact_identity'):
            raise ValueError('Older-run results cannot be reused')
        if origin == 'exact_identity' and (after != before or values['agent_edit_norm'] or values['patient_edit_norm']):
            raise ValueError('Exact identity requires zero edits and unchanged scores')
        if layer == LAYERS[-1] and (after != before or origin != 'scored'):
            raise ValueError('Final block must actually be scored and preserve the baseline')
    if seen != expected:
        raise ValueError('Incomplete layer/method/row coverage')
    for method in METHODS:
        condition = [r for r in records if r['method'] == method]
        origins = {r['result_origin'] for r in condition}
        if 'exact_identity' in origins and origins != {'exact_identity'}:
            raise ValueError('Identity reuse requires an entirely zero condition')


def evaluate(args, plan):
    """Evaluate 108 conditions, resuming only complete, hash-verified layers."""
    source, ids, vocabulary, lineage, training, calibration, extraction = validate_inputs(args, plan)
    out = Path(args.run_dir) / 'evaluation'
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, model, provenance = load_lm()
    if (model.config.num_hidden_layers != len(LAYERS) or model.config.hidden_size != HIDDEN_SIZE
            or provenance != extraction['model']):
        raise ValueError('Loaded model differs from extraction provenance or dimensions')
    baseline, baseline_hash = canonical_baseline(out, source, ids, tokenizer, model, provenance,
                                                lineage, args.batch_size)
    all_checks, summaries, identity_conditions = [], [], []
    for layer in LAYERS:
        results_path, metadata_path = out / f'layer_{layer}.csv', out / f'layer_{layer}_metadata.json'
        dependencies = dict(**lineage, baseline_sha256=baseline_hash,
            checkpoint_sha256={kind: training['checkpoint_sha256'][f'{kind}/layer_{layer}.pt'] for kind in ('tpr', 'soft')},
            calibration_sha256=calibration['calibration_sha256'][f'layer_{layer}.json'],
            layer=layer, methods=list(METHODS), alpha=1., row_ids=ids, batch_size=args.batch_size, model=provenance)
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata['status'] != 'complete' or any(metadata.get(k) != v for k, v in dependencies.items()):
                raise ValueError(f'Completed layer {layer} has incompatible provenance')
            _require_hash(results_path, metadata['results_sha256'])
            records = read_csv(results_path, RESULT_COLUMNS)
            validate_results(records, source, ids, baseline, layer, vocabulary)
            if metadata['result_rows'] != len(records):
                raise ValueError('Completed layer result count differs')
            print(f'Evaluation layer {layer}: verified completed results', flush=True)
        else:
            tpr = torch_load(Path(args.run_dir) / f'training/tpr/layer_{layer}.pt')
            soft = torch_load(Path(args.run_dir) / f'training/soft/layer_{layer}.pt')
            if tpr['fillers'] != sorted(vocabulary):
                raise ValueError('Checkpoint vocabulary differs from training rows')
            permutation = read_json(Path(args.run_dir) / f'calibration/layer_{layer}.json')['permutation']
            edits, checks = build_layer_edits(source, ids, layer, tpr, soft, permutation)
            identity = [method for method in METHODS if torch.count_nonzero(edits[method]).item() == 0]
            records = []
            for start in range(0, len(ids), args.batch_size):
                chosen = ids[start:start + args.batch_size]
                inputs, answers, positions = _batch(source, chosen, tokenizer, model)
                before = baseline[start:start + len(chosen)]
                for method in METHODS:
                    delta = edits[method][start:start + len(chosen)]
                    norms = delta.double().norm(dim=-1)
                    if method in identity and layer != LAYERS[-1]:
                        scores, origin = before, 'exact_identity'
                    else:
                        scores = _valid_scores(score_batch(model, inputs, answers, layer, positions, delta), len(chosen))
                        origin = 'scored'
                    if layer == LAYERS[-1]:
                        torch.testing.assert_close(scores, before, rtol=0, atol=0,
                                                   msg=f'Final-block actual {method} edits changed answer logits')
                    for j, index in enumerate(chosen):
                        row = source['rows'][index]
                        bm, em = float(before[j, 1] - before[j, 0]), float(scores[j, 1] - scores[j, 0])
                        records.append(dict(row_id=index, split=row['split'], voice=row['voice'], quad_id=row['quad_id'],
                            layer=layer, method=method, alpha=1., known_tokens=sum(row[r] in vocabulary for r in ROLES),
                            sentence_has_oov=any(row[r] not in vocabulary for r in ROLES),
                            baseline_margin=bm, edited_margin=em, margin_change=em-bm,
                            baseline_agent_win=bm < 0, patient_win=em > 0, flip=bm < 0 and em > 0,
                            agent_edit_norm=float(norms[j, 0]), patient_edit_norm=float(norms[j, 1]),
                            agent_edit_is_zero=bool(norms[j, 0] == 0), patient_edit_is_zero=bool(norms[j, 1] == 0),
                            result_origin=origin))
                if start % (16 * args.batch_size) == 0 or start + len(chosen) == len(ids):
                    print(f'Evaluation layer {layer}: {start + len(chosen)}/{len(ids)} rows across 3 methods', flush=True)
            validate_results(records, source, ids, baseline, layer, vocabulary)
            atomic_csv(results_path, records, RESULT_COLUMNS)
            metadata = dict(status='complete', **dependencies, result_rows=len(records),
                results_sha256=sha256(results_path), native_checks=checks, exact_identity_methods=identity,
                final_block_control='passed for all methods and rows' if layer == LAYERS[-1] else 'not final block')
            atomic_json(metadata_path, metadata)
            del edits, tpr, soft
        all_checks.extend(metadata['native_checks'])
        identity_conditions.extend(dict(layer=layer, method=method) for method in metadata['exact_identity_methods'])
        summaries.extend(summarize(records))
    # Concatenate only verified complete layers; an interrupted run has no final metadata.
    with tempfile.NamedTemporaryFile(mode='w', dir=out, prefix='results.', suffix='.tmp',
                                     newline='', delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        result_rows = 0
        for layer in LAYERS:
            records = read_csv(out / f'layer_{layer}.csv', RESULT_COLUMNS)
            validate_results(records, source, ids, baseline, layer, vocabulary)
            writer.writerows(records)
            result_rows += len(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, out / 'results.csv')
    summaries.sort(key=lambda r: (r['split'], r['voice'], r['layer'], r['method'], r['alpha']))
    atomic_csv(out / 'summary.csv', summaries)
    metadata = dict(status='complete', **lineage, model=provenance, rows=len(ids), row_ids=ids,
        layers=LAYERS, methods=list(METHODS), alphas=[1.], conditions=len(LAYERS) * len(METHODS),
        norm_matching=False, batch_size=args.batch_size, result_rows=result_rows,
        baseline_sha256=baseline_hash, results_sha256=sha256(out / 'results.csv'), summary_sha256=sha256(out / 'summary.csv'),
        layer_results_sha256={f'layer_{layer}.csv': sha256(out / f'layer_{layer}.csv') for layer in LAYERS},
        native_checks=all_checks, exact_identity_conditions=identity_conditions,
        controls='Zero edits matched baseline; all final-block methods and rows actually scored and matched baseline exactly',
        baseline_cache='This run only; exact first-batch scored sentinel on resume',
        result_reuse='Only hash-verified completed layers in this run; no older-run results')
    atomic_json(out / 'metadata.json', metadata)
    return metadata
