"""Evaluate reconstructed-centroid steering beside the frozen all-layer run."""

import argparse
from datetime import datetime, timezone
import fcntl
import math
from pathlib import Path
import time

import torch

import native_all_layers_evaluation as native
from reconstructed_centroids import METHODS, ROLES, centroid_edits, fit_centroids
from run_tpr_nov6 import load_lm, torch_load


ROOT = Path(__file__).resolve().parents[2]
LAYERS = list(range(36))
DEFAULT_RUN = ROOT / 'ac-tpr-cache/native-tpr-all-layers-20261109'


def source_hashes():
    return {str(path.relative_to(ROOT)): native.sha256(path) for path in (
        Path(__file__).resolve(), Path(__file__).with_name('reconstructed_centroids.py'))}


def validate_results(records, source, ids, baseline, layer, vocabulary):
    """Require complete scored conditions with unchanged corpus and baseline."""
    expected = {(i, method) for i in ids for method in METHODS}
    actual = [(int(r['row_id']), r['method']) for r in records]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError('Incomplete or duplicate reconstructed-centroid coverage')
    baseline_by_id = {i: float(baseline[j, 1] - baseline[j, 0]) for j, i in enumerate(ids)}
    for record in records:
        index = int(record['row_id'])
        row = source['rows'][index]
        if (int(record['layer']) != layer or float(record['alpha']) != 1.
                or record['result_origin'] != 'scored'
                or any(record[k] != row[k] for k in ('split', 'voice', 'quad_id'))):
            raise ValueError('Condition or corpus labels differ')
        known = sum(row[role] in vocabulary for role in ROLES)
        if int(record['known_tokens']) != known or native._boolean(record['sentence_has_oov']) != (known < 2):
            raise ValueError('Vocabulary annotations differ')
        values = {k: float(record[k]) for k in ('baseline_margin', 'edited_margin', 'margin_change',
                                               'agent_edit_norm', 'patient_edit_norm')}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Nonfinite scores or norms')
        before, after = values['baseline_margin'], values['edited_margin']
        if before != baseline_by_id[index] or values['margin_change'] != after - before:
            raise ValueError('Baseline or score change differs')
        flags = dict(baseline_agent_win=before < 0, patient_win=after > 0, flip=before < 0 and after > 0)
        for role in ROLES:
            norm = values[f'{role}_edit_norm']
            if norm < 0:
                raise ValueError('Negative edit norm')
            flags[f'{role}_edit_is_zero'] = norm == 0
        if any(native._boolean(record[k]) != v for k, v in flags.items()):
            raise ValueError('Stored flags disagree with scores')
        if layer == LAYERS[-1] and after != before:
            raise ValueError('Final-block control changed the baseline')


def prepare_centroids(args, source, lineage, training):
    out = args.run_dir / 'reconstructed_centroids'
    (out / 'centroids').mkdir(parents=True, exist_ok=True)
    protocol_path = out / 'protocol.json'
    protocol = dict(schema_version='reconstructed-centroids-v1', **lineage,
        extension_source_sha256=source_hashes(), layers=LAYERS, methods=list(METHODS),
        fit_split='train', grouping=['voice', 'role'], roles=list(ROLES),
        tpr_reconstruction='c + s * (((h-c)/s) @ M_flat) @ pinv(M_flat); float64; rtol=1e-10',
        soft_reconstruction='c + s * decoder(bind(quantize(unbind(encoder((h-c)/s)))))',
        direction='mean(reconstruction(patient)) - mean(reconstruction(agent)), by voice',
        intervention='Add direction to agent token and subtract from patient token, jointly at one layer',
        alpha=1.0, norm_matching=False, batch_size=args.batch_size,
        reconstruction_residual='Used to construct centroids only; original live hidden states receive additive edits')
    if protocol_path.exists():
        if native.read_json(protocol_path) != protocol:
            raise ValueError('Reconstructed-centroid protocol or sources changed')
    else:
        native.atomic_json(protocol_path, protocol)
    protocol_hash = native.sha256(protocol_path)
    hashes, checks = {}, []
    for layer in LAYERS:
        path = out / f'centroids/layer_{layer}.pt'
        metadata_path = out / f'centroids/layer_{layer}_metadata.json'
        dependencies = dict(protocol_sha256=protocol_hash, layer=layer,
            checkpoint_sha256={kind: training['checkpoint_sha256'][f'{kind}/layer_{layer}.pt'] for kind in ('tpr', 'soft')})
        if metadata_path.exists():
            metadata = native.read_json(metadata_path)
            if metadata['status'] != 'complete' or any(metadata.get(k) != v for k, v in dependencies.items()):
                raise ValueError('Centroid lineage changed')
            native._require_hash(path, metadata['centroids_sha256'])
        else:
            centroids = fit_centroids(source, layer,
                torch_load(args.run_dir / f'training/tpr/layer_{layer}.pt'),
                torch_load(args.run_dir / f'training/soft/layer_{layer}.pt'))
            temporary = path.with_suffix('.pt.tmp')
            torch.save(centroids, temporary)
            temporary.replace(path)
            metadata = dict(status='complete', **dependencies, centroids_sha256=native.sha256(path),
                            diagnostics=centroids['diagnostics'], train_row_ids=centroids['train_row_ids'])
            native.atomic_json(metadata_path, metadata)
        hashes[path.name] = metadata['centroids_sha256']
        checks.extend(metadata['diagnostics'])
        print(f'Centroids layer {layer}: complete (TPR and Soft TPR, train only)', flush=True)
    native.atomic_csv(out / 'centroids/diagnostics.csv', checks)
    native.atomic_json(out / 'centroids/metadata.json', dict(status='complete', protocol_sha256=protocol_hash,
        layers=LAYERS, methods=list(METHODS), fit_split='train', centroids_sha256=hashes,
        diagnostics_sha256=native.sha256(out / 'centroids/diagnostics.csv')))
    return out, protocol_hash, hashes


def evaluate(args, source, ids, vocabulary, lineage, extraction, out, protocol_hash, hashes):
    evaluation = out / 'evaluation'
    evaluation.mkdir(exist_ok=True)
    tokenizer, model, provenance = load_lm()
    if provenance != extraction['model'] or model.config.num_hidden_layers != len(LAYERS):
        raise ValueError('Loaded model differs from the frozen experiment')
    baseline, baseline_hash = native.canonical_baseline(args.run_dir / 'evaluation', source, ids,
        tokenizer, model, provenance, lineage, args.batch_size)
    dependencies = dict(protocol_sha256=protocol_hash, baseline_sha256=baseline_hash,
        model=provenance, row_ids=ids, batch_size=args.batch_size, methods=list(METHODS), alpha=1.)
    started = time.monotonic()
    for layer in args.layers:
        path, metadata_path = evaluation / f'layer_{layer}.csv', evaluation / f'layer_{layer}_metadata.json'
        expected = dict(**dependencies, layer=layer, centroids_sha256=hashes[f'layer_{layer}.pt'])
        if metadata_path.exists():
            metadata = native.read_json(metadata_path)
            if metadata['status'] != 'complete' or any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError('Evaluation lineage changed')
            native._require_hash(path, metadata['results_sha256'])
            validate_results(native.read_csv(path, native.RESULT_COLUMNS), source, ids, baseline, layer, vocabulary)
            print(f'Evaluation layer {layer}: verified completed results', flush=True)
            continue
        centroids = torch_load(out / f'centroids/layer_{layer}.pt')
        edits = centroid_edits(centroids, [source['rows'][i] for i in ids])
        records = []
        for start in range(0, len(ids), args.batch_size):
            chosen = ids[start:start + args.batch_size]
            inputs, answers, positions = native._batch(source, chosen, tokenizer, model)
            before = baseline[start:start + len(chosen)]
            for method in METHODS:
                delta = edits[method][start:start + len(chosen)]
                norms = delta.double().norm(dim=-1)
                scores = native._valid_scores(native.score_batch(model, inputs, answers, layer, positions, delta), len(chosen))
                if layer == LAYERS[-1]:
                    torch.testing.assert_close(scores, before, rtol=0, atol=0,
                        msg=f'Final-block {method} changed the answer logits')
                for j, index in enumerate(chosen):
                    row = source['rows'][index]
                    bm, em = float(before[j, 1] - before[j, 0]), float(scores[j, 1] - scores[j, 0])
                    records.append(dict(row_id=index, split=row['split'], voice=row['voice'], quad_id=row['quad_id'],
                        layer=layer, method=method, alpha=1., known_tokens=sum(row[r] in vocabulary for r in ROLES),
                        sentence_has_oov=any(row[r] not in vocabulary for r in ROLES), baseline_margin=bm,
                        edited_margin=em, margin_change=em-bm, baseline_agent_win=bm < 0,
                        patient_win=em > 0, flip=bm < 0 and em > 0,
                        agent_edit_norm=float(norms[j, 0]), patient_edit_norm=float(norms[j, 1]),
                        agent_edit_is_zero=bool(norms[j, 0] == 0), patient_edit_is_zero=bool(norms[j, 1] == 0),
                        result_origin='scored'))
            if start % (16 * args.batch_size) == 0 or start + len(chosen) == len(ids):
                print(f'Evaluation layer {layer}: {start + len(chosen)}/{len(ids)} rows across 2 centroid methods '
                      f'({time.monotonic()-started:.0f}s elapsed)', flush=True)
        validate_results(records, source, ids, baseline, layer, vocabulary)
        native.atomic_csv(path, records, native.RESULT_COLUMNS)
        native.atomic_json(metadata_path, dict(status='complete', **expected,
            result_rows=len(records), results_sha256=native.sha256(path)))
        del edits
    # A partial pilot cannot publish metadata claiming a complete all-layer run.
    if not all((evaluation / f'layer_{layer}_metadata.json').exists() for layer in LAYERS):
        print('Selected layers complete; the full 36-layer evaluation is not complete.', flush=True)
        return
    records = []
    layer_hashes = {}
    for layer in LAYERS:
        path = evaluation / f'layer_{layer}.csv'
        metadata = native.read_json(evaluation / f'layer_{layer}_metadata.json')
        expected = dict(**dependencies, layer=layer, centroids_sha256=hashes[f'layer_{layer}.pt'])
        if metadata['status'] != 'complete' or any(metadata.get(k) != v for k, v in expected.items()):
            raise ValueError('Completed layer lineage changed')
        layer_hashes[path.name] = native._require_hash(path, metadata['results_sha256'])
        layer_records = native.read_csv(path, native.RESULT_COLUMNS)
        validate_results(layer_records, source, ids, baseline, layer, vocabulary)
        records.extend(layer_records)
    native.atomic_csv(evaluation / 'results.csv', records, native.RESULT_COLUMNS)
    native.atomic_json(evaluation / 'metadata.json', dict(status='complete', **dependencies,
        layers=LAYERS, rows=len(ids), result_rows=len(records), norm_matching=False,
        results_sha256=native.sha256(evaluation / 'results.csv'), layer_results_sha256=layer_hashes,
        centroid_metadata_sha256=native.sha256(out / 'centroids/metadata.json'),
        completed_at=datetime.now(timezone.utc).isoformat(),
        controls='Original baseline verified with exact scored sentinel; zero-edit and final-block checks passed'))
    print(f'COMPLETE: {len(records):,} reconstructed-centroid results across all 36 layers', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'evaluate'))
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--layers', nargs='+', type=int, default=LAYERS)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if args.layers != sorted(set(args.layers)) or not set(args.layers) <= set(LAYERS):
        raise ValueError('Layers must be unique, sorted block indices 0 through 35')
    torch.set_num_threads(2)
    out = args.run_dir / 'reconstructed_centroids'
    out.mkdir(exist_ok=True)
    with (out / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = native.read_json(args.run_dir / 'plan.json')
        source, ids, vocabulary, lineage, training, _, extraction = native.validate_inputs(args, plan)
        out, protocol_hash, hashes = prepare_centroids(args, source, lineage, training)
        if args.stage == 'evaluate':
            evaluate(args, source, ids, vocabulary, lineage, extraction, out, protocol_hash, hashes)


if __name__ == '__main__':
    main()
