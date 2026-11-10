"""Test native TPR profile swaps and train-calibrated Soft slot permutations."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import csv
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import torch

from ac_tpr.dataset import load_corpus
from ac_tpr.interventions import score_batch
from run_tpr_nov6 import digest, load_lm, torch_load, write_csv, write_json
from native_tpr_interventions import native_role_swap
from native_soft_interventions import encode_native_soft, select_slot_permutation, native_slot_swap


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = Path('/home/ldomenichelli/Workspace/AC-TPR-tpr-nov6/ac-tpr-cache/tpr-role-filler-20261107')
SOFT_REFERENCE = Path('/home/ldomenichelli/Workspace/AC-TPR-soft-nov6/ac-tpr-cache/soft-tpr-nov6-20261107')
NORM_REFERENCE = Path('/home/ldomenichelli/Workspace/AC-TPR-tpr-norm-nov6/ac-tpr-cache/tpr-norm-nov6-20261107')
METHODS = ('raw_centroid', 'tpr_native_role_swap', 'soft_native_slot_swap')
VOICES = ('active', 'passive')
ROLES = ('agent', 'patient')
LAYERS = [20, 21, 22]
ALPHAS = [1.]


def read_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def source_paths():
    scripts = ['run_native_tpr_nov6.py', 'native_tpr_interventions.py', 'native_soft_interventions.py',
               'run_tpr_nov6.py', 'tpr_probe.py', 'tpr_nov6_probe.py', 'tpr_nov6_runtime.py',
               'tpr_role_filler_factors.py', 'resource_limited_run.py', 'soft_tpr_model.py']
    return [ROOT/'scripts/temporary'/name for name in scripts] + [ROOT/'src/ac_tpr'/name for name in
        ('model.py', 'dataset.py', 'prompts.py', 'activations.py', 'interventions.py')]


def snapshot(directory, plan):
    hashes = {}
    for source in source_paths():
        relative = source.relative_to(ROOT)
        target = directory/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = digest(target)
    if hashes != plan['source_sha256']:
        raise ValueError('Scientific sources differ from the frozen plan')
    write_json(directory/'runtime_source.json', dict(source_sha256=hashes, differences_from_plan={}))


def pair_rows(rows, ids):
    """Map local tokens to the same filler in the opposite role of its twin.

    Token order is [agent, patient] for each selected row. Both swapped values
    must be present within each quad/voice, in the same split. Lexical labels
    validate the mapping; they do not choose it. Calibration calls this only on
    train rows; postrun counterfactual diagnostics may use a held-out split.
    """
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= len(rows) for i in ids):
        raise ValueError('Require distinct, nonempty valid row IDs')
    selected = [rows[i] for i in ids]
    lookup = {}
    for i, row in enumerate(selected):
        if row['voice'] not in VOICES or row['swapped'] not in ('0', '1'):
            raise ValueError('Require known voices and binary swapped annotations')
        key = (row['quad_id'], row['voice'], row['swapped'])
        if key in lookup:
            raise ValueError('Duplicate quad/voice/swapped row')
        lookup[key] = i
    donors = []
    for row in selected:
        key = (row['quad_id'], row['voice'], str(1-int(row['swapped'])))
        if key not in lookup:
            raise ValueError('Missing opposite-role twin within selected rows')
        donor_index = lookup[key]
        donor = selected[donor_index]
        if donor['split'] != row['split']:
            raise ValueError('Twin rows cross experimental splits')
        for field in ('verb', 'tense', 'subset'):
            if field in row and donor.get(field) != row[field]:
                raise ValueError(f'Twin context annotation differs: {field}')
        for role_index, role in enumerate(ROLES):
            if row[role] != donor[ROLES[1-role_index]]:
                raise ValueError('Twin mapping does not preserve filler identity')
            donors.append(2*donor_index + 1-role_index)
    result = torch.tensor(donors, dtype=torch.long)
    if not torch.equal(result[result], torch.arange(len(result))):
        raise ValueError('Twin token mapping must be involutive')
    return result


def prepare(args):
    reference = args.reference_run.resolve()
    old = json.loads((reference/'plan.json').read_text())
    for stage in ('extraction', 'training', 'evaluation'):
        if json.loads((reference/stage/'metadata.json').read_text())['status'] != 'complete':
            raise ValueError(f'Reference stage is incomplete: {stage}')
    source = torch_load(reference/'extraction/states.pt')
    if old['layers'] != LAYERS or source['layers'] != LAYERS:
        raise ValueError('Reference layers differ')
    if source['rows'] != load_corpus(ROOT/'data/active_pass_long/corpus.csv'):
        raise ValueError('Reference rows differ from the corpus')
    splits = {name: {r['quad_id'] for r in source['rows'] if r['split'] == name}
              for name in ('train', 'val', 'test', 'gen_test')}
    if any(splits[a] & splits[b] for a in splits for b in splits if a != b):
        raise ValueError('Quad overlap between splits')
    counts = Counter(r['split'] for r in source['rows'])
    if dict(counts) != dict(train=3300, val=412, test=416, gen_test=1680):
        raise ValueError('Unexpected reference split sizes')
    pair_rows(source['rows'], [i for i, row in enumerate(source['rows']) if row['split']=='train'])
    names = ['plan.json', 'extraction/states.pt', 'extraction/metadata.json', 'training/metadata.json',
             'evaluation/baseline.csv', 'evaluation/results.csv', 'evaluation/summary.csv', 'evaluation/metadata.json']
    names += [f'training/layer_{layer}.pt' for layer in LAYERS]
    norm_files = ['plan.json'] + [f'{stage}/{name}' for stage in ('validation', 'evaluation')
        for name in ('metadata.json', 'baseline.csv', 'results.csv', 'summary.csv')]
    soft_files = ['plan.json', 'training/metadata.json'] + [f'training/layer_{layer}.pt' for layer in LAYERS]
    for directory, stages in ((NORM_REFERENCE, ('validation', 'evaluation')), (SOFT_REFERENCE, ('training',))):
        other = json.loads((directory/'plan.json').read_text())
        for key in ('model', 'revision', 'layers', 'corpus_sha256', 'prompt_template', 'demonstrations'):
            if other[key] != old[key]:
                raise ValueError(f'Reference protocol differs: {directory}/{key}')
        for stage in stages:
            if json.loads((directory/stage/'metadata.json').read_text())['status'] != 'complete':
                raise ValueError('Require complete frozen reference stages')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.run_dir/'plan.json', dict(
        status='prepared', created_at=datetime.now(timezone.utc).isoformat(),
        base_commit=old['base_commit'], model=old['model'], revision=old['revision'],
        reference_run=str(reference), reference_sha256={name: digest(reference/name) for name in names},
        norm_reference_run=str(NORM_REFERENCE), norm_reference_sha256={n: digest(NORM_REFERENCE/n) for n in norm_files},
        soft_reference_run=str(SOFT_REFERENCE), soft_reference_sha256={n: digest(SOFT_REFERENCE/n) for n in soft_files},
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in source_paths()},
        layers=LAYERS, layer_selection=old['layer_selection'], methods=METHODS, alphas=ALPHAS, fixed_alpha=1.,
        strength_selection='none; fixed native alpha=1', norm_matching=False, batch_size=args.batch_size,
        expected_counts=dict(counts), prompt_template=old['prompt_template'], demonstrations=old['demonstrations'],
        corpus_sha256=digest(ROOT/'data/active_pass_long/corpus.csv'),
        tpr_intervention='Swap the two native unbound role profiles using the original R, M and F; no new role head or fitted factors',
        soft_intervention='Permute the eight native quantized slots; preserve the target code multiset before decoding; add decoder difference to original hidden state',
        calibration=dict(split='train', sentences=3300, tokens=6600, shared_across_voices=True,
            candidates='764 involutions of eight slots, including identity',
            target='Same-filler opposite-role token in the same-voice swapped twin',
            objective='Mean squared error between permuted source quantized slots and twin quantized slots',
            ties='Within relative tolerance 1e-12 of minimum; fewer moved slots, then lexicographic order',
            heldout_donors='Never used to construct steering; no validation or held-out selection'),
        no_op='Exact zero directions remain zero; reuse baseline only when the entire layer/method condition is exactly zero',
        reuse='Raw alpha=1 only, after exact individual baseline logits and a first-batch scored sentinel for each reused layer',
        interpretation='Soft train-calibrated slot permutation is not an identified semantic role map; native latent constraints do not prove lexical identity preservation or exact realization after decoder/BF16',
        runtime=dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                     transformers=version('transformers'), nnsight=version('nnsight'))))
    print(f'Frozen plan: {args.run_dir}', flush=True)


def load_plan(args):
    plan = json.loads((args.run_dir/'plan.json').read_text())
    if (plan['layers'] != LAYERS or plan['alphas'] != ALPHAS or tuple(plan['methods']) != METHODS
            or plan['fixed_alpha'] != 1. or plan['norm_matching'] is not False or plan['batch_size'] != args.batch_size):
        raise ValueError('Runtime settings differ from the frozen native protocol')
    for prefix in ('reference', 'norm_reference', 'soft_reference'):
        directory = Path(plan[prefix+'_run'])
        for name, expected in plan[prefix+'_sha256'].items():
            if digest(directory/name) != expected:
                raise ValueError(f'Frozen reference changed: {prefix}/{name}')
    for path in source_paths():
        if digest(path) != plan['source_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError(f'Scientific source changed: {path}')
    return plan, Path(plan['reference_run'])


def calibrate(args):
    plan, reference = load_plan(args)
    out = args.run_dir/'calibration'
    out.mkdir(exist_ok=False)
    snapshot(out, plan)
    source = torch_load(reference/'extraction/states.pt')
    ids = [i for i, row in enumerate(source['rows']) if row['split']=='train']
    if len(ids) != plan['expected_counts']['train']:
        raise ValueError('Unexpected calibration training size')
    donors = pair_rows(source['rows'], ids)
    write_json(out/'train_pairing.json', dict(split='train', row_ids=ids, donor_token_local_ids=donors.tolist()))
    files = []
    for li, layer in enumerate(LAYERS):
        payload = torch_load(Path(plan['soft_reference_run'])/f'training/layer_{layer}.pt')
        h = source['states'][ids, li].float().flatten(0, 1)
        q, codes = encode_native_soft(payload, h)
        if q.shape[:2] != (2*len(ids), 8) or codes.shape != q.shape[:2]:
            raise ValueError('Require eight aligned native slots for each training token')
        permutation, diagnostics = select_slot_permutation(q, q[donors])
        filename = f'layer_{layer}.json'
        write_json(out/filename, dict(layer=layer, permutation=list(map(int, permutation)),
            training_tokens=len(h), diagnostics=diagnostics))
        files.append(filename)
        print(f'Calibrated native Soft slots L{layer}: {list(map(int, permutation))}', flush=True)
    vocabulary = sorted({source['rows'][i][role] for i in ids for role in ROLES})
    write_json(out/'metadata.json', dict(status='complete', layers=LAYERS, split='train',
        training_sentences=len(ids), training_tokens=2*len(ids), shared_across_voices=True,
        vocabulary=vocabulary, calibration_sha256={name: digest(out/name) for name in files},
        train_pairing_sha256=digest(out/'train_pairing.json')))


def verify_calibration(args):
    out = args.run_dir/'calibration'
    metadata = json.loads((out/'metadata.json').read_text())
    if metadata['status'] != 'complete' or metadata['split'] != 'train' or metadata['layers'] != LAYERS:
        raise ValueError('Native calibration must be complete and train-only')
    expected = {f'layer_{layer}.json' for layer in LAYERS}
    if set(metadata['calibration_sha256']) != expected:
        raise ValueError('Incomplete native calibration')
    for name, sha in metadata['calibration_sha256'].items():
        if digest(out/name) != sha:
            raise ValueError('Slot permutation changed after calibration')
        permutation = json.loads((out/name).read_text())['permutation']
        if (sorted(permutation) != list(range(8))
                or any(permutation[permutation[i]] != i for i in range(8))):
            raise ValueError('Require an involutive permutation of eight native slots')
    if digest(out/'train_pairing.json') != metadata['train_pairing_sha256']:
        raise ValueError('Training pair map changed after calibration')


def build_edits(source, reference, ids, args, plan):
    """Build native deltas without norm matching, donor access, or filler labels."""
    rows = [source['rows'][i] for i in ids]
    if not rows or any(row['voice'] not in VOICES for row in rows):
        raise ValueError('Require nonempty rows with known voices')
    edits, checks = {}, []
    for li, layer in enumerate(LAYERS):
        tpr = torch_load(reference/f'training/layer_{layer}.pt')
        soft = torch_load(Path(plan['soft_reference_run'])/f'training/layer_{layer}.pt')
        permutation = json.loads((args.run_dir/f'calibration/layer_{layer}.json').read_text())['permutation']
        h = source['states'][ids, li].float()
        raw = torch.empty_like(h)
        for voice in VOICES:
            selected = [i for i, row in enumerate(rows) if row['voice']==voice]
            if selected:
                delta = tpr['raw_means'][voice][1]-tpr['raw_means'][voice][0]
                raw[selected] = torch.stack((delta, -delta)).expand(len(selected), -1, -1)
        tpr_delta, tpr_checks = native_role_swap(tpr, h.flatten(0, 1))
        soft_delta, soft_checks = native_slot_swap(soft, h.flatten(0, 1), permutation)
        edits[layer] = dict(raw_centroid=raw, tpr_native_role_swap=tpr_delta.reshape(h.shape),
                            soft_native_slot_swap=soft_delta.reshape(h.shape))
        for method, delta in edits[layer].items():
            if delta.shape != h.shape or not torch.isfinite(delta).all():
                raise ValueError('Invalid native edit shape or values')
            norms = delta.double().norm(dim=-1)
            checks.append(dict(layer=layer, method=method, tokens=norms.numel(),
                native_norm_min=float(norms.min()), native_norm_mean=float(norms.mean()), native_norm_max=float(norms.max()),
                exact_zero_tokens=int((norms==0).sum()), exact_zero_sentences=int((norms==0).all(dim=1).sum()),
                diagnostics=tpr_checks if method=='tpr_native_role_swap' else soft_checks if method=='soft_native_slot_swap' else {}))
    return edits, checks


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row['split'], row['voice'], int(row['layer']), row['method'], float(row['alpha']))].append(row)
    output = []
    for key, rows in sorted(groups.items()):
        before = np.array([float(r['baseline_margin']) for r in rows])
        after = np.array([float(r['edited_margin']) for r in rows])
        eligible = before < 0
        flips = int((eligible & (after > 0)).sum())
        zeros = np.array([[float(r['agent_edit_norm'])==0, float(r['patient_edit_norm'])==0] for r in rows])
        output.append(dict(split=key[0], voice=key[1], layer=key[2], method=key[3], alpha=key[4],
            sentences=len(rows), baseline_agent_wins=int(eligible.sum()), flips=flips,
            conditional_flip=flips/int(eligible.sum()) if eligible.any() else None,
            patient_preference=float((after > 0).mean()), mean_margin_change=float((after-before).mean()),
            mean_agent_edit_norm=float(np.mean([float(r['agent_edit_norm']) for r in rows])),
            mean_patient_edit_norm=float(np.mean([float(r['patient_edit_norm']) for r in rows])),
            zero_sentence_edits=int(zeros.all(axis=1).sum()), zero_token_edits=int(zeros.sum())))
    return output


def evaluate_stage(args, plan, reference, source, tokenizer, model, provenance, stage, alphas=ALPHAS):
    if stage not in ('validation', 'evaluation') or list(alphas) != ALPHAS:
        raise ValueError('Native evaluation requires validation/evaluation and fixed alpha=1')
    out = args.run_dir/stage
    out.mkdir(exist_ok=False)
    snapshot(out, plan)
    wanted = ('val',) if stage=='validation' else ('test', 'gen_test')
    ids = [i for i, row in enumerate(source['rows']) if row['split'] in wanted]
    edits, checks = build_edits(source, reference, ids, args, plan)
    identity_conditions = {(layer, method) for layer in LAYERS for method in METHODS
                           if torch.count_nonzero(edits[layer][method]).item()==0}
    cache = Path(plan['norm_reference_run'])/stage
    baseline_rows = read_csv(cache/'baseline.csv')
    old_baseline = {int(r['row_id']): r for r in baseline_rows}
    raw_rows = [r for r in read_csv(cache/'results.csv') if r['method']=='raw_centroid' and float(r['alpha'])==1.]
    old_raw = {(int(r['row_id']), int(r['layer'])): r for r in raw_rows}
    if len(old_baseline)!=len(baseline_rows) or set(old_baseline)!=set(ids) or len(old_raw)!=len(raw_rows):
        raise ValueError('Duplicate or differing reference IDs/conditions')
    vocabulary = set(json.loads((args.run_dir/'calibration/metadata.json').read_text())['vocabulary'])
    results, verified_raw = [], set()
    with (out/'baseline.csv').open('x', newline='') as bf, (out/'results.csv').open('x', newline='') as rf:
        bw = rw = None
        for start in range(0, len(ids), args.batch_size):
            end = min(start+args.batch_size, len(ids))
            chosen = ids[start:end]
            inputs = tokenizer.pad({'input_ids': [source['input_ids'][i] for i in chosen]},
                                   padding=True, return_tensors='pt').to('cuda')
            answers, positions = source['answers'][chosen], source['positions'][chosen]
            baseline = score_batch(model, inputs, answers)
            for j, i in enumerate(chosen):
                row, old = source['rows'][i], old_baseline[i]
                if (float(baseline[j, 0]), float(baseline[j, 1])) != (float(old['agent_logit']), float(old['patient_logit'])):
                    raise ValueError(f'Baseline differs from reference at row {i}')
                if any(row[k]!=old[k] for k in ('split', 'voice', 'agent', 'patient')):
                    raise ValueError('Reference row labels differ')
                record = dict(row_id=i, split=row['split'], voice=row['voice'], agent=row['agent'], patient=row['patient'],
                    agent_logit=float(baseline[j, 0]), patient_logit=float(baseline[j, 1]),
                    extraction_margin=float(source['baseline'][i, 1]-source['baseline'][i, 0]))
                if bw is None:
                    bw = csv.DictWriter(bf, fieldnames=list(record)); bw.writeheader()
                bw.writerow(record)
            if start==0:
                zero = torch.zeros(len(chosen), 2, model.config.hidden_size)
                torch.testing.assert_close(score_batch(model, inputs, answers, LAYERS[0], positions, zero), baseline, rtol=0, atol=0)
                final_edit = edits[LAYERS[0]]['tpr_native_role_swap'][start:end]
                torch.testing.assert_close(score_batch(model, inputs, answers, 35, positions, final_edit), baseline, rtol=0, atol=0)
                for layer in LAYERS:
                    if not all((i, layer) in old_raw for i in chosen):
                        continue
                    raw = score_batch(model, inputs, answers, layer, positions, edits[layer]['raw_centroid'][start:end])
                    expected = torch.tensor([float(old_raw[i, layer]['edited_margin']) for i in chosen])
                    torch.testing.assert_close(raw[:, 1]-raw[:, 0], expected, rtol=0, atol=0)
                    verified_raw.add(layer)
            for layer in LAYERS:
                for method in METHODS:
                    delta = edits[layer][method][start:end]
                    actual = delta.double().norm(dim=-1)
                    if (layer, method) in identity_conditions:
                        scores, origin = baseline, 'exact_identity'
                    elif method=='raw_centroid' and layer in verified_raw and all((i, layer) in old_raw for i in chosen):
                        scores, origin = None, 'reference_raw'
                    else:
                        scores, origin = score_batch(model, inputs, answers, layer, positions, delta), 'scored'
                    for j, i in enumerate(chosen):
                        row = source['rows'][i]
                        bm = float(baseline[j, 1]-baseline[j, 0])
                        em = float(old_raw[i, layer]['edited_margin']) if scores is None else float(scores[j, 1]-scores[j, 0])
                        record = dict(row_id=i, split=row['split'], voice=row['voice'], quad_id=row['quad_id'],
                            layer=layer, method=method, alpha=1., known_tokens=sum(row[role] in vocabulary for role in ROLES),
                            sentence_has_oov=any(row[role] not in vocabulary for role in ROLES),
                            baseline_margin=bm, edited_margin=em, margin_change=em-bm,
                            baseline_agent_win=bm<0, patient_win=em>0, flip=bm<0 and em>0,
                            agent_edit_norm=float(actual[j, 0]), patient_edit_norm=float(actual[j, 1]),
                            agent_edit_is_zero=bool(actual[j, 0]==0), patient_edit_is_zero=bool(actual[j, 1]==0), result_origin=origin)
                        if rw is None:
                            rw = csv.DictWriter(rf, fieldnames=list(record)); rw.writeheader()
                        rw.writerow(record); results.append(record)
            bf.flush(); rf.flush()
            print(f'{stage}: {end}/{len(ids)} sentences across 9 native conditions', flush=True)
    write_csv(out/'summary.csv', summarize(results))
    write_json(out/'metadata.json', dict(status='complete', model=provenance, rows=len(ids), layers=LAYERS,
        alphas=ALPHAS, methods=METHODS, conditions=9, norm_matching=False, native_checks=checks,
        exact_identity_conditions=[dict(layer=layer, method=method) for layer, method in sorted(identity_conditions)],
        controls='Zero/final-block edits matched baseline; reused raw alpha=1 matched first-batch sentinels',
        exact_reference_baseline=True, result_rows=len(results), results_sha256=digest(out/'results.csv'),
        calibration_metadata_sha256=digest(args.run_dir/'calibration/metadata.json')))


def run(args):
    plan, reference = load_plan(args)
    verify_calibration(args)
    for stage in ('validation', 'evaluation'):
        out = args.run_dir/stage
        if out.exists() and not (out/'metadata.json').exists():
            raise ValueError(f'Inspect incomplete stage before resuming: {out}')
        if (out/'metadata.json').exists() and json.loads((out/'metadata.json').read_text())['status']!='complete':
            raise ValueError(f'Incomplete stage metadata: {stage}')
    if (args.run_dir/'evaluation/metadata.json').exists():
        return
    source = torch_load(reference/'extraction/states.pt')
    tokenizer, model, provenance = load_lm()
    for stage in ('validation', 'evaluation'):
        if not (args.run_dir/stage/'metadata.json').exists():
            evaluate_stage(args, plan, reference, source, tokenizer, model, provenance, stage, ALPHAS)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'calibrate', 'run'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--reference-run', type=Path, default=REFERENCE)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    globals()[args.stage](args)
