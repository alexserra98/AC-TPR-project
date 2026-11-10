"""Postrun CPU diagnostics of cached states plus native TPR/Soft interventions.

States are simulated as BF16(h.float()+delta), NOT captured at live LM hooks.
The common filler readout is the original TPR classifier, with source-role
conditioning before and target-role conditioning after. OOV prediction retention
means stability within its known vocabulary, not preserved lexical identity.
Paired Soft alignment on validation/test is descriptive and never selects edits.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from native_soft_interventions import encode_native_soft
from run_tpr_nov6 import digest, torch_load, write_csv, write_json
from tpr_probe import BilinearTPRProbe

ROLES = ('agent', 'patient')
VOICES = ('active', 'passive')
METHODS = ('raw_centroid', 'tpr_native_role_swap', 'soft_native_slot_swap')


def simulate_local_edit(h, delta):
    if h.ndim != 2 or delta.shape != h.shape or not len(h):
        raise ValueError('Require aligned nonempty [token, hidden] states and edits')
    if h.device.type != 'cpu' or delta.device.type != 'cpu':
        raise ValueError('Diagnostics use CPU states')
    if not torch.isfinite(h).all() or not torch.isfinite(delta).all():
        raise ValueError('Nonfinite state or edit')
    result = (h.float()+delta.float()).bfloat16().float()
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite locally rounded state')
    return result


def require_completed_evaluation(run_dir, layers):
    metadata = json.loads((run_dir/'evaluation/metadata.json').read_text())
    if metadata['status'] != 'complete':
        raise ValueError('LM evaluation must be complete before local diagnostics')
    if metadata['layers'] != layers or metadata['methods'] != list(METHODS):
        raise ValueError('Evaluation layers/methods differ from the frozen protocol')
    if metadata['alphas'] != [1.] or metadata['norm_matching'] is not False:
        raise ValueError('Require native alpha=1 without norm matching')
    if metadata['results_sha256'] != digest(run_dir/'evaluation/results.csv'):
        raise ValueError('Completed evaluation results changed')
    if metadata['calibration_metadata_sha256'] != digest(run_dir/'calibration/metadata.json'):
        raise ValueError('Calibration metadata changed after evaluation')
    return metadata


def verify_evaluated_edits(records, ids, rows, layers, edits):
    """Check every evaluated row/method/layer, native norm, and exact-zero flag."""
    positions = {row_id: i for i, row_id in enumerate(ids)}
    if len(positions) != len(ids) or len(rows) != len(ids):
        raise ValueError('Require distinct and aligned selected row IDs')
    expected = {(i, layer, method) for i in ids for layer in layers for method in METHODS}
    seen, error_max, zero_tokens = set(), 0., 0
    for record in records:
        key = (int(record['row_id']), int(record['layer']), record['method'])
        if key not in expected or key in seen:
            raise ValueError('Duplicate or unexpected evaluated edit')
        seen.add(key)
        row_id, layer, method = key
        local = positions[row_id]
        if float(record['alpha']) != 1. or any(record[k] != rows[local][k] for k in ('split','voice')):
            raise ValueError('Evaluated edit metadata differs')
        norms = edits[layer][method][local].double().norm(dim=-1)
        for role, norm in zip(ROLES, norms):
            actual, wanted = float(record[role+'_edit_norm']), float(norm)
            if not math.isfinite(actual) or not math.isclose(actual,wanted,rel_tol=1e-6,abs_tol=1e-10):
                raise ValueError('Evaluated edit norm differs from reconstruction')
            flag = record[role+'_edit_is_zero']
            if flag not in (True,False,'True','False'):
                raise ValueError('Unknown exact-zero flag representation')
            recorded_zero = flag is True or flag == 'True'
            if recorded_zero != (wanted == 0.):
                raise ValueError('Evaluated exact-zero flag differs from reconstruction')
            zero_tokens += int(recorded_zero)
            error_max = max(error_max,abs(actual-wanted)/max(wanted,1e-12))
    if seen != expected:
        raise ValueError('Incomplete evaluated edit coverage')
    return dict(verified_result_rows=len(records), verified_token_edits=2*len(records),
                zero_token_edits=zero_tokens, maximum_relative_norm_error=error_max,
                coverage='Exact row_id x layer x method; alpha=1; both token norms and zero flags')


def _groups(token_rows, known):
    if known.shape != (len(token_rows),):
        raise ValueError('Known-vocabulary mask does not match token rows')
    groups = defaultdict(list)
    for i,row in enumerate(token_rows):
        if row['split'] not in ('test','gen_test') or row['voice'] not in VOICES or row['source_role'] not in ROLES:
            raise ValueError('Fidelity/filler metrics require valid held-out token rows')
        for vocabulary in ('all','known' if known[i] else 'oov'):
            groups[(row['split'],row['voice'],row['source_role'],vocabulary)].append(i)
    return groups


def aggregate_filler_metrics(token_rows, before, after, gold_ids, zero_edits, *, layer, method):
    if before.ndim != 3 or before.shape[1] != 2 or after.shape != before.shape or len(before) != len(token_rows):
        raise ValueError('Require aligned [token, two roles, vocabulary] logits')
    if not torch.isfinite(before).all() or not torch.isfinite(after).all():
        raise ValueError('Nonfinite filler logits')
    n, _, size = before.shape
    if gold_ids.shape != (n,) or zero_edits.shape != (n,) or (gold_ids < -1).any() or (gold_ids >= size).any():
        raise ValueError('Invalid gold labels or zero-edit mask')
    roles = torch.tensor([ROLES.index(row['source_role']) for row in token_rows])
    before_label = before[torch.arange(n),roles].argmax(-1)
    after_label = after[torch.arange(n),1-roles].argmax(-1)
    known = gold_ids >= 0
    output = []
    for (split,voice,role,vocabulary),ids in sorted(_groups(token_rows,known).items()):
        supported = [i for i in ids if known[i]]
        correct_before = int((before_label[supported]==gold_ids[supported]).sum())
        correct_after = int((after_label[supported]==gold_ids[supported]).sum())
        target = before[ids].double().flip(1)
        profile_error = float((after[ids].double()-target).norm()/target.norm().clamp_min(1e-12))
        output.append(dict(layer=int(layer),method=method,split=split,voice=voice,source_role=role,vocabulary=vocabulary,
            tokens=len(ids),known_tokens=len(supported),filler_before_correct=correct_before,
            filler_after_target_correct=correct_after,
            filler_before_accuracy=correct_before/len(supported) if supported else None,
            filler_after_target_accuracy=correct_after/len(supported) if supported else None,
            filler_predicted_label_retention=float((before_label[ids]==after_label[ids]).double().mean()),
            native_ideal_logits_relative_error=profile_error if method=='tpr_native_role_swap' else None,
            zero_edit_tokens=int(zero_edits[ids].sum())))
    return output


def aggregate_soft_fidelity(token_rows, before_codes, after_codes, permutation, known, *, layer):
    if before_codes.ndim != 2 or after_codes.shape != before_codes.shape or len(before_codes)!=len(token_rows):
        raise ValueError('Require aligned [token, slot] code indices')
    slots = before_codes.shape[1]
    p = list(permutation)
    if sorted(p)!=list(range(slots)) or any(p[p[i]]!=i for i in range(slots)):
        raise ValueError('Require the frozen involutive slot permutation')
    if (before_codes<0).any() or (after_codes<0).any():
        raise ValueError('Code indices must be nonnegative')
    target_codes = before_codes[:,p]
    target_bag = (target_codes.sort(1).values==before_codes.sort(1).values).all(1)
    if not target_bag.all():
        raise AssertionError('Permuted latent target must preserve every code bag exactly')
    slot_match = after_codes==target_codes
    all_match = slot_match.all(1)
    bag_retained = (after_codes.sort(1).values==before_codes.sort(1).values).all(1)
    output=[]
    for (split,voice,role,vocabulary),ids in sorted(_groups(token_rows,known).items()):
        matches=int(slot_match[ids].sum()); count=len(ids); total=count*slots
        output.append(dict(layer=int(layer),split=split,voice=voice,source_role=role,vocabulary=vocabulary,
            tokens=count,slots_per_token=slots,slot_matches=matches,total_slots=total,
            slot_code_match_fraction=matches/total,all_codes_match_tokens=int(all_match[ids].sum()),
            all_codes_match_fraction=float(all_match[ids].double().mean()),
            bag_retained_tokens=int(bag_retained[ids].sum()),code_bag_retention=float(bag_retained[ids].double().mean()),
            target_bag_preserved_tokens=int(target_bag[ids].sum()),target_code_bag_preservation=1.))
    return output


def alignment_metrics(q, donor_q, token_rows, permutation, *, layer, split):
    if q.ndim!=3 or donor_q.shape!=q.shape or len(q)!=len(token_rows) or not len(q):
        raise ValueError('Require aligned source/donor quantized tensors')
    if split not in ('val','test','gen_test') or any(row['split']!=split for row in token_rows):
        raise ValueError('Alignment cohorts must be a single held-out/validation split')
    if not torch.isfinite(q).all() or not torch.isfinite(donor_q).all():
        raise ValueError('Nonfinite native quantized features')
    output=[]
    for voice in VOICES:
        ids=[i for i,row in enumerate(token_rows) if row['voice']==voice]
        if not ids: continue
        source,target=q[ids].double(),donor_q[ids].double()
        identity=float((source-target).square().mean())
        permuted=float((source[:,permutation]-target).square().mean())
        output.append(dict(layer=int(layer),split=split,voice=voice,tokens=len(ids),identity_mse=identity,
            permuted_mse=permuted,relative_mse_reduction=(identity-permuted)/identity if identity else None))
    return output


@torch.no_grad()
def _logits(probe,payload,h):
    return torch.cat([probe((batch-payload['center'])/payload['scale']) for batch in h.split(256)])


def _tokens(rows):
    return [dict(split=row['split'],voice=row['voice'],source_role=role,filler=row[role])
            for row in rows for role in ROLES]


def diagnose(run_dir):
    from run_native_tpr_nov6 import build_edits, load_plan, pair_rows, read_csv, verify_calibration
    run_dir=Path(run_dir).resolve()
    if (run_dir/'diagnostics').exists():
        raise ValueError('Diagnostics output already exists')
    declared=json.loads((run_dir/'plan.json').read_text())
    args=SimpleNamespace(run_dir=run_dir,batch_size=declared['batch_size'])
    plan,reference=load_plan(args)
    verify_calibration(args)
    layers=plan['layers']
    evaluation=require_completed_evaluation(run_dir,layers)
    for stage in ('calibration','evaluation'):
        runtime=json.loads((run_dir/stage/'runtime_source.json').read_text())
        if runtime['source_sha256']!=plan['source_sha256'] or runtime['differences_from_plan']:
            raise ValueError('Stage runtime sources differ from the frozen plan')
        for name,expected in plan['source_sha256'].items():
            if digest(run_dir/stage/'source_snapshot'/name)!=expected:
                raise ValueError('Frozen stage source snapshot changed')
    calibration=json.loads((run_dir/'calibration/metadata.json').read_text())
    source=torch_load(reference/'extraction/states.pt')
    if source['layers']!=layers or source['states'].dtype!=torch.bfloat16:
        raise ValueError('Require frozen layers and BF16 baseline states')
    ids=[i for i,row in enumerate(source['rows']) if row['split'] in ('test','gen_test')]
    rows=[source['rows'][i] for i in ids]
    token_rows=_tokens(rows)
    if len(ids)!=evaluation['rows']:
        raise ValueError('Held-out row count differs from evaluation')
    edits,native_checks=build_edits(source,reference,ids,args,plan)
    records=read_csv(run_dir/'evaluation/results.csv')
    if len(records)!=evaluation['result_rows']:
        raise ValueError('Evaluation result count differs from metadata')
    edit_verification=verify_evaluated_edits(records,ids,rows,layers,edits)
    filler_records=[]; fidelity_records=[]; alignment_records=[]; pairing_hashes={}
    for li,layer in enumerate(layers):
        payload=torch_load(reference/f'training/layer_{layer}.pt')
        soft=torch_load(Path(plan['soft_reference_run'])/f'training/layer_{layer}.pt')
        permutation=json.loads((run_dir/f'calibration/layer_{layer}.json').read_text())['permutation']
        with torch.random.fork_rng(devices=[]):
            probe=BilinearTPRProbe(**payload['config'])
        probe.load_state_dict(payload['state_dict']); probe.eval().requires_grad_(False)
        vocabulary=payload['fillers']
        if len(vocabulary)!=19 or set(vocabulary)!=set(calibration['vocabulary']):
            raise ValueError('Require the original 19-filler training vocabulary')
        gold=torch.tensor([vocabulary.index(row['filler']) if row['filler'] in vocabulary else -1 for row in token_rows])
        h=source['states'][ids,li].float().flatten(0,1)
        before=_logits(probe,payload,h)
        q,codes=encode_native_soft(soft,h)
        for method in METHODS:
            delta=edits[layer][method].flatten(0,1)
            after=simulate_local_edit(h,delta)
            after_logits=_logits(probe,payload,after)
            filler_records+=aggregate_filler_metrics(token_rows,before,after_logits,gold,
                delta.double().norm(dim=-1)==0,layer=layer,method=method)
            if method=='soft_native_slot_swap':
                _,after_codes=encode_native_soft(soft,after)
                fidelity_records+=aggregate_soft_fidelity(token_rows,codes,after_codes,permutation,gold>=0,layer=layer)
        for split in ('val','test','gen_test'):
            cohort_ids=[i for i,row in enumerate(source['rows']) if row['split']==split]
            donors=pair_rows(source['rows'],cohort_ids)
            cohort_rows=_tokens([source['rows'][i] for i in cohort_ids])
            if split=='val':
                cohort_q,_=encode_native_soft(soft,source['states'][cohort_ids,li].float().flatten(0,1))
            else:
                token_ids=[i for i,row in enumerate(token_rows) if row['split']==split]
                cohort_q=q[token_ids]
            alignment_records+=alignment_metrics(cohort_q,cohort_q[donors],cohort_rows,permutation,layer=layer,split=split)
            if li==0:
                import hashlib
                pairing_hashes[split]=hashlib.sha256(json.dumps(dict(row_ids=cohort_ids,
                    donor_token_local_ids=donors.tolist()),sort_keys=True).encode()).hexdigest()
        print(f'Native local diagnostics L{layer} complete',flush=True)
    # Each all-vocabulary partition counts each token exactly once per condition.
    for layer in layers:
        for method in METHODS:
            if sum(r['tokens'] for r in filler_records if r['layer']==layer and r['method']==method and r['vocabulary']=='all')!=len(token_rows):
                raise ValueError('Filler diagnostic coverage is incomplete')
        if sum(r['tokens'] for r in fidelity_records if r['layer']==layer and r['vocabulary']=='all')!=len(token_rows):
            raise ValueError('Soft fidelity coverage is incomplete')
        for split in ('val','test','gen_test'):
            if sum(r['tokens'] for r in alignment_records if r['layer']==layer and r['split']==split)!=2*plan['expected_counts'][split]:
                raise ValueError('Soft alignment coverage is incomplete')
    out=run_dir/'diagnostics'; out.mkdir(exist_ok=False)
    tables={'filler_metrics.csv':filler_records,'soft_fidelity.csv':fidelity_records,'soft_alignment.csv':alignment_records}
    for name,table in tables.items(): write_csv(out/name,table)
    inputs=['plan.json','calibration/metadata.json','calibration/train_pairing.json','calibration/runtime_source.json',
            'evaluation/metadata.json','evaluation/results.csv','evaluation/runtime_source.json']
    inputs += ['calibration/'+name for name in calibration['calibration_sha256']]
    write_json(out/'metadata.json',dict(status='complete',created_at=datetime.now(timezone.utc).isoformat(),
        layers=layers,methods=list(METHODS),alpha=1.,norm_matching=False,tokens_per_condition=len(token_rows),
        metrics_rows={name:len(table) for name,table in tables.items()},
        files_sha256={name:digest(out/name) for name in tables},input_sha256={name:digest(run_dir/name) for name in inputs},
        reference_run=str(reference),soft_reference_run=plan['soft_reference_run'],
        verified_reference_sha256=plan['reference_sha256'],verified_soft_reference_sha256=plan['soft_reference_sha256'],
        scientific_source_sha256=plan['source_sha256'],diagnostic_script_sha256=digest(Path(__file__)),
        edit_verification=edit_verification,native_checks=native_checks,alignment_pairing_sha256=pairing_hashes,
        semantics=dict(state_source='Local CPU BF16(h.float()+delta), NOT actual live hook activations',
            edit_verification='Every completed evaluation row norm and exact-zero flag checked against the same reconstructed native edit',
            filler='Original TPR readout for all methods; source-role head before, target-role head after',
            filler_accuracy='Gold accuracy only for 19 training fillers; known_tokens denominator; None when unsupported',
            filler_retention='Equality of predicted labels in known vocabulary; OOV retention is NOT lexical identity preservation',
            native_profile='TPR full role/filler logits after BF16 compared with source logits flipped along the role axis',
            soft_target='Frozen permutation q[:,p] and codes[:,p]; exact target code-bag preservation is asserted',
            soft_fidelity='Re-encoded simulated hidden states compared with target codes; bag retention compares with the original code multiset',
            soft_alignment='Frozen source/donor quantized tensors on val/test/gen_test, paired by same-filler opposite-role twins after evaluation; no refitting or selection',
            vocabulary='all includes known+oov; empty subsets omitted; source_role is agent or patient',
            inference='No extra role head, new training, live LM execution, or intervention tuning')))
    print(f'Wrote {out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    diagnose(args.run_dir)
