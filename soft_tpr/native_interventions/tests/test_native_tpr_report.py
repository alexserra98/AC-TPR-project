"""Independent CSV/provenance checks for the frozen native intervention report."""

from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
import hashlib
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/temporary'))
import report_native_tpr_nov6 as report


def put_json(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')


def put_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def corpus():
    return report.read_csv(ROOT/'data/active_pass_long/corpus.csv')


def stage_rows(stage='validation'):
    source=corpus(); splits={'val'} if stage=='validation' else {'test','gen_test'}
    vocabulary={r[role] for r in source if r['split']=='train' for role in ('agent','patient')}
    base=[]; results=[]; reference=[]
    for row_id,row in enumerate(source):
        if row['split'] not in splits: continue
        bm=0. if row_id==next((i for i,r in enumerate(source) if r['split']=='gen_test' and r['voice']=='passive'),-1) else -1.
        base.append(dict(row_id=row_id,split=row['split'],voice=row['voice'],agent=row['agent'],patient=row['patient'],
                         agent_logit=2.,patient_logit=2.+bm))
        for layer in report.LAYERS:
            for method in report.METHODS:
                zero=method=='soft_native_slot_swap' and layer==22
                em=bm if zero else .25 if method=='raw_centroid' or row_id%3==0 else -0.5
                norm=0. if zero else 10. if method=='raw_centroid' else 2.
                known=sum(row[role] in vocabulary for role in ('agent','patient'))
                record=dict(row_id=row_id,split=row['split'],voice=row['voice'],quad_id=row['quad_id'],layer=layer,
                    method=method,alpha=1.,known_tokens=known,sentence_has_oov=known<2,
                    baseline_margin=bm,edited_margin=em,margin_change=em-bm,baseline_agent_win=bm<0,
                    patient_win=em>0,flip=bm<0 and em>0,agent_edit_norm=norm,patient_edit_norm=norm,
                    agent_edit_is_zero=zero,patient_edit_is_zero=zero,
                    result_origin='exact_identity' if zero else 'reference_raw' if method=='raw_centroid' else 'scored')
                results.append(record)
                if method=='raw_centroid': reference.append(dict(record))
    return base,results,reference


def independent_summary(rows):
    groups=defaultdict(list)
    for r in rows: groups[r['split'],r['voice'],r['layer'],r['method']].append(r)
    output=[]
    for key,group in sorted(groups.items()):
        eligible=[r for r in group if r['baseline_margin']<0]
        flips=len([r for r in eligible if r['edited_margin']>0])
        output.append(dict(split=key[0],voice=key[1],layer=key[2],method=key[3],alpha=1.,sentences=len(group),
            baseline_agent_wins=len(eligible),flips=flips,conditional_flip=flips/len(eligible),
            patient_preference=len([r for r in group if r['edited_margin']>0])/len(group),
            mean_margin_change=sum(r['margin_change'] for r in group)/len(group),
            mean_agent_edit_norm=sum(r['agent_edit_norm'] for r in group)/len(group),
            mean_patient_edit_norm=sum(r['patient_edit_norm'] for r in group)/len(group),
            zero_sentence_edits=len([r for r in group if r['agent_edit_norm']==r['patient_edit_norm']==0]),
            zero_token_edits=sum(int(r[role+'_edit_norm']==0) for r in group for role in ('agent','patient'))))
    return output


def stage_metadata(rows, run):
    checks=[]
    for layer in report.LAYERS:
        for method in report.METHODS:
            selected=[r for r in rows if (r['layer'],r['method'])==(layer,method)]
            norms=[r[role+'_edit_norm'] for r in selected for role in ('agent','patient')]
            checks.append(dict(layer=layer,method=method,tokens=len(norms),native_norm_min=min(norms),
                native_norm_mean=sum(norms)/len(norms),native_norm_max=max(norms),
                exact_zero_tokens=norms.count(0.),exact_zero_sentences=sum(r['agent_edit_is_zero'] and r['patient_edit_is_zero'] for r in selected)))
    return dict(status='complete',layers=list(report.LAYERS),methods=list(report.METHODS),alphas=[1.],conditions=9,
        norm_matching=False,exact_reference_baseline=True,rows=len(rows)//9,result_rows=len(rows),native_checks=checks,
        exact_identity_conditions=[dict(layer=22,method='soft_native_slot_swap')],
        calibration_metadata_sha256=report.digest(run/'calibration/metadata.json'))


def calibration_fixture(run):
    source=corpus(); ids=[i for i,r in enumerate(source) if r['split']=='train']
    lookup={(source[i]['quad_id'],source[i]['voice'],source[i]['swapped']):local for local,i in enumerate(ids)}
    donors=[]
    for i in ids:
        r=source[i]; twin=lookup[r['quad_id'],r['voice'],str(1-int(r['swapped']))]
        donors.extend([2*twin+1,2*twin])
    put_json(run/'calibration/train_pairing.json',dict(split='train',row_ids=ids,donor_token_local_ids=donors))
    files={}
    for layer in report.LAYERS:
        permutation=[1,0,2,3,4,5,6,7]
        obj=dict(layer=layer,permutation=permutation,training_tokens=6600,diagnostics=dict(
            tokens=6600,slots=8,filler_dim=32,candidate_count=764,permutation=permutation,
            best=dict(permutation=permutation,objective_sse=6600*8*32*.75,mse=.75),
            identity=dict(permutation=list(range(8)),objective_sse=6600*8*32,mse=1.),relative_improvement=.25))
        path=run/f'calibration/layer_{layer}.json';put_json(path,obj);files[path.name]=report.digest(path)
    metadata=dict(status='complete',layers=list(report.LAYERS),split='train',training_sentences=3300,training_tokens=6600,
        shared_across_voices=True,vocabulary=sorted({source[i][role] for i in ids for role in ('agent','patient')}),
        calibration_sha256=files,train_pairing_sha256=report.digest(run/'calibration/train_pairing.json'))
    put_json(run/'calibration/metadata.json',metadata)
    put_json(run/'calibration/runtime_source.json',dict(source_sha256={},differences_from_plan={}))
    return metadata


def diagnostic_fixture(run,plan,calibration,rows):
    source=corpus(); vocab=set(calibration['vocabulary'])
    groups=defaultdict(list)
    for row_id,r in enumerate(source):
        if r['split'] not in ('test','gen_test'):continue
        for role in ('agent','patient'):
            known=r[role] in vocab
            for subset in ('all','known' if known else 'oov'):
                groups[r['split'],r['voice'],role,subset].append((row_id,known))
    filler=[]; fidelity=[]; alignment=[]
    for layer in report.LAYERS:
        for key, tokens in sorted(groups.items()):
            split,voice,role,subset=key; n=len(tokens); k=sum(known for _,known in tokens)
            for method in report.METHODS:
                zero=layer==22 and method=='soft_native_slot_swap'
                filler.append(dict(layer=layer,method=method,split=split,voice=voice,source_role=role,vocabulary=subset,
                    tokens=n,known_tokens=k,filler_before_correct=k,filler_after_target_correct=k,
                    filler_before_accuracy=1. if k else None,filler_after_target_accuracy=1. if k else None,
                    filler_predicted_label_retention=1.,native_ideal_logits_relative_error=.001 if method=='tpr_native_role_swap' else None,
                    zero_edit_tokens=n if zero else 0))
            fidelity.append(dict(layer=layer,split=split,voice=voice,source_role=role,vocabulary=subset,tokens=n,
                slots_per_token=8,slot_matches=6*n,total_slots=8*n,slot_code_match_fraction=.75,
                all_codes_match_tokens=0,all_codes_match_fraction=0.,bag_retained_tokens=n,code_bag_retention=1.,
                target_bag_preserved_tokens=n,target_code_bag_preservation=1.))
        for split in ('val','test','gen_test'):
            for voice in ('active','passive'):
                n=2*sum((r['split'],r['voice'])==(split,voice) for r in source)
                alignment.append(dict(layer=layer,split=split,voice=voice,tokens=n,identity_mse=1.,permuted_mse=.9,relative_mse_reduction=.1))
    tables={'filler_metrics.csv':filler,'soft_fidelity.csv':fidelity,'soft_alignment.csv':alignment}
    for name,table in tables.items():put_csv(run/'diagnostics'/name,table)
    names=['plan.json','calibration/metadata.json','calibration/train_pairing.json','calibration/runtime_source.json',
        'evaluation/metadata.json','evaluation/results.csv','evaluation/runtime_source.json']
    names+=['calibration/'+n for n in calibration['calibration_sha256']]
    pairing_hashes={}
    for split in ('val','test','gen_test'):
        ids=[i for i,r in enumerate(source) if r['split']==split]
        lookup={(source[i]['quad_id'],source[i]['voice'],source[i]['swapped']):local for local,i in enumerate(ids)}
        donors=[]
        for i in ids:
            r=source[i];twin=lookup[r['quad_id'],r['voice'],str(1-int(r['swapped']))]
            donors.extend([2*twin+1,2*twin])
        pairing_hashes[split]=hashlib.sha256(json.dumps(dict(row_ids=ids,donor_token_local_ids=donors),sort_keys=True).encode()).hexdigest()
    metadata=dict(status='complete',layers=list(report.LAYERS),methods=list(report.METHODS),alpha=1.,norm_matching=False,
        alignment_pairing_sha256=pairing_hashes,
        tokens_per_condition=4192,files_sha256={name:report.digest(run/'diagnostics'/name) for name in tables},
        metrics_rows={name:len(table) for name,table in tables.items()},scientific_source_sha256=plan['source_sha256'],
        verified_reference_sha256=plan['reference_sha256'],verified_soft_reference_sha256=plan['soft_reference_sha256'],
        diagnostic_script_sha256=report.digest(ROOT/'scripts/temporary/diagnose_native_tpr_nov6.py'),
        input_sha256={name:report.digest(run/name) for name in names},
        edit_verification=dict(verified_result_rows=len(rows),verified_token_edits=2*len(rows),
            zero_token_edits=sum(r[role+'_edit_is_zero'] for r in rows for role in ('agent','patient')),
            maximum_relative_norm_error=0.))
    put_json(run/'diagnostics/metadata.json',metadata)
    return tables,metadata


@pytest.fixture
def valid_stage():
    base,rows,reference=stage_rows()
    return base,rows,independent_summary(rows),deepcopy(base),reference,'validation'


def test_complete_native_validation_has_nine_conditions_and_identity(valid_stage):
    result=report.audit_stage(*valid_stage)
    assert len(result)==18 and sum(r['sentences'] for r in result)==3708
    assert all(r['conditional_flip']==1. for r in result if r['method']=='raw_centroid')
    assert all(r['zero_token_edits']==412 and r['flips']==0 for r in result if r['method']=='soft_native_slot_swap' and r['layer']==22)


def test_heldout_baseline_tie_is_excluded_from_flip_denominator():
    base,rows,reference=stage_rows('evaluation')
    result=report.audit_stage(base,rows,independent_summary(rows),deepcopy(base),reference,'evaluation')
    group=next(r for r in result if (r['split'],r['voice'],r['layer'],r['method'])==('gen_test','passive',20,'raw_centroid'))
    assert group['sentences']==840 and group['baseline_agent_wins']==839 and group['flips']==839
    assert group['conditional_flip']==group['patient_preference']==1.


@pytest.mark.parametrize('change,match',[
    (lambda args:args[0][0].update(agent_logit=2.00001),'Baseline differs'),
    (lambda args:args[1].pop(),'Incomplete condition'),
    (lambda args:args[1].append(dict(args[1][0])),'Duplicate/unexpected'),
    (lambda args:args[1][0].update(alpha=4),'alpha=1'),
    (lambda args:args[1][0].update(agent_edit_is_zero=True),'norm/zero'),
    (lambda args:args[1][0].update(flip=False),'outcome flag'),
    (lambda args:args[1][0].update(edited_margin=2.,margin_change=3.),'Reused raw output'),
    (lambda args:args[1][1].update(result_origin='exact_identity'),'Identity condition'),
    (lambda args:args[2][0].update(flips=0),'Summary differs'),
    (lambda args:args[4].append(dict(args[4][0])),'Duplicate reference'),
])
def test_corrupt_stage_rejected(valid_stage,change,match):
    args=list(deepcopy(valid_stage));change(args)
    with pytest.raises(ValueError,match=match):report.audit_stage(*args)


def test_calibration_is_train_only_with_real_corpus_pairing(tmp_path):
    calibration_fixture(tmp_path)
    metadata,perms=report.audit_calibration(tmp_path,dict(source_sha256={}),corpus(),{})
    assert metadata['training_tokens']==6600 and len(perms)==3
    pair_path=tmp_path/'calibration/train_pairing.json';pair=json.loads(pair_path.read_text())
    pair['donor_token_local_ids'][:2]=reversed(pair['donor_token_local_ids'][:2]);put_json(pair_path,pair)
    metadata['train_pairing_sha256']=report.digest(pair_path);put_json(tmp_path/'calibration/metadata.json',metadata)
    with pytest.raises(ValueError,match='opposite-role'):report.audit_calibration(tmp_path,dict(source_sha256={}),corpus(),{})


@pytest.mark.parametrize('field,value', [('split','val'),('shared_across_voices',False),('training_tokens',6598)])
def test_calibration_cannot_change_cohort(tmp_path,field,value):
    metadata=calibration_fixture(tmp_path);metadata[field]=value;put_json(tmp_path/'calibration/metadata.json',metadata)
    with pytest.raises(ValueError,match='non-train'):report.audit_calibration(tmp_path,dict(source_sha256={}),corpus(),{})


def test_calibration_file_hash_and_involution_are_checked(tmp_path):
    metadata=calibration_fixture(tmp_path);path=tmp_path/'calibration/layer_20.json'
    obj=json.loads(path.read_text());obj['permutation']=[1,2,0,3,4,5,6,7];put_json(path,obj)
    with pytest.raises(ValueError,match='Calibration changed'):report.audit_calibration(tmp_path,dict(source_sha256={}),corpus(),{})
    metadata['calibration_sha256'][path.name]=report.digest(path);put_json(tmp_path/'calibration/metadata.json',metadata)
    with pytest.raises(ValueError,match='involution'):report.audit_calibration(tmp_path,dict(source_sha256={}),corpus(),{})


def test_runtime_source_snapshot_must_match_plan(tmp_path):
    path=tmp_path/'evaluation/source_snapshot/scientific.py';path.parent.mkdir(parents=True);path.write_text('original')
    plan=dict(source_sha256={'scientific.py':report.digest(path)})
    put_json(tmp_path/'evaluation/runtime_source.json',dict(source_sha256=plan['source_sha256'],differences_from_plan={}))
    report.audit_runtime(tmp_path,'evaluation',plan,{})
    path.write_text('changed')
    with pytest.raises(ValueError,match='snapshot changed'):report.audit_runtime(tmp_path,'evaluation',plan,{})


def test_stage_metadata_matches_calibration_zero_conditions_and_norms(tmp_path):
    calibration_fixture(tmp_path);_,rows,_=stage_rows();metadata=stage_metadata(rows,tmp_path)
    report.audit_stage_metadata(metadata,rows,tmp_path)
    metadata['native_checks'][0]['native_norm_mean']*=2
    with pytest.raises(ValueError,match='native check'):report.audit_stage_metadata(metadata,rows,tmp_path)
    metadata=stage_metadata(rows,tmp_path);metadata['exact_identity_conditions']=[]
    with pytest.raises(ValueError,match='identity metadata'):report.audit_stage_metadata(metadata,rows,tmp_path)
    metadata=stage_metadata(rows,tmp_path);metadata['calibration_metadata_sha256']='changed'
    with pytest.raises(ValueError,match='calibration metadata'):report.audit_stage_metadata(metadata,rows,tmp_path)


@pytest.fixture
def valid_diagnostics(tmp_path):
    calibration=calibration_fixture(tmp_path)
    plan=dict(source_sha256={},reference_sha256={},soft_reference_sha256={});put_json(tmp_path/'plan.json',plan)
    _,rows,_=stage_rows('evaluation');put_csv(tmp_path/'evaluation/results.csv',rows)
    put_json(tmp_path/'evaluation/metadata.json',stage_metadata(rows,tmp_path))
    put_json(tmp_path/'evaluation/runtime_source.json',dict(source_sha256={},differences_from_plan={}))
    tables,metadata=diagnostic_fixture(tmp_path,plan,calibration,rows)
    return tmp_path,plan,corpus(),calibration,rows,{},tables,metadata


def test_complete_diagnostics_keep_lexical_and_code_fidelity_distinct(valid_diagnostics):
    args=valid_diagnostics
    tables,metadata=report.audit_diagnostics(*args[:6])
    filler,fidelity,alignment,profile=report.diagnostic_tables(tables)
    assert (len(filler),len(fidelity),len(alignment),len(profile))==(12,4,6,4)
    assert all(r[-1]=='100%' and r[4]=='75.00%–75.00%' and r[5]=='0.00%–0.00%' for r in fidelity)
    assert all(r['known_tokens']=='0' and r['filler_after_target_accuracy']=='' for r in tables['filler_metrics.csv'] if r['vocabulary']=='oov')
    assert metadata['tokens_per_condition']==4192


@pytest.mark.parametrize('name,field,value,match',[
    ('filler_metrics.csv','known_tokens',0,'denominator'),
    ('filler_metrics.csv','filler_after_target_accuracy',0.,'accuracy'),
    ('filler_metrics.csv','zero_edit_tokens',1,'zero-token'),
    ('soft_fidelity.csv','total_slots',1,'denominator'),
    ('soft_fidelity.csv','slot_code_match_fraction',1.,'fidelity fraction'),
    ('soft_fidelity.csv','target_bag_preserved_tokens',0,'fidelity fraction'),
    ('soft_alignment.csv','tokens',1,'denominator'),
    ('soft_alignment.csv','relative_mse_reduction',.8,'alignment reduction'),
])
def test_corrupt_diagnostic_metrics_rejected_even_with_updated_hash(valid_diagnostics,name,field,value,match):
    run,plan,source,calibration,rows,inputs,tables,metadata=valid_diagnostics
    tables[name][0][field]=value;put_csv(run/'diagnostics'/name,tables[name])
    metadata['files_sha256'][name]=report.digest(run/'diagnostics'/name);put_json(run/'diagnostics/metadata.json',metadata)
    with pytest.raises(ValueError,match=match):report.audit_diagnostics(run,plan,source,calibration,rows,inputs)


def test_diagnostic_csv_and_input_hashes_are_required(valid_diagnostics):
    run,plan,source,calibration,rows,inputs,tables,metadata=valid_diagnostics
    path=run/'diagnostics/soft_alignment.csv';path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='CSV changed'):report.audit_diagnostics(run,plan,source,calibration,rows,inputs)
    metadata['files_sha256'][path.name]=report.digest(path);put_json(run/'diagnostics/metadata.json',metadata)
    (run/'evaluation/results.csv').write_text('changed')
    with pytest.raises(ValueError,match='input changed'):report.audit_diagnostics(run,plan,source,calibration,rows,inputs)


def test_filler_partitions_cannot_disagree_while_each_accuracy_is_valid(valid_diagnostics):
    run,plan,source,calibration,rows,inputs,tables,metadata=valid_diagnostics
    row=next(r for r in tables['filler_metrics.csv'] if r['vocabulary']=='all')
    row['filler_after_target_correct']-=1
    row['filler_after_target_accuracy']=row['filler_after_target_correct']/row['known_tokens']
    name='filler_metrics.csv';put_csv(run/'diagnostics'/name,tables[name])
    metadata['files_sha256'][name]=report.digest(run/'diagnostics'/name);put_json(run/'diagnostics/metadata.json',metadata)
    with pytest.raises(ValueError,match='partitions disagree'):report.audit_diagnostics(run,plan,source,calibration,rows,inputs)


def test_all_slots_matching_requires_all_tokens_matching(valid_diagnostics):
    run,plan,source,calibration,rows,inputs,tables,metadata=valid_diagnostics
    row=tables['soft_fidelity.csv'][0];row['slot_matches']=row['total_slots'];row['slot_code_match_fraction']=1.
    name='soft_fidelity.csv';put_csv(run/'diagnostics'/name,tables[name])
    metadata['files_sha256'][name]=report.digest(run/'diagnostics'/name);put_json(run/'diagnostics/metadata.json',metadata)
    with pytest.raises(ValueError,match='Inconsistent fidelity'):report.audit_diagnostics(run,plan,source,calibration,rows,inputs)


def test_diagnostic_pairing_hash_cannot_change(valid_diagnostics):
    run,plan,source,calibration,rows,inputs,tables,metadata=valid_diagnostics
    metadata['alignment_pairing_sha256']['test']='changed';put_json(run/'diagnostics/metadata.json',metadata)
    with pytest.raises(ValueError,match='pairing hash'):report.audit_diagnostics(run,plan,source,calibration,rows,inputs)
