"""Check native-only calibration, zero edits, and exact cached raw controls."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
import run_native_tpr_nov6 as driver


def quad_rows(split='train', quad='q0'):
    return [dict(split=split, quad_id=quad, voice=voice, swapped=str(swapped),
                 agent='Anna' if swapped==0 else 'David', patient='David' if swapped==0 else 'Anna', verb='kick')
            for swapped in (0, 1) for voice in driver.VOICES]


def test_pairing_is_local_same_voice_same_filler_and_involutive():
    rows=quad_rows()+quad_rows('val','q1')
    ids=[3, 0, 2, 1]
    donors=driver.pair_rows(rows,ids)
    assert donors.dtype==torch.long and donors.shape==(8,)
    assert torch.equal(donors[donors],torch.arange(8))
    for i,global_id in enumerate(ids):
        for role in range(2):
            other=int(donors[2*i+role])
            donor=rows[ids[other//2]]
            assert donor['quad_id']==rows[global_id]['quad_id']
            assert donor['voice']==rows[global_id]['voice']
            assert donor['swapped']!=rows[global_id]['swapped']
            assert other%2==1-role
            assert donor[driver.ROLES[other%2]]==rows[global_id][driver.ROLES[role]]


@pytest.mark.parametrize('problem',('missing_twin','cross_split','wrong_filler','duplicate'))
def test_invalid_training_twins_are_rejected(problem):
    rows=quad_rows();ids=list(range(4))
    if problem=='missing_twin':ids.pop()
    elif problem=='cross_split':rows[2]['split']='val'
    elif problem=='wrong_filler':rows[2]['agent']='Mary'
    else:ids.append(0)
    with pytest.raises(ValueError):driver.pair_rows(rows,ids)


def test_calibration_uses_only_train_and_one_permutation_for_both_voices(tmp_path,monkeypatch):
    run=tmp_path/'run';run.mkdir()
    rows=quad_rows()+quad_rows('val','q1')+quad_rows('test','q2')
    source=dict(rows=rows,states=torch.arange(len(rows)*3*2*2,dtype=torch.float32).reshape(len(rows),3,2,2))
    plan=dict(soft_reference_run=str(tmp_path/'soft'),expected_counts={'train':4})
    args=SimpleNamespace(run_dir=run,batch_size=2)
    encoded=[];selected=[]
    monkeypatch.setattr(driver,'load_plan',lambda args:(plan,tmp_path/'reference'))
    monkeypatch.setattr(driver,'snapshot',lambda *args:None)
    monkeypatch.setattr(driver,'torch_load',lambda path:source if path.name=='states.pt' else {})
    def encode(payload,h):
        encoded.append(h.clone())
        return h[:,None,:].repeat(1,8,1),torch.zeros(len(h),8,dtype=torch.long)
    def select(q,donor_q):
        selected.append((q.clone(),donor_q.clone()))
        return list(range(8)),dict(candidate_count=764)
    monkeypatch.setattr(driver,'encode_native_soft',encode)
    monkeypatch.setattr(driver,'select_slot_permutation',select)
    driver.calibrate(args)
    assert len(encoded)==len(selected)==3
    donors=driver.pair_rows(rows,list(range(4)))
    for li,(h,(q,donor_q)) in enumerate(zip(encoded,selected)):
        torch.testing.assert_close(h,source['states'][:4,li].flatten(0,1))
        torch.testing.assert_close(donor_q,q[donors])
        assert len(h)==8
    metadata=json.loads((run/'calibration/metadata.json').read_text())
    assert metadata['shared_across_voices'] and metadata['training_tokens']==8
    assert len(metadata['calibration_sha256'])==3
    driver.verify_calibration(args)
    path=run/'calibration/layer_20.json'
    value=json.loads(path.read_text());value['permutation']=[1,0,2,3,4,5,6,7]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='changed after calibration'):driver.verify_calibration(args)


class Inputs(dict):
    def to(self,device):
        assert device=='cuda'
        return self


class Tokenizer:
    def pad(self,values,padding,return_tensors):
        assert padding and return_tensors=='pt'
        return Inputs(input_ids=torch.tensor(values['input_ids']))


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    run=tmp_path/'run';(run/'calibration').mkdir(parents=True)
    reference=tmp_path/'reference';cache=tmp_path/'norm'
    rows=[dict(split=split,voice='active' if i%2==0 else 'passive',agent='Anna',patient='David',quad_id=f'q{i}')
          for i,split in enumerate(('val','val','test','test','gen_test','gen_test'))]
    source=dict(rows=rows,input_ids=[[i] for i in range(6)],answers=torch.zeros(6,2,dtype=torch.long),
                positions=torch.zeros(6,2,dtype=torch.long),baseline=torch.tensor([[10.+i,1.+i] for i in range(6)]))
    args=SimpleNamespace(run_dir=run,batch_size=2)
    plan=dict(norm_reference_run=str(cache),soft_reference_run=str(tmp_path/'soft'))
    calls=[]
    def score(model,inputs,answers,layer=None,positions=None,delta=None):
        ids=inputs['input_ids'][:,0].tolist()
        calls.append((ids,layer,None if delta is None else delta.clone()))
        logits=source['baseline'][ids].clone()
        if delta is not None and layer!=35:logits[:,1]+=2*delta.sum(dim=(1,2))
        return logits
    def build(source,reference,ids,args,plan):
        scales=torch.tensor([1. if source['rows'][i]['voice']=='active' else 2. for i in ids])
        values=dict(raw_centroid=[[3.,0.],[0.,4.]],tpr_native_role_swap=[[.25,0.],[0.,-.5]],soft_native_slot_swap=[[0.,0.],[0.,0.]])
        return {layer:{method:torch.tensor(value).repeat(len(ids),1,1)*scales[:,None,None]
                       for method,value in values.items()} for layer in driver.LAYERS},[]
    for stage,ids in (('validation',(0,1)),('evaluation',(2,3,4,5))):
        (cache/stage).mkdir(parents=True)
        driver.write_csv(cache/stage/'baseline.csv',[dict(row_id=i,**{k:rows[i][k] for k in ('split','voice','agent','patient')},
            agent_logit=float(source['baseline'][i,0]),patient_logit=float(source['baseline'][i,1])) for i in ids])
        driver.write_csv(cache/stage/'results.csv',[dict(row_id=i,layer=layer,method='raw_centroid',alpha=alpha,
            edited_margin=-9.+14*alpha*(1 if rows[i]['voice']=='active' else 2))
            for i in ids for layer in driver.LAYERS for alpha in (1.,4.)])
    driver.write_json(run/'calibration/metadata.json',dict(vocabulary=['Anna','David']))
    for layer in driver.LAYERS:driver.write_json(run/f'calibration/layer_{layer}.json',dict(permutation=list(range(8))))
    original_build=driver.build_edits
    monkeypatch.setattr(driver,'snapshot',lambda *args:None)
    monkeypatch.setattr(driver,'build_edits',build)
    monkeypatch.setattr(driver,'score_batch',score)
    def evaluate(stage='evaluation',alphas=(1.,)):
        driver.evaluate_stage(args,plan,reference,source,Tokenizer(),
            SimpleNamespace(config=SimpleNamespace(hidden_size=2)),{},stage,list(alphas))
        return driver.read_csv(run/stage/'results.csv')
    return SimpleNamespace(run=run,args=args,plan=plan,reference=reference,cache=cache,source=source,
                           score=score,build=build,original_build=original_build,evaluate=evaluate,calls=calls)


@pytest.mark.parametrize('stage',('validation','evaluation'))
def test_baseline_requires_individual_logits_not_only_margin(experiment,monkeypatch,stage):
    monkeypatch.setattr(driver,'score_batch',lambda *args:experiment.score(*args)+1)
    with pytest.raises(ValueError,match='Baseline differs'):experiment.evaluate(stage)
    assert len(experiment.calls)==1


def test_native_norms_remain_unequal_and_identity_is_exact(experiment):
    records=experiment.evaluate()
    assert len(records)==4*9 and {float(r['alpha']) for r in records}=={1.}
    for row in records:
        assert 'target_agent_norm' not in row
        scale=1 if row['voice']=='active' else 2
        if row['method']=='tpr_native_role_swap':
            assert float(row['agent_edit_norm'])==.25*scale
            assert float(row['patient_edit_norm'])==.5*scale
        elif row['method']=='soft_native_slot_swap':
            assert row['result_origin']=='exact_identity'
            assert row['agent_edit_is_zero']==row['patient_edit_is_zero']=='True'
            assert float(row['edited_margin'])==float(row['baseline_margin'])
        else:assert row['result_origin']=='reference_raw'
    summary=driver.read_csv(experiment.run/'evaluation/summary.csv')
    assert all(int(r['zero_token_edits'])==2*int(r['sentences']) for r in summary if r['method']=='soft_native_slot_swap')
    # Two baselines, two controls, three raw sentinels, and six TPR forwards.
    assert len(experiment.calls)==13


def test_partial_zero_condition_is_scored_even_for_an_all_zero_batch(experiment,monkeypatch):
    def mixed(*args):
        edits,checks=experiment.build(*args)
        for layer in driver.LAYERS:edits[layer]['soft_native_slot_swap'][2:,0,0]=.125
        return edits,checks
    monkeypatch.setattr(driver,'build_edits',mixed)
    records=experiment.evaluate()
    soft=[r for r in records if r['method']=='soft_native_slot_swap']
    assert all(r['result_origin']=='scored' for r in soft)
    assert all(r['patient_edit_is_zero']=='True' for r in soft)
    assert all(r['agent_edit_is_zero']=='True' for r in soft if int(r['row_id']) in (2,3))


def test_raw_reuse_has_one_scored_sentinel_per_layer_and_never_reuses_alpha4(experiment):
    path=experiment.cache/'evaluation/results.csv'
    rows=driver.read_csv(path)
    for row in rows:
        if float(row['alpha'])==4.:row['edited_margin']=12345
    path.unlink();driver.write_csv(path,rows)
    experiment.evaluate()
    for layer in driver.LAYERS:
        raw=experiment.build(experiment.source,None,[2,3],None,None)[0][layer]['raw_centroid']
        calls=[(ids,l) for ids,l,delta in experiment.calls if l==layer and delta is not None and torch.equal(delta,raw)]
        assert calls==[([2,3],layer)]


def test_incomplete_first_batch_cache_never_enables_later_unsentinelled_reuse(experiment):
    path=experiment.cache/'evaluation/results.csv'
    rows=[r for r in driver.read_csv(path) if not(int(r['row_id'])==2 and int(r['layer'])==20 and float(r['alpha'])==1.)]
    path.unlink();driver.write_csv(path,rows)
    result=experiment.evaluate()
    assert all(r['result_origin']=='scored' for r in result if r['method']=='raw_centroid' and int(r['layer'])==20)


def test_raw_sentinel_rejects_changed_reference(experiment):
    path=experiment.cache/'evaluation/results.csv';rows=driver.read_csv(path)
    rows[0]['edited_margin']=100
    path.unlink();driver.write_csv(path,rows)
    with pytest.raises(AssertionError):experiment.evaluate()


@pytest.mark.parametrize('alphas',((0.,),(4.,),(1.,4.)))
def test_native_protocol_rejects_other_strengths_before_creating_outputs(experiment,alphas):
    with pytest.raises(ValueError,match='fixed alpha=1'):experiment.evaluate(alphas=alphas)
    assert not (experiment.run/'evaluation').exists()


def test_build_edits_needs_no_filler_labels_or_counterfactual_donor(experiment,monkeypatch):
    source=deepcopy(experiment.source)
    source['states']=torch.arange(6*3*2*2,dtype=torch.float32).reshape(6,3,2,2)
    means={voice:torch.tensor([[0.,0.],[3.,4.]]) for voice in driver.VOICES}
    monkeypatch.setattr(driver,'torch_load',lambda path:dict(raw_means=means))
    monkeypatch.setattr(driver,'native_role_swap',lambda payload,h:(h*.01,{}))
    monkeypatch.setattr(driver,'native_slot_swap',lambda payload,h,permutation:(torch.zeros_like(h),{}))
    ids=[2,3]
    before,_=experiment.original_build(source,experiment.reference,ids,experiment.args,experiment.plan)
    # No filler, quad, swapped, or donor metadata remains available to steering.
    source['rows']=[dict(voice=row['voice']) for row in source['rows']]
    after,_=experiment.original_build(source,experiment.reference,ids,experiment.args,experiment.plan)
    for layer in driver.LAYERS:
        for method in driver.METHODS:torch.testing.assert_close(before[layer][method],after[layer][method],rtol=0,atol=0)


def test_run_verifies_calibration_before_loading_lm_and_keeps_fixed_alpha(tmp_path,monkeypatch):
    events=[];args=SimpleNamespace(run_dir=tmp_path,batch_size=2)
    monkeypatch.setattr(driver,'load_plan',lambda args:({},tmp_path/'reference'))
    monkeypatch.setattr(driver,'verify_calibration',lambda args:events.append('verify_calibration'))
    monkeypatch.setattr(driver,'torch_load',lambda path:{})
    def load():
        assert events==['verify_calibration'];events.append('load_lm')
        return None,None,{}
    monkeypatch.setattr(driver,'load_lm',load)
    def evaluate(args,plan,reference,source,tokenizer,model,provenance,stage,alphas):
        assert alphas==[1.];events.append(stage)
    monkeypatch.setattr(driver,'evaluate_stage',evaluate)
    driver.run(args)
    assert events==['verify_calibration','load_lm','validation','evaluation']
    assert not (tmp_path/'selection.json').exists()
