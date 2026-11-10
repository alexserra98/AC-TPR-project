"""Check native postrun identity readouts, reconstruction failures, and guards."""

import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/temporary'))
from diagnose_native_tpr_nov6 import (
    METHODS, aggregate_filler_metrics, aggregate_soft_fidelity, alignment_metrics,
    require_completed_evaluation, simulate_local_edit, verify_evaluated_edits,
)
from run_tpr_nov6 import digest


def rows(n=2,split='gen_test',role='agent'):
    return [dict(split=split,voice='active',source_role=role) for _ in range(n)]


def test_filler_readout_uses_target_role_and_keeps_oov_out_of_accuracy():
    before=torch.tensor([[[4.,0.],[0.,3.]],[[0.,4.],[3.,0.]]])
    after=before.flip(1).clone()
    gold=torch.tensor([0,-1])
    result=aggregate_filler_metrics(rows(),before,after,gold,torch.tensor([False,False]),
                                    layer=20,method='tpr_native_role_swap')
    group={r['vocabulary']:r for r in result}
    assert group['all']['tokens']==2 and group['all']['known_tokens']==1
    assert group['all']['filler_after_target_accuracy']==1.
    assert group['all']['filler_predicted_label_retention']==1.
    assert group['all']['native_ideal_logits_relative_error']==0.
    assert group['oov']['known_tokens']==0 and group['oov']['filler_after_target_accuracy'] is None
    assert group['oov']['filler_predicted_label_retention']==1.
    assert group['all']['tokens']==group['known']['tokens']+group['oov']['tokens']


def test_patient_source_requires_agent_destination_and_can_change_filler():
    before=torch.tensor([[[0.,4.],[3.,0.]]])
    after=before.clone()
    result=aggregate_filler_metrics(rows(1,split='test',role='patient'),before,after,torch.tensor([0]),
        torch.tensor([True]),layer=20,method='soft_native_slot_swap')
    for row in result:
        assert row['filler_before_accuracy']==1.
        assert row['filler_after_target_accuracy']==row['filler_predicted_label_retention']==0.
        assert row['zero_edit_tokens']==1 and row['native_ideal_logits_relative_error'] is None


def test_exact_target_code_bag_does_not_imply_reencoded_target_fidelity():
    codes=torch.arange(8).reshape(1,8)
    permutation=[1,0,2,3,4,5,6,7]
    result=aggregate_soft_fidelity(rows(1),codes,codes.clone(),permutation,torch.tensor([False]),layer=20)
    for row in result:
        assert row['target_code_bag_preservation']==1.
        assert row['code_bag_retention']==1.
        assert row['slot_code_match_fraction']==.75
        assert row['all_codes_match_fraction']==0.
    corrupted=codes[:,permutation].clone(); corrupted[0,7]=99
    result=aggregate_soft_fidelity(rows(1),codes,corrupted,permutation,torch.tensor([True]),layer=20)
    for row in result:
        assert row['target_code_bag_preservation']==1.
        assert row['code_bag_retention']==0.
        assert row['slot_code_match_fraction']==7/8


def test_exact_reencoding_matches_permuted_slots_and_bag():
    codes=torch.arange(16).reshape(2,8)
    permutation=[1,0,2,3,4,5,6,7]
    result=aggregate_soft_fidelity(rows(),codes,codes[:,permutation],permutation,torch.tensor([True,False]),layer=20)
    for row in result:
        assert row['slot_matches']==row['total_slots']
        assert row['all_codes_match_fraction']==row['code_bag_retention']==row['target_code_bag_preservation']==1.


def test_local_simulation_rounds_after_float32_sum():
    h=torch.tensor([[1.,2.]],dtype=torch.bfloat16)
    delta=torch.tensor([[.001,.01]])
    result=simulate_local_edit(h,delta)
    torch.testing.assert_close(result,torch.tensor([[1.,2.015625]]),rtol=0,atol=0)
    assert not torch.equal(result,h.float()+delta)
    torch.testing.assert_close(simulate_local_edit(h,torch.zeros_like(delta)),h.float(),rtol=0,atol=0)


def test_alignment_compares_frozen_permutation_without_reselection():
    q=torch.arange(32,dtype=torch.float32).reshape(2,8,2)
    permutation=[1,0,2,3,4,5,6,7]
    target=q[:,permutation]
    result=alignment_metrics(q,target,rows(split='val'),permutation,layer=20,split='val')
    assert len(result)==1 and result[0]['tokens']==2
    assert result[0]['identity_mse']>0 and result[0]['permuted_mse']==0.
    assert result[0]['relative_mse_reduction']==1.
    wrong=alignment_metrics(q,target,rows(split='val'),list(range(8)),layer=20,split='val')
    assert wrong[0]['permuted_mse']==wrong[0]['identity_mse']


def edit_fixture():
    edits={20:{method:torch.tensor([[[3.,4.],[0.,0.]]]) for method in METHODS}}
    records=[dict(row_id=9,layer=20,method=method,alpha=1.,split='test',voice='active',
                  agent_edit_norm=5.,patient_edit_norm=0.,agent_edit_is_zero='False',patient_edit_is_zero='True')
             for method in METHODS]
    return edits,records


def test_completed_csv_edit_norms_flags_and_coverage_are_checked():
    edits,records=edit_fixture()
    checked=verify_evaluated_edits(records,[9],[dict(split='test',voice='active')],[20],edits)
    assert checked['verified_token_edits']==6 and checked['zero_token_edits']==3
    assert checked['maximum_relative_norm_error']==0.
    with pytest.raises(ValueError,match='Incomplete'):
        verify_evaluated_edits(records[:-1],[9],[dict(split='test',voice='active')],[20],edits)
    records[0]['agent_edit_norm']=4.
    with pytest.raises(ValueError,match='norm'):
        verify_evaluated_edits(records,[9],[dict(split='test',voice='active')],[20],edits)
    records[0]['agent_edit_norm']=5.; records[0]['patient_edit_is_zero']='False'
    with pytest.raises(ValueError,match='zero'):
        verify_evaluated_edits(records,[9],[dict(split='test',voice='active')],[20],edits)


def evaluation_fixture(tmp_path):
    evaluation=tmp_path/'evaluation'; evaluation.mkdir()
    calibration=tmp_path/'calibration'; calibration.mkdir()
    (calibration/'metadata.json').write_text('{}')
    (evaluation/'results.csv').write_text('row_id\n0\n')
    metadata=dict(status='complete',layers=[20],methods=list(METHODS),alphas=[1.],norm_matching=False,
        results_sha256=digest(evaluation/'results.csv'),calibration_metadata_sha256=digest(calibration/'metadata.json'))
    (evaluation/'metadata.json').write_text(json.dumps(metadata))
    return metadata


def test_incomplete_and_changed_results_or_calibration_are_rejected(tmp_path):
    metadata=evaluation_fixture(tmp_path)
    require_completed_evaluation(tmp_path,[20])
    metadata['status']='running'
    (tmp_path/'evaluation/metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='complete'):
        require_completed_evaluation(tmp_path,[20])
    metadata['status']='complete'
    (tmp_path/'evaluation/metadata.json').write_text(json.dumps(metadata))
    (tmp_path/'evaluation/results.csv').write_text('row_id\n1\n')
    with pytest.raises(ValueError,match='results changed'):
        require_completed_evaluation(tmp_path,[20])
    (tmp_path/'evaluation/results.csv').write_text('row_id\n0\n')
    (tmp_path/'calibration/metadata.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='Calibration metadata changed'):
        require_completed_evaluation(tmp_path,[20])
