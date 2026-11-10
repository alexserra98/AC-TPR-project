"""Fit TPR role/filler factors and jointly steer noun tokens in the Nov 6 protocol.

The LM, prompt, corpus, sites, scoring and alpha=1 match dev@20d3b0a.
This exploratory comparison fixes layers 20,21,22 from the historical raw sweep.
Only training rows estimate vectors; validation selects probe checkpoints.
Role edits preserve the chosen continuous filler factor. Reconstruction residuals
can change the factor recovered afterwards; LM identity preservation is not implied.
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from ac_tpr.activations import extract_batch
from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION, load_corpus
from ac_tpr.interventions import score_batch
from ac_tpr.model import load_model
from ac_tpr.prompts import FEW_SHOT, PROMPT_TEMPLATE, prepare_prompts
from tpr_probe import BilinearTPRProbe
from tpr_nov6_probe import fit_probe, normalization, token_metrics
from tpr_nov6_runtime import capture_and_score

ROOT = Path(__file__).resolve().parents[2]
ROLES = ('agent', 'patient')
VOICES = ('active', 'passive')
METHODS = ('raw_centroid', 'tpr_centroid', 'tpr_role_filler')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')


def write_csv(path, records):
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def torch_load(path):
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def source_paths():
    return [Path(__file__).resolve(), ROOT/'src/ac_tpr/prompts.py', ROOT/'src/ac_tpr/dataset.py',
            ROOT/'src/ac_tpr/model.py', ROOT/'src/ac_tpr/activations.py',
            ROOT/'src/ac_tpr/interventions.py', Path(__file__).resolve().with_name('tpr_probe.py'),
            Path(__file__).resolve().with_name('tpr_nov6_probe.py'),
            Path(__file__).resolve().with_name('tpr_role_filler_factors.py'), Path(__file__).resolve().with_name('tpr_nov6_runtime.py')]


def snapshot_sources(output, stage, plan_path):
    """Record actual stage sources alongside the unchanged original plan."""
    planned=json.loads(plan_path.read_text())['source_sha256']
    snapshot=output/'source_snapshot'
    snapshot.mkdir(exist_ok=False)
    hashes={}
    for path in source_paths():
        relative=path.relative_to(ROOT)
        target=snapshot/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        hashes[str(relative)]=digest(target)
        if hashes[str(relative)]!=digest(path): raise ValueError('Source changed during snapshot')
    provenance=dict(stage=stage,created_at=datetime.now(timezone.utc).isoformat(),
        plan_sha256=digest(plan_path),source_sha256=hashes,
        differences_from_original_plan={key:dict(planned=planned.get(key),actual=value)
                                        for key,value in hashes.items() if planned.get(key)!=value},
        torch=str(torch.__version__),numpy=str(np.__version__))
    write_json(output/'runtime_source.json',provenance)
    return digest(output/'runtime_source.json')


def prepare(args):
    args.run_dir.mkdir(parents=True, exist_ok=False)
    corpus = ROOT / 'data/active_pass_long/corpus.csv'
    code = source_paths()
    write_json(args.run_dir/'plan.json', dict(
        status='prepared', created_at=datetime.now(timezone.utc).isoformat(),
        base_commit='20d3b0a780e5cb744622ec9d09c60d57b7dead47',
        model=TOKENIZER_ID, revision=TOKENIZER_REVISION,
        corpus=str(corpus), corpus_sha256=digest(corpus),
        prompt_template=PROMPT_TEMPLATE, demonstrations=FEW_SHOT,
        layers=args.layers, layer_selection='Fixed from historical raw peaks, exploratory; no new test selection',
        roles=ROLES, voices=VOICES, methods=METHODS, alpha=1.0,
        probe=dict(role_dim=2, filler_dim=8, seeds=[42,43], epochs=args.epochs,
                   patience=25, learning_rate=.01, batch_size=512,
                   objective='Observed-role filler CE, reused from train_tpr_centroids.py',
                   selection='Minimum validation CE; ties smaller seed'),
        factors='Train-only cell means by filler, semantic role and voice, equal cell weight; outer-product fit per voice',
        steering='At both noun positions, change fitted mean role using the inferred filler factor; factor re-unbinding and LM identity may change',
        heldout='Vocabulary and factors trained only on train; infer continuous filler from each focal binding for every split',
        evaluation='Test and gen_test separate by voice; no strength or layer selection on new test results',
        batch_size=args.batch_size, source_sha256={str(p.relative_to(ROOT)):digest(p) for p in code},
        runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                     note='GB10 runtime differs from original Torch 2.7.1; same pinned weights, BF16, eager, NNsight hooks'),
    ))


def load_lm():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    model, provenance = load_model(tokenizer, TOKENIZER_REVISION, 'cuda', 'bfloat16')
    return tokenizer, model, provenance


def extract(args):
    plan = json.loads((args.run_dir/'plan.json').read_text())
    if plan['layers']!=args.layers: raise ValueError('Layers differ from frozen plan')
    out = args.run_dir/'extraction'
    out.mkdir(exist_ok=False)
    rows = load_corpus(Path(plan['corpus']))
    tokenizer, model, provenance = load_lm()
    ids, positions, answers = prepare_prompts(rows, tokenizer)
    states = torch.empty((len(rows),len(args.layers),2,model.config.hidden_size), dtype=torch.bfloat16)
    baseline = torch.empty(len(rows),2)
    for start in range(0,len(rows),args.batch_size):
        end=min(start+args.batch_size,len(rows))
        inputs=tokenizer.pad({'input_ids':ids[start:end]},padding=True,return_tensors='pt').to('cuda')
        captured,scores=capture_and_score(model,inputs,positions[start:end],answers[start:end],args.layers)
        states[start:end]=captured
        baseline[start:end]=scores
        if start==0:
            torch.testing.assert_close(captured,extract_batch(model,inputs,positions[start:end])[:,args.layers],rtol=0,atol=0)
            torch.testing.assert_close(scores,score_batch(model,inputs,answers[start:end]),rtol=0,atol=0)
            zero=torch.zeros(end-start,2,model.config.hidden_size)
            torch.testing.assert_close(score_batch(model,inputs,answers[start:end],args.layers[0],positions[start:end],zero),baseline[start:end],rtol=0,atol=0)
            torch.testing.assert_close(score_batch(model,inputs,answers[start:end],35,positions[start:end],torch.ones_like(zero)),baseline[start:end],rtol=0,atol=0)
        if not torch.isfinite(states[start:end]).all() or not torch.isfinite(baseline[start:end]).all():
            raise ValueError('Nonfinite extraction')
        if start % (16*args.batch_size)==0 or end==len(rows):
            print(f'Extracted and scored {end}/{len(rows)}',flush=True)
    torch.save(dict(states=states,baseline=baseline,rows=rows,input_ids=ids,positions=positions,answers=answers,layers=args.layers),out/'states.pt')
    write_json(out/'metadata.json',dict(status='complete',model=provenance,shape=list(states.shape),
        layers=args.layers,prompt_template=PROMPT_TEMPLATE,demonstrations=FEW_SHOT,
        corpus_sha256=plan['corpus_sha256'],batch_size=args.batch_size,zero_and_final_controls='passed',
        states_sha256=digest(out/'states.pt')))


def factor_change_diagnostics(z,role_ids,factor,inferred,edited):
    """Measure reconstruction and recovered filler drift, without lexical labels."""
    from tpr_role_filler_factors import infer_filler_factors
    recovered=infer_filler_factors(edited,1-role_ids,factor)
    prediction=factor['bias']+np.einsum('ni,nj->nij',factor['roles'][role_ids],inferred)
    residual=z-prediction
    centered=z-factor['bias']
    difference=recovered-inferred
    original_norm=np.linalg.norm(inferred,axis=1)
    recovered_norm=np.linalg.norm(recovered,axis=1)
    cosine_valid=(original_norm>1e-12)&(recovered_norm>1e-12)
    cosine=np.sum(inferred[cosine_valid]*recovered[cosine_valid],axis=1)/(original_norm[cosine_valid]*recovered_norm[cosine_valid])
    diagnostics=dict(
        source_reconstruction_relative_error=float(np.linalg.norm(residual)/max(np.linalg.norm(centered),1e-12)),
        mean_source_reconstruction_relative_error=float(np.mean(np.linalg.norm(residual,axis=(1,2))/np.maximum(np.linalg.norm(centered,axis=(1,2)),1e-12))),
        inferred_filler_change_relative_norm=float(np.linalg.norm(difference)/max(np.linalg.norm(inferred),1e-12)),
        mean_inferred_filler_change_relative_norm=float(np.mean(np.linalg.norm(difference,axis=1)/np.maximum(original_norm,1e-12))),
        mean_inferred_filler_cosine=float(np.mean(np.clip(cosine,-1,1))) if len(cosine) else None,
    )
    return recovered,diagnostics


def factor_split_metrics(z,token_rows,fillers,factors,probe):
    """Assess lexical recovery; all nearest-filler prototypes use training rows.

    Labels outside the training vocabulary are counted and excluded from lexical
    accuracy. They still contribute to continuous reconstruction diagnostics.
    No labels used here influence factors, edits, probe selection, or evaluation.
    """
    from tpr_role_filler_factors import infer_filler_factors,role_swap_deltas
    records=[];prototypes={}
    vocabulary={name:i for i,name in enumerate(fillers)}
    for voice in VOICES:
        ids=np.array([i for i,row in enumerate(token_rows) if row['voice']==voice])
        rows=[token_rows[i] for i in ids]
        role_ids=np.array([ROLES.index(row['role']) for row in rows])
        labels=np.array([vocabulary.get(row['filler'],-1) for row in rows])
        train=np.array([row['split']=='train' for row in rows])
        factor=factors[voice]
        inferred=infer_filler_factors(z[ids],role_ids,factor)
        edited=z[ids]+role_swap_deltas(inferred,role_ids,factor)
        recovered,_=factor_change_diagnostics(z[ids],role_ids,factor,inferred,edited)
        prototypes[voice]=np.stack([np.mean([inferred[train&(labels==f)&(role_ids==r)].mean(0)
                                             for r in range(2)],axis=0) for f in range(len(fillers))])
        if not np.isfinite(prototypes[voice]).all(): raise ValueError('Incomplete train filler prototypes')
        before=((inferred[:,None]-prototypes[voice][None])**2).sum(-1).argmin(-1)
        after=((recovered[:,None]-prototypes[voice][None])**2).sum(-1).argmin(-1)
        with torch.no_grad():
            original_logits=torch.einsum('nij,ri,fj->nrf',torch.as_tensor(z[ids],dtype=probe.R.dtype),probe.R,probe.F)
            edited_logits=torch.einsum('nij,ri,fj->nrf',torch.as_tensor(edited,dtype=probe.R.dtype),probe.R,probe.F)
        probe_before=original_logits[torch.arange(len(ids)),torch.from_numpy(role_ids)].argmax(-1).numpy()
        probe_after=edited_logits[torch.arange(len(ids)),torch.from_numpy(1-role_ids)].argmax(-1).numpy()
        for split in ('train','val','test','gen_test'):
            mask=np.array([row['split']==split for row in rows])
            if not mask.any(): continue
            supported=mask&(labels>=0)
            eligible=supported&(before==labels)
            _,diagnostics=factor_change_diagnostics(z[ids][mask],role_ids[mask],factor,inferred[mask],edited[mask])
            records.append(dict(split=split,voice=voice,tokens=int(mask.sum()),
                supported_tokens=int(supported.sum()),unsupported_tokens=int((mask&~supported).sum()),
                supported_fraction=float(supported.sum()/mask.sum()),
                nearest_filler_accuracy_before=float(np.mean(before[supported]==labels[supported])) if supported.any() else None,
                nearest_filler_accuracy_after=float(np.mean(after[supported]==labels[supported])) if supported.any() else None,
                nearest_filler_retention=float(np.mean(after[eligible]==labels[eligible])) if eligible.any() else None,
                probe_source_role_filler_accuracy_before=float(np.mean(probe_before[supported]==labels[supported])) if supported.any() else None,
                probe_target_role_filler_accuracy_after=float(np.mean(probe_after[supported]==labels[supported])) if supported.any() else None,
                **diagnostics))
    return records,prototypes


def train(args):
    from tpr_role_filler_factors import training_cell_means, fit_outer, predict_outer, factor_diagnostics as diagnose_factors
    plan=json.loads((args.run_dir/'plan.json').read_text())
    if plan['layers']!=args.layers: raise ValueError('Layers differ from frozen plan')
    out=args.run_dir/'training'
    out.mkdir(exist_ok=False)
    runtime_hash=snapshot_sources(out,'train',args.run_dir/'plan.json')
    source=torch_load(args.run_dir/'extraction/states.pt')
    if source['layers']!=args.layers: raise ValueError('Layers differ from extracted states')
    rows=source['rows']
    token_rows=[dict(filler=row[role],role=role,voice=row['voice'],split=row['split']) for row in rows for role in ROLES]
    fillers=sorted({r['filler'] for r in token_rows if r['split']=='train'})
    labels=torch.tensor([fillers.index(r['filler']) if r['filler'] in fillers else -1 for r in token_rows])
    roles=torch.arange(2).repeat(len(rows))
    splits={s:torch.tensor([i for i,r in enumerate(token_rows) if r['split']==s]) for s in ('train','val','test','gen_test')}
    if any((labels[splits[s]]<0).any() for s in ('train','val','test')): raise ValueError('Unexpected unsupported vocabulary')
    records=[];factor_records=[]
    for li,layer in enumerate(args.layers):
        print(f'Training TPR layer {layer}',flush=True)
        h=source['states'][:,li].float().flatten(0,1)
        center,scale=normalization(h[splits['train']])
        x=(h-center)/scale
        config=dict(hidden_dim=h.shape[-1],num_roles=2,num_fillers=len(fillers),role_dim=2,filler_dim=8)
        settings=SimpleNamespace(**{k:plan['probe'][k] for k in ('epochs','patience','learning_rate','batch_size')})
        candidates=[]
        for seed in plan['probe']['seeds']:
            torch.manual_seed(seed)
            probe=BilinearTPRProbe(**config).to(args.training_device)
            tr,va=splits['train'],splits['val']
            result=fit_probe(probe,x[tr].to(args.training_device),labels[tr].to(args.training_device),roles[tr].to(args.training_device),
                             x[va].to(args.training_device),labels[va].to(args.training_device),roles[va].to(args.training_device),
                             settings,seed,out/f'layer_{layer}_seed_{seed}.jsonl')
            probe.cpu()
            candidates.append((result['validation']['loss'],seed,probe,result))
        _,seed,probe,fit=min(candidates,key=lambda c:(c[0],c[1]))
        probe.eval()
        with torch.no_grad():
            z=probe.binding(x).numpy()
            logits=probe(x)
        for split,indices in splits.items():
            for voice in VOICES:
                ids=torch.tensor([i for i in indices.tolist() if token_rows[i]['voice']==voice])
                metric=token_metrics(logits[ids],roles[ids],labels[ids])
                records.append(dict(layer=layer,split=split,voice=voice,**metric))
        table=training_cell_means(z,token_rows)
        factors={}
        factor_diagnostics={}
        for vi,voice in enumerate(VOICES):
            factor=fit_outer(table['means'][vi])
            factors[voice]=factor
            predicted=predict_outer(factor)
            target=table['means'][vi]
            factor_diagnostics[voice]=dict(**diagnose_factors(target,factor),
                                          centroid_relative_rmse=float(np.linalg.norm(predicted-target)/max(np.linalg.norm(target-target.mean((0,1),keepdims=True)),1e-12)))
        metrics,prototypes=factor_split_metrics(z,token_rows,fillers,factors,probe)
        factor_records.extend(dict(layer=layer,**record) for record in metrics)
        pinv,svd=probe.binding_pseudoinverse()
        train_ids=splits['train']
        raw={v:torch.stack([h[[i for i in train_ids.tolist() if token_rows[i]['voice']==v and token_rows[i]['role']==r]].mean(0) for r in ROLES]) for v in VOICES}
        payload=dict(config=config,state_dict=probe.state_dict(),center=center,scale=scale,
                     pseudoinverse=pinv,raw_means=raw,fillers=fillers,seed=seed,
                     empirical_filler_prototypes={v:torch.from_numpy(a) for v,a in prototypes.items()},
                     factors={v:{k:torch.from_numpy(a) if isinstance(a,np.ndarray) else a for k,a in f.items()} for v,f in factors.items()})
        torch.save(payload,out/f'layer_{layer}.pt')
        write_json(out/f'layer_{layer}_diagnostics.json',dict(seed=seed,fit=fit,pseudoinverse=svd,
                  factor_diagnostics=factor_diagnostics,cell_counts=np.asarray(table['counts']).tolist(),fillers=table['fillers']))
        np.savez(out/f'layer_{layer}_factors.npz',**{v+'_'+k:a for v,f in factors.items() for k,a in f.items() if isinstance(a,np.ndarray)},**{v+'_empirical_filler_prototypes':a for v,a in prototypes.items()},centroids=table['means'])
    write_csv(out/'probe_metrics.csv',records)
    write_csv(out/'factor_metrics.csv',factor_records)
    write_json(out/'metadata.json',dict(status='complete',layers=args.layers,fillers=fillers,
        extraction_sha256=digest(args.run_dir/'extraction/states.pt'),
        runtime_source_sha256=runtime_hash,
        filler_metric='Euclidean nearest prototype of LS-unbound training fillers, equal role weighting, per voice; unsupported labels excluded',
        probe_edit_metric='Readout at target semantic role of intended edited native bindings; not a language-model identity test',
        factor_code_sha256=digest(Path(__file__).resolve().with_name('tpr_role_filler_factors.py'))))


def prepare_edits(payload,h,rows):
    from tpr_role_filler_factors import infer_filler_factors, role_swap_deltas
    probe=BilinearTPRProbe(**payload['config'])
    probe.load_state_dict(payload['state_dict']); probe.eval()
    with torch.no_grad():
        z=probe.binding((h.flatten(0,1)-payload['center'])/payload['scale']).numpy()
    result={method:torch.empty_like(h) for method in METHODS}
    factor_checks=[]
    for voice in VOICES:
        selected=[i for i,r in enumerate(rows) if r['voice']==voice]
        if not selected: continue
        index=torch.tensor(selected)
        flat=np.array([2*i+r for i in selected for r in range(2)])
        role_ids=np.tile(np.arange(2),len(selected))
        factor={k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in payload['factors'][voice].items()}
        inferred=infer_filler_factors(z[flat],role_ids,factor)
        dz=role_swap_deltas(inferred,role_ids,factor)
        lifted=payload['scale']*torch.from_numpy(dz.reshape(len(flat),-1)).float()@payload['pseudoinverse']
        raw_delta=payload['raw_means'][voice][1]-payload['raw_means'][voice][0]
        raw=torch.stack((raw_delta,-raw_delta)).expand(len(selected),-1,-1)
        projected=raw@probe.M.flatten(1).detach()@payload['pseudoinverse']
        result['raw_centroid'][index]=raw
        result['tpr_centroid'][index]=projected
        result['tpr_role_filler'][index]=lifted.reshape(len(selected),2,-1)
        realized=(lifted/payload['scale'])@probe.M.flatten(1).detach()
        relative=float((realized-torch.from_numpy(dz.reshape(len(flat),-1))).norm()/max(np.linalg.norm(dz),1e-12))
        if relative>1e-3: raise ValueError('Requested binding change is not realized')
        edited=z[flat]+realized.numpy().reshape(z[flat].shape)
        _,diagnostics=factor_change_diagnostics(z[flat],role_ids,factor,inferred,edited)
        factor_checks.append(dict(voice=voice,lifting_relative_error=relative,**diagnostics))
    if not all(torch.isfinite(v).all() for v in result.values()): raise ValueError('Nonfinite edits')
    return result,factor_checks


def evaluate(args):
    plan=json.loads((args.run_dir/'plan.json').read_text())
    if plan['layers']!=args.layers: raise ValueError('Layers differ from frozen plan')
    out=args.run_dir/'evaluation';out.mkdir(exist_ok=False)
    runtime_hash=snapshot_sources(out,'evaluate',args.run_dir/'plan.json')
    source=torch_load(args.run_dir/'extraction/states.pt')
    if source['layers']!=args.layers: raise ValueError('Layers differ from extracted states')
    ids=[i for i,r in enumerate(source['rows']) if r['split'] in ('test','gen_test')]
    rows=[source['rows'][i] for i in ids]
    edits={}; checks=[]
    for li,layer in enumerate(args.layers):
        payload=torch_load(args.run_dir/f'training/layer_{layer}.pt')
        edits[layer],diagnostic=prepare_edits(payload,source['states'][ids,li].float(),rows)
        checks.extend(dict(layer=layer,**d) for d in diagnostic)
    tokenizer,model,provenance=load_lm()
    baseline_records=[];results=[]
    for start in range(0,len(ids),args.batch_size):
        end=min(start+args.batch_size,len(ids)); chosen=ids[start:end]
        inputs=tokenizer.pad({'input_ids':[source['input_ids'][i] for i in chosen]},padding=True,return_tensors='pt').to('cuda')
        answers=source['answers'][chosen];positions=source['positions'][chosen]
        baseline=score_batch(model,inputs,answers)
        # Different batch grouping can change BF16 ties; record the actual baseline.
        for j,i in enumerate(chosen):
            row=source['rows'][i]
            baseline_records.append(dict(row_id=i,split=row['split'],voice=row['voice'],agent=row['agent'],patient=row['patient'],
                agent_logit=float(baseline[j,0]),patient_logit=float(baseline[j,1]),
                extraction_margin=float(source['baseline'][i,1]-source['baseline'][i,0])))
        if start==0:
            zero=torch.zeros(len(chosen),2,model.config.hidden_size)
            torch.testing.assert_close(score_batch(model,inputs,answers,args.layers[0],positions,zero),baseline,rtol=0,atol=0)
            torch.testing.assert_close(score_batch(model,inputs,answers,35,positions,edits[args.layers[0]]['tpr_role_filler'][start:end]),baseline,rtol=0,atol=0)
        for layer in args.layers:
            for method in METHODS:
                delta=edits[layer][method][start:end]
                scores=score_batch(model,inputs,answers,layer,positions,delta)
                for j,i in enumerate(chosen):
                    row=source['rows'][i]
                    bm=float(baseline[j,1]-baseline[j,0]);em=float(scores[j,1]-scores[j,0])
                    results.append(dict(row_id=i,split=row['split'],voice=row['voice'],quad_id=row.get('quad_id',''),
                        layer=layer,method=method,baseline_margin=bm,edited_margin=em,margin_change=em-bm,
                        baseline_agent_win=bm<0,patient_win=em>0,flip=bm<0 and em>0,
                        agent_edit_norm=float(delta[j,0].norm()),patient_edit_norm=float(delta[j,1].norm())))
        if start % (8*args.batch_size)==0 or end==len(ids): print(f'Evaluated {end}/{len(ids)} across {len(args.layers)*len(METHODS)} conditions',flush=True)
    write_csv(out/'baseline.csv',baseline_records)
    write_csv(out/'results.csv',results)
    groups=defaultdict(list)
    for row in results: groups[tuple(row[k] for k in ('split','voice','layer','method'))].append(row)
    summary=[]
    for key,group in groups.items():
        eligible=sum(r['baseline_agent_win'] for r in group)
        summary.append(dict(zip(('split','voice','layer','method'),key),sentences=len(group),baseline_agent_wins=eligible,
            mean_margin_change=float(np.mean([r['margin_change'] for r in group])),
            patient_preference=float(np.mean([r['patient_win'] for r in group])),
            conditional_flip=sum(r['flip'] for r in group)/eligible if eligible else None,
            mean_agent_edit_norm=float(np.mean([r['agent_edit_norm'] for r in group])),
            mean_patient_edit_norm=float(np.mean([r['patient_edit_norm'] for r in group]))))
    write_csv(out/'summary.csv',summary)
    write_json(out/'metadata.json',dict(status='complete',model=provenance,layers=args.layers,
        runtime_source_sha256=runtime_hash,
        controls='Zero edits and final-block actual factor edits passed exactly on first batch',lifting=checks,
        rows=len(rows),conditions=len(args.layers)*len(METHODS),results_sha256=digest(out/'results.csv')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','extract','train','evaluate'])
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--layers',type=int,nargs='+',default=[20,21,22])
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--training-device',default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    globals()[args.stage](args)


if __name__=='__main__': main()
