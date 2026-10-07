"""Quick train/validation-only AE/PCA controls at two frozen steering settings."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from ac_tpr.activations import ROLES
from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.interventions import prepare_prompts, role_deltas, score_batch
from ac_tpr.model import load_model, validate_activation_provenance
from ac_tpr.soft_tpr import SoftTPRAutoencoder
from ac_tpr.syntactic_vectors import VOICES
from analyze_soft_tpr_pilot import cluster_interval

METHODS = ['raw', 'soft', 'ae_256_best', 'ae_256_matched', 'pca_256', 'pca_matched']
# Frozen from the prior validation experiment, not chosen using this run's test scores.
CONDITIONS = {'active': (24, 'by_voice', 'agent'), 'passive': (25, 'pooled', 'patient')}


def mse(model, x):
    with torch.no_grad():
        return float(F.mse_loss(model(x), x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('ac-tpr-cache/local-soft-tpr-v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(42)
    experiment = args.root / 'training-seed42'
    corpus_path = Path('data/generated/corpus.csv')
    source = json.loads((experiment/'role_means/metadata.json').read_text())
    validate_activation_provenance(source)
    assert hashlib.sha256(corpus_path.read_bytes()).hexdigest() == source['corpus_sha256']
    plan = dict(layers=[24,25], conditions=CONDITIONS, seed=42, epochs=args.epochs,
                batch_size=args.batch_size, methods=METHODS,
                train_only='Normalization, PCA fitting, AE gradients, and centroids',
                validation_only='AE checkpoint selection and PCA rank matching to Soft TPR normalized MSE',
                ae='4096 -> 256 GELU -> 256 linear latent -> 4096 linear decoder',
                compression='256 continuous coordinates; Soft TPR has 8 x 32 quantized fillers (48 code-index bits), so bit rates differ',
                steering='Native norms, strength 1; both test and held-out nouns; fixed earlier validation conditions',
                source_corpus_sha256=source['corpus_sha256'],
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    data=torch.load(args.root/'activations/activations.pt',weights_only=True,map_location='cpu',mmap=True)
    rows=data['rows']; states=data['activations']
    train_ids=[i for i,r in enumerate(rows) if r['split']=='train']
    val_ids=[i for i,r in enumerate(rows) if r['split']=='val']
    train_rows=[rows[i] for i in train_ids]
    raw=torch.load(experiment/'role_means/syntactic_vectors.pt',weights_only=True,map_location='cpu')
    soft=torch.load(experiment/'soft_tpr_vectors/syntactic_vectors.pt',weights_only=True,map_location='cpu')
    vectors={'raw':raw,'soft':soft}
    for name in METHODS[2:]:
        vectors[name]={k:torch.zeros_like(v) for k,v in raw.items()}
    metrics=[]; diagnostics=[]; history=[]
    for layer in (24,25):
        ckpt=torch.load(experiment/f'checkpoints/layer_{layer:02d}.pt',weights_only=True,map_location='cpu')
        center,scale=ckpt['center'],ckpt['scale']
        train=states[train_ids,layer].float().flatten(0,1)
        val=states[val_ids,layer].float().flatten(0,1)
        # Confirm the identical train-only normalization used by Soft TPR.
        torch.testing.assert_close(center, train.mean(0))
        torch.testing.assert_close(scale, (train-center).square().mean().sqrt())
        x=((train-center)/scale).cuda(); v=((val-center)/scale).cuda()
        model=SoftTPRAutoencoder(**ckpt['config']).cuda()
        model.load_state_dict(ckpt['state_dict']);model.eval()
        with torch.no_grad():
            rt=model(x)['reconstruction'];rv=model(v)['reconstruction']
            target=float(F.mse_loss(rv,v))
            metrics.append(dict(layer=layer,method='soft',latent_dim=256,epoch=ckpt['best_epoch'],
                                train_mse=float(F.mse_loss(rt,x)),val_mse=target,target_val_mse=target))
        # Re-exported Soft TPR centroids must agree with the saved artifacts.
        reconstructed=(rt.cpu()*scale+center).reshape_as(states[train_ids,layer])
        torch.testing.assert_close(reconstructed.mean(0),soft['pooled'][layer],rtol=1e-5,atol=1e-5)
        del model,rt,rv,reconstructed
        torch.manual_seed(42+layer)
        ae=nn.Sequential(nn.Linear(x.shape[1],256),nn.GELU(),nn.Linear(256,256),nn.Linear(256,x.shape[1])).cuda()
        optimizer=torch.optim.Adam(ae.parameters(),lr=.001)
        best_error=closest=float('inf'); best_state=matched_state=None
        generator=torch.Generator().manual_seed(42+layer)
        for epoch in range(1,args.epochs+1):
            ae.train()
            for step,indices in enumerate(torch.randperm(len(x),generator=generator).split(128),1):
                batch=x[indices.cuda()]
                loss=F.mse_loss(ae(batch),batch)
                if not torch.isfinite(loss): raise ValueError('Nonfinite AE loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(ae.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                # Fine checkpoint resolution early in training helps quality matching.
                if epoch<=5 or step==int(np.ceil(len(x)/128)):
                    ae.eval();error=mse(ae,v)
                    history.append(dict(layer=layer,epoch=epoch,step=step,val_mse=error))
                    if error<best_error:
                        best_error=error;best_epoch=epoch;best_step=step
                        best_state={k:t.detach().cpu().clone() for k,t in ae.state_dict().items()}
                    if abs(error-target)<closest:
                        closest=abs(error-target);matched_epoch=epoch;matched_step=step
                        matched_state={k:t.detach().cpu().clone() for k,t in ae.state_dict().items()}
                    ae.train()
            if epoch==1 or epoch%25==0:
                print(f'Layer {layer}: AE epoch {epoch}, best val MSE {best_error:.6f}, Soft TPR target {target:.6f}',flush=True)
        reconstructions={}
        for name,weights,epoch,step in [('ae_256_best',best_state,best_epoch,best_step),
                                        ('ae_256_matched',matched_state,matched_epoch,matched_step)]:
            ae.load_state_dict(weights);ae.eval()
            with torch.no_grad():
                pred=ae(x)
                metrics.append(dict(layer=layer,method=name,latent_dim=256,epoch=epoch,step=step,
                                    train_mse=float(F.mse_loss(pred,x)),val_mse=mse(ae,v),target_val_mse=target))
                reconstructions[name]=pred.cpu()*scale+center
            torch.save(dict(state_dict=weights,center=center,scale=scale,epoch=epoch,step=step),out/f'{name}_layer{layer}.pt')
        del ae,optimizer
        print(f'Layer {layer}: fitting PCA on training tokens only',flush=True)
        # Exact SVD; rank selection only uses validation reconstruction quality.
        with torch.no_grad():
            u,singular,vt=torch.linalg.svd(x.cpu().double(),full_matrices=False)
            basis=vt[:256].T.float().cuda().contiguous()
            torch.testing.assert_close(basis.T@basis,torch.eye(256,device='cuda'),rtol=1e-4,atol=1e-5)
            del u
            scores=v@basis
            errors=(v.square().sum()-scores.square().sum(0).cumsum(0))/v.numel()
            rank=int((errors-target).abs().argmin())+1
            for name,k in [('pca_256',256),('pca_matched',rank)]:
                b=basis[:,:k];pred=(x@b)@b.T
                measured=float(F.mse_loss((v@b)@b.T,v))
                torch.testing.assert_close(torch.tensor(measured,device='cuda'),errors[k-1],rtol=1e-3,atol=1e-6)
                metrics.append(dict(layer=layer,method=name,latent_dim=k,epoch=None,
                                    train_mse=float(F.mse_loss(pred,x)),val_mse=measured,target_val_mse=target))
                reconstructions[name]=pred.cpu()*scale+center
                torch.save(dict(basis=b.cpu(),center=center,scale=scale,rank=k),out/f'{name}_layer{layer}.pt')
        for name,recon in reconstructions.items():
            tokens=recon.reshape(len(train_ids),2,-1)
            vectors[name]['pooled'][layer]=tokens.mean(0)
            for vi,voice in enumerate(VOICES):
                mask=torch.tensor([r['voice']==voice for r in train_rows])
                vectors[name]['by_voice'][vi,layer]=tokens[mask].mean(0)
        for name in METHODS:
            for means,voice in [('pooled',None),('by_voice','active'),('by_voice','passive')]:
                m=vectors[name][means][layer] if voice is None else vectors[name][means][VOICES.index(voice),layer]
                reference=raw[means][layer] if voice is None else raw[means][VOICES.index(voice),layer]
                d=m[1]-m[0];dr=reference[1]-reference[0]
                diagnostics.append(dict(layer=layer,method=name,means=means,voice=voice or 'pooled',
                                        cosine=float(F.cosine_similarity(d,dr,dim=0)),norm_ratio=float(d.norm()/dr.norm()),
                                        relative_direction_error=float((d-dr).norm()/dr.norm())))
        del x,v,basis,vt,singular,reconstructions,pred,train,val,scores
        torch.cuda.empty_cache()
    metric_frame=pd.DataFrame(metrics)
    metric_frame['val_error_ratio_to_soft']=metric_frame.val_mse/metric_frame.target_val_mse
    metric_frame.to_csv(out/'reconstruction.csv',index=False)
    pd.DataFrame(history).to_csv(out/'ae_training.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'directions.csv',index=False)
    torch.save(vectors,out/'vectors.pt')
    print(metric_frame.to_string(index=False),flush=True)
    del data,states
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER_ID,revision=TOKENIZER_REVISION,use_fast=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='right'
    assert hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()==source['tokenizer']['backend_sha256']
    lm,execution=load_model(tokenizer,source['model']['commit'],'cuda','float32')
    assert execution==source['model']
    records=[]
    for split in ('test','gen_test'):
        for voice in VOICES:
            selected=[r for r in rows if r['split']==split and r['voice']==voice]
            layer,means,role=CONDITIONS[voice]
            for start in range(0,len(selected),args.batch_size):
                batch=selected[start:start+args.batch_size]
                ids,positions,answers=prepare_prompts(batch,tokenizer)
                inputs=tokenizer.pad({'input_ids':ids},padding=True,return_tensors='pt').to('cuda')
                base=score_batch(lm,inputs,answers)
                if start==0:
                    zero=score_batch(lm,inputs,answers,layer,positions[:,ROLES.index(role)],torch.zeros(len(batch),lm.cfg.d_model))
                    assert torch.equal(zero,base),'Zero control failed'
                for method in METHODS:
                    delta=role_deltas(vectors[method],batch,means,layer,role)
                    if start==0:
                        final=score_batch(lm,inputs,answers,lm.cfg.n_layers-1,positions[:,ROLES.index(role)],delta)
                        assert torch.equal(final,base),'Final block control failed'
                    after=score_batch(lm,inputs,answers,layer,positions[:,ROLES.index(role)],delta)
                    for row,before,score in zip(batch,base.tolist(),after.tolist(),strict=True):
                        records.append(dict(**row,layer=layer,mean_type=means,edited_role=role,method=method,
                                            baseline_logit_diff=before[1]-before[0],logit_diff=score[1]-score[0],
                                            logit_diff_change=score[1]-score[0]-before[1]+before[0]))
                print(f'Steering {split}/{voice} {min(start+args.batch_size,len(selected))}/{len(selected)}; elapsed {time.monotonic()-started:.0f}s',flush=True)
            pd.DataFrame(records).to_csv(out/'results.csv',index=False)
    frame=pd.DataFrame(records)
    summaries=[];gains=[]
    for (split,voice),group in frame.groupby(['split','voice']):
        original=group[group.method=='raw'].set_index('pair_id')
        for method in METHODS:
            g=group[group.method==method].set_index('pair_id').loc[original.index].copy()
            assert np.array_equal(g.baseline_logit_diff,original.baseline_logit_diff)
            wins=g.baseline_logit_diff<0;flips=wins&(g.logit_diff>0)
            summaries.append(dict(split=split,voice=voice,method=method,n=len(g),
                                  mean_change=g.logit_diff_change.mean(),conditional_flip_rate=flips.sum()/wins.sum(),
                                  agent_wins=int(wins.sum()),flips=int(flips.sum())))
            g['cluster']=g.apply(lambda r:'|'.join([*sorted((r.agent,r.patient)),r.verb]),axis=1)
            g['gain']=g.logit_diff_change-original.logit_diff_change
            low,high=cluster_interval(g,'gain')
            gains.append(dict(split=split,voice=voice,method=method,mean_gain=g.gain.mean(),ci_low=low,ci_high=high))
    summary=pd.DataFrame(summaries);summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(gains).to_csv(out/'paired_gains.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    groups=[(s,v) for s in ('test','gen_test') for v in VOICES]
    for i,method in enumerate(METHODS):
        g=summary[summary.method==method].set_index(['split','voice']).loc[groups]
        pos=np.arange(4)+(i-2.5)*.12
        axes[0].plot(pos,g.mean_change,'o',label=method.replace('_',' '))
        axes[1].plot(pos,100*g.conditional_flip_rate,'o')
    for ax in axes:
        ax.set_xticks(range(4),['Test\nactive','Test\npassive','Held-out nouns\nactive','Held-out nouns\npassive']);ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean patient-minus-agent logit change');axes[1].set_ylabel('Flips among baseline agent wins (%)')
    axes[0].legend(fontsize=8)
    fig.suptitle('Reconstruction baselines: fixed active block 24 / passive block 25\nPythia 6.9B; native direction norms; one training seed')
    for ext in ('png','pdf'):fig.savefig(out/f'comparison.{ext}',dpi=170)
    plan.update(controls_passed=True,execution_model=execution,result_rows=len(frame),elapsed_seconds=time.monotonic()-started,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [experiment/'role_means/metadata.json',experiment/'soft_tpr_vectors/metadata.json']})
    (out/'metadata.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)
    print('Completed AE/PCA comparison',flush=True)


if __name__=='__main__':main()
