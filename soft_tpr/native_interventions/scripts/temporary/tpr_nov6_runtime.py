"""Capture selected noun states and baseline logits in one original NNsight pass."""
import torch


@torch.inference_mode()
def capture_and_score(model, inputs, positions, answer_ids, layers):
    batch=torch.arange(len(positions),device=inputs['input_ids'].device)
    last=inputs['attention_mask'].sum(1)-1
    positions=positions.to(batch.device)
    answer_ids=answer_ids.to(batch.device)
    with model.trace(inputs,use_cache=False):
        states=[]
        for layer in layers:
            states.append(model.model.layers[layer].output[batch[:,None],positions].cpu())
        captured=torch.stack(states,dim=1).save()
        model.lm_head.input=model.lm_head.input[batch,last][:,None,:]
        scores=model.output.logits[:,0].gather(1,answer_ids).float().cpu().save()
    return captured,scores
