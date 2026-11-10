"""Noun-probe training helpers reused from train_tpr_centroids.py (26 October).

The loss predicts the focal filler conditioned on its supplied semantic role.
Role/filler factor estimation is a separate supervised centroid analysis.
"""
import math
import json
import time
import torch
from torch.nn import functional as F

def normalization(train_states):
    """Match Soft TPR's train-only mean and single centered RMS scalar."""
    center = train_states.mean(0)
    scale = (train_states - center).square().mean().sqrt().clamp_min(1e-6)
    return center, scale


def observed_logits(logits, role_ids):
    return logits[torch.arange(len(role_ids), device=logits.device), role_ids]


def masked_filler_loss(logits, role_ids, filler_ids):
    """Only the supplied observed role contributes to the filler CE."""
    return F.cross_entropy(observed_logits(logits, role_ids), filler_ids)


@torch.no_grad()
def token_metrics(logits, role_ids, filler_ids):
    supported = filler_ids >= 0
    result = {"tokens": len(filler_ids), "supported_tokens": int(supported.sum()),
              "unsupported_tokens": int((~supported).sum()),
              "role_conditioned_filler_accuracy": None, "loss": None}
    if supported.any():
        selected = observed_logits(logits[supported], role_ids[supported])
        targets = filler_ids[supported]
        result.update(role_conditioned_filler_accuracy=(selected.argmax(-1) == targets).float().mean().item(),
                      loss=F.cross_entropy(selected, targets).item())
    return result


def fit_probe(model, train_x, train_y, train_roles, val_x, val_y, val_roles, args, seed, history_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=.01, betas=(.9, .99))
    generator = torch.Generator(device=train_x.device).manual_seed(seed)
    best_loss, best_state, best_epoch, stale = math.inf, None, 0, 0
    started = time.monotonic()
    with history_path.open("x") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            permutation = torch.randperm(len(train_x), generator=generator, device=train_x.device)
            loss_sum = 0.
            for ids in permutation.split(args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = masked_filler_loss(model(train_x[ids]), train_roles[ids], train_y[ids])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * len(ids)
            model.eval()
            with torch.no_grad():
                validation = token_metrics(model(val_x), val_roles, val_y)
            log.write(json.dumps({"epoch": epoch, "train_loss": loss_sum / len(train_x),
                                  "val_loss": validation["loss"],
                                  "val_role_conditioned_filler_accuracy":
                                  validation["role_conditioned_filler_accuracy"]}) + "\n")
            log.flush()
            if validation["loss"] < best_loss:
                best_loss, best_epoch, stale = validation["loss"], epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation = token_metrics(model(val_x), val_roles, val_y)
    return {"best_epoch": best_epoch, "epochs_run": epoch, "validation": validation,
            "elapsed_seconds": time.monotonic() - started,
            "stopped_at_epoch_limit": epoch == args.epochs}
