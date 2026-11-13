"""Train the unpacking decoder and TPR with validation-selected checkpoints."""

import gc
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    eval_every: int = 100
    patience: int = 5
    max_lr_reductions: int = 5
    seed: int = 42


def fit_model(
    model: nn.Module, train_dataset: Dataset, val_dataset: Dataset,
    loss_fn: Callable[[nn.Module, tuple[torch.Tensor, ...]], tuple[torch.Tensor, int]],
    output_dir: str | Path, config: TrainingConfig, device: str | torch.device,
) -> dict:
    """Minimize a summed loss/count callback and restore the best validation state.

    Validation and reported training losses are weighted by the callback's count,
    so partial batches and variable sentence lengths have the correct weight.
    A plateau restores the best weights and halves Adam's learning rate; training
    stops after the configured number of reductions has been exhausted.
    """
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Training and validation datasets must both be nonempty")
    if min(config.batch_size, config.epochs, config.eval_every, config.patience) <= 0:
        raise ValueError("Batch size, epochs, evaluation interval, and patience must be positive")
    if config.learning_rate <= 0 or config.max_lr_reductions < 0:
        raise ValueError("Learning rate must be positive and reduction count nonnegative")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Training output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    model.to(device)
    learning_rate = config.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"config": asdict(config), "evaluations": []}
    best_loss = float("inf")
    plateau = reductions = step = 0
    accumulated_loss = 0.0
    accumulated_count = 0
    stop = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            batch = tuple(tensor.to(device) for tensor in batch)
            summed_loss, count = loss_fn(model, batch)
            if count <= 0 or not torch.isfinite(summed_loss):
                raise ValueError("Training loss must be finite with a positive normalizer")
            optimizer.zero_grad(set_to_none=True)
            (summed_loss / count).backward()
            optimizer.step()
            accumulated_loss += summed_loss.detach().item()
            accumulated_count += count
            step += 1
            final_step = epoch == config.epochs and batch_index + 1 == len(train_loader)
            if step % config.eval_every and not final_step:
                continue

            model.eval()
            valid_loss, valid_count = 0.0, 0
            with torch.no_grad():
                for valid_batch in val_loader:
                    valid_batch = tuple(tensor.to(device) for tensor in valid_batch)
                    total, count = loss_fn(model, valid_batch)
                    if count <= 0 or not torch.isfinite(total):
                        raise ValueError("Validation loss must be finite with a positive normalizer")
                    valid_loss += total.item()
                    valid_count += count
            valid_loss /= valid_count
            print(f"Step {step}, epoch {epoch}: validation loss {valid_loss:.6g}", flush=True)
            history["evaluations"].append({
                "step": step, "epoch": epoch, "train_loss": accumulated_loss / accumulated_count,
                "validation_loss": valid_loss, "learning_rate": learning_rate,
            })
            accumulated_loss = 0.0
            accumulated_count = 0
            if valid_loss < best_loss:
                best_loss = valid_loss
                plateau = 0
                history.update(best_step=step, best_epoch=epoch, best_validation_loss=best_loss)
                torch.save(model.state_dict(), best_path)
            else:
                plateau += 1

            if plateau >= config.patience:
                if reductions >= config.max_lr_reductions:
                    stop = True
                    break
                reductions += 1
                learning_rate /= 2
                model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                plateau = 0
            model.train()
        if stop:
            break

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    history.update(steps=step, epochs=epoch, learning_rate_reductions=reductions)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return history


def model_directory(root: Path, layer: int, site: str) -> Path:
    return root / f"layer_{layer:02d}" / site



def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
