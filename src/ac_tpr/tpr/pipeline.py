"""Run DISCOVER-style decomposition and evaluate role swaps in the original LLM.

1. Extract full-prompt states at sentence_end and prompt_end; estimate training
   normalization and keep unseen filler-role bindings out of all model fitting.
2. Train an unpacker from each standardized state to the target sentence only.
3. Independently fit a TPR from annotated bindings to the same standardized state.
4. Freeze both models and compare free-running reconstruction from real, TPR,
   and edited states. The unpacker is a diagnostic, not a steering component.
5. Add the unstandardized TPR swap difference at the original LLM hook, then
   score agent/patient logits at the final prompt token with the existing scorer.

Each stage writes a separate directory. Existing stage outputs are rejected;
individual stages can consume completed outputs from earlier invocations.
"""

import json
import math
from pathlib import Path

from ac_tpr.prompts import PROMPT_TEMPLATE
from ac_tpr.tpr.decomposition import TPRConfig, fit_tprs, load_tprs
from ac_tpr.tpr.extraction import (
    DEFAULT_BINDINGS, DEFAULT_LAYERS, extract_encodings, load_cache, load_auxiliary_metadata,
)
from ac_tpr.tpr.reconstruction import evaluate_reconstruction
from ac_tpr.tpr.training import TrainingConfig, release_memory
from ac_tpr.tpr.unpacking import UnpackingConfig, train_unpackers


def run_pipeline(
    corpus_path: Path, output_dir: Path, *, stage: str = "all", layers: tuple[int, ...] = DEFAULT_LAYERS,
    held_out_bindings: tuple[tuple[str, str], ...] = DEFAULT_BINDINGS,
    unpacking: UnpackingConfig = UnpackingConfig(), tpr: TPRConfig = TPRConfig(),
    unpacker_training: TrainingConfig = TrainingConfig(learning_rate=1e-4),
    tpr_training: TrainingConfig = TrainingConfig(), device: str = "cuda", dtype: str = "bfloat16",
    batch_size: int = 8, alpha: float = 1.0, prompt_template: str = PROMPT_TEMPLATE,
) -> dict:
    """Run all stages or one stage against a common experiment directory."""
    stages = ("extract", "unpack", "fit", "reconstruct", "steer")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    if stage != "all" and stage not in stages:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    if stage == "all" and output_dir.exists():
        raise FileExistsError(output_dir)
    results = {}
    encodings = output_dir / "encodings"
    for current in stages if stage == "all" else (stage,):
        if current == "extract":
            results[current] = extract_encodings(corpus_path, encodings, layers, batch_size, device, dtype,
                                                prompt_template=prompt_template, held_out_bindings=held_out_bindings)
        elif current == "unpack":
            results[current] = train_unpackers(encodings, output_dir / "unpackers", config=unpacking,
                                               training=unpacker_training, device=device)
        elif current == "fit":
            results[current] = fit_tprs(encodings, output_dir / "tprs", config=tpr, training=tpr_training, device=device)
        elif current == "reconstruct":
            results[current] = evaluate_reconstruction(encodings, output_dir / "unpackers", output_dir / "tprs",
                                                       output_dir / "reconstruction", device=device,
                                                       batch_size=batch_size, alpha=alpha)
        else:
            from ac_tpr.tpr.steering import evaluate_steering

            cache = load_cache(encodings)
            models = load_tprs(output_dir / "tprs", cache)
            results[current] = evaluate_steering(cache, models, output_dir / "steering", device=device,
                                                 dtype=dtype, batch_size=batch_size, alpha=alpha,
                                                 prompt_template=prompt_template)
            results[current]["tprs"] = load_auxiliary_metadata(output_dir / "tprs", cache, "tpr_models")
            (output_dir / "steering" / "metadata.json").write_text(json.dumps(results[current], indent=2) + "\n")
            del cache, models
        release_memory()
    return results
