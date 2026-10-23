"""Validate NNsight against saved TL3 float32 outputs with identical inputs."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

from ac_tpr.activations import extract_batch, tokenize_roles
from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.interventions import MEAN_TYPES, prepare_prompts, role_deltas, score_batch
from ac_tpr.model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, default=Path("ac-tpr-cache/migration-reference-tl4"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    started = time.monotonic()
    if args.device == "cpu":
        reference = torch.load(args.reference_dir / "tiny.pt", weights_only=True)
        config = GPTNeoXConfig.from_dict(reference["config"])
        config._attn_implementation = "eager"
        config._commit_hash = "tiny-reference"
        config.architectures = ["GPTNeoXForCausalLM"]
        hf_model = GPTNeoXForCausalLM(config).eval()
        hf_model.load_state_dict(reference["state_dict"], strict=True)
        with patch("ac_tpr.model.AutoConfig.from_pretrained", return_value=config), \
             patch("ac_tpr.model.AutoModelForCausalLM.from_pretrained", return_value=hf_model):
            model, _ = load_model(tokenizer, config._commit_hash, "cpu", "float32")
        rows, vectors = reference["rows"], reference["vectors"]
    else:
        directory = args.reference_dir / "pythia"
        reference = torch.load(directory / "activations.pt", weights_only=True)
        metadata = json.loads((directory / "metadata.json").read_text())
        model, _ = load_model(tokenizer, metadata["model"]["commit"], "cuda", "float32")
        rows = reference["rows"]
        vectors = torch.load(directory / "syntactic_vectors.pt", weights_only=True)

    classic_dir = Path(f"ac-tpr-cache/migration-comparison-tl3-{'cpu' if args.device == 'cpu' else 'gpu'}")
    expected = torch.load(classic_dir / "outputs.pt", weights_only=True)
    expected_baseline, expected_scores = expected["baseline"], expected["scores"]
    rtol, score_atol = (1e-5, 1e-6) if args.device == "cpu" else (1e-4, 1e-4)
    activation_atol = 1e-6 if args.device == "cpu" else 5e-4

    loaded_seconds = time.monotonic() - started
    ids, positions = tokenize_roles(rows, tokenizer)
    if "input_ids" in reference:
        assert ids == reference["input_ids"]
        torch.testing.assert_close(positions, reference["role_positions"], rtol=0, atol=0)
    sentence_inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(args.device)
    prompt_ids, prompt_positions, answers = prepare_prompts(rows, tokenizer)
    inputs = tokenizer.pad({"input_ids": prompt_ids}, padding=True, return_tensors="pt").to(args.device)
    if args.device == "cuda":
        score_batch(model, inputs, answers)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    if args.device == "cpu":
        actual_activations = extract_batch(model, sentence_inputs, positions)
    else:
        actual_activations = torch.empty_like(reference["activations"], dtype=model.dtype)
        extraction_batches = torch.load(directory / "extraction_batches.pt", weights_only=True)
        for batch in extraction_batches:
            extraction_inputs = tokenizer.pad({"input_ids": batch["input_ids"]}, padding=True,
                                              return_tensors="pt").to(args.device)
            states = extract_batch(model, extraction_inputs, batch["role_positions"])
            actual_activations[batch["reference_indices"]] = states[batch["batch_indices"]]
    baseline = score_batch(model, inputs, answers)
    scores = {}
    for role_index in range(2):
        zero = score_batch(model, inputs, answers, 0, prompt_positions[:, role_index],
                           torch.zeros(len(rows), model.config.hidden_size))
        torch.testing.assert_close(zero, baseline, rtol=0, atol=0)
    for layer in range(model.config.num_hidden_layers):
        for mean_type in MEAN_TYPES:
            for role_index, role in enumerate(("agent", "patient")):
                key = f"{layer}/{mean_type}/{role}"
                deltas = role_deltas(vectors, rows, mean_type, layer, role)
                scores[key] = score_batch(model, inputs, answers, layer, prompt_positions[:, role_index], deltas)
                if layer == model.config.num_hidden_layers - 1:
                    torch.testing.assert_close(scores[key], baseline, rtol=0, atol=0)
    if args.device == "cuda":
        torch.cuda.synchronize()
    measurements = {}
    failures = []
    pairs = [("activations", actual_activations, expected["activations"]),
             ("baseline", baseline, expected_baseline)]
    assert set(scores) == set(expected_scores)
    pairs.extend((key, scores[key], expected_scores[key]) for key in scores)
    for name, actual, expected in pairs:
        difference = (actual.float() - expected.float()).abs()
        measurements[name] = {"max_abs": difference.max().item(), "mean_abs": difference.mean().item()}
        try:
            torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=activation_atol if name == "activations" else score_atol)
        except AssertionError as error:
            failures.append({"name": name, "error": str(error)})
    reference_activations = pairs[0][2].float()
    relative_rms = ((actual_activations - reference_activations).square().mean()
                    / reference_activations.square().mean()).sqrt().item()
    if not torch.isfinite(torch.tensor(relative_rms)) or relative_rms >= 1e-5:
        failures.append({"name": "activation_relative_rms", "error": str(relative_rms)})
    actual_diffs = torch.stack([scores[key][:, 1] - scores[key][:, 0] for key in scores])
    expected_diffs = torch.stack([expected_scores[key][:, 1] - expected_scores[key][:, 0] for key in scores])
    report = {
        "passed": not failures, "device": args.device, "dtype": str(model.dtype), "rows": len(rows),
        "conditions": len(scores), "rtol": rtol, "score_atol": score_atol,
        "activation_atol": activation_atol, "activation_relative_rms": relative_rms,
        "activation_relative_rms_limit": 1e-5,
        "controls": {"zero_vector": "exact", "final_block": "exact"},
        "reference_directory": str(classic_dir.resolve()),
        "checkpoint_commit": model.config._commit_hash,
        "attention_implementation": model.config._attn_implementation,
        "load_seconds": loaded_seconds, "evaluation_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "max_logit_difference_change": (actual_diffs - expected_diffs).abs().max().item(),
        "preference_changes": ((actual_diffs > 0) != (expected_diffs > 0)).sum().item(),
        "versions": {name: version(name) for name in ("torch", "transformers", "nnsight")},
        "measurements": measurements, "failures": failures,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"activations": actual_activations, "baseline": baseline, "scores": scores},
               args.output_dir / "outputs.pt")
    print(json.dumps({key: value for key, value in report.items() if key not in {"measurements", "failures"}}, indent=2))
    if failures:
        raise ValueError(f"Reference comparison failed for {len(failures)} tensors; inspect comparison.json")


if __name__ == "__main__":
    main()
