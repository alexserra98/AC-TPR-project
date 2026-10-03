"""Compare TransformerLens outputs with the captured pre-migration references."""

import argparse
import csv
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
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--report-only", action="store_true", help="Record historical precision differences without failing the job")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.init_kwargs["revision"] = TOKENIZER_REVISION
    started = time.monotonic()
    if args.device == "cpu":
        reference = torch.load(args.reference_dir / "tiny.pt", weights_only=True)
        config = GPTNeoXConfig.from_dict(reference["config"])
        config._attn_implementation = "sdpa"
        config._commit_hash = "tiny-reference"
        config.architectures = ["GPTNeoXForCausalLM"]
        hf_model = GPTNeoXForCausalLM(config).eval()
        hf_model.load_state_dict(reference["state_dict"], strict=True)
        with patch("ac_tpr.model.AutoConfig.from_pretrained", return_value=config), \
             patch("ac_tpr.model.AutoModelForCausalLM.from_pretrained", return_value=hf_model):
            model, _ = load_model(tokenizer, config._commit_hash, "cpu", "float32")
        rows, vectors = reference["rows"], reference["vectors"]
        expected_baseline, expected_scores = reference["baseline"], reference["scores"]
        rtol, atol = 1e-5, 1e-6
    else:
        directory = args.reference_dir / "pythia"
        reference = torch.load(directory / "activations.pt", weights_only=True)
        metadata = json.loads((directory / "metadata.json").read_text())
        model, _ = load_model(tokenizer, metadata["model"]["commit"], "cuda", args.dtype)
        rows = reference["rows"]
        vectors = torch.load(directory / "syntactic_vectors.pt", weights_only=True)
        with (directory / "baseline.csv").open() as handle:
            records = list(csv.DictReader(handle))
        assert [(r["pair_id"], r["voice"]) for r in rows] == [(r["pair_id"], r["voice"]) for r in records]
        expected_baseline = torch.tensor([
            [float(row["baseline_agent_logit"]), float(row["baseline_patient_logit"])] for row in records
        ])
        with (directory / "results.csv").open() as handle:
            records = list(csv.DictReader(handle))
        expected_scores = {}
        for row in records:
            key = f"{row['layer']}/{row['mean_type']}/{row['edited_role']}"
            expected_scores.setdefault(key, []).append([
                float(row["intervened_agent_logit"]), float(row["intervened_patient_logit"]),
            ])
        expected_scores = {key: torch.tensor(value) for key, value in expected_scores.items()}
        # FP16 comparisons permit small kernel rounding differences across library versions.
        rtol, atol = 1e-3, 0.05

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
        actual_activations = torch.empty_like(reference["activations"], dtype=model.cfg.dtype)
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
                           torch.zeros(len(rows), model.cfg.d_model))
        torch.testing.assert_close(zero, baseline, rtol=0, atol=0)
    for layer in range(model.cfg.n_layers):
        for mean_type in MEAN_TYPES:
            for role_index, role in enumerate(("agent", "patient")):
                key = f"{layer}/{mean_type}/{role}"
                deltas = role_deltas(vectors, rows, mean_type, layer, role)
                scores[key] = score_batch(model, inputs, answers, layer, prompt_positions[:, role_index], deltas)
                if layer == model.cfg.n_layers - 1:
                    torch.testing.assert_close(scores[key], baseline, rtol=0, atol=0)
    if args.device == "cuda":
        torch.cuda.synchronize()
    measurements = {}
    failures = []
    pairs = [("activations", actual_activations, reference["activations"]),
             ("baseline", baseline, expected_baseline)]
    pairs.extend((key, scores[key], expected_scores[key]) for key in scores)
    for name, actual, expected in pairs:
        difference = (actual.float() - expected.float()).abs()
        measurements[name] = {"max_abs": difference.max().item(), "mean_abs": difference.mean().item()}
        try:
            torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
        except AssertionError as error:
            failures.append({"name": name, "error": str(error)})
    actual_diffs = torch.stack([scores[key][:, 1] - scores[key][:, 0] for key in scores])
    expected_diffs = torch.stack([expected_scores[key][:, 1] - expected_scores[key][:, 0] for key in scores])
    report = {
        "passed": not failures, "device": args.device, "dtype": str(model.cfg.dtype), "rows": len(rows),
        "conditions": len(scores), "rtol": rtol, "atol": atol,
        "load_seconds": loaded_seconds, "evaluation_seconds": time.monotonic() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if args.device == "cuda" else None,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "max_logit_difference_change": (actual_diffs - expected_diffs).abs().max().item(),
        "preference_changes": ((actual_diffs > 0) != (expected_diffs > 0)).sum().item(),
        "versions": {name: version(name) for name in ("torch", "transformers", "transformer-lens")},
        "measurements": measurements, "failures": failures,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save({"activations": actual_activations, "baseline": baseline, "scores": scores},
               args.output_dir / "outputs.pt")
    print(json.dumps({key: value for key, value in report.items() if key not in {"measurements", "failures"}}, indent=2))
    if failures and not args.report_only:
        raise ValueError(f"Reference comparison failed for {len(failures)} tensors; inspect comparison.json")


if __name__ == "__main__":
    main()
