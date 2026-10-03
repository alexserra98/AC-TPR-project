"""Compare classic Pythia outputs with an independent HF float32 forward."""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.interventions import MEAN_TYPES, prepare_prompts, role_deltas


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference_dir = Path("ac-tpr-cache/migration-reference-tl4/pythia")
    reference = torch.load(reference_dir / "activations.pt", weights_only=True)
    vectors = torch.load(reference_dir / "syntactic_vectors.pt", weights_only=True)
    metadata = json.loads((reference_dir / "metadata.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_ID, revision=metadata["model"]["commit"], dtype=torch.float32,
        attn_implementation="sdpa",
    ).eval().to("cuda")
    model.requires_grad_(False)
    rows = reference["rows"]
    ids, positions, answers = prepare_prompts(rows, tokenizer)
    inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
    positions, answers = positions.cuda(), answers.cuda()
    batch_indices = torch.arange(len(rows), device="cuda")
    last = inputs["attention_mask"].sum(1) - 1

    def score(layer=None, role_index=None, delta=None):
        def translate(module, inputs, hidden):
            result = hidden.clone()
            result[batch_indices, positions[:, role_index]] += delta.cuda()
            return result

        handle = None if layer is None else model.gpt_neox.layers[layer].register_forward_hook(translate)
        try:
            logits = model(**inputs, use_cache=False).logits
            return logits[batch_indices, last].gather(1, answers).cpu()
        finally:
            if handle is not None:
                handle.remove()

    started = time.monotonic()
    states = torch.empty_like(reference["activations"], dtype=torch.float32)
    for batch in torch.load(reference_dir / "extraction_batches.pt", weights_only=True):
        encoded = tokenizer.pad({"input_ids": batch["input_ids"]}, padding=True, return_tensors="pt").to("cuda")
        role_positions = batch["role_positions"].cuda()
        indices = torch.arange(len(role_positions), device="cuda")[:, None]
        captured = []

        def capture(module, inputs, hidden):
            captured.append(hidden[indices, role_positions].cpu())

        handles = [block.register_forward_hook(capture) for block in model.gpt_neox.layers]
        try:
            model.gpt_neox(**encoded, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
        states[batch["reference_indices"]] = torch.stack(captured, dim=1)[batch["batch_indices"]]
    baseline = score()
    scores = {}
    for layer in range(model.config.num_hidden_layers):
        for mean_type in MEAN_TYPES:
            for role_index, role in enumerate(("agent", "patient")):
                delta = role_deltas(vectors, rows, mean_type, layer, role)
                scores[f"{layer}/{mean_type}/{role}"] = score(layer, role_index, delta)
    outputs = {"activations": states, "baseline": baseline, "scores": scores}
    torch.save(outputs, args.output_dir / "outputs.pt")
    classic = torch.load(args.classic_dir / "outputs.pt", weights_only=True)
    pairs = [(name, classic[name], outputs[name]) for name in ("activations", "baseline")]
    pairs += [(key, classic["scores"][key], scores[key]) for key in scores]
    measurements, failures = {}, []
    for name, actual, expected in pairs:
        difference = (actual - expected).abs()
        measurements[name] = {"max_abs": difference.max().item(), "mean_abs": difference.mean().item()}
        try:
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-4 if name == "activations" else 1e-4)
        except AssertionError as error:
            failures.append({"name": name, "error": str(error)})
    relative_rms = ((classic["activations"] - states).square().mean() / states.square().mean()).sqrt().item()
    if relative_rms >= 1e-5 or not torch.isfinite(torch.tensor(relative_rms)):
        failures.append({"name": "activation_relative_rms", "error": str(relative_rms)})
    report = {"passed": not failures, "dtype": "float32", "rtol": 1e-4,
              "score_atol": 1e-4, "activation_atol": 5e-4, "activation_relative_rms": relative_rms,
              "seconds": time.monotonic() - started, "measurements": measurements, "failures": failures}
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "measurements"}, indent=2))
    assert not failures, "Inspect the float32 HF comparison report"


if __name__ == "__main__":
    main()
