"""Evaluate TPR reconstruction and role swaps in the original frozen LLM."""

import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.interventions import score_batch
from ac_tpr.model import load_model
from ac_tpr.prompts import PROMPT_TEMPLATE, prepare_prompts


CONDITIONS = ("zero", "reconstruction", "swap", "donor_swap")
GROUP_COLUMNS = [
    "split", "voice", "layer", "site", "condition", "source_tpr_split",
    "target_tpr_split", "target_binding_held_out", "source_seen_in_training",
]
BASELINE_FIELDS = [
    "source_index", "target_index", "corpus_index", "target_corpus_index",
    "source_tpr_split", "target_tpr_split",
    "target_binding_held_out", "source_seen_in_training", "target_sentence",
    "agent_token_id", "patient_token_id", "sentence_end_position", "prompt_end_position",
    "baseline_agent_logit", "baseline_patient_logit", "baseline_logit_diff",
    "donor_baseline_agent_logit", "donor_baseline_patient_logit", "donor_baseline_logit_diff",
]
RESULT_FIELDS = [
    "layer", "site", "condition", "alpha", "intervened_agent_logit",
    "intervened_patient_logit", "intervened_logit_diff", "logit_diff_change",
]


def summarize_steering(results_path: Path, output_path: Path) -> int:
    """Aggregate strict answer preferences, preserving the direction of binding tests.

    The ordinary flip rate divides by all rows. The conditional flip rate divides
    by baseline agent wins, and is empty when that denominator is zero.
    """
    groups = defaultdict(Counter)
    with results_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            totals = groups[tuple(row[key] for key in GROUP_COLUMNS)]
            baseline = float(row["baseline_logit_diff"])
            intervened = float(row["intervened_logit_diff"])
            totals.update({
                "count": 1,
                "baseline_mean_logit_diff": baseline,
                "intervened_mean_logit_diff": intervened,
                "mean_logit_diff_change": float(row["logit_diff_change"]),
                "baseline_agent_preference_rate": int(baseline < 0),
                "baseline_patient_preference_rate": int(baseline > 0),
                "intervened_agent_preference_rate": int(intervened < 0),
                "intervened_patient_preference_rate": int(intervened > 0),
                "agent_to_patient_flip_rate": int(baseline < 0 and intervened > 0),
            })
    if not groups:
        raise ValueError("Steering results must contain at least one row")
    fields = GROUP_COLUMNS + list(next(iter(groups.values()))) + [
        "baseline_agent_wins", "conditional_agent_to_patient_flip_rate",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, totals in sorted(groups.items()):
            wins = totals["baseline_agent_preference_rate"]
            writer.writerow({
                **dict(zip(GROUP_COLUMNS, key)),
                **{name: value if name == "count" else value / totals["count"]
                   for name, value in totals.items()},
                "baseline_agent_wins": wins,
                "conditional_agent_to_patient_flip_rate": (
                    totals["agent_to_patient_flip_rate"] / wins if wins else ""
                ),
            })
    return len(groups)


@torch.inference_mode()
def evaluate_steering(
    cache: dict, models: dict[tuple[int, str], torch.nn.Module], output_dir: Path,
    *, device: str = "cuda", dtype: str = "bfloat16", batch_size: int = 8,
    alpha: float = 1.0, prompt_template: str = PROMPT_TEMPLATE,
) -> dict:
    """Patch one cached block/site at a time and score the original answer choices.

    TPR outputs use training-normalized coordinates. Reconstruction restores the
    mean and scale; role-swap differences restore only the scale and preserve the
    unexplained residual. Donors are real same-voice sentences with exchanged
    agent and patient. They are an evaluation control, not TPR training targets.
    """
    from ac_tpr.tpr.extraction import validate_cache

    started = time.monotonic()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if batch_size < 1 or not math.isfinite(alpha):
        raise ValueError("batch_size must be positive and alpha must be finite")
    validate_cache(cache)
    source = cache["metadata"]
    if source["prompt_template"] != prompt_template:
        raise ValueError("Extraction and steering must use the same prompt template")
    if source["model"]["dtype"] != dtype:
        raise ValueError("Steering dtype must match the cached activation dtype")
    layers, sites = source["layer_indices"], source["sites"]
    if sites != ["sentence_end", "prompt_end"]:
        raise ValueError("Steering requires sentence_end and prompt_end sites in that order")
    if set(models) != {(layer, site) for layer in layers for site in sites}:
        raise ValueError("Provide one TPR model for every cached layer and site")

    rows = cache["rows"]
    indices = [i for i, row in enumerate(rows)
               if row["split"] == "test" or row["tpr_split"] == "binding_test"]
    if not indices:
        raise ValueError("No test or binding_test rows to evaluate")
    donors = cache["donor_indices"].long()
    needed = sorted(set(indices) | {int(donors[i]) for i in indices})
    answers = cache["answer_ids"]
    if not torch.equal(answers[donors[indices]], answers[indices].flip(-1)):
        raise ValueError("Donor answers must exchange the source agent and patient")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True,
    )
    input_ids, _, prepared_answers = prepare_prompts(rows, tokenizer, prompt_template)
    if input_ids != cache["input_ids"] or not torch.equal(prepared_answers, answers):
        raise ValueError("Cached token IDs do not match the rendered steering prompts")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if tokenizer_hash != source["tokenizer"]["backend_sha256"]:
        raise ValueError("Cached tokenizer does not match the scoring tokenizer")
    model, execution_model = load_model(tokenizer, TOKENIZER_REVISION, device, dtype)
    if max(layers) >= model.config.num_hidden_layers:
        raise ValueError("A cached layer is outside the scoring model")
    if cache["activations"].shape[-1] != model.config.hidden_size:
        raise ValueError("Cached states do not match the scoring model hidden size")

    # Keep logits for both source and donor prompts in their own semantic order.
    baseline_scores = {}
    source_indices = set(indices)
    for baseline_indices in (indices, [i for i in needed if i not in source_indices]):
        for start in range(0, len(baseline_indices), batch_size):
            batch_indices = baseline_indices[start:start + batch_size]
            inputs = tokenizer.pad(
                {"input_ids": [input_ids[i] for i in batch_indices]},
                padding=True, return_tensors="pt",
            ).to(device)
            scores = score_batch(model, inputs, answers[batch_indices])
            baseline_scores.update(zip(batch_indices, scores, strict=True))

    records = {}
    for index in indices:
        target = int(donors[index])
        agent, patient = baseline_scores[index].tolist()
        donor_agent, donor_patient = baseline_scores[target].flip(-1).tolist()
        records[index] = {
            **rows[index], "source_index": index, "target_index": target,
            "corpus_index": int(cache["corpus_indices"][index]),
            "target_corpus_index": int(cache["corpus_indices"][target]),
            "source_tpr_split": rows[index]["tpr_split"],
            "target_tpr_split": rows[target]["tpr_split"],
            "target_binding_held_out": rows[target]["tpr_split"] == "binding_test",
            "source_seen_in_training": rows[index]["tpr_split"] == "train",
            "target_sentence": rows[target]["sentence"],
            "agent_token_id": int(answers[index, 0]), "patient_token_id": int(answers[index, 1]),
            "sentence_end_position": int(cache["positions"][index, 0]),
            "prompt_end_position": int(cache["positions"][index, 1]),
            "baseline_agent_logit": agent, "baseline_patient_logit": patient,
            "baseline_logit_diff": patient - agent,
            "donor_baseline_agent_logit": donor_agent,
            "donor_baseline_patient_logit": donor_patient,
            "donor_baseline_logit_diff": donor_patient - donor_agent,
        }

    output_dir.mkdir(parents=True, exist_ok=False)
    baseline_columns = list(rows[0]) + BASELINE_FIELDS
    with (output_dir / "baseline.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=baseline_columns)
        writer.writeheader()
        writer.writerows(records.values())
    result_count = 0
    final_layer = model.config.num_hidden_layers - 1
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=baseline_columns + RESULT_FIELDS)
        writer.writeheader()
        for layer_index, layer in enumerate(layers):
            for site_index, site in enumerate(sites):
                tpr_model = models[layer, site].to(device).eval()
                mean = cache["mean"][layer_index, site_index].float()
                scale = cache["scale"][layer_index, site_index].float()
                try:
                    for start in range(0, len(indices), batch_size):
                        batch_indices = indices[start:start + batch_size]
                        inputs = tokenizer.pad(
                            {"input_ids": [input_ids[i] for i in batch_indices]},
                            padding=True, return_tensors="pt",
                        ).to(device)
                        fillers = cache["filler_ids"][batch_indices].to(device)
                        roles = cache["role_ids"][batch_indices].to(device)
                        swapped_fillers = fillers.clone()
                        swapped_fillers[:, [0, 2]] = fillers[:, [2, 0]]
                        original = tpr_model(fillers, roles).float().cpu()
                        swapped = tpr_model(swapped_fillers, roles).float().cpu()
                        raw = cache["activations"][batch_indices, layer_index, site_index].float()
                        donor = cache["activations"][donors[batch_indices], layer_index, site_index].float()
                        deltas = {
                            "zero": torch.zeros_like(raw),
                            "reconstruction": mean + scale * original - raw,
                            "swap": alpha * scale * (swapped - original),
                            "donor_swap": donor - raw,
                        }
                        baseline = torch.stack([baseline_scores[i] for i in batch_indices])
                        for condition, delta in deltas.items():
                            scores = score_batch(
                                model, inputs, answers[batch_indices], layer,
                                cache["positions"][batch_indices, site_index], delta,
                                allow_final_token=site == "prompt_end",
                            )
                            if condition == "zero" and not torch.equal(scores, baseline):
                                raise ValueError("Zero-vector control changed baseline logits")
                            if layer == final_layer and site == "sentence_end" and not torch.equal(scores, baseline):
                                raise ValueError("Final-block sentence-end control changed answer logits")
                            if layer == final_layer and site == "prompt_end" and condition == "donor_swap":
                                expected = torch.stack([
                                    baseline_scores[int(donors[i])].flip(-1) for i in batch_indices
                                ])
                                rtol, atol = (0.01, 0.125) if dtype == "bfloat16" else (1e-5, 1e-6)
                                if not torch.allclose(scores, expected, rtol=rtol, atol=atol):
                                    raise ValueError("Final-block prompt-end donor patch did not reproduce donor logits")
                            for index, (agent, patient) in zip(batch_indices, scores.tolist(), strict=True):
                                difference = patient - agent
                                writer.writerow({
                                    **records[index], "layer": layer, "site": site,
                                    "condition": condition, "alpha": alpha if condition == "swap" else 1.0,
                                    "intervened_agent_logit": agent, "intervened_patient_logit": patient,
                                    "intervened_logit_diff": difference,
                                    "logit_diff_change": difference - records[index]["baseline_logit_diff"],
                                })
                                result_count += 1
                    handle.flush()
                    print(f"Evaluated TPR steering at block {layer}, {site}", flush=True)
                finally:
                    tpr_model.cpu()

    summary_count = summarize_steering(output_dir / "results.csv", output_dir / "summary.csv")
    metadata = {
        "tpr_schema_version": 1, "artifact_type": "tpr_steering",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "source_cache": source, "execution_model": execution_model,
        "prompt_template": prompt_template, "layer_indices": layers, "sites": sites,
        "conditions": list(CONDITIONS), "alpha": alpha,
        "selection": "original split test or tpr_split binding_test",
        "source_tpr_split_counts": dict(Counter(rows[i]["tpr_split"] for i in indices)),
        "target_tpr_split_counts": dict(Counter(rows[int(donors[i])]["tpr_split"] for i in indices)),
        "original_answer": "source agent", "counterfactual_answer": "source patient",
        "logit_diff": "patient_logit - agent_logit",
        "logit_diff_change": "intervened_logit_diff - baseline_logit_diff",
        "donor_baseline_candidates": "source agent and source patient, in that order",
        "flip_rate": "baseline_logit_diff < 0 and intervened_logit_diff > 0, divided by all rows",
        "conditional_flip_rate": "same flips divided by baseline agent wins; empty when undefined",
        "interventions": {
            "zero": "h + 0",
            "reconstruction": "h + (mean + scale * T(original) - cached_h)",
            "swap": "h + alpha * scale * (T(swapped) - T(original))",
            "donor_swap": "h + (cached_donor_h - cached_h)",
        },
        "controls": {
            "zero_vector": "passed",
            "final_block_sentence_end": "passed" if final_layer in layers else "not evaluated",
            "final_block_prompt_end_donor": "passed" if final_layer in layers else "not evaluated",
        },
        "baseline_rows": len(indices), "baseline_forward_rows": len(needed),
        "result_rows": result_count, "summary_rows": summary_count,
        "batch_size": batch_size, "device": device,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return metadata
