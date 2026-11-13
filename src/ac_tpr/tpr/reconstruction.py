"""Measure sentence recovery from real, approximated, and edited LLM states."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from ac_tpr.tpr.decomposition import load_tprs
from ac_tpr.tpr.extraction import cache_fingerprint, load_cache, load_auxiliary_metadata, normalized_states
from ac_tpr.tpr.training import model_directory, release_memory
from ac_tpr.tpr.unpacking import EOS_ID, SentenceUnpacker, UnpackingConfig


@torch.inference_mode()
def evaluate_reconstruction(
    encodings_dir: Path, unpackers_dir: Path, tprs_dir: Path, output_dir: Path, *,
    device: str = "cuda", batch_size: int = 32, alpha: float = 1.0,
) -> dict:
    """Compare free-running sentence recovery and swaps, without teacher forcing."""
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if batch_size < 1 or not math.isfinite(alpha):
        raise ValueError("batch_size must be positive and alpha must be finite")
    cache = load_cache(encodings_dir)
    decoder_metadata = load_auxiliary_metadata(unpackers_dir, cache, "tpr_unpackers")
    models = load_tprs(tprs_dir, cache)
    indices = [i for i, row in enumerate(cache["rows"])
               if row["split"] == "test" or row["tpr_split"] == "binding_test"]
    if not indices:
        raise ValueError("No test examples to reconstruct")
    tokens_by_id = {index: token for token, index in cache["token_vocab"].items()}
    output_dir.mkdir(parents=True)
    fields = ["corpus_index", "pair_id", "voice", "split", "tpr_split", "target_tpr_split",
              "target_binding_held_out", "layer", "site", "condition", "exact_match",
              "correct_tokens", "target_tokens", "activation_mse", "expected_sentence", "predicted_tokens"]
    groups = defaultdict(lambda: {"count": 0, "exact_match": 0, "correct_tokens": 0,
                                  "target_tokens": 0, "activation_mse": 0.0})
    with (output_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for li, layer in enumerate(cache["metadata"]["layer_indices"]):
            for si, site in enumerate(cache["metadata"]["sites"]):
                print(f"Evaluating reconstruction: block {layer}, {site}", flush=True)
                decoder = SentenceUnpacker(cache["metadata"]["hidden_size"], len(cache["token_vocab"]),
                                            UnpackingConfig(**decoder_metadata["config"]))
                decoder.load_state_dict(torch.load(model_directory(unpackers_dir, layer, site) / "best.pt",
                                                   map_location="cpu", weights_only=True))
                decoder.to(device).eval().requires_grad_(False)
                tpr = models[layer, site].to(device)
                z = normalized_states(cache, li, si)
                for start in range(0, len(indices), batch_size):
                    selected = indices[start:start + batch_size]
                    donors = cache["donor_indices"][selected].tolist()
                    fillers, roles = cache["filler_ids"][selected].to(device), cache["role_ids"][selected].to(device)
                    swapped = fillers[:, [2, 1, 0, 3]]
                    real = z[selected].to(device)
                    predicted = tpr(fillers, roles)
                    edited = real + alpha * (tpr(swapped, roles) - predicted)
                    for condition, states, targets in (("real", real, selected),
                                                        ("tpr", predicted, selected),
                                                        ("swap", edited, donors)):
                        guesses = decoder.generate(states).cpu().tolist()
                        mse = (states.cpu() - z[targets]).square().mean(dim=1).tolist()
                        for j, (source, target, guess) in enumerate(zip(selected, targets, guesses, strict=True)):
                            if EOS_ID in guess:
                                guess = guess[:guess.index(EOS_ID) + 1]
                            gold = [*cache["target_ids"][target], EOS_ID]
                            correct = sum(a == b for a, b in zip(guess, gold))
                            row, target_row = cache["rows"][source], cache["rows"][target]
                            held = target_row["tpr_split"] == "binding_test"
                            writer.writerow({
                                "corpus_index": int(cache["corpus_indices"][source]),
                                **{key: row[key] for key in ("pair_id", "voice", "split", "tpr_split")},
                                "target_tpr_split": target_row["tpr_split"], "target_binding_held_out": held,
                                "layer": layer, "site": site, "condition": condition,
                                "exact_match": int(guess == gold), "correct_tokens": correct,
                                "target_tokens": len(gold), "activation_mse": mse[j],
                                "expected_sentence": target_row["sentence"],
                                "predicted_tokens": json.dumps([tokens_by_id[token] for token in guess]),
                            })
                            key = (row["tpr_split"], target_row["tpr_split"], row["voice"], layer, site, condition, held)
                            totals = groups[key]
                            totals["count"] += 1
                            totals["exact_match"] += int(guess == gold)
                            totals["correct_tokens"] += correct
                            totals["target_tokens"] += len(gold)
                            totals["activation_mse"] += mse[j]
                tpr.cpu()
                del decoder, z
                release_memory()
    group_columns = ["tpr_split", "target_tpr_split", "voice", "layer", "site", "condition", "target_binding_held_out"]
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_columns + ["count", "exact_match_rate", "token_accuracy", "activation_mse"])
        writer.writeheader()
        for key, totals in sorted(groups.items()):
            writer.writerow({**dict(zip(group_columns, key)), "count": totals["count"],
                             "exact_match_rate": totals["exact_match"] / totals["count"],
                             "token_accuracy": totals["correct_tokens"] / totals["target_tokens"],
                             "activation_mse": totals["activation_mse"] / totals["count"]})
    metadata = {"artifact_type": "tpr_reconstruction", "cache_fingerprint": cache_fingerprint(cache),
                "unpackers": decoder_metadata, "tprs": load_auxiliary_metadata(tprs_dir, cache, "tpr_models"),
                "alpha": alpha, "evaluation_rows": len(indices), "summary_rows": len(groups),
                "generation": "greedy, BOS only, EOS termination; no teacher forcing"}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
