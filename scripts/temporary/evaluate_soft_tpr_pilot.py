"""Select steering conditions on validation, then evaluate them on held-out rows."""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time

import torch
from transformers import AutoTokenizer

from ac_tpr.activations import ROLES
from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.interventions import prepare_prompts, role_deltas, score_batch, PROMPT_TEMPLATE
from ac_tpr.model import load_model, validate_activation_provenance
from ac_tpr.syntactic_vectors import VOICES


def matched_delta(soft, raw):
    """Match each soft direction's norm to its raw counterpart without rotation."""
    norm = soft.norm(dim=-1, keepdim=True)
    if (norm == 0).any():
        raise ValueError("Cannot norm-match a collapsed Soft TPR direction")
    return soft * raw.norm(dim=-1, keepdim=True) / norm


def evaluate(model, tokenizer, rows, vectors, conditions, output, phase, batch_size):
    """Score all methods within the same model and batch; stream every example."""
    baseline_records, results = [], []
    started = time.monotonic()
    with (output / f"{phase}_results.csv").open("w", newline="") as handle:
        writer = None
        for split, voice in itertools.product(sorted({r["split"] for r in rows}), VOICES):
            selected = [r for r in rows if r["split"] == split and r["voice"] == voice]
            for start in range(0, len(selected), batch_size):
                batch_rows = selected[start:start + batch_size]
                ids, positions, answers = prepare_prompts(batch_rows, tokenizer)
                inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to(model.cfg.device)
                baseline = score_batch(model, inputs, answers)
                for row, scores in zip(batch_rows, baseline.tolist(), strict=True):
                    baseline_records.append({**row, "agent_logit": scores[0], "patient_logit": scores[1],
                                             "baseline_logit_diff": scores[1] - scores[0]})
                if start == 0:
                    for role in range(2):
                        zero = score_batch(model, inputs, answers, 0, positions[:, role], torch.zeros(len(batch_rows), model.cfg.d_model))
                        if not torch.equal(zero, baseline):
                            raise ValueError("Zero-vector control failed")
                        for method in ("raw", "soft"):
                            delta = role_deltas(vectors[method], batch_rows, "pooled", model.cfg.n_layers - 1, ROLES[role])
                            final = score_batch(model, inputs, answers, model.cfg.n_layers - 1, positions[:, role], delta)
                            if not torch.equal(final, baseline):
                                raise ValueError("Final-block control failed")
                for layer, mean_type, edited_role in conditions[voice]:
                    raw = role_deltas(vectors["raw"], batch_rows, mean_type, layer, edited_role)
                    soft = role_deltas(vectors["soft"], batch_rows, mean_type, layer, edited_role)
                    for method, delta in (("raw", raw), ("soft", soft), ("soft_matched", matched_delta(soft, raw))):
                        scores = score_batch(model, inputs, answers, layer, positions[:, ROLES.index(edited_role)], delta)
                        for row, before, after in zip(batch_rows, baseline.tolist(), scores.tolist(), strict=True):
                            record = {
                                **row, "layer": layer, "mean_type": mean_type, "edited_role": edited_role,
                                "method": method, "baseline_logit_diff": before[1] - before[0],
                                "logit_diff": after[1] - after[0],
                                "logit_diff_change": (after[1] - after[0]) - (before[1] - before[0]),
                            }
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=list(record))
                                writer.writeheader()
                            writer.writerow(record)
                            results.append(record)
                handle.flush()
                print(f"{phase} {split}/{voice} {min(start + batch_size, len(selected))}/{len(selected)} ({time.monotonic()-started:.1f}s)", flush=True)
    with (output / f"{phase}_baseline.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_records[0]))
        writer.writeheader()
        writer.writerows(baseline_records)
    return results


def select_conditions(records):
    """Independently select each method's largest validation mean logit change."""
    if not records or any(record["split"] != "val" for record in records):
        raise ValueError("Condition selection requires validation records only")
    totals = {}
    for record in records:
        key = tuple(record[k] for k in ("voice", "method", "layer", "mean_type", "edited_role"))
        total, count = totals.get(key, (0.0, 0))
        totals[key] = total + record["logit_diff_change"], count + 1
    chosen = {}
    for voice, method in itertools.product(VOICES, ("raw", "soft", "soft_matched")):
        candidates = [(key, total / count) for key, (total, count) in totals.items() if key[:2] == (voice, method)]
        key, score = max(candidates, key=lambda item: item[1])
        chosen[f"{voice}/{method}"] = dict(layer=key[2], mean_type=key[3], edited_role=key[4], validation_mean_change=score)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24, 25, 28])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batch_size < 1 or len(set(args.layers)) != len(args.layers):
        raise ValueError("Use positive batch_size and distinct layers")
    sources, vectors = {}, {}
    for method, name in (("raw", "role_means"), ("soft", "soft_tpr_vectors")):
        path = args.experiment_dir / name
        sources[method] = json.loads((path / "metadata.json").read_text())
        validate_activation_provenance(sources[method])
        if sources[method]["corpus_sha256"] != hashlib.sha256(args.corpus.read_bytes()).hexdigest():
            raise ValueError("Corpus differs from vector training corpus")
        vectors[method] = torch.load(path / "syntactic_vectors.pt", weights_only=True, map_location="cpu")
    for key in ("model", "tokenizer", "layer_indices", "roles", "voices", "shape", "sentences", "sentences_by_voice"):
        if sources["raw"][key] != sources["soft"][key]:
            raise ValueError(f"Vector provenance mismatch: {key}")
    if not set(args.layers) <= set(sources["raw"]["layer_indices"]):
        raise ValueError("Invalid candidate layers")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
    tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "right"
    if hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() != sources["raw"]["tokenizer"]["backend_sha256"]:
        raise ValueError("Tokenizer differs from vector provenance")
    with args.corpus.open() as handle:
        rows = list(csv.DictReader(handle))
    model, execution = load_model(tokenizer, sources["raw"]["model"]["commit"], args.device, "float32")
    if execution["commit"] != sources["raw"]["model"]["commit"]:
        raise ValueError("Checkpoint differs from vector provenance")
    args.output_dir.mkdir(parents=True)
    plan = dict(candidate_layers=args.layers, selection="Largest validation mean patient-minus-agent logit change, independently per voice/method",
                heldout_splits=["test", "gen_test"], batch_size=args.batch_size, execution_model=execution,
                corpus_sha256=sources["raw"]["corpus_sha256"], prompt=PROMPT_TEMPLATE,
                vector_sources=sources, strength=1.0,
                runtime={"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda},
                controls="Zero edits at block 0 and raw/soft edits at final block, both roles, first batch of each split/voice",
                gpu_max_allocated_gib=None)
    (args.output_dir / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    candidates = list(itertools.product(args.layers, ("pooled", "by_voice"), ROLES))
    validation = evaluate(model, tokenizer, [r for r in rows if r["split"] == "val"], vectors,
                          {voice: candidates for voice in VOICES}, args.output_dir, "validation", args.batch_size)
    chosen = select_conditions(validation)
    (args.output_dir / "selection.json").write_text(json.dumps(chosen, indent=2) + "\n")
    selected = {voice: sorted({(c["layer"], c["mean_type"], c["edited_role"])
                for key, c in chosen.items() if key.startswith(voice + "/")}) for voice in VOICES}
    evaluate(model, tokenizer, [r for r in rows if r["split"] in ("test", "gen_test")], vectors,
             selected, args.output_dir, "heldout", args.batch_size)
    plan["controls_passed"] = True
    plan["gpu_max_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None
    (args.output_dir / "metadata.json").write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
