"""Run the fixed Qwen3 chat-prompt role intervention experiment."""

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ac_tpr.activations import ROLES
from ac_tpr.dataset import CORPUS_COLUMNS
from ac_tpr.interventions import (
    BASELINE_COLUMNS, MEAN_TYPES, RESULT_COLUMNS, role_deltas, summarize_results,
)
from ac_tpr.syntactic_vectors import VOICES

MODEL_ID = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
QUESTION = "Who is performing the action? Answer only with the noun identifying that person."
RUN_NAME = "qwen3-8b-generated-agent-chat-nnsight"


def prepare_chat_prompts(rows, tokenizer):
    """Locate single-token nouns within the sentence span of each full chat prompt."""
    prompts, input_ids, positions, answers = [], [], [], []
    for row in rows:
        sentence = row["sentence"]
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": sentence + "\n" + QUESTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        assert prompt.count(sentence) == 1
        sentence_start = prompt.index(sentence)
        encoding = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        tokens = encoding["input_ids"]
        role_positions, answer_ids = [], []
        assert row["agent"] != row["patient"]
        for role in ROLES:
            matches = list(re.finditer(rf"\b{re.escape(row[role])}\b", sentence))
            assert len(matches) == 1, (sentence, role)
            start = sentence_start + matches[0].start()
            end = sentence_start + matches[0].end()
            indices = [
                i for i, (a, b) in enumerate(encoding["offset_mapping"])
                if a < end and b > start
            ]
            assert len(indices) == 1, (sentence, role, indices)
            position = indices[0]
            a, b = encoding["offset_mapping"][position]
            assert prompt[a:b].strip() == row[role]
            assert tokenizer.decode([tokens[position]]).strip() == row[role]
            assert position < len(tokens) - 1
            answer = " " + row[role]
            candidate = tokenizer.encode(answer, add_special_tokens=False)
            assert len(candidate) == 1, answer
            assert tokenizer.encode(prompt + answer, add_special_tokens=False) == tokens + candidate
            role_positions.append(position)
            answer_ids.append(candidate[0])
        prompts.append(prompt)
        input_ids.append(tokens)
        positions.append(role_positions)
        answers.append(answer_ids)
    return prompts, input_ids, torch.tensor(positions), torch.tensor(answers)


def padded_inputs(tokenizer, ids, positions, device):
    """Right-pad prompts and confirm the role indices still select the same IDs."""
    assert tokenizer.padding_side == "right"
    inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    expected = torch.tensor([
        [tokens[p] for p in pair]
        for tokens, pair in zip(ids, positions.tolist(), strict=True)
    ])
    assert torch.equal(inputs["input_ids"].gather(1, positions), expected)
    assert inputs["attention_mask"].gather(1, positions).all()
    return inputs.to(device)


@torch.inference_mode()
def extract_batch(model, inputs, positions):
    positions = positions.to(inputs["input_ids"].device)
    batch = torch.arange(len(positions), device=positions.device)[:, None]
    with model.trace(inputs, use_cache=False) as tracer:
        states = []
        for layer in range(model.config.num_hidden_layers):
            states.append(model.model.layers[layer].output[batch, positions].cpu())
        result = torch.stack(states, dim=1).save()
        tracer.stop()
    assert torch.isfinite(result).all()
    return result


@torch.inference_mode()
def score_batch(model, inputs, answer_ids, layer=None, positions=None, deltas=None):
    """Read candidate logits at the final prompt position, optionally editing one role."""
    assert not model.training
    device = inputs["input_ids"].device
    batch = torch.arange(inputs["input_ids"].shape[0], device=device)
    last = inputs["attention_mask"].sum(dim=1) - 1
    answer_ids = answer_ids.to(device)
    if layer is not None:
        positions = positions.to(device)
        deltas = deltas.to(device=device, dtype=torch.float32)
        assert ((positions >= 0) & (positions < last)).all()
    with model.trace(inputs, use_cache=False):
        if layer is not None:
            hidden = model.model.layers[layer].output.clone()
            updated = (hidden[batch, positions].float() + deltas).to(hidden.dtype)
            assert torch.isfinite(updated).all()
            hidden[batch, positions] = updated
            model.model.layers[layer].output = hidden
        model.lm_head.input = model.lm_head.input[batch, last][:, None, :]
        scores = model.output.logits[:, 0].gather(1, answer_ids).float().cpu().save()
    assert torch.isfinite(scores).all()
    return scores


def write_metadata(directory, metadata):
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def extract_training(model, tokenizer, rows, ids, positions, directory, metadata, batch_size):
    """Save training role states and their pooled and per-voice float32 means."""
    assert all(row["split"] == "train" for row in rows)
    activation_dir, vector_dir = directory / "activations", directory / "vectors"
    activation_dir.mkdir()
    vector_dir.mkdir()
    shape = (len(rows), model.config.num_hidden_layers, 2, model.config.hidden_size)
    activations = torch.empty(shape, dtype=torch.bfloat16)
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        inputs = padded_inputs(tokenizer, ids[start:end], positions[start:end], model.device)
        activations[start:end] = extract_batch(model, inputs, positions[start:end])
        print(f"Extracted training rows {end}/{len(rows)}", flush=True)
    torch.save({"activations": activations, "rows": rows, "input_ids": ids,
                "role_positions": positions}, activation_dir / "activations.pt")
    write_metadata(activation_dir, {
        **metadata, "axes": ["sentence", "layer", "role", "hidden"], "shape": list(shape),
        "sentences_by_split": dict(Counter(row["split"] for row in rows)),
        "sentences_by_voice": dict(Counter(row["voice"] for row in rows)),
    })
    pooled = torch.empty(shape[1:], dtype=torch.float32)
    by_voice = torch.empty((2, *shape[1:]), dtype=torch.float32)
    masks = [torch.tensor([row["voice"] == voice for row in rows]) for voice in VOICES]
    assert all(mask.any() for mask in masks)
    for layer in range(shape[1]):
        states = activations[:, layer].float()
        pooled[layer] = states.mean(dim=0)
        for voice_index, mask in enumerate(masks):
            by_voice[voice_index, layer] = states[mask].mean(dim=0)
    vectors = {"pooled": pooled, "by_voice": by_voice}
    assert all(torch.isfinite(value).all() for value in vectors.values())
    torch.save(vectors, vector_dir / "syntactic_vectors.pt")
    write_metadata(vector_dir, {
        **metadata, "activations_dir": str(activation_dir.resolve()), "split": "train",
        "sentences": len(rows), "sentences_by_voice": dict(Counter(row["voice"] for row in rows)),
        "dtype": "float32", "aggregation": "Arithmetic mean over training sentences",
        "axes": {"pooled": ["layer", "role", "hidden"], "by_voice": ["voice", "layer", "role", "hidden"]},
        "shape": {key: list(value.shape) for key, value in vectors.items()},
    })
    return vectors


def evaluate(model, tokenizer, rows, ids, positions, answers, vectors, output_dir, metadata, batch_size, smoke):
    """Stream baseline and intervention CSVs using the existing experiment schema."""
    output_dir.mkdir()
    started = time.monotonic()
    layers, hidden = model.config.num_hidden_layers, model.config.hidden_size
    result_count = 0
    with (output_dir / "baseline.csv").open("w", newline="") as base_handle, \
         (output_dir / "results.csv").open("w", newline="") as result_handle:
        baseline_writer = csv.DictWriter(base_handle, fieldnames=BASELINE_COLUMNS)
        result_writer = csv.DictWriter(result_handle, fieldnames=RESULT_COLUMNS)
        baseline_writer.writeheader()
        result_writer.writeheader()
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch_rows = rows[start:end]
            inputs = padded_inputs(tokenizer, ids[start:end], positions[start:end], model.device)
            baseline = score_batch(model, inputs, answers[start:end])
            if start == 0:
                for role in range(2):
                    zero = score_batch(model, inputs, answers[start:end], 0,
                                       positions[start:end, role], torch.zeros(end - start, hidden))
                    assert torch.equal(zero, baseline), "Zero-vector control changed logits"
            records = []
            for index, row in enumerate(batch_rows):
                agent, patient = baseline[index].tolist()
                record = {
                    **row, "agent_token_id": answers[start + index, 0].item(),
                    "patient_token_id": answers[start + index, 1].item(),
                    "agent_position": positions[start + index, 0].item(),
                    "patient_position": positions[start + index, 1].item(),
                    "baseline_agent_logit": agent, "baseline_patient_logit": patient,
                    "baseline_logit_diff": patient - agent,
                }
                baseline_writer.writerow(record)
                records.append(record)
            for layer in range(layers):
                for mean_type in MEAN_TYPES:
                    for role, edited_role in enumerate(ROLES):
                        deltas = role_deltas(vectors, batch_rows, mean_type, layer, edited_role)
                        scores = score_batch(model, inputs, answers[start:end], layer,
                                             positions[start:end, role], deltas)
                        if layer == layers - 1:
                            assert torch.equal(scores, baseline), "Final-block control changed logits"
                        for record, (agent, patient) in zip(records, scores.tolist(), strict=True):
                            difference = patient - agent
                            result_writer.writerow({
                                **record, "layer": layer, "mean_type": mean_type, "edited_role": edited_role,
                                "intervened_agent_logit": agent, "intervened_patient_logit": patient,
                                "intervened_logit_diff": difference,
                                "logit_diff_change": difference - record["baseline_logit_diff"],
                            })
                            result_count += 1
            base_handle.flush()
            result_handle.flush()
            print(f"{output_dir.name}: evaluated {end}/{len(rows)} rows ({time.monotonic() - started:.1f}s)", flush=True)
    summary_count = summarize_results(output_dir / "results.csv", output_dir / "summary.csv")
    assert result_count == len(rows) * layers * 4
    assert summary_count == 4 * layers * 4
    write_metadata(output_dir, {
        **metadata, "elapsed_seconds": time.monotonic() - started,
        "execution_model": metadata["model"], "scoring_hook": "lm_head.input",
        "splits": ["test", "gen_test"], "limit": len(rows) if smoke else None,
        "selection": "Two examples per split and voice" if smoke else "All test and gen_test rows",
        "sentences_by_split": dict(Counter(row["split"] for row in rows)),
        "sentences_by_voice": dict(Counter(row["voice"] for row in rows)),
        "baseline_rows": len(rows), "result_rows": result_count, "summary_rows": summary_count,
        "logit_diff": "patient_logit - agent_logit",
        "logit_diff_change": "intervened_logit_diff - baseline_logit_diff",
        "preference_rates": "Strict comparisons; denominator is all rows; ties favor neither answer",
        "flip_rate": "baseline_logit_diff < 0 and intervened_logit_diff > 0; denominator is all rows",
        "intervention": "h + (mean_other_role - mean_edited_role), float32 update cast to residual dtype",
        "strength": 1.0, "mean_types": MEAN_TYPES, "edited_roles": ROLES,
        "original_answer": "agent", "counterfactual_answer": "patient",
        "controls": {"zero_vector_first_batch": "passed", "final_block_all_rows": "passed"},
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/generated/corpus.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("ac-tpr-cache") / RUN_NAME)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--check-indices", action="store_true", help="Validate the entire corpus without loading weights")
    args = parser.parse_args()
    assert args.batch_size > 0
    with args.corpus.open(newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CORPUS_COLUMNS
        rows = list(reader)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prompts, ids, positions, answers = prepare_chat_prompts(rows, tokenizer)
    for start in range(0, len(rows), args.batch_size):
        padded_inputs(tokenizer, ids[start:start + args.batch_size], positions[start:start + args.batch_size], "cpu")
    print(f"Validated {len(rows)} chat prompts, {2 * len(rows)} role indices, answer boundaries, and right padding.", flush=True)
    if args.check_indices:
        return
    assert not args.output_dir.exists(), f"Output directory already exists: {args.output_dir}"
    assert torch.cuda.is_bf16_supported(), "This experiment requires a BF16-capable GPU"
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=REVISION, dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True,
    ).eval().to("cuda").requires_grad_(False)
    assert hf_model.config._commit_hash == REVISION
    model = LanguageModel(hf_model, tokenizer=tokenizer, config_model=hf_model.config)
    assert model.config.num_hidden_layers == 36
    assert max(map(len, ids)) <= model.config.max_position_embeddings
    torch.cuda.reset_peak_memory_stats()
    args.output_dir.mkdir(parents=True)
    metadata = {
        "schema_version": 4, "created_at": datetime.now(timezone.utc).isoformat(),
        "corpus": str(args.corpus.resolve()), "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "model": {"name": MODEL_ID, "revision": REVISION, "commit": REVISION, "dtype": "bfloat16",
                  "attention_implementation": "eager", "backend": "nnsight", "api": "LanguageModel",
                  "weight_processing": "none", "nnsight_version": version("nnsight"),
                  "transformers_version": version("transformers")},
        "tokenizer": {"name": MODEL_ID, "revision": REVISION, "padding_side": "right",
                      "pad_token_id": tokenizer.pad_token_id, "add_special_tokens": False,
                      "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()},
        "question": QUESTION, "prompt_template": "{sentence}\n" + QUESTION,
        "chat_template": tokenizer.chat_template, "enable_thinking": False, "add_generation_prompt": True,
        "example_rendered_prompt": prompts[0], "answer_prefix": " ",
        "role_index_validation": {"sentences": len(rows), "roles": 2 * len(rows), "right_padding": "passed"},
        "activation_site": "Transformer block output, before final RMSNorm",
        "hook_names": [f"model.layers.{layer}.output" for layer in range(36)],
        "layer_indices": list(range(36)), "roles": ROLES, "voices": VOICES,
        "batch_size": args.batch_size, "device": "cuda", "gpu": torch.cuda.get_device_name(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "versions": {name: version(name) for name in ("torch", "transformers", "nnsight")},
    }
    train = [i for i, row in enumerate(rows) if row["split"] == "train"]
    vectors = extract_training(model, tokenizer, [rows[i] for i in train], [ids[i] for i in train],
                               positions[train], args.output_dir, metadata, args.batch_size)
    vector_dir = args.output_dir / "vectors"
    metadata.update(vectors_dir=str(vector_dir.resolve()),
                    vectors_sha256=hashlib.sha256((vector_dir / "syntactic_vectors.pt").read_bytes()).hexdigest(),
                    vectors_metadata_sha256=hashlib.sha256((vector_dir / "metadata.json").read_bytes()).hexdigest())
    smoke = []
    for split in ("test", "gen_test"):
        for voice in VOICES:
            selected = [i for i, row in enumerate(rows) if row["split"] == split and row["voice"] == voice][:2]
            assert len(selected) == 2
            smoke.extend(selected)
    evaluation = [i for i, row in enumerate(rows) if row["split"] in {"test", "gen_test"}]
    for name, indices in (("smoke", smoke), ("interventions", evaluation)):
        evaluate(model, tokenizer, [rows[i] for i in indices], [ids[i] for i in indices], positions[indices],
                 answers[indices], vectors, args.output_dir / name, metadata, args.batch_size, name == "smoke")


if __name__ == "__main__":
    main()
