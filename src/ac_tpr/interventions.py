"""Evaluate separate agent/patient role translations at Pythia block outputs."""

import csv
import hashlib
import json
import os
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
import transformers
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from ac_tpr.activations import ROLES, tokenize_roles
from ac_tpr.dataset import CORPUS_COLUMNS, TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.model import BLOCK_HOOK, SCHEMA_VERSION, load_model, validate_activation_provenance
from ac_tpr.syntactic_vectors import VOICES

PROMPT_TEMPLATE = "{sentence}\nQuestion: Who is performing the action?\nAnswer:"
MEAN_TYPES = ["pooled", "by_voice"]
BASELINE_COLUMNS = CORPUS_COLUMNS + [
    "agent_token_id", "patient_token_id", "agent_position", "patient_position",
    "baseline_agent_logit", "baseline_patient_logit", "baseline_logit_diff",
]
RESULT_COLUMNS = BASELINE_COLUMNS + [
    "layer", "mean_type", "edited_role", "intervened_agent_logit",
    "intervened_patient_logit", "intervened_logit_diff", "logit_diff_change",
]
GROUP_COLUMNS = ["split", "voice", "layer", "mean_type", "edited_role"]


def prepare_prompts(
    rows: list[dict[str, str]], tokenizer: PreTrainedTokenizerFast
) -> tuple[list[list[int]], torch.Tensor, torch.Tensor]:
    """Return prompt IDs, sentence role positions, and single-token answer IDs.

    Roles and candidate answers are ordered [agent, patient]. Candidate answers
    include one leading space. Their tokenization must append exactly one token
    to the prompt, and the sentence must remain an unchanged token prefix.
    """
    sentence_ids, positions = tokenize_roles(rows, tokenizer)
    prompts = [PROMPT_TEMPLATE.format(sentence=row["sentence"]) for row in rows]
    prompt_ids = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    answers = []
    for row, prompt, tokens, prefix in zip(rows, prompts, prompt_ids, sentence_ids, strict=True):
        if tokens[:len(prefix)] != prefix:
            raise ValueError("Prompt must preserve sentence tokenization")
        answer_ids = []
        for role in ROLES:
            answer = " " + row[role]
            ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"The {role} answer must be one token: {answer!r}")
            if tokenizer(prompt + answer, add_special_tokens=False)["input_ids"] != tokens + ids:
                raise ValueError("Answer must append exactly one token to the prompt")
            answer_ids.append(ids[0])
        answers.append(answer_ids)
    return prompt_ids, positions, torch.tensor(answers, dtype=torch.long)


def role_deltas(
    vectors: dict[str, torch.Tensor], rows: list[dict[str, str]],
    mean_type: str, layer: int, edited_role: str,
) -> torch.Tensor:
    """Select float32 destination-minus-source role means for each sentence."""
    role = ROLES.index(edited_role)
    if mean_type == "pooled":
        means = vectors["pooled"][layer].expand(len(rows), -1, -1)
    elif mean_type == "by_voice":
        voices = [VOICES.index(row["voice"]) for row in rows]
        means = vectors["by_voice"][voices, layer]
    else:
        raise ValueError(f"Unknown mean type: {mean_type}")
    return means[:, 1 - role].float() - means[:, role].float()


@torch.inference_mode()
def score_batch(
    model: HookedTransformer, inputs: dict[str, torch.Tensor], answer_ids: torch.Tensor,
    layer: int | None = None, positions: torch.Tensor | None = None,
    deltas: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return CPU float32 [agent, patient] logits at each last non-padding token.

    Inputs use right padding. An optional hook translates one sentence token per
    row after the selected block; other tokens and block outputs are preserved.
    Updates use float32 arithmetic and are cast back to the residual dtype.
    """
    if model.training:
        raise ValueError("Scoring requires model.eval()")
    device = inputs["input_ids"].device
    batch = torch.arange(inputs["input_ids"].shape[0], device=device)
    last = inputs["attention_mask"].sum(dim=1) - 1
    # answer_ids: [batch, 2] -> [batch, 2] logits for agent and patient
    answer_ids = answer_ids.to(device)
    if answer_ids.shape != (len(batch), len(ROLES)):
        raise ValueError("Answer IDs must have shape [batch, 2]")
    def select_answer_position(hidden, hook):
        return hidden[batch, last][:, None, :]

    hooks = [("unembed.hook_in", select_answer_position)]
    if layer is None:
        if positions is not None or deltas is not None:
            raise ValueError("An edit requires a layer, positions, and deltas together")
    else:
        if not 0 <= layer < model.cfg.n_layers:
            raise ValueError("Intervention layer is outside the model")
        if positions is None or deltas is None:
            raise ValueError("An edit requires a layer, positions, and deltas together")
        positions, deltas = positions.to(device), deltas.to(device=device, dtype=torch.float32)
        if positions.shape != (len(batch),) or deltas.shape != (len(batch), model.cfg.d_model):
            raise ValueError("Edit positions or deltas have incompatible shapes")
        if ((positions < 0) | (positions >= last)).any():
            raise ValueError("Edited tokens must precede the final prompt token")

        def translate(hidden, hook):
            hidden = hidden.clone()
            updated = (hidden[batch, positions].float() + deltas).to(hidden.dtype)
            if not torch.isfinite(updated).all():
                raise ValueError("Nonfinite edited activations")
            hidden[batch, positions] = updated
            return hidden

        hooks.append((BLOCK_HOOK.format(layer=layer), translate))
    logits = model.run_with_hooks(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        prepend_bos=False, past_kv_cache=None, fwd_hooks=hooks,
    )
    scores = logits[:, 0].gather(1, answer_ids).float().cpu()
    if not torch.isfinite(scores).all():
        raise ValueError("Nonfinite answer logits")
    return scores


def summarize_results(results_path: Path, output_path: Path) -> int:
    """Write means and strict pairwise preference rates for each condition.

    Rates use all sentences in the group as their denominator. A tie contributes
    to neither answer's preference rate. Flip rate counts baseline agent wins
    that become patient wins, again divided by all sentences in the group.
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
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GROUP_COLUMNS + list(next(iter(groups.values()))))
        writer.writeheader()
        for key, totals in sorted(groups.items()):
            writer.writerow({
                **dict(zip(GROUP_COLUMNS, key)),
                **{name: value if name == "count" else value / totals["count"]
                   for name, value in totals.items()},
            })
    return len(groups)


def run_interventions(
    corpus_path: Path, vectors_dir: Path, output_dir: Path,
    splits: tuple[str, ...] = ("test", "gen_test"), batch_size: int = 32,
    device: str = "cuda", dtype: str = "float32", limit: int | None = None,
) -> dict:
    """Run the fixed agent question with both mean types and separate token edits.

    All layers and selected rows are evaluated. Baseline and intervention CSVs
    are streamed by batch. Zero updates on the first batch and final-block edits
    on every batch must reproduce baseline logits exactly. metadata.json is
    written only after scoring and summary output succeed.

    IMPORTANT: The function expect that in all prompts the AGENT is the correct answer
    """
    started = time.monotonic()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("batch_size and limit must be positive")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Select distinct, nonempty splits")
    source = json.loads((vectors_dir / "metadata.json").read_text(encoding="utf-8"))
    validate_activation_provenance(source)
    corpus_hash = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    if source["split"] != "train" or source["corpus_sha256"] != corpus_hash:
        raise ValueError("Vectors must use training rows from this exact corpus")
    if source["roles"] != ROLES or source["voices"] != VOICES:
        raise ValueError("Vector role or voice ordering is incompatible")
    if source["activation_site"] != "Transformer block output, before final_layer_norm":
        raise ValueError("Vectors must describe block outputs before final_layer_norm")
    for component in ("model", "tokenizer"):
        if source[component]["name"] != TOKENIZER_ID or source[component]["revision"] != TOKENIZER_REVISION:
            raise ValueError("Vectors must match the dataset checkpoint and tokenizer")
    if not source["model"]["commit"]:
        raise ValueError("Vector provenance must identify the resolved model commit")
    if source["axes"] != {
        "pooled": ["layer", "role", "hidden"],
        "by_voice": ["voice", "layer", "role", "hidden"],
    }:
        raise ValueError("Vector axes are incompatible")
    vectors_path = vectors_dir / "syntactic_vectors.pt"
    # vectors: pooled [layer, role, hidden], by_voice [voice, layer, role, hidden]
    vectors = torch.load(vectors_path, map_location="cpu", weights_only=True)
    for name in MEAN_TYPES:
        if list(vectors[name].shape) != source["shape"][name]:
            raise ValueError("Vector shapes do not match their metadata")
        if vectors[name].dtype != torch.float32 or not torch.isfinite(vectors[name]).all():
            raise ValueError("Vectors must contain finite float32 means")
    with corpus_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CORPUS_COLUMNS:
            raise ValueError(f"Corpus CSV columns must be {CORPUS_COLUMNS}")
        corpus = list(reader)
    if not corpus or any(
        None in row or any(value is None or value == "" for value in row.values())
        for row in corpus
    ):
        raise ValueError("Corpus must contain complete nonempty rows")
    if len({(row["pair_id"], row["voice"]) for row in corpus}) != len(corpus):
        raise ValueError("Corpus (pair_id, voice) keys must be unique")
    train_counts = dict(Counter(row["voice"] for row in corpus if row["split"] == "train"))
    if train_counts != source["sentences_by_voice"] or sum(train_counts.values()) != source["sentences"]:
        raise ValueError("Vector training counts do not match the corpus")
    if not set(splits) <= {row["split"] for row in corpus}:
        raise ValueError("A requested split is absent from the corpus")
    rows = [row for row in corpus if row["split"] in splits][:limit]
    if not rows or any(row["voice"] not in VOICES for row in rows):
        raise ValueError("Evaluation requires rows with active/passive voices")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if tokenizer_hash != source["tokenizer"]["backend_sha256"]:
        raise ValueError("Tokenizer backend does not match vector provenance")
    if (
        source["tokenizer"]["add_special_tokens"]
        or source["tokenizer"]["padding_side"] != "right"
        or source["tokenizer"]["pad_token_id"] != tokenizer.pad_token_id
    ):
        raise ValueError("Source tokenization settings are incompatible")
    input_ids, positions, answer_ids = prepare_prompts(rows, tokenizer)
    print(f"Loading {TOKENIZER_ID} at {TOKENIZER_REVISION} ({dtype}, {device})", flush=True)
    model, execution_model = load_model(tokenizer, source["model"]["commit"], device, dtype)
    layers, hidden = model.cfg.n_layers, model.cfg.d_model
    if execution_model["commit"] != source["model"]["commit"]:
        raise ValueError("Loaded model commit does not match vector provenance")
    if (
        source["layer_indices"] != list(range(layers))
        or vectors["pooled"].shape != (layers, 2, hidden)
        or vectors["by_voice"].shape != (2, layers, 2, hidden)
    ):
        raise ValueError("Vector layer or hidden dimensions do not match the model")
    if max(map(len, input_ids)) > model.cfg.n_ctx:
        raise ValueError("A prompt exceeds the model context length")

    output_dir.mkdir(parents=True, exist_ok=False)
    result_count = 0
    with (
        (output_dir / "baseline.csv").open("w", newline="", encoding="utf-8") as base_handle,
        (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as result_handle,
    ):
        # Just defining the structure of the output CSVs
        baseline_writer = csv.DictWriter(base_handle, fieldnames=BASELINE_COLUMNS)
        result_writer = csv.DictWriter(result_handle, fieldnames=RESULT_COLUMNS)
        baseline_writer.writeheader()
        result_writer.writeheader()
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch_rows = rows[start:end]
            inputs = tokenizer.pad(
                {"input_ids": input_ids[start:end]}, padding=True, return_tensors="pt"
            ).to(device)
            baseline = score_batch(model, inputs, answer_ids[start:end])
            if start == 0:
                for role in range(len(ROLES)):
                    zero = score_batch(
                        model, inputs, answer_ids[start:end], 0, positions[start:end, role],
                        torch.zeros((end - start, hidden)),
                    )
                    if not torch.equal(zero, baseline):
                        raise ValueError("Zero-vector control changed baseline logits")
            records = []
            for index, row in enumerate(batch_rows):
                agent, patient = baseline[index].tolist()
                record = {
                    **row,
                    "agent_token_id": answer_ids[start + index, 0].item(),
                    "patient_token_id": answer_ids[start + index, 1].item(),
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
                        scores = score_batch(
                            model, inputs, answer_ids[start:end], layer,
                            positions[start:end, role], deltas,
                        )
                        if layer == layers - 1 and not torch.equal(scores, baseline):
                            raise ValueError("Final-block control changed answer logits")
                        for record, (agent, patient) in zip(records, scores.tolist(), strict=True):
                            # The function expect that in all prompts the AGENT is the correct answer
                            difference = patient - agent
                            result_writer.writerow({
                                **record, "layer": layer, "mean_type": mean_type,
                                "edited_role": edited_role, "intervened_agent_logit": agent,
                                "intervened_patient_logit": patient,
                                "intervened_logit_diff": difference,
                                "logit_diff_change": difference - record["baseline_logit_diff"],
                            })
                            result_count += 1
            base_handle.flush()
            result_handle.flush()
            print(
                f"Evaluated {end}/{len(rows)} sentences across all {layers} layers "
                f"({time.monotonic() - started:.1f}s)", flush=True,
            )
    summary_count = summarize_results(output_dir / "results.csv", output_dir / "summary.csv")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "corpus": str(corpus_path.resolve()), "corpus_sha256": corpus_hash,
        "vectors_dir": str(vectors_dir.resolve()),
        "vectors_sha256": hashlib.sha256(vectors_path.read_bytes()).hexdigest(),
        "vectors_metadata_sha256": hashlib.sha256((vectors_dir / "metadata.json").read_bytes()).hexdigest(),
        "model": source["model"], "execution_model": execution_model,
        "tokenizer": source["tokenizer"], "activation_site": source["activation_site"],
        "hook_names": source["hook_names"], "scoring_hook": "unembed.hook_in",
        "prompt_template": PROMPT_TEMPLATE, "answer_prefix": " ",
        "original_answer": "agent", "counterfactual_answer": "patient",
        "logit_diff": "patient_logit - agent_logit",
        "logit_diff_change": "intervened_logit_diff - baseline_logit_diff",
        "preference_rates": (
            "Strict logit comparisons; denominator is all sentences in the group; "
            "ties favor neither answer"
        ),
        "flip_rate": (
            "baseline_logit_diff < 0 and intervened_logit_diff > 0; "
            "denominator is all sentences in the group"
        ),
        "intervention": "h + (mean_other_role - mean_edited_role), float32 update cast to residual dtype",
        "strength": 1.0, "mean_types": MEAN_TYPES, "edited_roles": ROLES,
        "voices": VOICES, "layer_indices": list(range(layers)),
        "splits": list(splits), "limit": limit,
        "sentences_by_split": dict(Counter(row["split"] for row in rows)),
        "sentences_by_voice": dict(Counter(row["voice"] for row in rows)),
        "baseline_rows": len(rows), "result_rows": result_count, "summary_rows": summary_count,
        "controls": {"zero_vector_first_batch": "passed", "final_block_all_rows": "passed"},
        "batch_size": batch_size, "device": str(model.cfg.device),
        "gpu": torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__,
                     "transformer_lens": version("transformer-lens"), "cuda": torch.version.cuda},
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
