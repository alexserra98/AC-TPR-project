"""Save references using the pre-migration ac_tpr sources and Transformers 4.57.6."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

from ac_tpr.activations import extract_batch, tokenize_roles
from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION, generate_rows
from ac_tpr.interventions import MEAN_TYPES, prepare_prompts, role_deltas, score_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("ac-tpr-cache"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    rows = generate_rows(["boy", "girl"], [{"lemma": "see", "past": "saw", "participle": "seen"}], [])
    torch.manual_seed(123)
    config = GPTNeoXConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
        hidden_dropout=0.0, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = GPTNeoXForCausalLM(config).eval()
    sentence_ids, positions = tokenize_roles(rows, tokenizer)
    sentence_inputs = tokenizer.pad({"input_ids": sentence_ids}, padding=True, return_tensors="pt")
    activations = extract_batch(model.gpt_neox, sentence_inputs, positions)
    ids, positions, answers = prepare_prompts(rows, tokenizer)
    inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    generator = torch.Generator().manual_seed(7)
    vectors = {
        "pooled": torch.randn((2, 2, 32), generator=generator),
        "by_voice": torch.randn((2, 2, 2, 32), generator=generator),
    }
    scores = {}
    for layer in range(2):
        for mean_type in MEAN_TYPES:
            for role_index, role in enumerate(("agent", "patient")):
                deltas = role_deltas(vectors, rows, mean_type, layer, role)
                scores[f"{layer}/{mean_type}/{role}"] = score_batch(
                    model, inputs, answers, layer, positions[:, role_index], deltas,
                )
    torch.save({
        "config": config.to_dict(), "state_dict": model.state_dict(), "rows": rows,
        "vectors": vectors, "activations": activations,
        "baseline": score_batch(model, inputs, answers), "scores": scores,
        "versions": {"torch": str(torch.__version__), "transformers": transformers.__version__},
    }, args.output_dir / "tiny.pt")

    name = "pythia-6.9b-step143000-generated"
    smoke = args.cache_dir / "interventions" / f"{name}-agent-smoke"
    target = args.output_dir / "pythia"
    target.mkdir()
    for filename in ("baseline.csv", "results.csv", "summary.csv", "metadata.json"):
        shutil.copyfile(smoke / filename, target / filename)
    with (smoke / "baseline.csv").open() as handle:
        selected = {(row["pair_id"], row["voice"]) for row in csv.DictReader(handle)}
    run = torch.load(args.cache_dir / "activations" / name / "activations.pt", mmap=True, weights_only=True)
    indices = [i for i, row in enumerate(run["rows"]) if (row["pair_id"], row["voice"]) in selected]
    torch.save({
        "rows": [run["rows"][i] for i in indices], "activations": run["activations"][indices],
        "role_positions": run["role_positions"][indices], "input_ids": [run["input_ids"][i] for i in indices],
    }, target / "activations.pt")
    shutil.copyfile(args.cache_dir / "activations" / name / "metadata.json", target / "activation_metadata.json")
    batch_size = json.loads((target / "activation_metadata.json").read_text())["batch_size"]
    batches = []
    for start in sorted({index // batch_size * batch_size for index in indices}):
        end = min(start + batch_size, len(run["rows"]))
        selected_indices = [i for i, index in enumerate(indices) if start <= index < end]
        batches.append({
            "input_ids": run["input_ids"][start:end],
            "role_positions": run["role_positions"][start:end].clone(),
            "reference_indices": selected_indices,
            "batch_indices": [indices[i] - start for i in selected_indices],
        })
    torch.save(batches, target / "extraction_batches.pt")
    vectors_dir = args.cache_dir / "syntactic_vectors" / f"{name}-train"
    shutil.copyfile(vectors_dir / "syntactic_vectors.pt", target / "syntactic_vectors.pt")
    shutil.copyfile(vectors_dir / "metadata.json", target / "vectors_metadata.json")
    print(json.dumps({"reference": str(args.output_dir), "pythia_rows": len(indices), "tiny_rows": len(rows)}))


if __name__ == "__main__":
    main()
