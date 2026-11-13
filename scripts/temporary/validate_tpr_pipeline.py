"""Smoke-test all TPR stages with the real Qwen checkpoint on one GPU."""

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ac_tpr.dataset import CORPUS_COLUMNS, TOKENIZER_ID, TOKENIZER_REVISION, generate_rows
from ac_tpr.prompts import prepare_prompts
from ac_tpr.tpr.extraction import load_cache

LAYERS = [5, 12, 15, 24, 31, 35]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New smoke-run directory.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke run requires an allocated CUDA GPU")

    rows = generate_rows(
        ["boy", "girl", "man", "woman"],
        [{"lemma": "help", "past": "helped", "participle": "helped"}],
        [],
    )
    for row in rows:
        nouns = {row["agent"], row["patient"]}
        row["split"] = "test" if nouns == {"boy", "woman"} else "val" if nouns == {"girl", "man"} else "train"
    output.mkdir(parents=True)
    corpus_path = output / "corpus.csv"
    with corpus_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    run = output / "pipeline"
    command = [
        sys.executable, "-u", str(Path(__file__).with_name("run_tpr_pipeline.py")),
        "--corpus", str(corpus_path), "--output-dir", str(run),
        "--layers", *map(str, LAYERS), "--held-out-bindings", "boy:agent",
        "--device", "cuda", "--dtype", "bfloat16", "--batch-size", "2",
        "--train-batch-size", "8", "--epochs", "1", "--eval-every", "100",
        "--decoder-width", "16", "--decoder-layers", "1", "--decoder-heads", "2",
        "--decoder-ffn", "32", "--dropout", "0", "--max-length", "16",
        "--filler-dim", "4", "--role-dim", "3",
    ]
    print(f"Running the five TPR stages; progress log: {output / 'pipeline.log'}", flush=True)
    with (output / "pipeline.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)

    cache = load_cache(run / "encodings")
    split_counts = dict(Counter(row["tpr_split"] for row in cache["rows"]))
    assert split_counts == {"train": 12, "val": 4, "test": 2, "binding_test": 6}
    assert cache["metadata"]["layer_indices"] == LAYERS
    assert cache["activations"].shape == (24, len(LAYERS), 2, 4096)
    assert torch.isfinite(cache["activations"]).all()
    metadata = json.loads((run / "steering/metadata.json").read_text())
    assert metadata["baseline_rows"] == 8
    assert metadata["result_rows"] == 8 * len(LAYERS) * 2 * 4
    assert metadata["controls"] == {
        "zero_vector": "passed", "final_block_sentence_end": "passed",
        "final_block_prompt_end_donor": "passed",
    }

    print("Pipeline finished; comparing baseline logits with an independent Transformers model", flush=True)
    with (run / "steering/baseline.csv").open(newline="") as handle:
        baseline = list(csv.DictReader(handle))
    eval_rows = [cache["rows"][int(row["source_index"])] for row in baseline]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    ids, _, answers = prepare_prompts(eval_rows, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda").eval().requires_grad_(False)
    expected = []
    with torch.inference_mode():
        for tokens, candidates in zip(ids, answers, strict=True):
            inputs = torch.tensor([tokens], device="cuda")
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                           use_cache=False, logits_to_keep=1).logits[0, -1]
            expected.append(logits[candidates.to("cuda")].float().cpu())
    expected = torch.stack(expected)
    actual = torch.tensor([[float(row["baseline_agent_logit"]), float(row["baseline_patient_logit"])]
                           for row in baseline])
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.125)

    numeric_fields = {
        "baseline_agent_logit", "baseline_patient_logit", "baseline_logit_diff",
        "donor_baseline_agent_logit", "donor_baseline_patient_logit", "donor_baseline_logit_diff",
        "intervened_agent_logit", "intervened_patient_logit", "intervened_logit_diff", "logit_diff_change",
        "activation_mse", "correct_tokens", "target_tokens", "exact_match", "exact_match_rate", "token_accuracy",
        "baseline_mean_logit_diff", "intervened_mean_logit_diff", "mean_logit_diff_change",
        "baseline_agent_preference_rate", "baseline_patient_preference_rate",
        "intervened_agent_preference_rate", "intervened_patient_preference_rate",
        "agent_to_patient_flip_rate", "count", "baseline_agent_wins",
        "conditional_agent_to_patient_flip_rate",
    }
    file_counts = {}
    for name in ("steering/baseline.csv", "steering/results.csv", "steering/summary.csv",
                 "reconstruction/results.csv", "reconstruction/summary.csv"):
        with (run / name).open(newline="") as handle:
            records = list(csv.DictReader(handle))
        file_counts[name] = len(records)
        for record in records:
            for key in numeric_fields.intersection(record):
                value = record[key]
                if key == "conditional_agent_to_patient_flip_rate" and value == "":
                    continue
                assert math.isfinite(float(value)), (name, key, value)
    assert file_counts["reconstruction/results.csv"] == 8 * len(LAYERS) * 2 * 3
    stage_configs = {}
    for name in ("unpackers", "tprs"):
        stage = json.loads((run / name / "metadata.json").read_text())
        assert len(stage["models"]) == len(LAYERS) * 2
        assert all(math.isfinite(model["best_validation_loss"]) for model in stage["models"])
        stage_configs[name] = {"config": stage["config"], "training": stage["training"]}
    report = {
        "passed": True, "purpose": "Pipeline smoke validation only; not a reconstruction or steering accuracy result",
        "model": TOKENIZER_ID, "revision": TOKENIZER_REVISION,
        "dtype": "bfloat16", "gpu": torch.cuda.get_device_name(),
        "corpus_rows": len(rows), "training_rows": split_counts["train"], "tpr_split_counts": split_counts,
        "held_out_bindings": [["boy", "agent"]],
        "requested_layers": LAYERS[:-1], "additional_control_layer": LAYERS[-1], "layers": LAYERS,
        "sites": cache["metadata"]["sites"], "llm_batch_size": 2,
        "stage_configs": stage_configs, "file_rows": file_counts,
        "baseline_rows": len(baseline), "intervention_rows": metadata["result_rows"],
        "controls": metadata["controls"],
        "hf_comparison": {"rtol": 0.01, "atol": 0.125,
                          "max_abs_error": (actual - expected).abs().max().item()},
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
