"""Run a 20-sentence Qwen Base pipeline smoke check on an allocated GPU."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION, load_corpus
from ac_tpr.prompts import prepare_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New smoke-run directory.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    corpus = load_corpus(Path("data/active_pass_long/corpus.csv"))
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prepare_prompts(corpus, tokenizer)
    print(f"Validated full prompts and role/answer tokens for {len(corpus)} corpus rows", flush=True)
    selected_pairs = set()
    for split, count in (("train", 5), ("test", 3), ("gen_test", 2)):
        pairs = [r["pair_id"] for r in corpus if r["split"] == split and r["voice"] == "active"][:count]
        assert len(pairs) == count
        selected_pairs.update(pairs)
    rows = [r for r in corpus if r["pair_id"] in selected_pairs]
    assert len(rows) == 20
    output.mkdir(parents=True)
    corpus_path = output / "corpus.csv"
    with corpus_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    commands = [
        ["scripts/extract_activations.py", "--corpus", str(corpus_path),
         "--output-dir", str(output / "activations"), "--batch-size", "2"],
        ["scripts/extract_syntactic_vectors.py", "--activations-dir", str(output / "activations"),
         "--output-dir", str(output / "vectors")],
        ["scripts/run_interventions.py", "--corpus", str(corpus_path),
         "--vectors-dir", str(output / "vectors"), "--output-dir", str(output / "interventions"),
         "--batch-size", "2"],
    ]
    for command in commands:
        subprocess.run([sys.executable, "-u", *command], check=True)

    metadata = json.loads((output / "interventions/metadata.json").read_text())
    assert metadata["baseline_rows"] == 10
    assert metadata["result_rows"] == 1440
    assert metadata["controls"] == {"zero_vector_first_batch": "passed", "final_block_all_rows": "passed"}
    with (output / "interventions/baseline.csv").open(newline="") as handle:
        baseline = list(csv.DictReader(handle))
    eval_rows = [r for r in rows if r["split"] in {"test", "gen_test"}]
    ids, _, answers = prepare_prompts(eval_rows, tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to("cuda").eval()
    model.requires_grad_(False)
    expected = []
    with torch.inference_mode():
        for tokens, candidates in zip(ids, answers, strict=True):
            inputs = torch.tensor([tokens], device="cuda")
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                           use_cache=False, logits_to_keep=1).logits[0, -1]
            expected.append(logits[candidates.to("cuda")].float().cpu())
    expected = torch.stack(expected)
    actual = torch.tensor([[float(r["baseline_agent_logit"]), float(r["baseline_patient_logit"])] for r in baseline])
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.125)
    with (output / "interventions/results.csv").open(newline="") as handle:
        results = list(csv.DictReader(handle))
    for row in results:
        assert all(torch.isfinite(torch.tensor(float(row[k]))) for k in (
            "intervened_agent_logit", "intervened_patient_logit", "logit_diff_change"
        ))
    report = {
        "passed": True, "purpose": "Smoke validation only; not research results",
        "model": TOKENIZER_ID, "revision": TOKENIZER_REVISION,
        "full_corpus_tokenization_rows": len(corpus), "training_rows": 10,
        "baseline_rows": len(baseline), "intervention_rows": len(results),
        "controls": metadata["controls"],
        "hf_comparison": {"rtol": 0.01, "atol": 0.125,
                          "max_abs_error": (actual - expected).abs().max().item()},
        "baseline_agent_preference": (actual[:, 0] > actual[:, 1]).float().mean().item(),
        "gpu": torch.cuda.get_device_name(),
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
