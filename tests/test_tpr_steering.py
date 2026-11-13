"""Check final-logit TPR interventions, donor controls, and binding-test summaries."""

import csv
import hashlib
import json

import pytest
import torch

from ac_tpr.activations import extract_batch
from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION, generate_rows
from ac_tpr.prompts import PROMPT_TEMPLATE
from ac_tpr.tpr.data import prepare_data
from ac_tpr.tpr.decomposition import TensorProductModel
from ac_tpr.tpr.extraction import training_statistics
from ac_tpr.tpr.steering import evaluate_steering, summarize_steering


@pytest.fixture
def steering_cache(tokenizer, model, model_metadata, monkeypatch):
    rows = generate_rows(
        ["boy", "girl", "man"], [{"lemma": "see", "past": "saw", "participle": "seen"}], [],
    )
    for row in rows:
        row["split"] = "test" if {row["agent"], row["patient"]} == {"boy", "girl"} else "train"
    cache = prepare_data(rows, tokenizer, held_out_bindings=(("boy", "agent"),))
    cache["corpus_indices"] = 2 * cache["corpus_indices"] + 7
    inputs = tokenizer.pad({"input_ids": cache["input_ids"]}, padding=True, return_tensors="pt")
    cache["activations"] = extract_batch(model, inputs, cache["positions"])
    cache["mean"], cache["scale"] = training_statistics(cache["activations"], cache["rows"])
    cache["metadata"] = {
        "schema_version": 5, "tpr_schema_version": 1, "artifact_type": "tpr_encodings",
        "prompt_template": PROMPT_TEMPLATE,
        "hook_names": ["model.layers.0.output", "model.layers.1.output"],
        "model": model_metadata,
        "tokenizer": {
            "name": TOKENIZER_ID, "revision": TOKENIZER_REVISION,
            "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "add_special_tokens": False, "padding_side": "right", "pad_token_id": tokenizer.pad_token_id,
        },
        "activation_site": "Transformer block output, before final RMSNorm",
        "layer_indices": [0, 1], "sites": ["sentence_end", "prompt_end"],
        "hidden_size": 32, "shape": list(cache["activations"].shape),
    }
    torch.manual_seed(52)
    models = {
        (layer, site): TensorProductModel(len(cache["filler_vocab"]), 32, filler_dim=3, role_dim=2)
        for layer in [0, 1] for site in ["sentence_end", "prompt_end"]
    }
    monkeypatch.setattr("ac_tpr.tpr.steering.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr("ac_tpr.tpr.steering.load_model", lambda *a, **kw: (model, model_metadata))
    return cache, models


def test_steering_roundtrip_and_site_specific_controls(steering_cache, tmp_path):
    cache, models = steering_cache
    output = tmp_path / "steering"
    metadata = evaluate_steering(cache, models, output, device="cpu", dtype="float32", batch_size=4)
    with (output / "baseline.csv").open() as handle:
        baseline = list(csv.DictReader(handle))
    with (output / "results.csv").open() as handle:
        results = list(csv.DictReader(handle))
    assert len(baseline) == metadata["baseline_rows"] == 6
    assert metadata["baseline_forward_rows"] == 8
    assert len(results) == metadata["result_rows"] == 6 * 2 * 2 * 4
    assert metadata["controls"] == {
        "zero_vector": "passed", "final_block_sentence_end": "passed",
        "final_block_prompt_end_donor": "passed",
    }
    assert {row["source_tpr_split"] for row in baseline} == {"test", "binding_test"}
    assert {row["target_binding_held_out"] for row in baseline} == {"True", "False"}
    assert all(row["source_seen_in_training"] == "False" for row in baseline)
    assert any(row["target_tpr_split"] == "train" for row in baseline)
    assert json.loads((output / "metadata.json").read_text()) == metadata
    for row in baseline:
        assert int(row["corpus_index"]) == 2 * int(row["source_index"]) + 7
        assert int(row["target_corpus_index"]) == 2 * int(row["target_index"]) + 7
    for row in results:
        assert int(row["corpus_index"]) == 2 * int(row["source_index"]) + 7
        assert int(row["target_corpus_index"]) == 2 * int(row["target_index"]) + 7
        assert float(row["intervened_logit_diff"]) == (
            float(row["intervened_patient_logit"]) - float(row["intervened_agent_logit"])
        )
        if row["condition"] == "zero" or (row["layer"] == "1" and row["site"] == "sentence_end"):
            assert float(row["logit_diff_change"]) == 0
        if row["layer"] == "1" and row["site"] == "prompt_end" and row["condition"] == "donor_swap":
            assert float(row["intervened_agent_logit"]) == pytest.approx(float(row["donor_baseline_agent_logit"]), abs=1e-6)
            assert float(row["intervened_patient_logit"]) == pytest.approx(float(row["donor_baseline_patient_logit"]), abs=1e-6)
    assert all(next(model.parameters()).device.type == "cpu" for model in models.values())
    with pytest.raises(FileExistsError):
        evaluate_steering(cache, models, output, device="cpu", dtype="float32")


def test_deltas_restore_scale_and_alpha_only_scales_swap(steering_cache, tmp_path, monkeypatch):
    cache, models = steering_cache
    calls = []

    def capture(_model, inputs, answer_ids, layer=None, positions=None, deltas=None, **kwargs):
        if layer is not None:
            calls.append((layer, kwargs["allow_final_token"], deltas.clone()))
        return torch.zeros((len(answer_ids), 2))

    monkeypatch.setattr("ac_tpr.tpr.steering.score_batch", capture)
    evaluate_steering(cache, models, tmp_path / "steering", device="cpu", dtype="float32", batch_size=20, alpha=2.5)
    indices = [i for i, row in enumerate(cache["rows"])
               if row["split"] == "test" or row["tpr_split"] == "binding_test"]
    fillers, roles = cache["filler_ids"][indices], cache["role_ids"][indices]
    swapped = fillers.clone()
    swapped[:, [0, 2]] = fillers[:, [2, 0]]
    for group, ((layer, site), tpr_model) in enumerate(models.items()):
        site_index = ["sentence_end", "prompt_end"].index(site)
        original, counterfactual = tpr_model(fillers, roles), tpr_model(swapped, roles)
        scale, mean = cache["scale"][layer, site_index], cache["mean"][layer, site_index]
        raw = cache["activations"][indices, layer, site_index]
        donor = cache["activations"][cache["donor_indices"][indices], layer, site_index]
        expected = [torch.zeros_like(raw), mean + scale * original - raw,
                    2.5 * scale * (counterfactual - original), donor - raw]
        for (_, allow_final, actual), target in zip(calls[group * 4:group * 4 + 4], expected, strict=True):
            assert allow_final == (site == "prompt_end")
            torch.testing.assert_close(actual, target)


def test_summary_preserves_target_binding_and_conditional_denominator(tmp_path):
    results = tmp_path / "results.csv"
    fields = ["split", "voice", "layer", "site", "condition", "source_tpr_split",
              "target_tpr_split", "target_binding_held_out", "source_seen_in_training",
              "baseline_logit_diff", "intervened_logit_diff", "logit_diff_change"]
    with results.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for target, baseline, intervened in [("binding_test", -2, 3), ("binding_test", -1, -2),
                                             ("binding_test", 0, 0), ("test", 2, -1)]:
            writer.writerow({
                "split": "test", "voice": "active", "layer": 1, "site": "prompt_end",
                "condition": "swap", "source_tpr_split": "test", "target_tpr_split": target,
                "target_binding_held_out": target == "binding_test", "source_seen_in_training": False,
                "baseline_logit_diff": baseline, "intervened_logit_diff": intervened,
                "logit_diff_change": intervened - baseline,
            })
    output = tmp_path / "summary.csv"
    assert summarize_steering(results, output) == 2
    with output.open() as handle:
        groups = {row["target_tpr_split"]: row for row in csv.DictReader(handle)}
    assert float(groups["binding_test"]["agent_to_patient_flip_rate"]) == pytest.approx(1 / 3)
    assert float(groups["binding_test"]["conditional_agent_to_patient_flip_rate"]) == 0.5
    assert int(groups["binding_test"]["baseline_agent_wins"]) == 2
    assert groups["test"]["conditional_agent_to_patient_flip_rate"] == ""


@pytest.mark.parametrize("problem", ["dtype", "prompt", "tokens", "model_keys"])
def test_incompatible_inputs_fail_before_model_loading(steering_cache, tmp_path, monkeypatch, problem):
    cache, models = steering_cache
    if problem == "dtype":
        cache["metadata"]["model"]["dtype"] = "bfloat16"
    elif problem == "prompt":
        cache["metadata"]["prompt_template"] = "{sentence}\nAnswer:"
    elif problem == "tokens":
        cache["input_ids"][0][0] += 1
    else:
        models.pop((0, "sentence_end"))

    def fail(*args, **kwargs):
        pytest.fail("Incompatible inputs must be rejected before loading the LLM")

    monkeypatch.setattr("ac_tpr.tpr.steering.load_model", fail)
    output = tmp_path / "steering"
    with pytest.raises(ValueError):
        evaluate_steering(cache, models, output, device="cpu", dtype="float32")
    assert not output.exists()
