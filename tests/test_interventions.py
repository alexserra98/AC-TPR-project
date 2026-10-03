"""Check intervention sites, answer logits, controls, and experiment artifacts."""

import csv
import hashlib
import json

import pytest
import torch

from ac_tpr.dataset import CORPUS_COLUMNS, TOKENIZER_ID, TOKENIZER_REVISION, generate_rows
from ac_tpr.interventions import (
    PROMPT_TEMPLATE, prepare_prompts, role_deltas, run_interventions, score_batch,
    summarize_results,
)


@pytest.fixture
def rows():
    rows = generate_rows(
        ["boy", "girl", "man"], [{"lemma": "see", "past": "saw", "participle": "seen"}], []
    )
    for i, row in enumerate(rows):
        row["split"] = "train" if i < 4 else "test" if i < 8 else "gen_test"
    return rows


@pytest.fixture
def inputs(rows, tokenizer):
    ids, positions, answers = prepare_prompts(rows[:3], tokenizer)
    inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    return inputs, positions, answers


@pytest.fixture
def source_run(tmp_path, rows, tokenizer, model, model_metadata, monkeypatch):
    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    vectors_dir = tmp_path / "vectors"
    vectors_dir.mkdir()
    generator = torch.Generator().manual_seed(7)
    torch.save({
        "pooled": torch.randn((2, 2, 32), generator=generator),
        "by_voice": torch.randn((2, 2, 2, 32), generator=generator),
    }, vectors_dir / "syntactic_vectors.pt")
    metadata = {
        "schema_version": 3,
        "hook_names": ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"],
        "split": "train", "sentences": 4, "sentences_by_voice": {"active": 2, "passive": 2},
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "model": model_metadata,
        "tokenizer": {
            "name": TOKENIZER_ID, "revision": TOKENIZER_REVISION,
            "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "add_special_tokens": False, "padding_side": "right", "pad_token_id": tokenizer.pad_token_id,
        },
        "activation_site": "Transformer block output, before final_layer_norm",
        "roles": ["agent", "patient"], "voices": ["active", "passive"],
        "layer_indices": [0, 1],
        "axes": {"pooled": ["layer", "role", "hidden"], "by_voice": ["voice", "layer", "role", "hidden"]},
        "shape": {"pooled": [2, 2, 32], "by_voice": [2, 2, 2, 32]},
    }
    (vectors_dir / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr("ac_tpr.interventions.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr("ac_tpr.interventions.load_model", lambda *a, **kw: (model, model_metadata))
    return corpus, vectors_dir


def test_prompt_roles_and_answer_boundaries(rows, tokenizer):
    ids, positions, answers = prepare_prompts(rows[:2], tokenizer)
    assert positions.tolist() == [[1, 4], [6, 1]]
    assert list(map(len, ids)) == [18, 20]
    for row, tokens, role_positions, candidates in zip(rows, ids, positions, answers):
        assert tokenizer.decode(tokens) == PROMPT_TEMPLATE.format(sentence=row["sentence"])
        for role, position, candidate in zip(["agent", "patient"], role_positions, candidates):
            assert tokens[position] == candidate.item()
            assert tokenizer.decode([candidate.item()]) == " " + row[role]


@pytest.mark.parametrize("edited_role,sign", [("agent", 1), ("patient", -1)])
def test_mean_selection_and_edit_direction(rows, edited_role, sign):
    pooled = torch.arange(128, dtype=torch.float32).reshape(2, 2, 32)
    vectors = {"pooled": pooled, "by_voice": torch.stack([pooled * 2, pooled * 3 + 100])}
    for layer in (0, 1):
        torch.testing.assert_close(
            role_deltas(vectors, rows[:2], "pooled", layer, edited_role),
            torch.full((2, 32), sign * 32, dtype=torch.float32),
        )
        torch.testing.assert_close(
            role_deltas(vectors, rows[:2], "by_voice", layer, edited_role),
            torch.tensor([sign * 64, sign * 96], dtype=torch.float32)[:, None].expand(2, 32),
        )


def test_scores_match_full_lm_and_unpadded_forwards(model, hf_model, inputs):
    batch_inputs, positions, answers = inputs
    actual = score_batch(model, batch_inputs, answers)
    last = batch_inputs["attention_mask"].sum(dim=1) - 1
    with torch.inference_mode():
        full = hf_model(**batch_inputs, use_cache=False).logits[torch.arange(3), last]
        classic_full = model(batch_inputs["input_ids"], attention_mask=batch_inputs["attention_mask"],
                            prepend_bos=False, past_kv_cache=None)[torch.arange(3), last]
    torch.testing.assert_close(actual, full.gather(1, answers), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual, classic_full.gather(1, answers), rtol=1e-5, atol=1e-6)
    for i in range(3):
        single = {key: value[i:i + 1, :last[i] + 1] for key, value in batch_inputs.items()}
        torch.testing.assert_close(actual[i:i + 1], score_batch(model, single, answers[i:i + 1]), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("role", [0, 1])
def test_only_requested_tokens_change_and_hooks_are_removed(model, inputs, role):
    batch_inputs, positions, answers = inputs
    before, after = [], []
    deltas = torch.arange(96, dtype=torch.float32).reshape(3, 32) / 32
    with model.hooks(fwd_hooks=[
        ("blocks.0.hook_resid_post", lambda hidden, hook: before.append(hidden.clone())),
        ("blocks.1.hook_resid_pre", lambda hidden, hook: after.append(hidden.clone())),
    ]):
        score_batch(model, batch_inputs, answers, 0, positions[:, role], deltas)
        assert model.hook_dict["blocks.0.hook_resid_post"].fwd_hooks
    expected = before[0].clone()
    expected[torch.arange(3), positions[:, role]] += deltas
    torch.testing.assert_close(after[0], expected, rtol=0, atol=0)
    assert all(not hook.fwd_hooks for hook in model.hook_dict.values())


def test_zero_edits_and_final_block_preserve_baseline(model, inputs):
    batch_inputs, positions, answers = inputs
    baseline = score_batch(model, batch_inputs, answers)
    for role in (0, 1):
        zero = score_batch(model, batch_inputs, answers, 0, positions[:, role], torch.zeros(3, 32))
        final = score_batch(model, batch_inputs, answers, 1, positions[:, role], torch.ones(3, 32))
        torch.testing.assert_close(zero, baseline, rtol=0, atol=0)
        torch.testing.assert_close(final, baseline, rtol=0, atol=0)
    earlier = score_batch(model, batch_inputs, answers, 0, positions[:, 0], torch.arange(32).expand(3, 32).float())
    assert not torch.allclose(earlier, baseline)


def test_scoring_unembeds_only_one_position(model, inputs, monkeypatch):
    batch_inputs, positions, answers = inputs
    shapes = []
    linear = torch.nn.functional.linear

    def record(input, weight, bias=None):
        if weight.shape == (model.cfg.d_vocab_out, model.cfg.d_model):
            shapes.append(tuple(input.shape))
        return linear(input, weight, bias)

    monkeypatch.setattr(torch.nn.functional, "linear", record)
    score_batch(model, batch_inputs, answers, 0, positions[:, 0], torch.ones(3, 32))
    assert shapes == [(3, 1, 32)]


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("mean_type", ["pooled", "by_voice"])
@pytest.mark.parametrize("role_index", [0, 1])
def test_interventions_match_unwrapped_hf(model, hf_model, rows, inputs, layer, mean_type, role_index):
    batch_inputs, positions, answers = inputs
    generator = torch.Generator().manual_seed(7)
    vectors = {"pooled": torch.randn(2, 2, 32, generator=generator),
               "by_voice": torch.randn(2, 2, 2, 32, generator=generator)}
    deltas = role_deltas(vectors, rows[:3], mean_type, layer, ["agent", "patient"][role_index])
    actual = score_batch(model, batch_inputs, answers, layer, positions[:, role_index], deltas)

    def translate(module, args, hidden):
        result = hidden.clone()
        result[torch.arange(3), positions[:, role_index]] += deltas
        return result

    handle = hf_model.gpt_neox.layers[layer].register_forward_hook(translate)
    try:
        with torch.inference_mode():
            logits = hf_model(**batch_inputs, use_cache=False).logits
    finally:
        handle.remove()
    last = batch_inputs["attention_mask"].sum(dim=1) - 1
    expected = logits[torch.arange(3), last].gather(1, answers)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_failed_forward_cleans_up_hook(model, inputs, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(model, "forward", fail)
    batch_inputs, positions, answers = inputs
    with pytest.raises(RuntimeError, match="forward failed"):
        score_batch(model, batch_inputs, answers, 0, positions[:, 0], torch.ones(3, 32))
    assert all(not hook.fwd_hooks for hook in model.hook_dict.values())


def test_nonfinite_edit_cleans_up_hook(model, inputs):
    batch_inputs, positions, answers = inputs
    with pytest.raises(ValueError, match="Nonfinite edited"):
        score_batch(model, batch_inputs, answers, 0, positions[:, 0], torch.full((3, 32), float("nan")))
    assert all(not hook.fwd_hooks for hook in model.hook_dict.values())


def test_run_roundtrip_short_batch_and_summary(source_run, tmp_path):
    output = tmp_path / "scores"
    metadata = run_interventions(*source_run, output, batch_size=3, device="cpu", dtype="float32", limit=7)
    with (output / "baseline.csv").open() as handle:
        baseline = list(csv.DictReader(handle))
    with (output / "results.csv").open() as handle:
        results = list(csv.DictReader(handle))
    with (output / "summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    assert len(baseline) == metadata["baseline_rows"] == 7
    assert len(results) == metadata["result_rows"] == 56
    assert len(summary) == metadata["summary_rows"] == 32
    assert metadata["sentences_by_split"] == {"test": 4, "gen_test": 3}
    assert metadata["controls"] == {"zero_vector_first_batch": "passed", "final_block_all_rows": "passed"}
    assert metadata["prompt_template"] == PROMPT_TEMPLATE
    assert metadata["model"]["commit"] == "test-commit"
    assert metadata["schema_version"] == 3
    assert metadata["execution_model"]["weight_processing"] == "none"
    assert metadata["scoring_hook"] == "unembed.hook_in"
    assert json.loads((output / "metadata.json").read_text()) == metadata
    assert set(row["split"] for row in baseline) == {"test", "gen_test"}
    assert len({(row["pair_id"], row["voice"], row["layer"], row["mean_type"], row["edited_role"]) for row in results}) == 56
    lookup = {(row["pair_id"], row["voice"]): row for row in baseline}
    for row in results:
        original = lookup[(row["pair_id"], row["voice"])]
        assert row["baseline_logit_diff"] == original["baseline_logit_diff"]
        expected = float(row["intervened_patient_logit"]) - float(row["intervened_agent_logit"])
        assert float(row["intervened_logit_diff"]) == expected
        assert float(row["logit_diff_change"]) == expected - float(row["baseline_logit_diff"])
        if row["layer"] == "1":
            assert float(row["logit_diff_change"]) == 0
    assert sum(int(row["count"]) for row in summary) == len(results)
    with pytest.raises(FileExistsError):
        run_interventions(*source_run, output, device="cpu", dtype="float32")


def test_summary_signs_ties_and_denominators(tmp_path):
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "split", "voice", "layer", "mean_type", "edited_role",
            "baseline_logit_diff", "intervened_logit_diff", "logit_diff_change",
        ])
        writer.writeheader()
        for baseline, intervened in [(-2, 3), (4, -1), (0, 0)]:
            writer.writerow({"split": "test", "voice": "active", "layer": 0,
                             "mean_type": "pooled", "edited_role": "agent",
                             "baseline_logit_diff": baseline, "intervened_logit_diff": intervened,
                             "logit_diff_change": intervened - baseline})
    output = tmp_path / "summary.csv"
    assert summarize_results(path, output) == 1
    with output.open() as handle:
        summary = next(csv.DictReader(handle))
    assert int(summary["count"]) == 3
    assert float(summary["mean_logit_diff_change"]) == 0
    assert float(summary["baseline_mean_logit_diff"]) == pytest.approx(2 / 3)
    for key, value in summary.items():
        if key.endswith("rate"):
            assert float(value) == pytest.approx(1 / 3)


def test_source_dtype_is_preserved_separately_from_execution(source_run, tmp_path):
    corpus, vectors_dir = source_run
    path = vectors_dir / "metadata.json"
    source = json.loads(path.read_text())
    source["model"]["dtype"] = "float16"
    path.write_text(json.dumps(source))
    metadata = run_interventions(corpus, vectors_dir, tmp_path / "scores", device="cpu", dtype="float32", limit=2)
    assert metadata["model"] == source["model"]
    assert metadata["execution_model"]["dtype"] == "float32"


@pytest.mark.parametrize("problem", ["split", "hash", "roles", "site", "counts", "tokenizer", "shape", "nonfinite", "commit", "schema", "hooks"])
def test_incompatible_inputs_fail_without_completion(source_run, tmp_path, problem):
    corpus, vectors_dir = source_run
    path = vectors_dir / "metadata.json"
    metadata = json.loads(path.read_text())
    if problem == "split":
        metadata["split"] = "test"
    elif problem == "hash":
        metadata["corpus_sha256"] = "wrong"
    elif problem == "roles":
        metadata["roles"].reverse()
    elif problem == "site":
        metadata["activation_site"] = "embeddings"
    elif problem == "counts":
        metadata["sentences"] = 99
    elif problem == "tokenizer":
        metadata["tokenizer"]["backend_sha256"] = "wrong"
    elif problem == "shape":
        metadata["shape"]["pooled"] = [2, 2, 64]
    elif problem == "commit":
        metadata["model"]["commit"] = "different-commit"
    elif problem == "schema":
        del metadata["schema_version"]
    elif problem == "hooks":
        metadata["hook_names"] = ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"]
    else:
        vectors = torch.load(vectors_dir / "syntactic_vectors.pt", weights_only=True)
        vectors["pooled"][0, 0, 0] = float("nan")
        torch.save(vectors, vectors_dir / "syntactic_vectors.pt")
    path.write_text(json.dumps(metadata))
    output = tmp_path / "scores"
    with pytest.raises(ValueError):
        run_interventions(corpus, vectors_dir, output, device="cpu", dtype="float32")
    assert not (output / "metadata.json").exists()


def test_nonfinite_scores_have_no_completion_marker(source_run, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("Nonfinite answer logits")

    monkeypatch.setattr("ac_tpr.interventions.score_batch", fail)
    output = tmp_path / "scores"
    with pytest.raises(ValueError, match="Nonfinite answer"):
        run_interventions(*source_run, output, device="cpu", dtype="float32")
    assert not (output / "metadata.json").exists()
