"""Check role alignment, residual hook locations, padding, and saved artifacts."""

import csv
import json

import pytest
import torch

from ac_tpr.activations import extract_activations, extract_batch, tokenize_roles
from ac_tpr.dataset import CORPUS_COLUMNS, TOKENIZER_REVISION, generate_rows
from ac_tpr.prompts import prepare_prompts, PROMPT_TEMPLATE


@pytest.fixture
def rows():
    return generate_rows(
        ["boy", "girl"], [{"lemma": "see", "past": "saw", "participle": "seen"}], []
    )


def test_roles_follow_semantics_in_both_voices(rows, tokenizer):
    ids, positions = tokenize_roles(rows, tokenizer)
    assert positions.tolist() == [[1, 4], [6, 1], [1, 4], [6, 1]]
    assert [len(tokens) for tokens in ids] == [6, 8, 6, 8]
    for row, tokens, role_positions in zip(rows, ids, positions):
        for role, position in zip(["agent", "patient"], role_positions):
            assert tokenizer.decode([tokens[position]]).strip() == row[role]


def test_batched_hooks_match_individual_residual_states(rows, tokenizer, model, hf_model):
    ids, positions = tokenize_roles(rows, tokenizer)
    batch_inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    actual = extract_batch(model, batch_inputs, positions)
    assert actual.shape == (4, 2, 2, 32)
    assert actual.device.type == "cpu"
    assert not actual.requires_grad
    for index, tokens in enumerate(ids):
        before_norm = []
        handle = hf_model.model.norm.register_forward_pre_hook(
            lambda _module, args: before_norm.append(args[0].clone())
        )
        with torch.inference_mode():
            output = hf_model.model(
                input_ids=torch.tensor([tokens]), use_cache=False, output_hidden_states=True
            )
        handle.remove()
        # hidden_states[1] is the first block output; the final entry is normalized.
        expected = torch.stack([
            output.hidden_states[1][0, positions[index]],
            before_norm[0][0, positions[index]],
        ])
        torch.testing.assert_close(actual[index], expected, atol=1e-6, rtol=1e-5)
        assert not torch.allclose(actual[index, -1], output.last_hidden_state[0, positions[index]])
    torch.testing.assert_close(extract_batch(model, batch_inputs, positions), actual, rtol=0, atol=0)


def test_trace_failure_does_not_affect_next_extraction(rows, tokenizer, model, monkeypatch):
    ids, positions = tokenize_roles(rows, tokenizer)
    inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    baseline = extract_batch(model, inputs, positions)

    def fail_forward(*args, **kwargs):
        raise RuntimeError("forward failed")

    with monkeypatch.context() as patch:
        patch.setattr(model.model.layers[1]._module, "forward", fail_forward)
        with pytest.raises(RuntimeError, match="forward failed"):
            extract_batch(model, inputs, positions)
    torch.testing.assert_close(extract_batch(model, inputs, positions), baseline, rtol=0, atol=0)


def test_extraction_stops_before_vocabulary_projection(rows, tokenizer, model, monkeypatch):
    ids, positions = tokenize_roles(rows, tokenizer)
    inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")

    def fail_projection(*args, **kwargs):
        raise AssertionError("Extraction must not project vocabulary logits")

    monkeypatch.setattr(model.lm_head._module, "forward", fail_projection)
    assert extract_batch(model, inputs, positions).shape == (4, 2, 2, 32)


@pytest.mark.parametrize("problem", ["missing", "duplicate", "multitoken"])
def test_invalid_role_tokens_are_rejected(rows, tokenizer, problem):
    if problem == "missing":
        rows[0]["agent"] = "man"
    elif problem == "duplicate":
        rows[0]["patient"] = rows[0]["agent"]
    else:
        rows[0]["sentence"] = rows[0]["sentence"].replace("boy", "neuroscientist")
        rows[0]["agent"] = "neuroscientist"
    with pytest.raises(ValueError):
        tokenize_roles(rows, tokenizer)


def test_extraction_roundtrip_and_partial_batch(tmp_path, rows, tokenizer, model, model_metadata, monkeypatch):
    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[:3])
    monkeypatch.setattr("ac_tpr.activations.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr("ac_tpr.activations.load_model", lambda *a, **kw: (model, model_metadata))
    output_dir = tmp_path / "activations"
    metadata = extract_activations(corpus, output_dir, batch_size=2, device="cpu", dtype="float32")
    saved = torch.load(output_dir / "activations.pt", weights_only=True)
    assert saved["rows"] == rows[:3]
    assert metadata["shape"] == [3, 2, 2, 32]
    assert metadata["roles"] == ["agent", "patient"]
    assert metadata["layer_indices"] == [0, 1]
    assert metadata["prompt_template"] == PROMPT_TEMPLATE
    assert metadata["schema_version"] == 5
    assert metadata["hook_names"] == ["model.layers.0.output", "model.layers.1.output"]
    assert metadata["sentences_by_voice"] == {"active": 2, "passive": 1}
    assert metadata["model"]["revision"] == TOKENIZER_REVISION
    assert json.loads((output_dir / "metadata.json").read_text()) == metadata
    assert torch.isfinite(saved["activations"]).all()
    ids, positions, _ = prepare_prompts(rows[:3], tokenizer)
    assert saved["input_ids"] == ids
    torch.testing.assert_close(saved["role_positions"], positions)
    inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")
    torch.testing.assert_close(saved["activations"], extract_batch(model, inputs, positions))
    with pytest.raises(FileExistsError):
        extract_activations(corpus, output_dir, device="cpu", dtype="float32")


def test_failed_extraction_has_no_completion_metadata(tmp_path, rows, tokenizer, model, model_metadata, monkeypatch):
    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr("ac_tpr.activations.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr("ac_tpr.activations.load_model", lambda *a, **kw: (model, model_metadata))

    def nonfinite(*args):
        return torch.full((4, 2, 2, 32), float("nan"))

    monkeypatch.setattr("ac_tpr.activations.extract_batch", nonfinite)
    output_dir = tmp_path / "activations"
    with pytest.raises(ValueError, match="Nonfinite"):
        extract_activations(corpus, output_dir, device="cpu", dtype="float32")
    assert not (output_dir / "metadata.json").exists()


def test_custom_prompt_survives_extraction_and_vector_provenance(tmp_path, rows, tokenizer, model, model_metadata, monkeypatch):
    from ac_tpr.syntactic_vectors import extract_syntactic_vectors
    corpus = tmp_path / "corpus.csv"
    for row in rows:
        row["split"] = "train"
    with corpus.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr("ac_tpr.activations.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    monkeypatch.setattr("ac_tpr.activations.load_model", lambda *a, **kw: (model, model_metadata))
    template = "The boy saw the girl.\nAnswer: boy\n\n{sentence}\nAnswer:"
    output = tmp_path / "activations"
    extract_activations(corpus, output, device="cpu", dtype="float32", prompt_template=template)
    metadata = extract_syntactic_vectors(output, tmp_path / "vectors")
    assert metadata["prompt_template"] == template
    saved = torch.load(output / "activations.pt", weights_only=True)
    ids, positions, _ = prepare_prompts(rows, tokenizer, template)
    assert saved["input_ids"] == ids
    torch.testing.assert_close(saved["role_positions"], positions)
