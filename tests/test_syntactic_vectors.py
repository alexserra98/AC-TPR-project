"""Check role means, training split isolation, and artifact compatibility."""

import json
from collections import Counter

import pytest
import torch

from ac_tpr.syntactic_vectors import extract_syntactic_vectors


@pytest.fixture
def source_run(tmp_path, model_metadata):
    run = tmp_path / "activations"
    run.mkdir()
    rows = [
        {"split": "train", "voice": "passive"},
        {"split": "val", "voice": "active"},
        {"split": "train", "voice": "active"},
        {"split": "test", "voice": "passive"},
        {"split": "train", "voice": "active"},
        {"split": "gen_test", "voice": "active"},
    ]
    base = torch.arange(12, dtype=torch.float16).reshape(2, 2, 3)
    offsets = torch.tensor([4, 300, -3, -300, 0, 600], dtype=torch.float16)
    activations = base + offsets[:, None, None, None]
    torch.save({"activations": activations, "rows": rows}, run / "activations.pt")
    metadata = {
        "schema_version": 3,
        "hook_names": ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"],
        "corpus": "/example/corpus.csv",
        "corpus_sha256": "a" * 64,
        "model": {
            **model_metadata,
            "name": "EleutherAI/pythia-6.9b", "revision": "step143000",
            "commit": "b" * 40, "dtype": "float16",
        },
        "tokenizer": {
            "name": "EleutherAI/pythia-6.9b", "revision": "step143000",
            "backend_sha256": "c" * 64, "add_special_tokens": False,
            "padding_side": "right", "pad_token_id": 0,
        },
        "activation_site": "Transformer block output, before final_layer_norm",
        "layer_indices": [0, 1],
        "roles": ["agent", "patient"],
        "axes": ["sentence", "layer", "role", "hidden"],
        "shape": list(activations.shape),
        "sentences_by_split": dict(Counter(row["split"] for row in rows)),
        "sentences_by_voice": dict(Counter(row["voice"] for row in rows)),
    }
    (run / "metadata.json").write_text(json.dumps(metadata))
    return run


def test_means_and_artifact_roundtrip(source_run, tmp_path):
    output = tmp_path / "means"
    metadata = extract_syntactic_vectors(source_run, output)
    saved = torch.load(output / "syntactic_vectors.pt", weights_only=True)
    base = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    torch.testing.assert_close(saved["pooled"], base + 1 / 3)
    torch.testing.assert_close(saved["by_voice"][0], base - 1.5)
    torch.testing.assert_close(saved["by_voice"][1], base + 4)
    assert not torch.allclose(saved["pooled"], saved["by_voice"].mean(dim=0))
    for tensor in saved.values():
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
    assert metadata["shape"] == {"pooled": [2, 2, 3], "by_voice": [2, 2, 2, 3]}
    assert metadata["axes"] == {
        "pooled": ["layer", "role", "hidden"],
        "by_voice": ["voice", "layer", "role", "hidden"],
    }
    assert metadata["dtype"] == "float32"
    assert metadata["voices"] == ["active", "passive"]
    assert metadata["split"] == "train"
    assert metadata["sentences"] == 3
    assert metadata["sentences_by_voice"] == {"active": 2, "passive": 1}
    assert metadata["activations_dir"] == str(source_run.resolve())
    source = json.loads((source_run / "metadata.json").read_text())
    for key in (
        "corpus", "corpus_sha256", "model", "tokenizer", "activation_site",
        "layer_indices", "roles", "schema_version", "hook_names",
    ):
        assert metadata[key] == source[key]
    assert json.loads((output / "metadata.json").read_text()) == metadata
    with pytest.raises(FileExistsError):
        extract_syntactic_vectors(source_run, output)


def test_held_out_activations_cannot_change_means(source_run, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    extract_syntactic_vectors(source_run, first)
    data = torch.load(source_run / "activations.pt", weights_only=True)
    for index, row in enumerate(data["rows"]):
        if row["split"] != "train":
            data["activations"][index] = float("nan")
    torch.save(data, source_run / "activations.pt")
    extract_syntactic_vectors(source_run, second)
    before = torch.load(first / "syntactic_vectors.pt", weights_only=True)
    after = torch.load(second / "syntactic_vectors.pt", weights_only=True)
    for name in ("pooled", "by_voice"):
        torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


@pytest.mark.parametrize("group", ["train", "active", "passive"])
def test_missing_training_groups_are_rejected(source_run, tmp_path, group):
    data = torch.load(source_run / "activations.pt", weights_only=True)
    for row in data["rows"]:
        if row["split"] == "train" and (group == "train" or row["voice"] == group):
            row["split"] = "val"
    torch.save(data, source_run / "activations.pt")
    metadata = json.loads((source_run / "metadata.json").read_text())
    metadata["sentences_by_split"] = dict(Counter(row["split"] for row in data["rows"]))
    (source_run / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "means"
    with pytest.raises(ValueError, match="training rows|Training rows"):
        extract_syntactic_vectors(source_run, output)
    assert not output.exists()


@pytest.mark.parametrize(("field", "value", "message"), [
    ("roles", ["patient", "agent"], "roles"),
    ("axes", ["sentence", "role", "layer", "hidden"], "axes"),
    ("shape", [6, 2, 2, 4], "shape"),
    ("layer_indices", [1, 2], "hook names"),
    ("activation_site", "Final normalized output", "block outputs"),
    ("sentences_by_split", {"train": 6}, "split counts"),
    ("schema_version", 1, "Regenerate"),
    ("hook_names", ["blocks.0.hook_resid_pre", "blocks.1.hook_resid_pre"], "hook names"),
])
def test_incompatible_metadata_is_rejected(source_run, tmp_path, field, value, message):
    metadata = json.loads((source_run / "metadata.json").read_text())
    metadata[field] = value
    (source_run / "metadata.json").write_text(json.dumps(metadata))
    output = tmp_path / "means"
    with pytest.raises(ValueError, match=message):
        extract_syntactic_vectors(source_run, output)
    assert not output.exists()


@pytest.mark.parametrize("problem", ["missing_file", "missing_field"])
def test_incomplete_source_metadata_is_rejected(source_run, tmp_path, problem):
    metadata_path = source_run / "metadata.json"
    if problem == "missing_file":
        metadata_path.unlink()
        expected_error = FileNotFoundError
    else:
        metadata = json.loads(metadata_path.read_text())
        del metadata["tokenizer"]
        metadata_path.write_text(json.dumps(metadata))
        expected_error = KeyError
    output = tmp_path / "means"
    with pytest.raises(expected_error):
        extract_syntactic_vectors(source_run, output)
    assert not output.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_training_activations_are_rejected(source_run, tmp_path, value):
    data = torch.load(source_run / "activations.pt", weights_only=True)
    data["activations"][0, 1, 0, 0] = value
    torch.save(data, source_run / "activations.pt")
    output = tmp_path / "means"
    with pytest.raises(ValueError, match="Nonfinite training activations at layer 1"):
        extract_syntactic_vectors(source_run, output)
    assert not output.exists()


def test_failed_save_has_no_completion_metadata(source_run, tmp_path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("save failed")

    monkeypatch.setattr("ac_tpr.syntactic_vectors.torch.save", fail_save)
    output = tmp_path / "means"
    with pytest.raises(OSError, match="save failed"):
        extract_syntactic_vectors(source_run, output)
    assert not (output / "metadata.json").exists()
