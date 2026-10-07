"""Check tensor binding, VQ gradients, held-out isolation, and steering integration."""

import csv
import hashlib
import json
from collections import Counter

import pytest
import torch

from ac_tpr.activations import extract_batch, tokenize_roles
from ac_tpr.dataset import CORPUS_COLUMNS, generate_rows
from ac_tpr.interventions import run_interventions
from ac_tpr.soft_tpr import (
    SoftTPRAutoencoder, decoded_states, fit_autoencoder, soft_tpr_loss, train_soft_tpr_vectors,
)
from ac_tpr.soft_tpr_comparison import compare_soft_tpr_runs


SMALL = dict(hidden_dim=12, num_roles=2, role_dim=3, filler_dim=3, num_codes=4)
FIT = dict(epochs=3, batch_size=5, learning_rate=0.01, form_weight=1.0,
           commitment_weight=0.25, seed=7, device="cpu")


def test_binding_and_nearest_codes():
    torch.manual_seed(10)
    model = SoftTPRAutoencoder(input_dim=6, **SMALL)
    fillers = torch.randn(5, 2, 3)
    expected = sum(fillers[:, slot, :, None] * model.roles[:, slot] for slot in range(2))
    torch.testing.assert_close(model.bind(fillers), expected)
    torch.testing.assert_close(model.unbind(expected), fillers)
    output = model(torch.randn(5, 6))
    distances = ((output["fillers"][:, :, None] - model.codebook.weight) ** 2).sum(-1)
    assert torch.equal(output["indices"], distances.argmin(-1))
    torch.testing.assert_close(output["reconstruction"], model.decoder(output["tpr"].flatten(1)))


def test_loss_gradient_routes():
    model = SoftTPRAutoencoder(input_dim=6, **SMALL)
    inputs = torch.randn(5, 6)
    losses = soft_tpr_loss(model(inputs), inputs)
    losses["reconstruction"].backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.decoder.weight.grad.abs().sum() > 0
    assert model.codebook.weight.grad is None
    model.zero_grad(set_to_none=True)
    soft_tpr_loss(model(inputs), inputs)["codebook"].backward()
    assert model.codebook.weight.grad.abs().sum() > 0
    assert model.encoder[0].weight.grad is None
    model.zero_grad(set_to_none=True)
    soft_tpr_loss(model(inputs), inputs)["commitment"].backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.codebook.weight.grad is None
    assert not model.roles.requires_grad


def test_form_loss_controls_component_outside_role_span():
    model = SoftTPRAutoencoder(input_dim=6, **SMALL)
    output = model(torch.randn(5, 6))
    reconstruction_gradient, = torch.autograd.grad(output["reconstruction"].square().mean(), output["soft"], retain_graph=True)
    torch.testing.assert_close(reconstruction_gradient, model.bind(model.unbind(reconstruction_gradient)))
    form_gradient, = torch.autograd.grad(soft_tpr_loss(output, torch.zeros(5, 6))["form"], output["soft"])
    assert (form_gradient - model.bind(model.unbind(form_gradient))).norm() > 0


def test_autoencoder_learns_and_restores_validation_checkpoint():
    generator = torch.Generator().manual_seed(11)
    centers = torch.randn(4, 6, generator=generator)
    train = centers.repeat(8, 1)
    val = centers.repeat(2, 1)
    model, center, scale, history, best = fit_autoencoder(
        train, val, {"input_dim": 6, **SMALL}, **{**FIT, "epochs": 40, "batch_size": 16},
    )
    validation = [row for row in history if row["split"] == "val"]
    assert min(row["reconstruction"] for row in validation) < validation[0]["reconstruction"] * 0.8
    assert best == min(validation, key=lambda row: row["reconstruction"])["epoch"]
    reconstruction = decoded_states(model, val, center, scale, 16, "cpu")
    mse = ((reconstruction - val) / scale).square().mean().item()
    assert mse == pytest.approx(validation[best - 1]["reconstruction"], abs=1e-6)
    torch.testing.assert_close(center, train.mean(0))
    torch.testing.assert_close(scale, (train - train.mean(0)).square().mean().sqrt())


@pytest.fixture
def activation_run(tmp_path, model_metadata):
    run = tmp_path / "activations"
    run.mkdir()
    rows = [{"split": split, "voice": voice}
            for split in ("train", "train", "val", "test", "gen_test")
            for voice in ("active", "passive")]
    # Unequal voice counts exercise weighting rather than averaging voice means.
    rows.append({"split": "train", "voice": "active"})
    states = torch.randn(len(rows), 2, 2, 6, generator=torch.Generator().manual_seed(12))
    torch.save({"activations": states, "rows": rows}, run / "activations.pt")
    metadata = {
        "schema_version": 3, "hook_names": [f"blocks.{layer}.hook_resid_post" for layer in range(2)],
        "corpus": "/example/corpus.csv", "corpus_sha256": "a" * 64, "model": model_metadata,
        "tokenizer": {}, "activation_site": "Transformer block output, before final_layer_norm",
        "layer_indices": [0, 1], "roles": ["agent", "patient"],
        "axes": ["sentence", "layer", "role", "hidden"], "shape": list(states.shape),
        "sentences_by_split": dict(Counter(row["split"] for row in rows)),
        "sentences_by_voice": dict(Counter(row["voice"] for row in rows)),
    }
    (run / "metadata.json").write_text(json.dumps(metadata))
    return run


def test_export_matches_decoded_train_means_and_roundtrip(activation_run, tmp_path):
    output = tmp_path / "experiment"
    metadata = train_soft_tpr_vectors(activation_run, output, **SMALL, **FIT)
    vectors = torch.load(output / "soft_tpr_vectors/syntactic_vectors.pt", weights_only=True)
    data = torch.load(activation_run / "activations.pt", weights_only=True)
    train_indices = [i for i, row in enumerate(data["rows"]) if row["split"] == "train"]
    for layer in (0, 1):
        checkpoint = torch.load(output / f"checkpoints/layer_{layer:02d}.pt", weights_only=True)
        model = SoftTPRAutoencoder(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        train = data["activations"][train_indices, layer]
        decoded = decoded_states(model, train.flatten(0, 1), checkpoint["center"], checkpoint["scale"], 5, "cpu").reshape_as(train)
        torch.testing.assert_close(vectors["pooled"][layer], decoded.mean(0))
        for voice_index, voice in enumerate(("active", "passive")):
            mask = [data["rows"][i]["voice"] == voice for i in train_indices]
            torch.testing.assert_close(vectors["by_voice"][voice_index, layer], decoded[mask].mean(0))
    assert metadata["vector_method"] == "soft_tpr_decoded_role_means"
    assert metadata["sentences_by_voice"] == {"active": 3, "passive": 2}
    assert metadata["soft_tpr"]["validation_sentences"] == 2
    assert (output / "role_means/metadata.json").exists()
    with pytest.raises(FileExistsError):
        train_soft_tpr_vectors(activation_run, output, **SMALL, **FIT)


def test_test_splits_cannot_influence_training_or_directions(activation_run, tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    train_soft_tpr_vectors(activation_run, before, **SMALL, **FIT)
    data = torch.load(activation_run / "activations.pt", weights_only=True)
    for i, row in enumerate(data["rows"]):
        if row["split"] in ("test", "gen_test"):
            data["activations"][i] = float("nan")
    torch.save(data, activation_run / "activations.pt")
    train_soft_tpr_vectors(activation_run, after, **SMALL, **FIT)
    for layer in (0, 1):
        a, b = [torch.load(directory / f"checkpoints/layer_{layer:02d}.pt", weights_only=True) for directory in (before, after)]
        for key in a["state_dict"]:
            torch.testing.assert_close(a["state_dict"][key], b["state_dict"][key], rtol=0, atol=0)
    a, b = [torch.load(directory / "soft_tpr_vectors/syntactic_vectors.pt", weights_only=True) for directory in (before, after)]
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


@pytest.mark.parametrize("problem", ["val", "nonfinite", "schema"])
def test_incomplete_or_invalid_training_input(activation_run, tmp_path, problem):
    if problem == "schema":
        path = activation_run / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["schema_version"] = 2
        path.write_text(json.dumps(metadata))
    else:
        data = torch.load(activation_run / "activations.pt", weights_only=True)
        for i, row in enumerate(data["rows"]):
            if row["split"] == "val":
                if problem == "val":
                    row["split"] = "test"
                else:
                    data["activations"][i] = float("nan")
        torch.save(data, activation_run / "activations.pt")
        path = activation_run / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["sentences_by_split"] = dict(Counter(row["split"] for row in data["rows"]))
        path.write_text(json.dumps(metadata))
    output = tmp_path / "failed"
    with pytest.raises(ValueError):
        train_soft_tpr_vectors(activation_run, output, **SMALL, **FIT)
    assert not (output / "metadata.json").exists()


def test_tiny_model_training_steering_and_comparison(tmp_path, tokenizer, model, model_metadata, monkeypatch):
    rows = generate_rows(["boy", "girl", "man"], [{"lemma": "see", "past": "saw", "participle": "seen"}], [])
    for i, row in enumerate(rows):
        row["split"] = "train" if i < 4 else "val" if i < 6 else "test" if i < 8 else "gen_test"
    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    ids, positions = tokenize_roles(rows, tokenizer)
    states = extract_batch(model, tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt"), positions)
    run = tmp_path / "activations"
    run.mkdir()
    torch.save({"rows": rows, "activations": states}, run / "activations.pt")
    metadata = {
        "schema_version": 3, "hook_names": [f"blocks.{layer}.hook_resid_post" for layer in range(2)],
        "corpus": str(corpus), "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "model": model_metadata, "tokenizer": {
            "name": model_metadata["name"], "revision": model_metadata["revision"],
            "backend_sha256": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
            "add_special_tokens": False, "padding_side": "right", "pad_token_id": tokenizer.pad_token_id,
        },
        "activation_site": "Transformer block output, before final_layer_norm",
        "layer_indices": [0, 1], "roles": ["agent", "patient"],
        "axes": ["sentence", "layer", "role", "hidden"], "shape": list(states.shape),
        "sentences_by_split": dict(Counter(row["split"] for row in rows)),
        "sentences_by_voice": dict(Counter(row["voice"] for row in rows)),
    }
    (run / "metadata.json").write_text(json.dumps(metadata))
    experiment = tmp_path / "experiment"
    train_soft_tpr_vectors(run, experiment, **SMALL, **FIT)
    monkeypatch.setattr("ac_tpr.interventions.load_model", lambda *a, **kw: (model, model_metadata))
    monkeypatch.setattr("ac_tpr.interventions.AutoTokenizer.from_pretrained", lambda *a, **kw: tokenizer)
    for vectors, output in (("role_means", "raw"), ("soft_tpr_vectors", "soft")):
        result = run_interventions(corpus, experiment / vectors, tmp_path / output, device="cpu", batch_size=3)
        assert result["controls"] == {"zero_vector_first_batch": "passed", "final_block_all_rows": "passed"}
    comparison = compare_soft_tpr_runs(tmp_path / "raw", tmp_path / "soft", tmp_path / "comparison")
    assert comparison["groups"] == 32
    with (tmp_path / "comparison/comparison.csv").open() as handle:
        comparisons = list(csv.DictReader(handle))
    with (tmp_path / "raw/summary.csv").open() as handle:
        raw_rows = {(r["split"], r["voice"], r["layer"], r["mean_type"], r["edited_role"]): r for r in csv.DictReader(handle)}
    with (tmp_path / "soft/summary.csv").open() as handle:
        soft_rows = {(r["split"], r["voice"], r["layer"], r["mean_type"], r["edited_role"]): r for r in csv.DictReader(handle)}
    for row in comparisons:
        key = tuple(row[k] for k in ("split", "voice", "layer", "mean_type", "edited_role"))
        expected = float(soft_rows[key]["mean_logit_diff_change"]) - float(raw_rows[key]["mean_logit_diff_change"])
        assert float(row["difference_mean_logit_diff_change"]) == pytest.approx(expected)
        if row["layer"] == "1":
            assert float(row["difference_mean_logit_diff_change"]) == 0
    path = tmp_path / "soft/metadata.json"
    metadata = json.loads(path.read_text())
    metadata["execution_model"]["dtype"] = "bfloat16"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="execution_model"):
        compare_soft_tpr_runs(tmp_path / "raw", tmp_path / "soft", tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
