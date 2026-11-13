"""Exercise cached DISCOVER stages with a tiny frozen Qwen model on CPU."""

import copy
import csv
import json
import math

import pytest
import torch

from ac_tpr.dataset import CORPUS_COLUMNS, generate_rows
from ac_tpr.prompts import TASK_TEMPLATE
from ac_tpr.tpr import extraction, pipeline
from ac_tpr.tpr.decomposition import TPRConfig
from ac_tpr.tpr.training import TrainingConfig, model_directory
from ac_tpr.tpr.unpacking import sentence_tensors, UnpackingConfig, unpacking_loss


@pytest.fixture
def tiny_corpus(tmp_path):
    rows = generate_rows(
        ["boy", "girl", "dog", "cat"],
        [{"lemma": "help", "past": "helped", "participle": "helped"}],
        [],
    )
    train_pairs = {frozenset(("boy", "girl")), frozenset(("dog", "cat"))}
    val_pairs = {frozenset(("boy", "dog")), frozenset(("girl", "cat"))}
    for row in rows:
        pair = frozenset((row["agent"], row["patient"]))
        row["split"] = "train" if pair in train_pairs else "val" if pair in val_pairs else "test"
    corpus = tmp_path / "corpus.csv"
    with corpus.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CORPUS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return corpus, rows


@pytest.fixture
def tiny_backend(monkeypatch, tokenizer, model, model_metadata):
    monkeypatch.setattr(extraction.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
    monkeypatch.setattr(extraction, "load_model", lambda *args, **kwargs: (model, model_metadata))
    monkeypatch.setattr("ac_tpr.tpr.steering.load_model", lambda *args, **kwargs: (model, model_metadata))
    return model


def test_training_statistics_exclude_validation_and_heldout_original_training():
    values = torch.tensor([1.0, 3.0, 1000.0, -1000.0])[:, None, None, None].expand(4, 2, 2, 3).clone()
    values[:, :, :, 2] = 7.0
    rows = [
        {"split": "train", "tpr_split": "train"},
        {"split": "train", "tpr_split": "train"},
        {"split": "val", "tpr_split": "val"},
        {"split": "train", "tpr_split": "binding_test"},
    ]
    mean, scale = extraction.training_statistics(values, rows)
    assert mean.shape == scale.shape == (2, 2, 3)
    assert mean.dtype == scale.dtype == torch.float32
    torch.testing.assert_close(mean[:, :, :2], torch.full((2, 2, 2), 2.0))
    torch.testing.assert_close(scale[:, :, :2], torch.ones(2, 2, 2))
    torch.testing.assert_close(mean[:, :, 2], torch.full((2, 2), 7.0))
    torch.testing.assert_close(scale[:, :, 2], torch.full((2, 2), 1e-10))
    with pytest.raises(ValueError, match="training examples"):
        extraction.training_statistics(values[2:], rows[2:])


def test_extraction_cache_preserves_site_and_requested_layer_order(
    tmp_path, tiny_corpus, tiny_backend, hf_model,
):
    corpus, rows = tiny_corpus
    directory = tmp_path / "encodings"
    metadata = extraction.extract_encodings(
        corpus, directory, layers=(1, 0), batch_size=5, device="cpu", dtype="float32",
        prompt_template=TASK_TEMPLATE, held_out_bindings=(),
    )
    cache = extraction.load_cache(directory)
    assert metadata == cache["metadata"]
    assert metadata["shape"] == [len(rows), 2, 2, 32]
    assert metadata["layer_indices"] == [1, 0]
    assert metadata["sites"] == ["sentence_end", "prompt_end"]
    assert cache["activations"].device.type == "cpu"
    assert cache["positions"][:, 1].tolist() == [len(ids) - 1 for ids in cache["input_ids"]]
    assert (cache["positions"][:, 0] < cache["positions"][:, 1]).all()
    for index in (0, len(rows) - 1):
        before_norm = []
        handle = hf_model.model.norm.register_forward_pre_hook(
            lambda _module, args: before_norm.append(args[0].clone())
        )
        try:
            with torch.inference_mode():
                output = hf_model.model(
                    input_ids=torch.tensor([cache["input_ids"][index]]),
                    use_cache=False, output_hidden_states=True,
                )
        finally:
            handle.remove()
        positions = cache["positions"][index]
        expected = torch.stack([
            before_norm[0][0, positions], output.hidden_states[1][0, positions],
        ])
        torch.testing.assert_close(cache["activations"][index], expected, atol=1e-6, rtol=1e-5)
    train = [i for i, row in enumerate(cache["rows"]) if row["tpr_split"] == "train"]
    torch.testing.assert_close(cache["mean"], cache["activations"][train].mean(dim=0))
    with pytest.raises(FileExistsError):
        extraction.extract_encodings(corpus, directory)
    corrupted = copy.deepcopy(cache)
    corrupted["positions"][0, 1] -= 1
    with pytest.raises(ValueError, match="boundary"):
        extraction.validate_cache(corrupted)


def test_interrupted_extraction_is_not_a_completed_cache(
    tmp_path, tiny_corpus, tiny_backend, monkeypatch,
):
    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted extraction")

    monkeypatch.setattr(extraction, "extract_batch", interrupted)
    directory = tmp_path / "incomplete"
    corpus, _ = tiny_corpus
    with pytest.raises(RuntimeError, match="interrupted extraction"):
        extraction.extract_encodings(
            corpus, directory, layers=(0,), device="cpu", dtype="float32",
            prompt_template=TASK_TEMPLATE, held_out_bindings=(),
        )
    assert directory.exists()
    assert not (directory / "metadata.json").exists()
    with pytest.raises(FileNotFoundError):
        extraction.load_cache(directory)
    with pytest.raises(FileExistsError):
        extraction.extract_encodings(corpus, directory)


def test_teacher_forcing_shift_supervises_sentence_and_eos_only():
    inputs, targets = sentence_tensors([[3, 4], [5]])
    assert inputs.tolist() == [[1, 3, 4], [1, 5, 0]]
    assert targets.tolist() == [[3, 4, 2], [5, 2, 0]]

    def uniform_logits(z, actual_inputs):
        assert torch.equal(actual_inputs, inputs)
        return torch.zeros(*actual_inputs.shape, 8)

    loss, count = unpacking_loss(uniform_logits, (torch.zeros(2, 32), inputs, targets))
    assert count == 5
    assert loss.item() == pytest.approx(5 * math.log(8))


def test_cpu_pipeline_trains_saves_reconstructs_and_steers(
    tmp_path, tiny_corpus, tiny_backend,
):
    corpus, source_rows = tiny_corpus
    output = tmp_path / "run"
    training = TrainingConfig(batch_size=4, learning_rate=1e-3, epochs=1, eval_every=1000)
    results = pipeline.run_pipeline(
        corpus, output, layers=(0, 1), held_out_bindings=(),
        unpacking=UnpackingConfig(hidden_size=16, n_layers=1, n_heads=2, ffn_size=32,
                                  dropout=0.0, max_length=12),
        tpr=TPRConfig(filler_dim=4, role_dim=3),
        unpacker_training=training, tpr_training=training,
        device="cpu", dtype="float32", batch_size=5, prompt_template=TASK_TEMPLATE,
    )
    assert set(results) == {"extract", "unpack", "fit", "reconstruct", "steer"}
    cache = extraction.load_cache(output / "encodings")
    fingerprint = extraction.cache_fingerprint(cache)
    assert results["unpack"]["cache_fingerprint"] == results["fit"]["cache_fingerprint"] == fingerprint
    assert len(results["unpack"]["models"]) == len(results["fit"]["models"]) == 4
    models = pipeline.load_tprs(output / "tprs", cache)
    assert set(models) == {(layer, site) for layer in (0, 1) for site in ("sentence_end", "prompt_end")}
    for key, model in models.items():
        assert not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        assert model(cache["filler_ids"][:2], cache["role_ids"][:2]).shape == (2, 32)
        for root in (output / "unpackers", output / "tprs"):
            history = json.loads((model_directory(root, *key) / "history.json").read_text())
            assert history["epochs"] == 1
            assert len(history["evaluations"]) == 1
            assert math.isfinite(history["best_validation_loss"])
    with (output / "reconstruction/results.csv").open(newline="") as handle:
        reconstructions = list(csv.DictReader(handle))
    n_test = sum(row["split"] == "test" for row in source_rows)
    assert len(reconstructions) == n_test * 4 * 3
    assert {row["condition"] for row in reconstructions} == {"real", "tpr", "swap"}
    for row in reconstructions:
        assert math.isfinite(float(row["activation_mse"]))
        if row["condition"] == "real":
            assert float(row["activation_mse"]) == 0.0
        assert int(row["correct_tokens"]) <= int(row["target_tokens"])
    with (output / "steering/results.csv").open(newline="") as handle:
        steering = list(csv.DictReader(handle))
    assert len(steering) == n_test * 4 * 4
    assert {row["condition"] for row in steering} == {"zero", "reconstruction", "swap", "donor_swap"}
    for row in steering:
        baseline = float(row["baseline_logit_diff"])
        intervened = float(row["intervened_logit_diff"])
        assert math.isfinite(intervened)
        assert float(row["logit_diff_change"]) == pytest.approx(intervened - baseline)
        if row["condition"] == "zero" or (row["layer"] == "1" and row["site"] == "sentence_end"):
            assert intervened == baseline
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(corpus, output)
    manifest_path = output / "tprs/metadata.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cache_fingerprint"] = "a different extraction"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="do not match this extraction"):
        pipeline.load_tprs(output / "tprs", cache)
