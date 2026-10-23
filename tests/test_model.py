"""Check checkpoint pinning, offline tokenizer setup, and artifact provenance."""

import copy

import pytest
import torch

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.model import load_model, validate_activation_provenance


def test_loader_pins_commit_and_preserves_input_tokenizer(tokenizer, hf_model, monkeypatch):
    calls = []
    before = tokenizer.backend_tokenizer.to_str()

    def boot(name, **kwargs):
        calls.append((name, kwargs))
        return copy.deepcopy(hf_model)

    monkeypatch.setattr("ac_tpr.model.AutoConfig.from_pretrained", lambda *a, **kw: hf_model.config)
    monkeypatch.setattr("ac_tpr.model.AutoModelForCausalLM.from_pretrained", boot)
    model, provenance = load_model(tokenizer, TOKENIZER_REVISION, "cpu", "float32")
    assert calls == [(TOKENIZER_ID, {"revision": "test-commit", "dtype": torch.float32,
                                  "attn_implementation": "eager"})]
    assert model.tokenizer is tokenizer
    assert tokenizer.backend_tokenizer.to_str() == before
    assert tokenizer.padding_side == "right"
    assert provenance["commit"] == "test-commit"
    assert provenance["attention_implementation"] == "eager"
    assert provenance["api"] == "LanguageModel"
    assert provenance["weight_processing"] == "none"
    assert model.config is model._model.config
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("schema", [None, 1, 2, 3, 5])
def test_old_or_unknown_schema_requires_regeneration(schema):
    source = {} if schema is None else {"schema_version": schema}
    with pytest.raises(ValueError, match="Regenerate"):
        validate_activation_provenance(source)


def test_pythia_float16_is_rejected_before_loading(tokenizer):
    with pytest.raises(ValueError, match="overflow in float16"):
        load_model(tokenizer, TOKENIZER_REVISION, "cpu", "float16")


@pytest.mark.parametrize("field,value", [
    ("backend", "transformers"), ("api", "TransformerBridge"),
    ("weight_processing", "centered"), ("nnsight_version", "0.0.0"), ("transformers_version", "0.0.0"),
    ("attention_implementation", "sdpa"),
])
def test_incompatible_processing_is_rejected(model_metadata, field, value):
    source = {"schema_version": 4, "model": copy.deepcopy(model_metadata),
              "layer_indices": [0, 1], "hook_names": ["gpt_neox.layers.0.output", "gpt_neox.layers.1.output"]}
    source["model"][field] = value
    with pytest.raises(ValueError, match="raw weights"):
        validate_activation_provenance(source)


def test_float32_attention_handles_large_queries_and_keys(hf_model, tokenizer, model):
    with torch.no_grad():
        for layer in hf_model.gpt_neox.layers:
            layer.attention.query_key_value.weight.view(4, 3, 8, 32)[:, :2].zero_()
            layer.attention.query_key_value.bias.view(4, 3, 8)[:, :2].fill_(100)
    model, _ = load_model(tokenizer, TOKENIZER_REVISION, "cpu", "float32")
    tokens = torch.tensor([[1, 2, 3], [1, 4, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
    with torch.inference_mode():
        expected = hf_model(tokens, attention_mask=mask, use_cache=False).logits
        with model.trace({"input_ids": tokens, "attention_mask": mask}, use_cache=False):
            actual = model.output.logits.save()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
