"""Check checkpoint pinning, offline tokenizer setup, and artifact provenance."""

import copy

import pytest
import torch

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.model import load_model, validate_activation_provenance


def test_loader_pins_commit_and_preserves_input_tokenizer(tokenizer, model, hf_model, monkeypatch):
    calls = []
    config = hf_model.config
    before = tokenizer.backend_tokenizer.to_str()

    def boot(name, **kwargs):
        calls.append((name, kwargs))
        return model

    monkeypatch.setattr("ac_tpr.model.AutoConfig.from_pretrained", lambda *a, **kw: config)
    monkeypatch.setattr("ac_tpr.model.HookedTransformer.from_pretrained_no_processing", boot)
    actual, provenance = load_model(tokenizer, TOKENIZER_REVISION, "cpu", "float32")
    assert actual is model
    assert calls[0][0] == TOKENIZER_ID
    assert calls[0][1]["revision"] == "test-commit"
    assert calls[0][1]["dtype"] == torch.float32
    assert calls[0][1]["hf_model"].config._commit_hash == "test-commit"
    assert calls[0][1]["default_prepend_bos"] is False
    assert calls[0][1]["default_padding_side"] == "right"
    assert calls[0][1]["n_devices"] == 1
    assert tokenizer.init_kwargs["revision"] == TOKENIZER_REVISION
    assert tokenizer.backend_tokenizer.to_str() == before
    assert provenance["commit"] == "test-commit"
    assert provenance["attention_implementation"] == "hooked_transformer_eager"
    assert provenance["api"] == "HookedTransformer"
    assert provenance["weight_processing"] == "none"
    assert not model.training
    assert not model.cfg.default_prepend_bos
    assert all(not parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("schema", [None, 1, 2, 4])
def test_old_or_unknown_schema_requires_regeneration(schema):
    source = {} if schema is None else {"schema_version": schema}
    with pytest.raises(ValueError, match="Regenerate"):
        validate_activation_provenance(source)


def test_pythia_float16_is_rejected_before_loading(tokenizer):
    with pytest.raises(ValueError, match="overflow in float16"):
        load_model(tokenizer, TOKENIZER_REVISION, "cpu", "float16")


def test_attention_internals_remain_hookable(model):
    captured = {}

    def capture(value, hook):
        captured[hook.name] = value.shape

    names = [f"blocks.0.attn.hook_{name}" for name in ("q", "k", "v", "attn_scores", "pattern")]
    model.run_with_hooks(torch.tensor([[1, 2, 3]]), prepend_bos=False,
                         fwd_hooks=[(name, capture) for name in names])
    assert set(captured) == set(names)
    assert captured["blocks.0.attn.hook_q"] == (1, 3, 4, 8)
    assert captured["blocks.0.attn.hook_pattern"] == (1, 4, 3, 3)


@pytest.mark.parametrize("field,value", [
    ("backend", "transformers"), ("api", "TransformerBridge"),
    ("weight_processing", "centered"), ("transformer_lens_version", "4.0.0"),
    ("attention_implementation", "sdpa"),
])
def test_incompatible_processing_is_rejected(model_metadata, field, value):
    source = {"schema_version": 3, "model": copy.deepcopy(model_metadata),
              "layer_indices": [0, 1], "hook_names": ["blocks.0.hook_resid_post", "blocks.1.hook_resid_post"]}
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
        actual = model(tokens, attention_mask=mask, prepend_bos=False)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
