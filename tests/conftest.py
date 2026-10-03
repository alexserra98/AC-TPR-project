"""Small matched Hugging Face and TransformerLens models for CPU checks."""

import copy
from importlib.metadata import version

import pytest
import torch
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.model import load_model


@pytest.fixture(scope="module")
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.init_kwargs["revision"] = TOKENIZER_REVISION
    return tokenizer


@pytest.fixture
def hf_model(tokenizer):
    torch.manual_seed(123)
    config = GPTNeoXConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=32,
        hidden_dropout=0.0, attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    config._commit_hash = "test-commit"
    config.architectures = ["GPTNeoXForCausalLM"]
    return GPTNeoXForCausalLM(config).eval()


@pytest.fixture
def model(tokenizer, hf_model, monkeypatch):
    monkeypatch.setattr("ac_tpr.model.AutoConfig.from_pretrained", lambda *a, **kw: hf_model.config)
    monkeypatch.setattr("ac_tpr.model.AutoModelForCausalLM.from_pretrained", lambda *a, **kw: copy.deepcopy(hf_model))
    model, _ = load_model(tokenizer, TOKENIZER_REVISION, "cpu", "float32")
    return model


@pytest.fixture
def model_metadata():
    return {
        "name": TOKENIZER_ID, "revision": TOKENIZER_REVISION, "commit": "test-commit",
        "dtype": "float32", "attention_implementation": "hooked_transformer_eager",
        "backend": "transformer_lens", "api": "HookedTransformer",
        "transformer_lens_version": version("transformer-lens"), "weight_processing": "none",
    }
