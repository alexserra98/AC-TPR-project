"""Load the experiment's raw Qwen3-8B-Base checkpoint for local NNsight traces."""

from importlib.metadata import version

import torch
from nnsight import LanguageModel
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION

SCHEMA_VERSION = 5
BLOCK_HOOK = "model.layers.{layer}.output"


def load_model(
    tokenizer: PreTrainedTokenizerFast, revision: str, device: str, dtype: str,
) -> tuple[LanguageModel, dict]:
    """Load raw weights at a resolved commit and return their execution provenance."""
    if dtype not in {"bfloat16", "float32"}:
        raise ValueError("Use bfloat16 or float32 for Qwen3-8B-Base")
    config = AutoConfig.from_pretrained(TOKENIZER_ID, revision=revision)
    if config.model_type != "qwen3":
        raise ValueError("Expected a Qwen3 checkpoint")
    commit = config._commit_hash
    if not commit:
        raise ValueError("Model loading requires a resolved checkpoint commit")
    if commit != TOKENIZER_REVISION:
        raise ValueError("Expected the pinned Qwen3-8B-Base commit")
    hf_model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_ID, revision=commit, dtype=getattr(torch, dtype),
        attn_implementation="eager",
    ).eval()
    if hf_model.config._commit_hash != commit:
        raise ValueError("Loaded model commit does not match the requested checkpoint")
    hf_model.to(device).requires_grad_(False)
    model = LanguageModel(hf_model, tokenizer=tokenizer, config_model=hf_model.config)
    provenance = {
        "name": TOKENIZER_ID, "revision": TOKENIZER_REVISION, "commit": commit,
        "dtype": str(hf_model.dtype).removeprefix("torch."),
        "attention_implementation": hf_model.config._attn_implementation,
        "backend": "nnsight", "api": "LanguageModel",
        "nnsight_version": version("nnsight"),
        "transformers_version": version("transformers"),
        "weight_processing": "none",
    }
    return model, provenance


def validate_activation_provenance(source: dict) -> None:
    """Require raw block outputs from a matching schema-5 NNsight extraction."""
    if source.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Regenerate activations and role vectors with NNsight LanguageModel (schema_version 5)")
    for component in ("model", "tokenizer"):
        if source[component]["name"] != TOKENIZER_ID or source[component]["revision"] != TOKENIZER_REVISION:
            raise ValueError("Regenerate artifacts with the pinned Qwen3-8B-Base model and tokenizer")
    if source["model"]["commit"] != TOKENIZER_REVISION:
        raise ValueError("Regenerate artifacts with the pinned Qwen3-8B-Base commit")
    if not isinstance(source["prompt_template"], str) or source["prompt_template"].count("{sentence}") != 1:
        raise ValueError("Source prompt template must contain exactly one {sentence}")
    model = source["model"]
    if (
        model["backend"] != "nnsight"
        or model["api"] != "LanguageModel"
        or model["weight_processing"] != "none"
        or model["attention_implementation"] != "eager"
        or model["nnsight_version"] != version("nnsight")
        or model["transformers_version"] != version("transformers")
    ):
        raise ValueError("Regenerate artifacts using the installed NNsight and Transformers versions with raw weights and eager attention")
    if source["hook_names"] != [BLOCK_HOOK.format(layer=layer) for layer in source["layer_indices"]]:
        raise ValueError("Source hook names must identify the ordered block outputs")
