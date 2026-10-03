"""Load the experiment's unprocessed Pythia checkpoint with TransformerLens."""

from importlib.metadata import version

import torch
from transformer_lens import HookedTransformer
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION

SCHEMA_VERSION = 3
BLOCK_HOOK = "blocks.{layer}.hook_resid_post"


def load_model(
    tokenizer: PreTrainedTokenizerFast, revision: str, device: str, dtype: str,
) -> tuple[HookedTransformer, dict]:
    """Load raw weights at a resolved commit and return their execution provenance."""
    if dtype == "float16":
        raise ValueError(
            "Pythia HookedTransformer attention can overflow in float16; use float32"
        )
    config = AutoConfig.from_pretrained(TOKENIZER_ID, revision=revision)
    commit = config._commit_hash
    if not commit:
        raise ValueError("Model loading requires a resolved checkpoint commit")
    # Tokenizer reconstruction must use the cached revision.
    tokenizer.init_kwargs["revision"] = TOKENIZER_REVISION
    hf_model = AutoModelForCausalLM.from_pretrained(
        TOKENIZER_ID, revision=commit, dtype=getattr(torch, dtype),
        attn_implementation="eager",
    ).eval()
    if hf_model.config._commit_hash != commit:
        raise ValueError("Loaded model commit does not match the requested checkpoint")
    model = HookedTransformer.from_pretrained_no_processing(
        TOKENIZER_ID, revision=commit, hf_model=hf_model, tokenizer=tokenizer,
        device=device, dtype=getattr(torch, dtype), n_devices=1,
        default_prepend_bos=False, default_padding_side="right",
    ).eval()
    model.requires_grad_(False)
    provenance = {
        "name": TOKENIZER_ID, "revision": TOKENIZER_REVISION, "commit": commit,
        "dtype": str(model.cfg.dtype).removeprefix("torch."),
        "attention_implementation": "hooked_transformer_eager",
        "backend": "transformer_lens", "api": "HookedTransformer",
        "transformer_lens_version": version("transformer-lens"),
        "weight_processing": "none",
    }
    return model, provenance


def validate_activation_provenance(source: dict) -> None:
    """Require raw HookedTransformer block outputs from a schema-3 extraction."""
    if source.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Regenerate activations and role vectors with TransformerLens 3 HookedTransformer (schema_version 3)")
    model = source["model"]
    if (
        model["backend"] != "transformer_lens"
        or model["api"] != "HookedTransformer"
        or model["weight_processing"] != "none"
        or model["attention_implementation"] != "hooked_transformer_eager"
        or model["transformer_lens_version"] != version("transformer-lens")
    ):
        raise ValueError("Regenerate artifacts using the installed HookedTransformer version with raw weights")
    if source["hook_names"] != [BLOCK_HOOK.format(layer=layer) for layer in source["layer_indices"]]:
        raise ValueError("Source hook names must identify the ordered block outputs")
