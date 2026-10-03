"""Locate nonfinite Pythia activations in native and bridged FP16 forwards."""

import json
from pathlib import Path
from types import MethodType

import torch
from transformer_lens.model_bridge import TransformerBridge
from transformer_lens.model_bridge.generalized_components.attention import AttentionBridge
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION
from ac_tpr.interventions import prepare_prompts


def describe(name, value):
    print(name, str(value.dtype), tuple(value.shape), "finite", torch.isfinite(value).all().item(),
          "max", value.abs().max().item(), flush=True)


tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.init_kwargs["revision"] = TOKENIZER_REVISION
reference = torch.load("ac-tpr-cache/migration-reference-tl4/pythia/activations.pt", weights_only=True)
metadata = json.loads(Path("ac-tpr-cache/migration-reference-tl4/pythia/metadata.json").read_text())
ids, positions, answers = prepare_prompts(reference["rows"], tokenizer)
inputs = tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt").to("cuda")
model = GPTNeoXForCausalLM.from_pretrained(
    TOKENIZER_ID, revision=metadata["model"]["commit"], dtype=torch.float16,
    attn_implementation="sdpa",
).to("cuda").eval()
handles = [block.register_forward_hook(lambda module, args, out, i=i: describe(f"HF block {i}", out))
           for i, block in enumerate(model.gpt_neox.layers)]
with torch.inference_mode():
    native_logits = model(**inputs, use_cache=False).logits
    describe("HF logits", native_logits)
for handle in handles:
    handle.remove()
bridge = TransformerBridge.boot_transformers(
    TOKENIZER_ID, hf_model=model, tokenizer=tokenizer, device="cuda", dtype=torch.float16,
    hf_config_overrides={"attn_implementation": "sdpa", "output_attentions": False, "use_cache": False},
).eval()
print("ATTENTION", bridge.cfg.attn_implementation, bridge.original_model.config._attn_implementation, flush=True)
with torch.inference_mode():
    result = bridge.run_with_hooks(
        inputs["input_ids"], attention_mask=inputs["attention_mask"], prepend_bos=False, use_cache=False,
        fwd_hooks=[(name, lambda x, hook: describe(hook.name, x)) for name in bridge.hook_dict
                   if name in {"embed.hook_out", "ln_final.hook_out"}
                   or name in {f"blocks.{i}.hook_out" for i in range(bridge.cfg.n_layers)}],
    )
    describe("Bridge logits", result)
    for block in bridge.blocks:
        object.__setattr__(block.attn, "forward", MethodType(AttentionBridge.forward, block.attn))
    result = bridge(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                    prepend_bos=False, use_cache=False)
    describe("Bridge with native attention logits", result)
    print("Native attention maximum logit difference", (result - native_logits).abs().max().item())
