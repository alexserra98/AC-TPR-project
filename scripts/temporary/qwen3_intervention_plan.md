# Qwen3-8B role-intervention experiment

## Experimental setup
- Use `Qwen/Qwen3-8B`, pinned to cached revision `b968826d9c46dd6066d109eabc6255188de91218`, with BF16 weights and eager attention.
- Apply Qwen's native chat template with `enable_thinking=False` and `add_generation_prompt=True`.
- Use this exact user message:
  ```text
  {sentence}
  Who is performing the action? Answer only with the noun identifying that person.
  ```
- Keep the existing corpus and splits. Compute role means from training rows using this same chat context; evaluate all `test` and `gen_test` rows.

## Simple logit evaluation
- Tokenize `" " + agent` and `" " + patient` as candidate answers. Assert each is exactly one token and appends cleanly to the tokenized prompt.
- Read both candidate logits at the last non-padding prompt position.
- Keep `logit_diff = patient_logit - agent_logit`: negative favors the agent; positive favors the patient.
- Preserve existing tie handling and flip-rate definitions. No generated-answer evaluation or multi-token scoring.

## Interventions and role positions
- Extract residual states after each of Qwen's 36 transformer blocks using NNsight.
- Compute float32 pooled and per-voice agent/patient means from training rows.
- Preserve `h <- h + mean_other_role - mean_edited_role` at strength 1.
- Evaluate agent and patient edits separately, for every layer and both mean types.
- Locate role tokens through character offsets in the fully rendered chat prompt, restricted to the sentence span. For every row, assert each role occupies one token whose decoded text matches the expected noun after stripping whitespace.
- Use those verified indices for extraction and intervention, with right padding so padding does not shift them.
- Retain existing zero-vector and final-block controls. Additional validation focuses on chat-template role indexing.

## Implementation, execution, and results
- Keep the experiment isolated in `scripts/temporary`, with a launcher in `scripts/slurm/temporary`. Reuse existing role-delta and summary helpers without introducing a general model framework.
- Save Qwen activations, vectors, results, and model/prompt provenance in separate Qwen-specific directories; preserve Pythia artifacts and the current result CSV schema and metrics.
- Run an eight-example smoke experiment covering both voices and evaluation splits, then the full experiment on one A100: batch size 8, account `lade`, DGX partition, 96 GB host memory, 12-hour limit.
- Create `notebook/role_interventions_qwen3.ipynb` as a copy of `notebook/role_interventions.ipynb`. Change only artifact paths, model-specific text, and layer ticks; retain its tables, plots, and analysis.
- Execute the analysis notebook after completion and report baseline agent preference and intervention effects.

## Entry points and artifacts
- Validate indices without loading weights: `.venv/bin/python scripts/temporary/run_qwen3_interventions.py --check-indices` (using the shared Hugging Face cache).
- Submit from the repository root: `sbatch scripts/slurm/temporary/run_qwen3_interventions.sh`.
- Artifacts: `ac-tpr-cache/qwen3-8b-generated-agent-chat-nnsight/{activations,vectors,smoke,interventions}`.
- Figures: `notebook/figures/role_interventions_qwen3`.
- Existing output directories are rejected to prevent accidental overwrite.
