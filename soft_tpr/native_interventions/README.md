# Qwen3 native TPR interventions: all 36 layers

Start here for the TPR probe, Soft TPR, and raw-centroid comparison on
Qwen3-8B-Base. This folder includes the saved results and the experiment sources.
The reports compare five methods on the same 2,096 held-out sentences:
raw centroids, native TPR role swaps, native Soft TPR slot swaps, and the two
reconstructed-centroid interventions. Both nouns are edited together at one
layer, with alpha = 1 and unmatched perturbation norms.

## Open the reports

- [Compare all methods](notebook/role_interventions_comparison_qwen3.ipynb)
- [TPR probe](notebook/role_interventions_tpr_qwen3.ipynb)
- [Soft TPR](notebook/role_interventions_soft_tpr_qwen3.ipynb)

Saved outputs can be read immediately. To rerun every report cell, change filters,
or redraw plots, use the CPU setup below. No model download or GPU is needed.

From the main repository root:

```bash
cd soft_tpr/native_interventions
python3 -m venv .venv-tpr-reports
source .venv-tpr-reports/bin/activate
python -m pip install -r notebook/requirements-tpr-reports.txt
python scripts/temporary/prepare_tpr_notebook_data.py
python -m ipykernel install --user --name ac-tpr-reports --display-name "AC-TPR reports (CPU)"
```

Select **AC-TPR reports (CPU)** in your notebook editor and run the cells in order.
Keep `PREPARE_CENTROIDS = False` and `RUN_CENTROID_EVALUATION = False`.
Figures stay inline; the reports write CSV tables and JSON provenance, but do not
export image files. Generated caches and table exports are ignored by Git.

## Regenerate the experiment

The complete [setup and regeneration guide](notebook/README_tpr.md) explains the
GPU environment and the stages. Regeneration is a separate, explicit command:

```bash
python scripts/temporary/regenerate_tpr_notebook_data.py --dry-run
python scripts/temporary/regenerate_tpr_notebook_data.py
```

Run these commands from this folder in a configured GPU environment. The second
command downloads the pinned Qwen model and repeats extraction, training,
calibration, and evaluation; it can take hours. It writes a new run, separate
from the bundled report data. GPU regeneration was not repeated while preparing
this package; the launcher calls the original experiment stages.

The `src/` and `scripts/temporary/` files here preserve the versions used by this
experiment. Their hashes are recorded in the run metadata. Use these copies when
regenerating this experiment. Model weights and cached activations are not
included in Git; the ready-results archive contains only nine CSV/JSON files.

The three linked `native_*` walkthroughs describe the earlier layers 20–22 run.
They include outputs for reading, but rerunning their checkpoint-based examples
requires that older run's artifacts. The bundled data support the three all-layer
reports linked above.

## Message for colleagues

> The all-layer Qwen TPR comparison is in `soft_tpr/native_interventions/`.
> Start with its README and the comparison notebook. The saved data are included:
> after installing the CPU dependencies and running `prepare_tpr_notebook_data.py`,
> you can run the reports and change subgroup filters without a GPU. Leave the
> centroid recomputation flags off. The optional regeneration launcher repeats
> the full GPU experiment. Plots are shown inline and are not exported as images.
