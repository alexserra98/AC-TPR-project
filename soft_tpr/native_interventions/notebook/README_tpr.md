# Run the TPR reports

The ready-to-use data cover all 36 Qwen3-8B-Base layers, 2,096 held-out sentences,
and all five methods. These three reports can run on CPU:

- `role_interventions_comparison_qwen3.ipynb`
- `role_interventions_tpr_qwen3.ipynb`
- `role_interventions_soft_tpr_qwen3.ipynb`

The archive in `data/tpr_notebook_results/` contains nine CSV/JSON files, compressed
from about 54 MiB to 4.8 MiB. It includes the baseline, native interventions,
reconstructed-centroid interventions, and their metadata. The corpus is already
in `data/active_pass_long/`. No model weights, cached activations, or image files
are included in this archive.

## Use the saved results

Run the following from `soft_tpr/native_interventions/` in the shared repository. Use Python 3.11
or newer; the reports were validated with Python 3.12. The report environment is
separate from the experiment dependencies in `pyproject.toml`.

```bash
python3 -m venv .venv-tpr-reports
source .venv-tpr-reports/bin/activate
python -m pip install -r notebook/requirements-tpr-reports.txt
python scripts/temporary/prepare_tpr_notebook_data.py
python -m ipykernel install --user --name ac-tpr-reports --display-name "AC-TPR reports (CPU)"
```

Open one of the three notebooks in your notebook editor, choose **AC-TPR reports
(CPU)**, and run all cells. Start the kernel in `soft_tpr/native_interventions/`
or its `notebook/` directory. Leave
`PREPARE_CENTROIDS = False` and `RUN_CENTROID_EVALUATION = False`. You can change
`FILTERS` to analyze a subgroup and rerun the cells below it.

The installer validates the archive, corpus, and source hashes, then places the
results at the path already used by the notebooks. It leaves matching files in
place and refuses to overwrite different results. It is safe to run again.

For a browser-based editor, install and launch JupyterLab in the same environment:

```bash
python -m pip install jupyterlab
python -m jupyterlab notebook/
```

Figures are displayed inline. The notebooks export CSV tables and JSON provenance
under `notebook/figures/`, but do not write separate plot image files.

The linked `native_*` notebooks document the earlier layers 20–22 experiment and
include saved outputs. Executing those walkthroughs also needs that older run's
checkpoints and cached activations, which are not in this report bundle.

## Regenerate all data

This is a full model experiment: extraction, TPR and Soft TPR training, slot
calibration, three native conditions, and two reconstructed-centroid conditions
on all 36 layers. It needs Linux, a CUDA GPU supporting BF16, room for Qwen3-8B-Base
plus the fixed batch size of 32, and disk space for weights, activations, and
checkpoints. Allow hours of computation. Running the reports above does not
start this experiment.

Use a separate GPU environment. Install CUDA-enabled PyTorch appropriate to the
machine, then install the other experiment dependencies:

```bash
python -m pip install 'transformers==5.9.0' 'nnsight==0.7.0' numpy psutil
```

The saved experiment used Python 3.12.3, PyTorch 2.13.0+cu130, CUDA 13.0,
Transformers 5.9.0, and NNsight 0.7.0. These versions are also recorded in the bundle
manifest. The project's older PyTorch pin is not the environment used for this
saved run. Installing PyTorch and selecting a compatible CUDA build remains
machine-specific. Results from another runtime need not be bitwise identical.

The portable launcher calls the existing scientific stages without modifying
them. First inspect its commands:

```bash
python scripts/temporary/regenerate_tpr_notebook_data.py --dry-run
```

Then run it explicitly in the GPU environment:

```bash
python scripts/temporary/regenerate_tpr_notebook_data.py
```

By default it downloads the pinned Qwen checkpoint
`Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, then writes a new
run to `ac-tpr-cache/tpr-all-layers-rerun`. Use `--skip-download` if this revision
is already in the Hugging Face cache. `--run-dir` selects another directory inside
`soft_tpr/native_interventions/`. Do not use the ready-results directory for a new experiment;
it contains no training checkpoints.

Existing completed stages are checked and reused by the experiment scripts. The
launcher does not call the historical `render` stage, which depends on a local
kernel configuration and would run the reports before the centroid results exist.
After the data finish, set the run path **before starting the notebook kernel**:

```bash
export AC_TPR_NATIVE_RUN="$PWD/ac-tpr-cache/tpr-all-layers-rerun"
```

Then use the report environment and run the three notebooks normally. No image
files are exported. GPU regeneration has not been rerun as part of packaging
these results; the launcher reuses the stages that produced the completed run.

## Commit and share

`notebook/tpr_share_files.txt` lists the reports, linked method walkthroughs,
required experiment sources, tests, setup scripts, and the compact data bundle.
It deliberately excludes old figure directories, model weights, and raw caches.
The paths in that list are relative to the main Git repository. From its root:

```bash
git add -- soft_tpr/native_interventions/
git diff --cached --stat
git commit -m "Share TPR notebooks, results and reproduction scripts"
git push origin main
```

After the commit is pushed to `main`, colleagues can clone the repository with:

```bash
git clone git@github.com:alexserra98/AC-TPR-project.git
cd AC-TPR-project/soft_tpr/native_interventions
```

A short message to accompany the link:

> The three all-layer TPR report notebooks now include the data needed to rerun
> the plots and subgroup analyses on CPU. Start in `soft_tpr/native_interventions/`
> and follow `notebook/README_tpr.md`: install
> the report dependencies, run `prepare_tpr_notebook_data.py`, and leave the two
> centroid recomputation flags off. The optional regeneration script repeats the
> full GPU experiment; it is not needed to inspect or plot the saved results.
