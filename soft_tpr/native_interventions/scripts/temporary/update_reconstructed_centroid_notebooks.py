"""Add reconstructed-centroid cells and render the three all-layer reports."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / 'ac-tpr-cache/native-tpr-all-layers-20261109'
REPORTS = {
    'role_interventions_tpr_qwen3.ipynb': ['tpr_reconstructed_centroid'],
    'role_interventions_soft_tpr_qwen3.ipynb': ['soft_tpr_reconstructed_centroid'],
    'role_interventions_comparison_qwen3.ipynb': ['tpr_reconstructed_centroid', 'soft_tpr_reconstructed_centroid'],
}

EXPLANATION = r'''## Steering with reconstructed centroids

Construct a **fixed training direction per layer and voice** separately for TPR and Soft TPR. These are additional interventions; the native role/slot swaps above remain separate methods.

For TPR, flatten the probe's binding map as $M$. Its minimum-norm reconstruction in residual coordinates is
$$\hat h_{\mathrm{TPR}}=c+s\left[\left(\frac{h-c}{s}M\right)M^+\right].$$
This projects onto the learned binding subspace; it does not decode predicted lexical labels or swap role channels.

For Soft TPR, encode the normalized state, quantize the unbound fillers, bind the **original slots**, and decode:
$$\hat h_{\mathrm{Soft}}=c+s\,D(\operatorname{bind}(q(h))).$$
There is no slot permutation in this centroid method.

For each method, layer, and active/passive voice, average these reconstructed **training** noun states separately by agent and patient role:
$$d=\mu_{\mathrm{patient}}-\mu_{\mathrm{agent}},\qquad
h'_{\mathrm{agent}}=h_{\mathrm{agent}}+d,\qquad
h'_{\mathrm{patient}}=h_{\mathrm{patient}}-d.$$
Both noun positions are edited jointly with **α = 1 and no norm matching**. Checkpoints are unchanged; no validation/test examples enter the centroid means. Subgroup filters select evaluation sentences only and never refit the directions.
'''

COMPUTE = '''# Cached results are used by default. Enable a stage to prepare or resume it.
# Preparation uses CPU; evaluation requires the experiment's GPU environment.
import subprocess
import sys
from datetime import datetime, timezone

CENTROID_RUN = RUN / "reconstructed_centroids"
PREPARE_CENTROIDS = False
RUN_CENTROID_EVALUATION = False
centroid_script = ROOT / "scripts/temporary/evaluate_reconstructed_centroids.py"

if PREPARE_CENTROIDS or RUN_CENTROID_EVALUATION:
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT / "scripts/temporary"),
                                  environment.get("PYTHONPATH", "")]),
        OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
    )
    if PREPARE_CENTROIDS:
        subprocess.run([sys.executable, str(centroid_script), "prepare", "--run-dir", str(RUN)],
                       cwd=ROOT, env=environment, check=True)
    if RUN_CENTROID_EVALUATION:
        # The report kernel disables CUDA; the evaluation subprocess uses GPU 0 by default.
        environment["CUDA_VISIBLE_DEVICES"] = os.environ.get("AC_TPR_CUDA_VISIBLE_DEVICES", "0")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        subprocess.run([
            sys.executable, "-u", str(ROOT / "scripts/temporary/resource_limited_run.py"),
            "--resource-log", str(CENTROID_RUN / f"resources-notebook-{stamp}.jsonl"),
            "--max-rss-gib", "48", "--min-available-gib", "32", "--gpu-fraction", "0.28",
            str(centroid_script), "evaluate", "--run-dir", str(RUN),
        ], cwd=ROOT, env=environment, check=True)
'''

LOAD = '''CENTROID_METHODS = __METHODS__
CENTROID_ALL_METHODS = ["tpr_reconstructed_centroid", "soft_tpr_reconstructed_centroid"]
centroid_paths = {name: CENTROID_RUN / name for name in (
    "protocol.json", "centroids/metadata.json", "centroids/diagnostics.csv",
    "evaluation/metadata.json", "evaluation/results.csv",
)}
assert centroid_paths["evaluation/metadata.json"].is_file(), (
    "Reconstructed-centroid evaluation is not complete. Run the evaluation cell above or wait for the active run."
)
centroid_hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in centroid_paths.items()}
centroid_protocol = json.loads(centroid_paths["protocol.json"].read_text())
centroid_fit = json.loads(centroid_paths["centroids/metadata.json"].read_text())
centroid_metadata = json.loads(centroid_paths["evaluation/metadata.json"].read_text())
assert centroid_metadata["status"] == centroid_fit["status"] == "complete"
assert centroid_protocol["layers"] == centroid_fit["layers"] == centroid_metadata["layers"] == LAYERS
assert centroid_protocol["methods"] == centroid_metadata["methods"] == CENTROID_ALL_METHODS
assert centroid_protocol["fit_split"] == centroid_fit["fit_split"] == "train"
assert centroid_protocol["plan_sha256"] == source_hashes["plan.json"]
assert centroid_metadata["protocol_sha256"] == centroid_fit["protocol_sha256"] == centroid_hashes["protocol.json"]
assert centroid_metadata["results_sha256"] == centroid_hashes["evaluation/results.csv"]
assert centroid_metadata["centroid_metadata_sha256"] == centroid_hashes["centroids/metadata.json"]
assert centroid_fit["diagnostics_sha256"] == centroid_hashes["centroids/diagnostics.csv"]
assert centroid_metadata["baseline_sha256"] == source_hashes["evaluation/baseline.csv"]
assert centroid_metadata["model"] == metadata["model"]
assert centroid_metadata["alpha"] == centroid_protocol["alpha"] == 1.0
assert centroid_metadata["norm_matching"] is centroid_protocol["norm_matching"] is False
for relative, expected_hash in centroid_protocol["extension_source_sha256"].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected_hash, relative

centroid_results_all = pd.read_csv(centroid_paths["evaluation/results.csv"])
expected_centroid_keys = pd.MultiIndex.from_product(
    [baseline_all.row_id, LAYERS, CENTROID_ALL_METHODS], names=["row_id", "layer", "method"]
)
assert len(centroid_results_all) == centroid_metadata["result_rows"] == len(expected_centroid_keys)
assert centroid_metadata["rows"] == len(baseline_all)
assert not centroid_results_all.duplicated(["row_id", "layer", "method"]).any()
assert centroid_results_all.set_index(["row_id", "layer", "method"]).index.sort_values().equals(expected_centroid_keys.sort_values())
assert centroid_results_all.alpha.eq(1.0).all()
assert centroid_results_all.result_origin.eq("scored").all()
assert np.isfinite(centroid_results_all[["baseline_margin", "edited_margin", "agent_edit_norm", "patient_edit_norm"]]).all().all()
for column in ["split", "voice", "quad_id"]:
    np.testing.assert_array_equal(centroid_results_all[column], centroid_results_all.row_id.map(corpus[column]))
np.testing.assert_array_equal(-centroid_results_all.baseline_margin,
    centroid_results_all.row_id.map(baseline_all.set_index("row_id").agent_minus_patient))
final_centroids = centroid_results_all[centroid_results_all.layer == LAYERS[-1]]
assert final_centroids.baseline_margin.eq(final_centroids.edited_margin).all(), "Final-block control failed"
centroid_diagnostics = pd.read_csv(centroid_paths["centroids/diagnostics.csv"])
assert len(centroid_diagnostics) == len(LAYERS) * len(CENTROID_ALL_METHODS) * len(VOICES)
display(centroid_diagnostics[centroid_diagnostics.method.isin(CENTROID_METHODS)].round(5))
print(f"Verified {len(centroid_results_all):,} scored centroid interventions over {len(LAYERS)} layers.")
'''

AGGREGATE = '''METHOD_LABELS.update({
    "tpr_reconstructed_centroid": "TPR reconstructed centroid",
    "soft_tpr_reconstructed_centroid": "Soft TPR reconstructed centroid",
})
COLORS.update({"tpr_reconstructed_centroid": "#009E73", "soft_tpr_reconstructed_centroid": "#CC79A7"})
CENTROID_COMPARE_METHODS = list(dict.fromkeys(["raw_centroid", *METHODS, *CENTROID_METHODS]))
# Use the identical selected sentence IDs, including when only one voice matches FILTERS.
original_results = pd.read_csv(source_paths["evaluation/results.csv"])
centroid_comparison_rows = pd.concat([
    original_results[original_results.method.isin(CENTROID_COMPARE_METHODS)],
    centroid_results_all[centroid_results_all.method.isin(CENTROID_METHODS)],
], ignore_index=True).merge(baseline[["row_id"]], on="row_id", how="inner", validate="many_to_one")
assert len(centroid_comparison_rows) == len(baseline) * len(LAYERS) * len(CENTROID_COMPARE_METHODS)
assert not centroid_comparison_rows.duplicated(["row_id", "layer", "method"]).any()
centroid_comparison_rows["agent_minus_patient"] = -centroid_comparison_rows.edited_margin
before, after = -centroid_comparison_rows.baseline_margin, centroid_comparison_rows.agent_minus_patient
centroid_comparison_rows["prediction_flipped"] = ((before > 0) & (after < 0)) | ((before < 0) & (after > 0))
centroid_summary = centroid_comparison_rows.groupby(KEYS).agg(
    sentences=("agent_minus_patient", "size"),
    mean_agent_minus_patient=("agent_minus_patient", "mean"),
    flips=("prediction_flipped", "sum"), flip_rate=("prediction_flipped", "mean"),
    mean_agent_edit_norm=("agent_edit_norm", "mean"), mean_patient_edit_norm=("patient_edit_norm", "mean"),
).reset_index()
for group, selected_group in baseline.groupby(GROUPS):
    rows = centroid_summary[(centroid_summary.split == group[0]) & (centroid_summary.voice == group[1])]
    assert len(rows) == len(LAYERS) * len(CENTROID_COMPARE_METHODS)
    assert rows.sentences.eq(len(selected_group)).all()
centroid_summary["mean_edit_norm"] = (centroid_summary.mean_agent_edit_norm + centroid_summary.mean_patient_edit_norm) / 2
centroid_comparison = centroid_summary.merge(
    baseline_stats[["agent_preference"]].reset_index(), on=GROUPS, validate="many_to_one"
)
centroid_comparison["Method"] = centroid_comparison.method.map(METHOD_LABELS)
display(centroid_comparison[["split", "voice", "layer", "Method", "sentences", "agent_preference",
    "mean_agent_minus_patient", "flips", "flip_rate", "mean_edit_norm"]].round(4))
'''

PLOT_SCORES = '''plot_layers(
    "mean_agent_minus_patient", "Native swaps and reconstructed-centroid steering",
    "Mean agent − patient logit",
    "Centroids use training reconstructions only. Positive favors the agent; negative favors the patient.",
    reference=True, methods=CENTROID_COMPARE_METHODS, statistics=centroid_summary,
)
'''
PLOT_FLIPS = '''plot_layers(
    "flip_rate", "Prediction flips: native swaps and reconstructed centroids",
    "Flipped predictions / all sentences",
    "Strict agent ↔ patient flips / all selected sentences; ties do not count as flips.",
    percent=True, methods=CENTROID_COMPARE_METHODS, statistics=centroid_summary,
)
'''
PLOT_NORMS = '''plot_layers(
    "mean_edit_norm", "Edit norms: native swaps and reconstructed centroids",
    "Mean edit norm per noun token",
    "α = 1; perturbation norms are not matched. Means average the two noun-token norms.",
    methods=CENTROID_COMPARE_METHODS, statistics=centroid_summary,
)
'''
EXPORT = '''assert {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in centroid_paths.items()} == centroid_hashes
centroid_summary.to_csv(FIGURES / "reconstructed_centroid_layer_statistics.csv", index=False)
centroid_comparison.to_csv(FIGURES / "reconstructed_centroid_method_comparison.csv", index=False)
centroid_diagnostics[centroid_diagnostics.method.isin(CENTROID_METHODS)].to_csv(
    FIGURES / "reconstructed_centroid_training_diagnostics.csv", index=False
)
centroid_provenance = dict(provenance,
    methods=CENTROID_COMPARE_METHODS, reconstructed_centroid_methods=CENTROID_METHODS,
    centroid_fit_split="train", centroid_grouping=["voice", "role"],
    centroid_protocol=centroid_protocol,
    centroid_source_paths={name: str(path) for name, path in centroid_paths.items()},
    centroid_source_sha256=centroid_hashes,
)
(FIGURES / "reconstructed_centroid_plot_metadata.json").write_text(json.dumps(centroid_provenance, indent=2) + "\\n")
print(f"Reconstructed-centroid tables and provenance saved to {FIGURES}")
'''


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def libraries():
    for directory in ('/tmp/ac-tpr-native-notebook-deps', '/tmp/ac-tpr-nov6-extra'):
        if Path(directory).is_dir() and directory not in sys.path:
            sys.path.insert(0, directory)
    import nbformat
    return nbformat


def edit(run):
    nbformat = libraries()
    backup = run / 'reconstructed_centroids/previous_notebooks'
    backup.mkdir(parents=True, exist_ok=True)
    for filename, methods in REPORTS.items():
        path = ROOT / 'notebook' / filename
        notebook = nbformat.read(path, as_version=4)
        if any('reconstructed-centroids' in cell.metadata.get('tags', []) for cell in notebook.cells):
            raise ValueError(f'Centroid cells already exist: {filename}')
        if not (backup / filename).exists():
            shutil.copy2(path, backup / filename)
        for cell in notebook.cells:
            if cell.cell_type == 'code' and 'def plot_layers(' in cell.source:
                cell.source = cell.source.replace('reference=False, percent=False):',
                    'reference=False, percent=False, methods=None, statistics=None):')
                cell.source = cell.source.replace('    fig, axes = plt.subplots(2, 2,',
                    '    methods = METHODS if methods is None else methods\n'
                    '    statistics = summary if statistics is None else statistics\n'
                    '    fig, axes = plt.subplots(2, 2,')
                cell.source = cell.source.replace('data = summary[(summary.split == split) & (summary.voice == voice)]',
                    'data = statistics[(statistics.split == split) & (statistics.voice == voice)]')
                cell.source = cell.source.replace('for method in METHODS', 'for method in methods')
                cell.source = cell.source.replace('np.ceil(summary[metric].max()', 'np.ceil(statistics[metric].max()')
                cell.source = cell.source.replace('bottom=0.23', 'bottom=0.27 if len(methods) > 3 else 0.23')
                cell.source = cell.source.replace('Compare native methods', 'Compare intervention methods')
            if cell.cell_type == 'markdown' and cell.source.startswith('## Interpretation'):
                cell.source += '\n\nThe additional reconstructed-centroid curves use the same checkpoints, held-out sentences, baseline, and α. Their fixed directions are fitted exclusively on training reconstructions, separately by voice. Read prediction changes alongside edit norms; neither reconstruction nor a native swap guarantees lexical preservation in the language model.'
        added = []
        for kind, source in [
            ('markdown', EXPLANATION), ('code', COMPUTE), ('code', LOAD.replace('__METHODS__', repr(methods))),
            ('markdown', '### Compare on the selected sentences\n\nInclude the raw centroid, the corresponding native swap(s), and each reconstructed centroid as distinct methods.'),
            ('code', AGGREGATE), ('markdown', '### Answer logits with reconstructed-centroid steering'),
            ('code', PLOT_SCORES), ('markdown', '### Prediction-flip frequency with reconstructed-centroid steering'),
            ('code', PLOT_FLIPS), ('markdown', '### Compare perturbation sizes'), ('code', PLOT_NORMS),
            ('markdown', '### Export reconstructed-centroid results'), ('code', EXPORT),
        ]:
            cell = nbformat.v4.new_code_cell(source) if kind == 'code' else nbformat.v4.new_markdown_cell(source)
            cell.metadata['tags'] = ['reconstructed-centroids']
            added.append(cell)
        index = next(i for i, c in enumerate(notebook.cells) if c.cell_type == 'markdown' and c.source.startswith('## Interpretation'))
        notebook.cells[index:index] = added
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f'Added centroid calculation and plotting cells: {filename}', flush=True)


def render(run):
    nbformat = libraries()
    from nbclient import NotebookClient
    out = run / 'reconstructed_centroids'
    metadata = json.loads((out / 'evaluation/metadata.json').read_text())
    assert metadata['status'] == 'complete' and metadata['layers'] == list(range(36))
    assert metadata['result_rows'] == 2096 * 36 * 2
    assert digest(out / 'evaluation/results.csv') == metadata['results_sha256']
    os.environ.update(JUPYTER_PATH='/tmp/ac-tpr-native-jupyter',
                      JUPYTER_RUNTIME_DIR='/tmp/ac-tpr-native-jupyter/runtime', AC_TPR_NATIVE_RUN=str(run))
    staged = out / 'notebooks'
    staged.mkdir(exist_ok=True)
    records = []
    for filename in REPORTS:
        path = ROOT / 'notebook' / filename
        notebook = nbformat.read(path, as_version=4)
        before = digest(path)
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                cell.outputs, cell.execution_count = [], None
        started = time.monotonic()
        NotebookClient(notebook, timeout=300, kernel_name='ac-tpr-native-notebooks',
            resources={'metadata': {'path': str(ROOT / 'notebook')}}, allow_errors=False).execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, staged / filename)
        records.append(dict(notebook=filename, source_sha256=before, sha256=digest(staged / filename),
            code_cells=sum(c.cell_type == 'code' for c in notebook.cells), duration_seconds=time.monotonic()-started))
        print(f'Executed all cells: {filename}', flush=True)
    for record in records:
        target = ROOT / 'notebook' / record['notebook']
        if digest(target) != record['source_sha256']:
            raise RuntimeError(f'Notebook changed during rendering: {target}')
    for record in records:
        target = ROOT / 'notebook' / record['notebook']
        temporary = target.with_suffix('.ipynb.tmp')
        shutil.copy2(staged / record['notebook'], temporary)
        temporary.replace(target)
    (staged / 'metadata.json').write_text(json.dumps(dict(status='complete', layers=list(range(36)),
        evaluation_sha256=metadata['results_sha256'], notebooks=records), indent=2) + '\n')
    print('Published all three notebooks with executed reconstructed-centroid plots.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('edit', 'render'))
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    (edit if args.stage == 'edit' else render)(args.run_dir.resolve())
