"""Run all Qwen blocks for native TPR, native Soft TPR, and centroid steering."""

import argparse
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LAYERS = list(range(36))
METHODS = ['raw_centroid', 'tpr_native_role_swap', 'soft_native_slot_swap']
REPORTS = [
    'role_interventions_tpr_qwen3.ipynb',
    'role_interventions_soft_tpr_qwen3.ipynb',
    'role_interventions_comparison_qwen3.ipynb',
]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_paths():
    from run_native_tpr_nov6 import source_paths as native_sources
    return sorted(set(native_sources() + [
        Path(__file__).resolve(),
        ROOT / 'scripts/temporary/native_all_layers_training.py',
        ROOT / 'scripts/temporary/native_all_layers_evaluation.py',
    ]))


def prepare(args):
    from importlib.metadata import version
    import torch
    from transformers import AutoConfig
    from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION, load_corpus
    from ac_tpr.prompts import FEW_SHOT, PROMPT_TEMPLATE

    if (args.run_dir / 'plan.json').exists():
        load_plan(args)
        return
    args.run_dir.relative_to(ROOT)  # Reports and source snapshots use repository-relative paths.
    assert args.batch_size == 32, 'This experiment uses fixed batches of 32'
    config = AutoConfig.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=True)
    assert config.num_hidden_layers == len(LAYERS) and config.hidden_size == 4096
    corpus = ROOT / 'data/active_pass_long/corpus.csv'
    rows = load_corpus(corpus)
    counts = {split: sum(row['split'] == split for row in rows)
              for split in ('train', 'val', 'test', 'gen_test')}
    assert counts == dict(train=3300, val=412, test=416, gen_test=1680)
    quad_sets = {split: {row['quad_id'] for row in rows if row['split'] == split} for split in counts}
    for i, split in enumerate(counts):
        for other in list(counts)[i + 1:]:
            assert not quad_sets[split] & quad_sets[other], 'Quad leakage'
    paths = source_paths()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    plan = dict(
        schema_version='native-all-layers-v1', status='prepared',
        created_at=datetime.now(timezone.utc).isoformat(),
        model=TOKENIZER_ID, revision=TOKENIZER_REVISION,
        corpus=str(corpus), corpus_sha256=digest(corpus),
        hidden_size=config.hidden_size, layers=LAYERS,
        layer_selection='All model blocks, 0 through 35; no layer selection',
        methods=METHODS, alphas=[1.0], fixed_alpha=1.0, norm_matching=False,
        batch_size=args.batch_size, expected_counts=counts,
        prompt_template=PROMPT_TEMPLATE, demonstrations=FEW_SHOT,
        probe=dict(seeds=[42, 43], epochs=150, patience=25,
                   learning_rate=.01, batch_size=512,
                   objective='Observed-role filler cross entropy',
                   selection='Minimum validation cross entropy; ties smaller seed',
                   role_dim=2, filler_dim=8, device='cpu'),
        soft=dict(
            architecture=dict(input_dim=config.hidden_size, hidden_dim=256, num_roles=8,
                              role_dim=16, filler_dim=32, num_codes=64),
            training=dict(epochs=100, batch_size=128, learning_rate=.001,
                          form_weight=1.0, commitment_weight=.25, seeds=[42, 43],
                          device=args.training_device),
            selection='Minimum validation reconstruction MSE; ties smaller seed'),
        calibration=dict(split='train', shared_across_voices=True,
                         sentences=3300, tokens=6600,
                         candidates='764 involutions including identity',
                         target='Same filler in opposite role in same-voice swapped twin'),
        intervention='Both noun tokens jointly at one block; native alpha=1; no norm matching',
        evaluation='All test and gen_test sentences; separate split and voice panels; no test selection',
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in paths},
        runtime=dict(python=sys.version.split()[0], torch=str(torch.__version__),
                     cuda=torch.version.cuda, transformers=version('transformers'),
                     nnsight=version('nnsight')),
    )
    snapshot = args.run_dir / 'source_snapshot'
    for path in paths:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert digest(target) == plan['source_sha256'][str(path.relative_to(ROOT))]
    write_json(args.run_dir / 'plan.json', plan)
    print(f'Prepared all {len(LAYERS)} layers: {args.run_dir}', flush=True)


def load_plan(args):
    plan = json.loads((args.run_dir / 'plan.json').read_text())
    assert plan['schema_version'] == 'native-all-layers-v1'
    assert plan['layers'] == LAYERS and plan['methods'] == METHODS
    assert plan['fixed_alpha'] == 1.0 and plan['alphas'] == [1.0]
    assert plan['norm_matching'] is False and plan['batch_size'] == args.batch_size
    assert plan['soft']['training']['device'] == args.training_device
    assert digest(plan['corpus']) == plan['corpus_sha256'], 'Corpus changed'
    for name, expected in plan['source_sha256'].items():
        assert digest(ROOT / name) == expected, f'Scientific source changed: {name}'
        assert digest(args.run_dir / 'source_snapshot' / name) == expected, name
    return plan


def extract(args, plan):
    from run_tpr_nov6 import extract as extract_states
    metadata_path = args.run_dir / 'extraction/metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        assert metadata['status'] == 'complete' and metadata['layers'] == LAYERS
        assert metadata['corpus_sha256'] == plan['corpus_sha256']
        assert digest(args.run_dir / 'extraction/states.pt') == metadata['states_sha256']
        print('Verified completed extraction', flush=True)
        return
    output = args.run_dir / 'extraction'
    if output.exists():
        incomplete = args.run_dir / 'incomplete'
        incomplete.mkdir(exist_ok=True)
        output.rename(incomplete / f'extraction-{time.time_ns()}')
    args.layers = LAYERS
    extract_states(args)


def render(args, plan):
    """Publish the report notebooks only after all layers and their cells succeed."""
    for directory in ('/tmp/ac-tpr-native-notebook-deps', '/tmp/ac-tpr-nov6-extra'):
        if Path(directory).is_dir() and directory not in sys.path:
            sys.path.insert(0, directory)
    import nbformat
    from nbclient import NotebookClient
    import pandas as pd

    metadata = json.loads((args.run_dir / 'evaluation/metadata.json').read_text())
    assert metadata['status'] == 'complete' and metadata['layers'] == LAYERS
    assert metadata['result_rows'] == 2096 * 36 * 3
    assert digest(args.run_dir / 'evaluation/results.csv') == metadata['results_sha256']
    results = pd.read_csv(args.run_dir / 'evaluation/results.csv')
    assert sorted(results.layer.unique()) == LAYERS
    final = results[results.layer == LAYERS[-1]]
    assert len(final) == 2096 * 3
    assert (final.baseline_margin == final.edited_margin).all(), 'Final-block control failed'
    os.environ['JUPYTER_PATH'] = '/tmp/ac-tpr-native-jupyter'
    os.environ['JUPYTER_RUNTIME_DIR'] = '/tmp/ac-tpr-native-jupyter/runtime'
    os.environ['AC_TPR_NATIVE_RUN'] = str(args.run_dir)
    staged = args.run_dir / 'notebooks'
    staged.mkdir(exist_ok=True)
    records = []
    for filename in REPORTS:
        path = ROOT / 'notebook' / filename
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            source = cell.source
            if cell.cell_type == 'code':
                source = source.replace('ac-tpr-cache/native-tpr-nov6-20261108', str(args.run_dir.relative_to(ROOT)))
                source = source.replace('plan["reference_sha256"]["evaluation/baseline.csv"]', 'metadata["baseline_sha256"]')
                source = source.replace('ax.set_xticks(LAYERS)', 'ax.set_xticks(sorted(set(LAYERS[::4] + [LAYERS[-1]])))')
                source = source.replace(', marker="o"', '')
                # Keep the prior three-layer figures and tables alongside this new sweep.
                source = source.replace('FIGURES = FIGURES_ROOT / folder', 'FIGURES = FIGURES_ROOT / "all_layers" / folder')
                cell.outputs = []
                cell.execution_count = None
            else:
                source = source.replace('layers **20–22**', 'all **36 layers (0–35)**')
                source = source.replace(
                    'The plotted sweep contains layers 20–22. The run\'s separate first-batch final-block control is not a plotted condition.',
                    'All layers 0–35 are plotted. Layer 35 is a control: editing a noun after the final block cannot affect the later answer position. All three methods must reproduce the baseline there.')
                source = source.replace(
                    'Their perturbation norms differ, and layers 20–22 were historically selected from centroid peaks, so the comparison remains exploratory.',
                    'Their perturbation norms differ. All 36 layers are included without selecting a layer from test outcomes; these are descriptive comparisons on this corpus.')
            cell.source = source
        nbformat.validate(notebook)
        started = time.monotonic()
        NotebookClient(notebook, timeout=240, kernel_name='ac-tpr-native-notebooks',
                       resources={'metadata': {'path': str(ROOT / 'notebook')}},
                       allow_errors=False).execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, staged / filename)
        records.append(dict(notebook=filename, code_cells=sum(c.cell_type == 'code' for c in notebook.cells),
                            duration_seconds=round(time.monotonic() - started, 3),
                            sha256=digest(staged / filename)))
    backup = args.run_dir / 'previous_notebooks'
    backup.mkdir(exist_ok=True)
    for filename in REPORTS:
        target = ROOT / 'notebook' / filename
        if not (backup / filename).exists():
            shutil.copy2(target, backup / filename)
        temporary = target.with_suffix('.ipynb.tmp')
        shutil.copy2(staged / filename, temporary)
        temporary.replace(target)
    write_json(args.run_dir / 'notebooks/metadata.json', dict(status='complete', layers=LAYERS,
                evaluation_sha256=metadata['results_sha256'], notebooks=records))
    print('Published all-layer notebooks with executed outputs', flush=True)


def run(args):
    """Run stages in separate bounded processes; retain logs and resumable outputs."""
    prepare(args)
    lock = (args.run_dir / '.pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
    write_json(args.run_dir / 'pipeline.json', record)
    wrapper = ROOT / 'scripts/temporary/resource_limited_run.py'
    try:
        for stage in ('extract', 'train', 'calibrate', 'evaluate', 'render'):
            record.update(stage=stage, stage_started_at=datetime.now(timezone.utc).isoformat())
            write_json(args.run_dir / 'pipeline.json', record)
            resource_path = args.run_dir / f'resources-{stage}-{time.time_ns()}.jsonl'
            command = [sys.executable, '-u', str(wrapper), '--resource-log', str(resource_path),
                       '--max-rss-gib', '42', '--min-available-gib', '32', '--gpu-fraction', '0.28',
                       str(Path(__file__).resolve()), stage, '--run-dir', str(args.run_dir),
                       '--batch-size', str(args.batch_size), '--training-device', args.training_device]
            print(f'START {stage}: {datetime.now(timezone.utc).isoformat()}', flush=True)
            with (args.run_dir / f'{stage}.log').open('a', buffering=1) as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'COMPLETE {stage}: {datetime.now(timezone.utc).isoformat()}', flush=True)
        record.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat())
        write_json(args.run_dir / 'pipeline.json', record)
    except BaseException as error:
        record.update(status='failed', error=str(error), failed_at=datetime.now(timezone.utc).isoformat())
        write_json(args.run_dir / 'pipeline.json', record)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'extract', 'train', 'calibrate', 'evaluate', 'render', 'run'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--training-device', default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    if args.stage == 'run':
        run(args)
        return
    if args.stage == 'prepare':
        prepare(args)
        return
    plan = load_plan(args)
    if args.stage == 'extract':
        extract(args, plan)
    elif args.stage in ('train', 'calibrate'):
        from native_all_layers_training import train, calibrate
        {'train': train, 'calibrate': calibrate}[args.stage](args, plan)
    elif args.stage == 'evaluate':
        from native_all_layers_evaluation import evaluate
        evaluate(args, plan)
    else:
        render(args, plan)


if __name__ == '__main__':
    main()
