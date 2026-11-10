"""Regenerate the 36-layer TPR data in a new run using the existing experiment stages."""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'ac-tpr-cache/tpr-all-layers-rerun')
    parser.add_argument('--skip-download', action='store_true', help='Use an already cached pinned model')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without downloading or computing')
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    run.relative_to(ROOT)  # The existing experiment stores repository-relative source snapshots.
    if (run / 'plan.json').exists() and not (run / 'source_snapshot').is_dir():
        raise ValueError('This directory contains report data only. Choose a new run directory for regeneration.')
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), environment.get('PYTHONPATH', '')]),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        TOKENIZERS_PARALLELISM='false',
    )
    download = [sys.executable, '-c',
        'from huggingface_hub import snapshot_download; '
        'from ac_tpr.dataset import TOKENIZER_ID, TOKENIZER_REVISION; '
        'snapshot_download(repo_id=TOKENIZER_ID, revision=TOKENIZER_REVISION)']
    runner = ROOT / 'scripts/temporary/run_native_tpr_all_layers.py'
    commands = [[sys.executable, '-u', str(runner), stage, '--run-dir', str(run),
                 '--training-device', 'cuda']
                for stage in ('prepare', 'extract', 'train', 'calibrate', 'evaluate')]
    commands.append([sys.executable, '-u', str(ROOT / 'scripts/temporary/evaluate_reconstructed_centroids.py'),
                     'evaluate', '--run-dir', str(run)])
    if args.dry_run:
        for command in ([] if args.skip_download else [download]) + commands:
            print(shlex.join(command))
        return

    import torch
    from importlib.metadata import version
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Regeneration requires a CUDA GPU supporting BF16')
    for package, expected in [('transformers', '5.9.0'), ('nnsight', '0.7.0')]:
        if version(package) != expected:
            raise RuntimeError(f'This protocol requires {package}=={expected}')
    if not args.skip_download:
        download_environment = environment.copy()
        for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'):
            download_environment.pop(name, None)
        subprocess.run(download, cwd=ROOT, env=download_environment, check=True)
    environment.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for command in commands:
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    print(f'Data complete. Set AC_TPR_NATIVE_RUN={run} before opening the three report notebooks.')


if __name__ == '__main__':
    main()
