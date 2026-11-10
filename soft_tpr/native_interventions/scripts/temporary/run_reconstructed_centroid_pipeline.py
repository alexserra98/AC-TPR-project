"""Run the complete centroid evaluation and publish executed notebooks afterwards."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def write_status(path, record):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'ac-tpr-cache/native-tpr-all-layers-20261109')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    out = run / 'reconstructed_centroids'
    out.mkdir(exist_ok=True)
    environment = os.environ.copy()
    environment.update(PYTHONPATH=os.pathsep.join(['/tmp/ac-tpr-native-notebook-deps', '/tmp/ac-tpr-nov6-extra', str(ROOT / 'src')]),
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    status_path = out / 'pipeline.json'
    with (out / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = dict(status='running', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat())
        commands = [
            ('evaluate', [sys.executable, '-u', str(ROOT / 'scripts/temporary/resource_limited_run.py'),
                '--resource-log', str(out / f'resources-{stamp}.jsonl'), '--max-rss-gib', '48',
                '--min-available-gib', '32', '--gpu-fraction', '0.28',
                str(ROOT / 'scripts/temporary/evaluate_reconstructed_centroids.py'), 'evaluate', '--run-dir', str(run)]),
            ('render', [sys.executable, '-u', str(ROOT / 'scripts/temporary/update_reconstructed_centroid_notebooks.py'),
                'render', '--run-dir', str(run)]),
        ]
        try:
            for stage, command in commands:
                record.update(stage=stage, stage_started_at=datetime.now(timezone.utc).isoformat())
                write_status(status_path, record)
                with (out / f'{stage}.log').open('a', buffering=1) as log:
                    subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            record.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            record.update(status='failed', error=str(error), failed_at=datetime.now(timezone.utc).isoformat())
            raise
        finally:
            write_status(status_path, record)


if __name__ == '__main__':
    main()
