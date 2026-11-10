"""Install the bundled CSV/JSON results for the three all-layer TPR reports."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'data/tpr_notebook_results'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='Destination; defaults to the run used by the notebooks')
    parser.add_argument('--verify-only', action='store_true', help='Check the installed files without writing')
    args = parser.parse_args()
    manifest = json.loads((BUNDLE / 'manifest.json').read_text())
    target = (args.run_dir or ROOT / manifest['default_run']).expanduser().resolve()
    archive = BUNDLE / manifest['archive']
    if digest(archive.read_bytes()) != manifest['archive_sha256']:
        raise ValueError('Results archive checksum mismatch')
    sources = dict(manifest['required_source_sha256'])
    sources[manifest['corpus']] = manifest['corpus_sha256']
    for name, expected in sources.items():
        if digest((ROOT / name).read_bytes()) != expected:
            raise ValueError(f'Required source differs from the saved run: {name}')

    # Check every file before installing anything; existing results must match.
    payloads = {}
    with zipfile.ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != sorted(manifest['artifacts_sha256']):
            raise ValueError('Unexpected or duplicate archive members')
        for name, expected in manifest['artifacts_sha256'].items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Invalid artifact path: {name}')
            destination = target / name
            destination.resolve().relative_to(target)
            data = bundle.read(name)
            if digest(data) != expected:
                raise ValueError(f'Artifact checksum mismatch: {name}')
            if destination.exists():
                if digest(destination.read_bytes()) != expected:
                    raise ValueError(f'Existing file differs; use a separate --run-dir: {destination}')
            elif args.verify_only:
                raise FileNotFoundError(destination)
            payloads[destination] = data
    if not args.verify_only:
        for destination, data in payloads.items():
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('xb') as stream:
                    stream.write(data)
    print(f'Verified {len(payloads)} result files for all 36 layers: {target}')
    print('Keep PREPARE_CENTROIDS and RUN_CENTROID_EVALUATION set to False in the reports.')
    if target != (ROOT / manifest['default_run']).resolve():
        print(f'Set AC_TPR_NATIVE_RUN to {target} before starting the notebook kernel.')


if __name__ == '__main__':
    main()
