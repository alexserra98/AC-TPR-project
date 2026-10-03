"""Save pooled and per-voice agent/patient means from training activations on CPU."""

import argparse
from pathlib import Path

from ac_tpr.syntactic_vectors import extract_syntactic_vectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--activations-dir", type=Path, required=True,
        help="Completed extraction directory containing activations.pt and metadata.json.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="New directory for syntactic_vectors.pt and metadata.json.",
    )
    args = parser.parse_args()
    metadata = extract_syntactic_vectors(args.activations_dir, args.output_dir)
    print(f"Saved vectors {metadata['shape']} from {metadata['sentences']} training sentences")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
