"""Save agent/patient token states from every Pythia 6.9B transformer block."""

import argparse
from pathlib import Path

from ac_tpr.activations import extract_activations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/generated/corpus.csv"))
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="New directory for activations.pt and metadata.json; use scratch storage.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument(
        "--dtype", choices=["bfloat16", "float32"], default="float32"
    )
    args = parser.parse_args()
    metadata = extract_activations(
        args.corpus, args.output_dir, args.batch_size, args.device, args.dtype
    )
    print(f"Saved activations {metadata['shape']} to {args.output_dir}")


if __name__ == "__main__":
    main()
