"""Create a Pythia-tokenized active/passive corpus from a labelled filler CSV."""

import argparse
from pathlib import Path

from ac_tpr.dataset import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fillers", type=Path, default=Path("data/fillers.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--held-out-nouns",
        nargs="*",
        default=[],
        help="Nouns reserved for gen_test; omit for train/val/test only.",
    )
    args = parser.parse_args()
    metadata = generate_dataset(args.fillers, args.output_dir, args.held_out_nouns, args.seed)
    counts = metadata["counts"]
    print(
        f"Wrote {counts['sentences']} sentences ({counts['included_pairs']} pairs) "
        f"to {args.output_dir}"
    )
    for split, count in counts["sentences_by_split"].items():
        print(f"  {split}: {count} sentences")


if __name__ == "__main__":
    main()
