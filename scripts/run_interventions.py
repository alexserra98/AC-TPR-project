"""Evaluate separate agent/patient translations using pooled and per-voice means."""

import argparse
from pathlib import Path

from ac_tpr.interventions import run_interventions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/generated/corpus.csv"))
    parser.add_argument("--vectors-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for scores and metadata.")
    parser.add_argument("--splits", nargs="+", default=["test", "gen_test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="float32")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N selected rows in corpus order.")
    args = parser.parse_args()
    metadata = run_interventions(
        args.corpus, args.vectors_dir, args.output_dir, tuple(args.splits),
        args.batch_size, args.device, args.dtype, args.limit,
    )
    print(
        f"Saved {metadata['baseline_rows']} baseline and {metadata['result_rows']} "
        f"intervention rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
