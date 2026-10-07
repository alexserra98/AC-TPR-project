"""Train per-layer Soft TPR autoencoders and export compatible steering vectors."""

import argparse
from pathlib import Path

from ac_tpr.soft_tpr import train_soft_tpr_vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-roles", type=int, default=8, help="Unlabeled latent TPR slots, not sentence roles.")
    parser.add_argument("--role-dim", type=int, default=16)
    parser.add_argument("--filler-dim", type=int, default=32)
    parser.add_argument("--num-codes", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--form-weight", type=float, default=1.0)
    parser.add_argument("--commitment-weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metadata = train_soft_tpr_vectors(**vars(args))
    print(f"Saved {len(metadata['layer_indices'])} autoencoders and role vectors under {args.output_dir}")


if __name__ == "__main__":
    main()
