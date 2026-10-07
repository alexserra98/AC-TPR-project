"""Compare completed raw-mean and Soft TPR intervention runs."""

import argparse
from pathlib import Path

from ac_tpr.soft_tpr_comparison import compare_soft_tpr_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--soft-tpr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = compare_soft_tpr_runs(**vars(args))
    print(f"Compared {result['groups']} conditions; results in {args.output_dir}")


if __name__ == "__main__":
    main()
