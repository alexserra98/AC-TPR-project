"""Run sentence-unpacking, TPR fitting, and role-swap evaluation stages."""

import argparse
import json
from pathlib import Path

from ac_tpr.tpr.decomposition import TPRConfig
from ac_tpr.tpr.extraction import DEFAULT_BINDINGS, DEFAULT_LAYERS
from ac_tpr.tpr.pipeline import run_pipeline
from ac_tpr.tpr.training import TrainingConfig
from ac_tpr.tpr.unpacking import UnpackingConfig


def binding(value: str) -> tuple[str, str]:
    noun, separator, role = value.rpartition(":")
    if not separator or not noun or role not in {"agent", "patient"}:
        raise argparse.ArgumentTypeError("Use NOUN:agent or NOUN:patient (noun case is significant)")
    return noun, role


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/active_pass_long/corpus.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["all", "extract", "unpack", "fit", "reconstruct", "steer"], default="all")
    parser.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    parser.add_argument("--held-out-bindings", nargs="*", type=binding, default=list(DEFAULT_BINDINGS),
                        help="Default: John:agent doctor:patient. Supply the flag alone for no binding holdout.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8, help="LLM extraction and evaluation batch size.")
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-lr-reductions", type=int, default=5)
    parser.add_argument("--unpacker-lr", type=float, default=1e-4)
    parser.add_argument("--tpr-lr", type=float, default=1e-3)
    parser.add_argument("--decoder-width", type=int, default=1024)
    parser.add_argument("--decoder-layers", type=int, default=6)
    parser.add_argument("--decoder-heads", type=int, default=16)
    parser.add_argument("--decoder-ffn", type=int, default=4096)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--filler-dim", type=int, default=100)
    parser.add_argument("--role-dim", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    training = dict(batch_size=args.train_batch_size, seed=args.seed, epochs=args.epochs,
                    eval_every=args.eval_every, patience=args.patience, max_lr_reductions=args.max_lr_reductions)
    result = run_pipeline(
        args.corpus, args.output_dir, stage=args.stage, layers=tuple(args.layers),
        held_out_bindings=tuple(args.held_out_bindings), device=args.device, dtype=args.dtype,
        batch_size=args.batch_size, alpha=args.alpha,
        unpacking=UnpackingConfig(hidden_size=args.decoder_width, n_layers=args.decoder_layers,
                                  n_heads=args.decoder_heads, ffn_size=args.decoder_ffn,
                                  dropout=args.dropout, max_length=args.max_length),
        tpr=TPRConfig(filler_dim=args.filler_dim, role_dim=args.role_dim),
        unpacker_training=TrainingConfig(learning_rate=args.unpacker_lr, **training),
        tpr_training=TrainingConfig(learning_rate=args.tpr_lr, **training),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
