"""Plot complete, matched centroid and decoded Soft TPR steering sweeps."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ac_tpr.soft_tpr_comparison import compare_soft_tpr_runs

COLORS = {"agent": "#0072B2", "patient": "#D55E00"}
STYLES = {"pooled": "-", "by_voice": "--"}
CONDITIONS = [(m, r) for m in STYLES for r in COLORS]
GROUPS = [(s, v) for s in ("test", "gen_test") for v in ("active", "passive")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centroid-dir", type=Path, required=True)
    parser.add_argument("--soft-tpr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    # This validates provenance, complete controls, and identical baseline scores.
    compare_soft_tpr_runs(args.centroid_dir, args.soft_tpr_dir, args.output_dir)
    raw, soft = [pd.read_csv(p / "summary.csv") for p in (args.centroid_dir, args.soft_tpr_dir)]
    base = pd.read_csv(args.centroid_dir / "baseline.csv")
    baseline = base.groupby(["split", "voice"]).baseline_logit_diff.agg(["mean", "count"])
    keys = ["split", "voice", "layer", "mean_type", "edited_role"]
    for frame in (raw, soft):
        if len(frame) != 512 or frame.duplicated(keys).any():
            raise ValueError("Expected all 512 distinct split/voice/layer/edit conditions")
        for split, voice in GROUPS:
            for means, role in CONDITIONS:
                line = select(frame, split, voice, means, role)
                if line.layer.tolist() != list(range(32)) or not (line['count'] == baseline.loc[(split, voice), 'count']).all():
                    raise ValueError("Missing layers or unequal sentence coverage")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#94a3b8", "axes.labelcolor": "#334155",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    metric = "intervened_mean_logit_diff"
    all_values = np.concatenate([raw[metric], soft[metric], baseline['mean'], [0]])
    limits = (all_values.min() - .08, all_values.max() + .08)
    handles = [Line2D([], [], color=COLORS[r], ls=STYLES[m], lw=2,
                      label=f"{r.capitalize()} edit / {'pooled' if m == 'pooled' else 'by voice'}")
               for m, r in CONDITIONS]
    handles.append(Line2D([], [], color="#334155", ls=":", label="Unedited baseline"))
    for name, frame, title in (("centroid", raw, "Centroid"), ("soft_tpr", soft, "Soft TPR")):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True, sharey=True)
        for ax, (split, voice) in zip(axes.flat, GROUPS):
            for means, role in CONDITIONS:
                line = select(frame, split, voice, means, role)
                ax.plot(line.layer, line[metric], color=COLORS[role], ls=STYLES[means], lw=1.9)
            decorate(ax, split, voice, baseline, reference=True)
            ax.set_ylim(*limits)
        finish(fig, axes, f"Answer scores after role edits — {title}", handles,
               "Above zero favors the patient; below zero favors the agent. Dotted lines show the unedited baseline.")
        save(fig, args.output_dir, f"answer_scores_{name}")
    # Use separate method columns so the four original condition styles stay legible.
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, sharey=True)
    for row, (split, voice) in enumerate(GROUPS):
        for col, (frame, title) in enumerate(((raw, "Centroid"), (soft, "Soft TPR"))):
            ax = axes[row, col]
            for means, role in CONDITIONS:
                line = select(frame, split, voice, means, role)
                ax.plot(line.layer, line[metric], color=COLORS[role], ls=STYLES[means], lw=1.8)
            decorate(ax, split, voice, baseline, reference=True)
            ax.set_title(title + "  |  " + ax.get_title(loc="left"), loc="left")
            ax.set_ylim(*limits)
    finish(fig, axes, "Answer scores: centroid and Soft TPR", handles,
           "Same checkpoint, float32 runtime, sentences and edit strength. Curves are descriptive means.", tall=True)
    save(fig, args.output_dir, "answer_scores_comparison")
    paired = soft.set_index(keys)[metric] - raw.set_index(keys)[metric]
    difference = paired.rename("soft_minus_centroid").reset_index()
    difference.to_csv(args.output_dir / "answer_score_differences.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True, sharey=True)
    for ax, (split, voice) in zip(axes.flat, GROUPS):
        for means, role in CONDITIONS:
            line = select(difference, split, voice, means, role)
            ax.plot(line.layer, line.soft_minus_centroid, color=COLORS[role], ls=STYLES[means], lw=1.9)
        decorate(ax, split, voice, baseline, reference=False)
    finish(fig, axes, "Difference in answer scores: Soft TPR − centroid", handles[:-1],
           "Positive values mean stronger patient-directed steering with Soft TPR; negative values favor centroid steering.",
           ylabel="Soft TPR − centroid (mean logit difference)")
    save(fig, args.output_dir, "answer_scores_difference")
    metadata = {"comparison": "Descriptive full-layer sweep; no held-out layer selection or significance claims",
                "shared_answer_score_ylim": limits,
                "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "maximum_absolute_mean_difference": float(paired.abs().max()),
                "mean_absolute_mean_difference": float(paired.abs().mean())}
    (args.output_dir / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)
    print(f"Saved four PNG/PDF figures in {args.output_dir}", flush=True)


def select(frame, split, voice, means, role):
    return frame[(frame.split == split) & (frame.voice == voice) &
                 (frame.mean_type == means) & (frame.edited_role == role)].sort_values("layer")


def decorate(ax, split, voice, baseline, reference):
    stats = baseline.loc[(split, voice)]
    title = "Test" if split == "test" else "Generalisation test"
    ax.set_title(f"{title} / {voice}  |  n = {int(stats['count'])}", loc="left")
    ax.axhline(0, color="#64748b", lw=.9, zorder=0)
    if reference:
        ax.axhline(stats['mean'], color="#334155", ls=":", lw=1.5)
    ax.set_xlim(0, 31)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
    ax.grid(color="#e2e8f0", lw=.65)
    ax.set_axisbelow(True)


def finish(fig, axes, title, handles, footnote, ylabel="Mean patient − agent logit", tall=False):
    for ax in axes[-1]:
        ax.set_xlabel("Transformer block (0-based)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title, fontsize=18, weight="bold", y=.985)
    fig.text(.5, .953 if tall else .945,
             "Pythia 6.9B / agent question / intervention strength 1 / float32", ha="center", color="#64748b")
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .045 if tall else .065),
               ncol=3 if len(handles) == 5 else 2, frameon=False)
    fig.text(.5, .017, footnote, ha="center", color="#64748b", fontsize=9)
    fig.subplots_adjust(top=.91 if tall else .87, bottom=.14 if tall else .23, hspace=.35, wspace=.14)


def save(fig, output, name):
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
