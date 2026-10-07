"""Summarize validation-selected held-out steering and paired cluster intervals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cluster_interval(frame, column, seed=42):
    """Resample unordered noun-pair/verb groups, preserving reversed role orders."""
    grouped = frame.groupby("cluster")[column].agg(["sum", "count"])
    generator = np.random.default_rng(seed)
    indices = generator.integers(len(grouped), size=(2000, len(grouped)))
    draws = grouped["sum"].to_numpy()[indices].sum(1) / grouped["count"].to_numpy()[indices].sum(1)
    return np.quantile(draws, [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["controls_passed"]
    selection = json.loads((root / "selection.json").read_text())
    frame = pd.read_csv(root / "heldout_results.csv")
    frame["cluster"] = frame.apply(lambda r: "|".join([*sorted((r.agent, r.patient)), r.verb]), axis=1)
    selected = []
    for key, condition in selection.items():
        voice, method = key.split("/")
        subset = frame[(frame.voice == voice) & (frame.method == method)]
        for column in ("layer", "mean_type", "edited_role"):
            subset = subset[subset[column] == condition[column]]
        if subset.duplicated(["pair_id", "voice"]).any():
            raise ValueError("Duplicate selected results")
        selected.append(subset)
    selected = pd.concat(selected, ignore_index=True)
    summaries, gains = [], []
    for (split, voice, method), group in selected.groupby(["split", "voice", "method"]):
        before = group.baseline_logit_diff.to_numpy()
        after = group.logit_diff.to_numpy()
        wins = before < 0
        flips = wins & (after > 0)
        low, high = cluster_interval(group, "logit_diff_change")
        summaries.append({
            "split": split, "voice": voice, "method": method, "n": len(group),
            **selection[f"{voice}/{method}"], "baseline_agent_preference": wins.mean(),
            "patient_preference": (after > 0).mean(), "agent_to_patient_flips": int(flips.sum()),
            "baseline_agent_wins": int(wins.sum()),
            "conditional_flip_rate": flips.sum() / wins.sum() if wins.sum() else np.nan,
            "mean_logit_change": group.logit_diff_change.mean(), "ci_low": low, "ci_high": high,
        })
    for (split, voice), group in selected.groupby(["split", "voice"]):
        raw = group[group.method == "raw"].set_index("pair_id")
        for method in ("soft", "soft_matched"):
            candidate = group[group.method == method].set_index("pair_id")
            if set(raw.index) != set(candidate.index):
                raise ValueError("Methods were evaluated on different held-out examples")
            candidate = candidate.loc[raw.index]
            difference = candidate.copy()
            difference["gain"] = candidate.logit_diff_change - raw.logit_diff_change
            low, high = cluster_interval(difference, "gain")
            gains.append(dict(split=split, voice=voice, method=method, n=len(difference),
                              mean_gain_over_selected_raw=difference.gain.mean(), ci_low=low, ci_high=high))
    summary = pd.DataFrame(summaries)
    gain = pd.DataFrame(gains)
    summary.to_csv(root / "selected_summary.csv", index=False)
    gain.to_csv(root / "paired_gains.csv", index=False)
    report = {
        "summary": summaries, "paired_gains": gains,
        "interval": "2000 paired cluster bootstrap replicates; unordered noun-pair/verb clusters; seed 42; percentile 95% intervals",
        "limitations": ["Single training seed", "Five candidate layers informed by earlier repository results", "One prompt and controlled vocabulary",
                        "Intervals are conditional on validation-selected settings", "Counterfactual steering, not improved factual QA"],
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("metadata.json", "selection.json", "heldout_results.csv", "validation_results.csv")},
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(gain.to_string(index=False))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    methods = ("raw", "soft", "soft_matched")
    labels = ("Raw means", "Soft TPR", "Soft TPR, matched norm")
    colors = ("#64748b", "#2563eb", "#ea580c")
    groups = [(s, v) for s in ("test", "gen_test") for v in ("active", "passive")]
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = summary[summary.method == method].set_index(["split", "voice"]).loc[groups]
        positions = np.arange(4) + (index - 1) * 0.23
        axes[0].errorbar(positions, values.mean_logit_change,
                         yerr=np.stack([values.mean_logit_change-values.ci_low, values.ci_high-values.mean_logit_change]),
                         fmt="o", capsize=3, color=color, label=label)
        axes[1].bar(positions, 100 * values.conditional_flip_rate, width=0.22, color=color, label=label)
    for axis in axes:
        axis.set_xticks(range(4), ["Test\nactive", "Test\npassive", "Held-out\nnouns\nactive", "Held-out\nnouns\npassive"])
        axis.grid(axis="y", alpha=0.2)
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel("Mean patient-minus-agent logit change")
    axes[0].set_title("Steering effect (95% cluster intervals)")
    axes[1].set_ylabel("Flips among baseline agent wins (%)")
    axes[1].set_title("Counterfactual answer preference")
    axes[0].legend(fontsize=8)
    figure.savefig(root / "steering_results.png", dpi=170)
    figure.savefig(root / "steering_results.pdf")


if __name__ == "__main__":
    main()
