"""Compare completed mean and Soft TPR steering sweeps on identical examples."""

import csv
import hashlib
import json
import math
from pathlib import Path

from ac_tpr.interventions import GROUP_COLUMNS


def read_keyed_csv(path: Path, keys: list[str]) -> dict:
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    indexed = {tuple(row[key] for key in keys): row for row in records}
    if not records or len(indexed) != len(records):
        raise ValueError(f"Empty or duplicate row keys in {path}")
    return indexed


def compare_soft_tpr_runs(baseline_dir: Path, soft_tpr_dir: Path, output_dir: Path) -> dict:
    """Require matching provenance, exact unedited scores, and matching sweep groups.

    Positive differences measure stronger counterfactual steering, not better
    factual QA. Conditional flip rates use baseline agent wins; a zero denominator
    is written as an empty CSV cell.
    """
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    reference, candidate = [
        json.loads((directory / "metadata.json").read_text())
        for directory in (baseline_dir, soft_tpr_dir)
    ]
    if reference.get("vector_method", "raw_role_means") != "raw_role_means":
        raise ValueError("The baseline run must use raw role means")
    if candidate.get("vector_method") != "soft_tpr_decoded_role_means":
        raise ValueError("The candidate run must use decoded Soft TPR role means")
    for key in (
        "schema_version", "corpus_sha256", "model", "execution_model", "tokenizer",
        "activation_site", "hook_names", "scoring_hook", "prompt_template", "answer_prefix",
        "original_answer", "counterfactual_answer", "logit_diff", "logit_diff_change",
        "preference_rates", "flip_rate", "strength", "mean_types", "edited_roles",
        "voices", "layer_indices", "splits", "limit", "batch_size",
        "baseline_rows", "result_rows", "summary_rows",
    ):
        if reference[key] != candidate[key]:
            raise ValueError(f"Runs have incompatible {key}")
    expected_controls = {"zero_vector_first_batch": "passed", "final_block_all_rows": "passed"}
    if reference["controls"] != expected_controls or candidate["controls"] != expected_controls:
        raise ValueError("Both runs must pass zero-vector and final-block controls")
    baseline_keys = ["pair_id", "voice"]
    original = read_keyed_csv(baseline_dir / "baseline.csv", baseline_keys)
    edited = read_keyed_csv(soft_tpr_dir / "baseline.csv", baseline_keys)
    if original != edited or len(original) != reference["baseline_rows"]:
        raise ValueError("Runs must have identical baseline examples and scores")
    summaries = [read_keyed_csv(directory / "summary.csv", GROUP_COLUMNS) for directory in (baseline_dir, soft_tpr_dir)]
    if summaries[0].keys() != summaries[1].keys() or len(summaries[0]) != reference["summary_rows"]:
        raise ValueError("Runs must contain identical summary groups")
    metrics = ("mean_logit_diff_change", "intervened_patient_preference_rate", "agent_to_patient_flip_rate")
    rows = []
    total = 0
    for key in sorted(summaries[0]):
        raw, soft = (summary[key] for summary in summaries)
        count = int(raw["count"])
        if count < 1 or count != int(soft["count"]):
            raise ValueError("Summary group counts differ or are empty")
        for record in (raw, soft):
            if any(not math.isfinite(float(record[name])) for name in (*metrics, "baseline_agent_preference_rate")):
                raise ValueError("Nonfinite comparison metric")
        if raw["baseline_agent_preference_rate"] != soft["baseline_agent_preference_rate"]:
            raise ValueError("Summary baseline preference rates differ")
        total += count
        row = {**dict(zip(GROUP_COLUMNS, key)), "count": count}
        for metric in metrics:
            a, b = float(raw[metric]), float(soft[metric])
            row.update({f"raw_{metric}": a, f"soft_tpr_{metric}": b, f"difference_{metric}": b - a})
        denominator = float(raw["baseline_agent_preference_rate"])
        raw_flip = float(raw["agent_to_patient_flip_rate"]) / denominator if denominator else None
        soft_flip = float(soft["agent_to_patient_flip_rate"]) / denominator if denominator else None
        row.update({
            "baseline_agent_preference_rate": denominator,
            "raw_conditional_flip_rate": raw_flip, "soft_tpr_conditional_flip_rate": soft_flip,
            "difference_conditional_flip_rate": soft_flip - raw_flip if denominator else None,
        })
        rows.append(row)
    if total != reference["result_rows"]:
        raise ValueError("Summary counts do not cover the completed sweep")
    metadata = {
        "baseline_dir": str(baseline_dir.resolve()), "soft_tpr_dir": str(soft_tpr_dir.resolve()),
        "groups": len(rows), "interpretation": "Positive differences mean stronger patient-directed steering, not factual QA accuracy",
        "uncertainty": "Descriptive paired-condition comparisons; no significance or best-layer claims",
        "source_sha256": {
            name: {file: hashlib.sha256((directory / file).read_bytes()).hexdigest()
                   for file in ("metadata.json", "baseline.csv", "summary.csv")}
            for name, directory in (("raw", baseline_dir), ("soft_tpr", soft_tpr_dir))
        },
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata
