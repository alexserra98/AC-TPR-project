"""Record full-experiment differences between the TL4 float16 and TL3 float32 runs."""

import json
from pathlib import Path

import pandas as pd
import torch


def tensor_difference(actual, expected):
    maximum, total, squared, reference_squared, count = 0.0, 0.0, 0.0, 0.0, 0
    for start in range(0, len(actual), 16):
        left, right = actual[start:start + 16].float(), expected[start:start + 16].float()
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        difference = left - right
        maximum = max(maximum, difference.abs().max().item())
        total += difference.abs().sum(dtype=torch.float64).item()
        squared += difference.square().sum(dtype=torch.float64).item()
        reference_squared += right.square().sum(dtype=torch.float64).item()
        count += difference.numel()
    return {"max_abs": maximum, "mean_abs": total / count,
            "relative_rms": (squared / reference_squared) ** 0.5}


def main():
    root = Path("ac-tpr-cache")
    stem = "pythia-6.9b-step143000-generated"
    old = torch.load(root / "activations" / f"{stem}-tl4/activations.pt", mmap=True, weights_only=True)
    new = torch.load(root / "activations" / f"{stem}-tl3/activations.pt", mmap=True, weights_only=True)
    for key in ("rows", "input_ids"):
        assert old[key] == new[key]
    assert torch.equal(old["role_positions"], new["role_positions"])
    report = {"sentences": len(new["rows"]), "rows_and_tokens_equal": True,
              "activations": tensor_difference(new["activations"], old["activations"])}
    old_vectors = torch.load(root / "syntactic_vectors" / f"{stem}-train-tl4/syntactic_vectors.pt", weights_only=True)
    new_vectors = torch.load(root / "syntactic_vectors" / f"{stem}-train-tl3/syntactic_vectors.pt", weights_only=True)
    report["vectors"] = {name: tensor_difference(new_vectors[name], old_vectors[name]) for name in old_vectors}
    for kind, suffix in (("smoke", "-smoke"), ("full", "")):
        old_dir = root / "interventions" / f"{stem}-agent{suffix}-tl4"
        new_dir = root / "interventions" / f"{stem}-agent{suffix}-tl3"
        metadata = json.loads((new_dir / "metadata.json").read_text())
        assert metadata["schema_version"] == 3
        assert metadata["execution_model"]["api"] == "HookedTransformer"
        assert metadata["execution_model"]["dtype"] == "float32"
        assert metadata["controls"] == {"zero_vector_first_batch": "passed", "final_block_all_rows": "passed"}
        comparison = {"metadata": metadata, "tables": {}}
        for name in ("baseline", "results", "summary"):
            previous, current = pd.read_csv(old_dir / f"{name}.csv"), pd.read_csv(new_dir / f"{name}.csv")
            assert list(previous.columns) == list(current.columns)
            keys = ["split", "voice", "layer", "mean_type", "edited_role"] if name == "summary" else ["pair_id", "voice"]
            if name == "results":
                keys += ["layer", "mean_type", "edited_role"]
            assert previous[keys].equals(current[keys])
            numeric = [column for column in current.select_dtypes("number") if column not in keys]
            difference = (current[numeric] - previous[numeric]).abs()
            table = {"rows": len(current), "max_abs": difference.max().to_dict(), "mean_abs": difference.mean().to_dict()}
            if name != "summary":
                table["baseline_preference_changes"] = int(((current.baseline_logit_diff > 0) != (previous.baseline_logit_diff > 0)).sum())
            if name == "results":
                table["intervened_preference_changes"] = int(((current.intervened_logit_diff > 0) != (previous.intervened_logit_diff > 0)).sum())
                assert (current.loc[current.layer == 31, "logit_diff_change"] == 0).all()
            comparison["tables"][name] = table
        report[kind] = comparison
    report["validation"] = json.loads((root / "migration-validation-tl3.json").read_text())
    report["note"] = "TL3 uses float32 and newly extracted means; TL4 used float16. Differences are reported, not asserted equal."
    output = root / "migration-artifact-comparison-tl3.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    print(json.dumps({kind: report[kind]["tables"] for kind in ("smoke", "full")}, indent=2))


if __name__ == "__main__":
    main()
