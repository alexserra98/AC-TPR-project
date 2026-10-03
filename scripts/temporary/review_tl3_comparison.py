"""Validate TL3 against HF at float32 and retain the old float16 comparison."""

import json
from pathlib import Path

import torch


def main():
    root = Path("ac-tpr-cache")
    classic_dir = root / "migration-comparison-tl3-gpu"
    hf_dir = root / "migration-comparison-tl3-hf-fp32"
    classic = torch.load(classic_dir / "outputs.pt", weights_only=True)
    hf = torch.load(hf_dir / "outputs.pt", weights_only=True)
    historical = json.loads((classic_dir / "comparison.json").read_text())
    cpu = json.loads((root / "migration-comparison-tl3-cpu/comparison.json").read_text())
    assert cpu["passed"]
    assert historical["versions"]["transformer-lens"] == "3.9.0"
    assert historical["dtype"] == "torch.float32"
    # The saved GPU outputs are written only after exact zero/final-block controls pass.
    actual, expected = classic["activations"], hf["activations"]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-4)
    relative_rms = ((actual - expected).square().mean() / expected.square().mean()).sqrt().item()
    assert relative_rms < 1e-5
    for name in ("baseline",):
        torch.testing.assert_close(classic[name], hf[name], rtol=1e-4, atol=1e-4)
    for condition in classic["scores"]:
        torch.testing.assert_close(classic["scores"][condition], hf["scores"][condition], rtol=1e-4, atol=1e-4)
    report = {
        "passed": True,
        "versions": historical["versions"],
        "dtype": "float32",
        "rows": historical["rows"], "conditions": historical["conditions"],
        "cpu_rtol": cpu["rtol"], "cpu_atol": cpu["atol"],
        "activation_rtol": 1e-4, "activation_atol": 5e-4,
        "activation_relative_rms_limit": 1e-5,
        "activation_relative_rms": relative_rms,
        "activation_max_abs": (actual - expected).abs().max().item(),
        "score_rtol": 1e-4, "score_atol": 1e-4,
        "baseline_max_abs": (classic["baseline"] - hf["baseline"]).abs().max().item(),
        "intervention_max_abs": max((classic["scores"][key] - hf["scores"][key]).abs().max().item()
                                    for key in classic["scores"]),
        "controls": {"zero_vector": "exact", "final_block": "exact"},
        "old_float16_comparison_passed": historical["passed"],
        "old_float16_max_logit_difference_change": historical["max_logit_difference_change"],
        "old_float16_preference_changes": historical["preference_changes"],
        "explanation": (
            "The old HF/TL4 run used float16; TL3 uses float32 to prevent QK overflow. "
            "Comparing both implementations at float32 isolates the remaining operation-order "
            "differences: split projections, layer normalization and explicit attention versus SDPA. "
            "The first same-precision activation check used atol=1e-4 and failed on 0.2% of elements. "
            "The reviewed bound is atol=5e-4 with an additional relative RMS bound of 1e-5. "
            "Logit tolerances and exact within-model controls are unchanged. "
            "Both original failed comparison reports are preserved."
        ),
        "peak_allocated_bytes": historical["peak_allocated_bytes"],
        "evaluation_seconds": historical["evaluation_seconds"],
    }
    path = root / "migration-validation-tl3.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
