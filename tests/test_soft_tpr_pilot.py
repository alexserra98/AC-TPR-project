"""Check norm controls and the small validation-selected steering experiment."""

import csv
from pathlib import Path
import runpy

import pytest
import pandas as pd
import torch

from ac_tpr.dataset import generate_rows


pilot = runpy.run_path(str(Path(__file__).parents[1] / "scripts/temporary/evaluate_soft_tpr_pilot.py"))


def test_cluster_interval_preserves_constant_paired_difference():
    analysis = runpy.run_path(str(Path(__file__).parents[1] / "scripts/temporary/analyze_soft_tpr_pilot.py"))
    frame = pd.DataFrame({"cluster": ["a", "a", "b", "b", "c", "c"], "gain": [0.5] * 6})
    assert analysis["cluster_interval"](frame, "gain") == [0.5, 0.5]


def test_norm_matching_preserves_direction_and_raw_length():
    soft = torch.tensor([[3., 4.], [1., -1.]])
    raw = torch.tensor([[0., 10.], [3., 4.]])
    matched = pilot["matched_delta"](soft, raw)
    torch.testing.assert_close(matched.norm(dim=1), raw.norm(dim=1))
    torch.testing.assert_close(torch.nn.functional.normalize(matched), torch.nn.functional.normalize(soft))
    with pytest.raises(ValueError, match="collapsed"):
        pilot["matched_delta"](torch.zeros_like(soft), raw)


def test_selection_is_independent_by_method_and_requires_validation():
    records = []
    for voice in ("active", "passive"):
        for method in ("raw", "soft", "soft_matched"):
            for layer in (1, 2):
                records.append(dict(split="val", voice=voice, method=method, layer=layer,
                                    mean_type="pooled", edited_role="agent",
                                    logit_diff_change=layer if method == "raw" else -layer))
    selected = pilot["select_conditions"](records)
    assert selected["active/raw"]["layer"] == 2
    assert selected["active/soft"]["layer"] == 1
    assert selected["passive/soft_matched"]["layer"] == 1
    records[0]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        pilot["select_conditions"](records)


def test_pilot_scores_and_controls(model, tokenizer, tmp_path):
    rows = generate_rows(["boy", "girl"], [{"lemma": "help", "past": "helped", "participle": "helped"}], [])
    for row in rows:
        row["split"] = "val"
    generator = torch.Generator().manual_seed(34)
    raw = {"pooled": torch.randn(2, 2, 32, generator=generator),
           "by_voice": torch.randn(2, 2, 2, 32, generator=generator)}
    vectors = {"raw": raw, "soft": {key: 2 * tensor for key, tensor in raw.items()}}
    records = pilot["evaluate"](model, tokenizer, rows, vectors,
                                {voice: [(0, "pooled", "agent"), (1, "by_voice", "patient")]
                                 for voice in ("active", "passive")}, tmp_path, "validation", 2)
    assert len(records) == len(rows) * 2 * 3
    for row in records:
        if row["layer"] == 1:
            assert row["logit_diff_change"] == 0
    raw_records = [r for r in records if r["method"] == "raw"]
    matched_records = [r for r in records if r["method"] == "soft_matched"]
    for raw, matched in zip(raw_records, matched_records, strict=True):
        assert raw["logit_diff_change"] == pytest.approx(matched["logit_diff_change"], abs=1e-6)
    with (tmp_path / "validation_baseline.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == len(rows)
