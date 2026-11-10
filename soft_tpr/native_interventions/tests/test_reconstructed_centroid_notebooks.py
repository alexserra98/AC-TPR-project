"""Exercise notebook comparisons on synthetic results, including empty panels."""

import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = ['role_interventions_tpr_qwen3.ipynb', 'role_interventions_soft_tpr_qwen3.ipynb',
         'role_interventions_comparison_qwen3.ipynb']
NATIVE = ['raw_centroid', 'tpr_native_role_swap', 'soft_native_slot_swap']
CENTROIDS = ['tpr_reconstructed_centroid', 'soft_tpr_reconstructed_centroid']


@pytest.mark.parametrize('filename', FILES)
@pytest.mark.parametrize('active_only', [False, True])
def test_comparison_cells_share_rows_and_plot_all_layers(tmp_path, filename, active_only):
    notebook = json.loads((ROOT / 'notebook' / filename).read_text())
    sources = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
    native_methods = NATIVE if 'comparison' in filename else [NATIVE[2] if 'soft' in filename else NATIVE[1]]
    centroid_methods = CENTROIDS if 'comparison' in filename else [CENTROIDS[1] if 'soft' in filename else CENTROIDS[0]]
    baseline = pd.DataFrame([dict(row_id=i, split=split, voice=voice, agent_minus_patient=1.)
        for i, (split, voice) in enumerate((s, v) for s in ['test', 'gen_test'] for v in ['active', 'passive'])])
    records = [dict(row_id=row.row_id, split=row.split, voice=row.voice, layer=layer, method=method,
        baseline_margin=-1., edited_margin=1. if method in CENTROIDS else -1.,
        agent_edit_norm=2., patient_edit_norm=2.)
        for row in baseline.itertuples() for layer in range(36) for method in NATIVE + CENTROIDS]
    data = pd.DataFrame(records)
    native_path = tmp_path / 'native.csv'
    data[data.method.isin(NATIVE)].to_csv(native_path, index=False)
    selected = baseline[baseline.voice.eq('active')] if active_only else baseline
    stats = selected.groupby(['split', 'voice']).agg(sentences=('row_id', 'size'),
        mean_agent_minus_patient=('agent_minus_patient', 'mean'))
    stats = stats.reindex(pd.MultiIndex.from_product([['test', 'gen_test'], ['active', 'passive']], names=['split', 'voice']))
    stats['sentences'] = stats.sentences.fillna(0).astype(int)
    stats['agent_preference'] = 1.
    figures = []
    namespace = dict(pd=pd, np=np, plt=plt, Line2D=Line2D, PercentFormatter=PercentFormatter,
        source_paths={'evaluation/results.csv': native_path}, baseline=selected, baseline_stats=stats,
        centroid_results_all=data[data.method.isin(CENTROIDS)], CENTROID_METHODS=centroid_methods,
        METHODS=native_methods, METHOD_LABELS=dict(zip(NATIVE, NATIVE)),
        COLORS=dict(zip(NATIVE, ['gray', 'blue', 'orange'])), LAYERS=list(range(36)),
        GROUPS=['split', 'voice'], KEYS=['split', 'voice', 'layer', 'method'],
        SPLITS=['test', 'gen_test'], VOICES=['active', 'passive'], FILTER_LABEL='Synthetic test fixture',
        display=lambda value: None, show_figure=lambda fig: figures.append(fig))
    aggregate = next(s for s in sources if s.startswith('METHOD_LABELS.update('))
    exec(aggregate, namespace)
    summary = namespace['centroid_summary']
    assert summary.sentences.eq(1).all()
    assert summary[summary.method.isin(CENTROIDS)].flip_rate.eq(1).all()
    assert summary[summary.method.isin(NATIVE)].flip_rate.eq(0).all()
    plot_source = next(s for s in sources if s.startswith('def plot_layers('))
    definition = ast.parse(plot_source).body[0]
    exec(compile(ast.Module(body=[definition], type_ignores=[]), '<plot_layers>', 'exec'), namespace)
    for source in sources:
        if source.startswith('plot_layers(') and 'methods=CENTROID_COMPARE_METHODS' in source:
            exec(source, namespace)
    assert len(figures) == 3
    for figure in figures:
        assert len(figure.legends[0].get_texts()) >= len(namespace['CENTROID_COMPARE_METHODS'])
        for index, axes in enumerate(figure.axes):
            if active_only and index % 2:
                assert not axes.axison
            else:
                method_lines = axes.lines[:len(namespace['CENTROID_COMPARE_METHODS'])]
                assert all(list(line.get_xdata()) == list(range(36)) for line in method_lines)
        figure.canvas.draw()
        plt.close(figure)
