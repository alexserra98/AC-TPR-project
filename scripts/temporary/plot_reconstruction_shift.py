"""Export the steering-shift panel using the intervention notebook's axis style."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir', type=Path,
                    default=Path('ac-tpr-cache/local-soft-tpr-v1/ae-pca-controls-v2'))
root = parser.parse_args().run_dir
summary = pd.read_csv(root / 'summary.csv')
methods = ['raw', 'soft', 'ae_256_best', 'pca_256']
colors = {'raw': 'tab:blue', 'soft': 'tab:orange', 'ae_256_best': 'tab:green', 'pca_256': 'tab:purple'}
groups = [(s, v) for s in ('test', 'gen_test') for v in ('active', 'passive')]
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#94a3b8', 'axes.labelcolor': '#334155',
    'xtick.color': '#475569', 'ytick.color': '#475569',
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'savefig.facecolor': 'white', 'figure.dpi': 130,
    'savefig.dpi': 200, 'pdf.fonttype': 42,
})
fig, ax = plt.subplots(figsize=(7.2, 5.1), layout='constrained')
for i, method in enumerate(methods):
    group = summary[summary.method == method].set_index(['split', 'voice']).loc[groups]
    ax.plot(np.arange(4) + (i - (len(methods) - 1) / 2) * .12, group.mean_change, 'o',
            label=method.replace('_', ' '), color=colors[method])
ax.set_xticks(range(4), [
    'Test / active', 'Test / passive',
    'Generalization test\n/ active', 'Generalization test\n/ passive',
])
ax.set_ylabel('Mean change in patient − agent logit')
ax.set_axisbelow(True)
ax.grid(color='#e2e8f0', linewidth=.65)
ax.legend(fontsize=8)
ax.set_title('Reconstruction baselines\nActive: block 24 / Passive: block 25', fontsize=12)
fig.text(.5, -.025, 'Pythia 6.9B; native direction norms; one training seed',
         ha='center', fontsize=9, color='#64748b')
for extension in ('png', 'pdf'):
    fig.savefig(root / f'steering_shift.{extension}', bbox_inches='tight')
plt.close(fig)
