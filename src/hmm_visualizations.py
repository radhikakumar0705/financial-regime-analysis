import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
import json
import pickle
import warnings
import os

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

ASSIGNMENTS_PATH = os.path.join(ROOT_DIR, "outputs", "hmm_regime_assignments.csv")
SUMMARY_PATH     = os.path.join(ROOT_DIR, "outputs", "hmm_regime_summary.json")
MODEL_PATH       = os.path.join(ROOT_DIR, "outputs", "hmm_best_model.pkl")
OUTPUT_PNG       = os.path.join(ROOT_DIR, "outputs", "hmm_results.png")

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update({
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : 'white',
    'axes.edgecolor'    : '#444444',
    'axes.labelcolor'   : '#222222',
    'axes.linewidth'    : 0.8,
    'xtick.color'       : '#444444',
    'ytick.color'       : '#444444',
    'xtick.labelsize'   : 8,
    'ytick.labelsize'   : 8,
    'text.color'        : '#222222',
    'grid.color'        : '#dddddd',
    'grid.linestyle'    : '--',
    'grid.linewidth'    : 0.6,
    'grid.alpha'        : 0.8,
    'font.family'       : 'DejaVu Sans',
    'axes.titlesize'    : 10,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 9,
    'legend.facecolor'  : 'white',
    'legend.edgecolor'  : '#cccccc',
    'legend.fontsize'   : 8,
    'legend.framealpha' : 0.9,
    'figure.dpi'        : 150,
    'savefig.dpi'       : 160,
    'savefig.bbox'      : 'tight',
    'savefig.facecolor' : 'white',
})

# FIXED DICTIONARY FOR COLORS
REGIME_COLORS = {
    'Bull'                 : '#2ca02c',   # green
    'Bear'                 : '#d62728',   # red
    'Sideways'             : '#ff7f0e',   # orange / amber
    'High-Vol'             : '#9467bd',   # purple
}
FALLBACK_COLORS = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd']

# Load data
df = pd.read_csv(ASSIGNMENTS_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

with open(SUMMARY_PATH) as f:
    summary = json.load(f)

with open(MODEL_PATH, 'rb') as f:
    saved = pickle.load(f)

model  = saved['model']
N      = summary['n_states']
labels = {int(k): v for k, v in summary['labels'].items()}
stats  = {int(k): v for k, v in summary['stats'].items()}
trans  = np.array(summary['transition_matrix'])

state_colors = {}
for s in range(N):
    lbl = labels[s]
    state_colors[s] = REGIME_COLORS.get(lbl, FALLBACK_COLORS[s % len(FALLBACK_COLORS)])

# Duration helper
def compute_durations(state_col, n_states):
    seq = state_col.values
    durations = {s: [] for s in range(n_states)}
    i = 0
    while i < len(seq):
        s = seq[i]; j = i
        while j < len(seq) and seq[j] == s:
            j += 1
        durations[s].append(j - i)
        i = j
    return durations

durations = compute_durations(df['hmm_state'], N)

fig = plt.figure(figsize=(16, 14)) 
fig.suptitle(
    'Hidden Markov Model — NSE NIFTY Regime Detection',
    fontsize=13, fontweight='bold', y=0.995
)

gs = gridspec.GridSpec(
    3, 3,
    figure=fig,
    hspace=0.50, wspace=0.35,
    # Changed 2.0 to 3.5 to make the top graph much taller!
    height_ratios=[3.5, 0.6, 1.6] 
)

# PLOT 1 — Price series
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df['date'], df['close'], color='#17202a', linewidth=0.8, alpha=0.92, zorder=3, label='NIFTY 50')

ylo = df['close'].min() * 0.96
yhi = df['close'].max() * 1.04

for s in range(N):
    mask = df['hmm_state'] == s
    ax1.fill_between(
        df['date'], ylo, yhi,
        where=mask,
        color=state_colors[s], alpha=0.22, zorder=1,
        label=f'R{s}: {labels[s]}'
    )

ax1.set_ylim(ylo, yhi)
ax1.set_ylabel('Index level')
ax1.set_title('Nifty 50 — close price with regime overlay')
ax1.legend(loc='upper left', ncol=N + 1, fontsize=8, framealpha=0.9)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax1.grid(True)
sns.despine(ax=ax1)

# PLOT 2 — Regime label strip
ax2 = fig.add_subplot(gs[1, :])
for s in range(N):
    mask = df['hmm_state'] == s
    ax2.scatter(
        df.loc[mask, 'date'],
        np.ones(mask.sum()),
        color=state_colors[s],
        s=2.5, marker='|',
        label=f'R{s}: {labels[s]}'
    )

ax2.set_yticks([])
ax2.set_ylim(0.5, 1.5)
ax2.set_title('Regime label per trading day')
ax2.legend(loc='upper right', ncol=N, fontsize=8, framealpha=0.9, markerscale=3)
ax2.set_xlim(ax1.get_xlim())
sns.despine(ax=ax2, left=True)

# PLOT 3 — Return & Volatility
ax3 = fig.add_subplot(gs[2, 0])
ann_rets = []
ann_vols = []

for s in range(N):
    r = stats[s]
    ret = r['mean_return'] * 252 * 100
    vol = r['mean_vol'] * np.sqrt(252) * 100
    ann_rets.append(ret)
    ann_vols.append(vol)
x = np.arange(N)
w = 0.36

bars1 = ax3.bar(x - w/2, ann_rets, w, color=[state_colors[s] for s in range(N)], alpha=0.85, edgecolor='white', linewidth=0.6, label='Ann. return %')
bars2 = ax3.bar(x + w/2, ann_vols, w, color=[state_colors[s] for s in range(N)], alpha=0.40, edgecolor='white', linewidth=0.6, hatch='//', label='Ann. vol %')

ax3.axhline(0, color='#555555', linewidth=0.8, linestyle='--')
ax3.set_xticks(x)
ax3.set_xticklabels([f'R{s}' for s in range(N)], fontsize=8)
ax3.set_ylabel('% (annualised)')
ax3.set_title('Return & vol per regime')
ax3.legend(fontsize=7)
ax3.grid(True, axis='y')
sns.despine(ax=ax3)

for bar in bars1:
    h = bar.get_height()
    va = 'bottom' if h >= 0 else 'top'
    offset = 0.3 if h >= 0 else -0.3
    ax3.text(bar.get_x() + bar.get_width()/2, h + offset, f'{h:.1f}', ha='center', va=va, fontsize=7)

# PLOT 4 — Transition matrix
ax4 = fig.add_subplot(gs[2, 1])
tick_labels = [f'R{s}' for s in range(N)]
sns.heatmap(
    trans, annot=True, fmt='.2f', cmap='Blues',
    xticklabels=tick_labels, yticklabels=tick_labels,
    ax=ax4, linewidths=0.5, linecolor='white',
    annot_kws={'size': 9, 'weight': 'bold'}, vmin=0, vmax=1,
    cbar_kws={'shrink': 0.85, 'label': 'Prob', 'pad': 0.02}
)
ax4.set_title('Regime transition matrix')
ax4.set_xlabel('To')
ax4.set_ylabel('From')
ax4.tick_params(labelsize=8)

# PLOT 5 — Regime duration
ax5 = fig.add_subplot(gs[2, 2])
all_dur = [d for s in range(N) for d in durations[s]]
max_dur = min(max(all_dur), 80) if all_dur else 60
bins = np.linspace(1, max_dur, 22)

for s in range(N):
    d = durations[s]
    mean = np.mean(d)
    ax5.hist(d, bins=bins, color=state_colors[s], alpha=0.55, edgecolor='white', linewidth=0.4, label=f'R{s} (μ={mean:.1f}d)')
    ax5.axvline(mean, color=state_colors[s], linewidth=1.6, linestyle='--', alpha=0.9)

ax5.set_xlabel('Duration (trading days)')
ax5.set_ylabel('Frequency')
ax5.set_title('Regime duration distribution')
ax5.legend(fontsize=7)
ax5.grid(True, axis='y')
sns.despine(ax=ax5)



import matplotlib.transforms as mtransforms


plt.savefig(OUTPUT_PNG, dpi=160, bbox_inches='tight', facecolor='white')
print(f"Saved Full Dashboard → {OUTPUT_PNG}")

fig.canvas.draw()
renderer = fig.canvas.get_renderer()

def save_single_plot(ax, filename, is_heatmap=False):

    bbox = ax.get_tightbbox(renderer)
    
    
    if is_heatmap and len(ax.collections) > 0:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar_bbox = cbar.ax.get_tightbbox(renderer)
            bbox = mtransforms.Bbox.union([bbox, cbar_bbox])
            
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    filepath = os.path.join(ROOT_DIR, "outputs", filename)
    fig.savefig(filepath, bbox_inches=extent.expanded(1.05, 1.15), dpi=160, facecolor='white')
    print(f"Saved Single Plot → {filepath}")


save_single_plot(ax1, "hmm_plot1_price_overlay.png")
save_single_plot(ax2, "hmm_plot2_regime_strip.png")
save_single_plot(ax3, "hmm_plot3_returns_vol.png")
save_single_plot(ax4, "hmm_plot4_transition_matrix.png", is_heatmap=True)
save_single_plot(ax5, "hmm_plot5_duration_dist.png")

plt.show()