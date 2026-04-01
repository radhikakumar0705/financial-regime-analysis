import pandas as pd
import numpy as np
import pickle
import json
import os
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(BASE_DIR)   # project root

FEATURES_PATH = os.path.join(ROOT_DIR, "outputs", "hmm_features.csv")
MODEL_PATH    = os.path.join(ROOT_DIR, "outputs", "hmm_best_model.pkl")

ASSIGNMENTS_OUTPUT = os.path.join(ROOT_DIR, "outputs", "hmm_regime_assignments.csv")
SUMMARY_OUTPUT     = os.path.join(ROOT_DIR, "outputs", "hmm_regime_summary.json")

os.makedirs(os.path.join(ROOT_DIR, "outputs"), exist_ok=True)

# load
df = pd.read_csv(FEATURES_PATH, parse_dates=['date']).sort_values('date').reset_index(drop=True)

with open(MODEL_PATH, 'rb') as f:
    saved = pickle.load(f)

model       = saved['model']
N_BEST      = saved['n_states']
SCALED_COLS = saved['scaled_cols']
X           = df[SCALED_COLS].values

print(f"Loaded model: {N_BEST} states")

# viterbi + probabilities
_, states = model.decode(X, algorithm='viterbi')
probs     = model.predict_proba(X)

df['state'] = states
df['state_prob_max'] = probs.max(axis=1)

for s in range(N_BEST):
    df[f'prob_state_{s}'] = probs[:, s]

# stats
regime_stats = {}
for s in range(N_BEST):
    mask = df['state'] == s
    ret  = df.loc[mask, 'log_return']
    vol  = df.loc[mask, 'rolling_vol']

    regime_stats[s] = {
        'count': int(mask.sum()),
        'pct_time': float(mask.mean()*100),
        'mean_return': float(ret.mean()),
        'mean_vol': float(vol.mean()),
        'sharpe': float(ret.mean()/ret.std()*np.sqrt(252)) if ret.std()>0 else 0.0,
        'skew': float(skew(ret)),
        'kurt': float(kurtosis(ret))
    }

# durations
def get_durations(seq):
    d = []
    i = 0
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        d.append(j - i)
        i = j
    return d

durations = {}
for s in range(N_BEST):
    seq = (df['state'] == s).astype(int).values
    durations[s] = get_durations(seq)

for s in range(N_BEST):
    d = durations[s]
    regime_stats[s]['dur_mean'] = float(np.mean(d))
    regime_stats[s]['dur_max']  = int(np.max(d))

# labeling
sorted_states = sorted(regime_stats.keys(), key=lambda s: regime_stats[s]['mean_return'])
avg_vol = np.mean([regime_stats[s]['mean_vol'] for s in regime_stats])

def label(rank, s):
    vol = regime_stats[s]['mean_vol']
    ret = regime_stats[s]['mean_return']

    if N_BEST == 2:
        return 'Bear' if rank == 0 else 'Bull'

    if N_BEST == 3:
        return ['Bear','Sideways','Bull'][rank]

    if rank == 0:
        return 'High Vol Bear' if vol > avg_vol else 'Bear'
    if rank == 1:
        return 'Bear' if ret < 0 else 'Sideways'
    if rank == 2:
        return 'Sideways'
    return 'Bull'

STATE_LABELS = {s: label(i, s) for i, s in enumerate(sorted_states)}

df['regime'] = df['state'].map(STATE_LABELS)

# transition matrix
trans = model.transmat_

print("\nTransition Matrix:")
print(pd.DataFrame(trans).round(3))

# output csv
out = df[['date','close','log_return','rolling_vol']].copy()
out['hmm_state']      = df['state']
out['hmm_regime']     = df['regime']
out['hmm_confidence'] = df['state_prob_max']

for s in range(N_BEST):
    out[f'hmm_prob_state_{s}'] = df[f'prob_state_{s}']

out.to_csv(ASSIGNMENTS_OUTPUT, index=False)

# summary json
summary = {
    'n_states': N_BEST,
    'labels': STATE_LABELS,
    'stats': regime_stats,
    'transition_matrix': trans.tolist()
}

with open(SUMMARY_OUTPUT, 'w') as f:
    json.dump(summary, f, indent=2)

print("\nSaved:", ASSIGNMENTS_OUTPUT)
print("Saved:", SUMMARY_OUTPUT)