import pandas as pd
import numpy as np
import pickle
import json
import os
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)  # project root

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

sorted_states = sorted(regime_stats.keys(), key=lambda s: regime_stats[s]['mean_return'])

def get_k4_labels(stats, sorted_by_ret):
    labels = {}
    
    # Lowest return is Bear
    labels[sorted_by_ret[0]] = 'Bear'
    # Highest return is Bull
    labels[sorted_by_ret[3]] = 'Bull'
    
    # The middle two states differentiated by volatility
    mid1, mid2 = sorted_by_ret[1], sorted_by_ret[2]
    
    if stats[mid1]['mean_vol'] > stats[mid2]['mean_vol']:
        labels[mid1] = 'High-Vol'
        labels[mid2] = 'Sideways'
    else:
        labels[mid1] = 'Sideways'
        labels[mid2] = 'High-Vol'
        
    return labels

# Generate labels based on number of states
if N_BEST == 2:
    STATE_LABELS = {sorted_states[0]: 'Bear', sorted_states[1]: 'Bull'}
elif N_BEST == 3:
    STATE_LABELS = {sorted_states[0]: 'Bear', sorted_states[1]: 'Sideways', sorted_states[2]: 'Bull'}
else: # N_BEST == 4
    STATE_LABELS = get_k4_labels(regime_stats, sorted_states)

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