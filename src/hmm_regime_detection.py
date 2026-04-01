import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

# base path (script-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES_PATH     = os.path.join(BASE_DIR, "..", "outputs", "hmm_features.csv")
MODEL_OUTPUT_PATH = os.path.join(BASE_DIR, "..", "outputs", "hmm_best_model.pkl")
SCORES_OUTPUT     = os.path.join(BASE_DIR, "..", "outputs", "hmm_model_scores.json")

# ensure output folder exists
os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)

# model config
N_RANGE = [2, 3, 4]
SEEDS   = [0, 7, 42]
N_ITER  = 500

# load data
df = pd.read_csv(FEATURES_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# scaled features only
SCALED_COLS = [c for c in df.columns if c.endswith('_scaled')]
X = df[SCALED_COLS].values

print("Running from:", BASE_DIR)
print("X shape:", X.shape)
print("Features:", SCALED_COLS, "\n")


# aic/bic
def compute_metrics(log_lik, n_states, n_features, n_obs):
    k = (
        n_states * n_features +
        n_states * (n_features ** 2) +
        n_states * (n_states - 1) +
        (n_states - 1)
    )
    aic = 2 * k - 2 * log_lik
    bic = k * np.log(n_obs) - 2 * log_lik
    return aic, bic, k


# training
print(f"{'States':>6} | {'Seed':>4} | {'LogL':>12} | {'AIC':>12} | {'BIC':>12}")
print("-" * 60)

records = []

for n in N_RANGE:
    for seed in SEEDS:
        model = GaussianHMM(
            n_components=n,
            covariance_type="full",
            n_iter=N_ITER,
            random_state=seed
        )

        model.fit(X)

        log_lik = model.score(X) * len(X)
        aic, bic, k = compute_metrics(log_lik, n, X.shape[1], len(X))

        print(f"{n:>6} | {seed:>4} | {log_lik:>12.2f} | {aic:>12.2f} | {bic:>12.2f}")

        records.append({
            'n_states': n,
            'seed': seed,
            'log_lik': log_lik,
            'aic': aic,
            'bic': bic,
            'n_params': k,
            'model': model
        })


# selecting best model
df_scores = pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in records])

best_idx = df_scores.sort_values(['bic', 'log_lik'], ascending=[True, False]).index[0]

best_row   = df_scores.loc[best_idx]
best_model = records[best_idx]['model']
N_BEST     = int(best_row['n_states'])

print("\n" + "="*50)
print(f"Best model → {N_BEST} states (seed={int(best_row['seed'])})")
print(f"LogL: {best_row['log_lik']:.2f}")
print(f"AIC : {best_row['aic']:.2f}")
print(f"BIC : {best_row['bic']:.2f}")
print("="*50 + "\n")


# quick state distribution
states = best_model.predict(X)
df['state'] = states

print("State distribution:")
for s in range(N_BEST):
    count = (df['state'] == s).sum()
    pct   = (df['state'] == s).mean() * 100
    print(f"State {s}: {count} ({pct:.1f}%)")


# save model
with open(MODEL_OUTPUT_PATH, 'wb') as f:
    pickle.dump({
        'model': best_model,
        'n_states': N_BEST,
        'scaled_cols': SCALED_COLS
    }, f)

# save scores
scores_out = df_scores.to_dict(orient='records')

with open(SCORES_OUTPUT, 'w') as f:
    json.dump({
        'scores': scores_out,
        'best_n_states': N_BEST
    }, f, indent=2)


print("\nSaved:", MODEL_OUTPUT_PATH)
print("Saved:", SCORES_OUTPUT)