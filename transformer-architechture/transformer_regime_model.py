import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.data import TimeSeriesDataSet, GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

def combine_all_data(folder_path, cache_path=None):
    # If cached file exists, load it directly
    if cache_path and os.path.exists(cache_path):
        print("Loading cached data...")
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        return df
    # Combining all the csv data files
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = [pd.read_csv(file) for file in all_files]
    df = pd.concat(df_list, ignore_index=True)
    # Converting the dates to datetime type
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])
    # Sort in chronological order
    df = df.sort_values("Date").reset_index(drop=True)
    # Save to cache
    if cache_path:
        df.to_csv(cache_path, index=False)
        print(f"Data cached to {cache_path}")
    return df

def feature_engineer(df):
    # Momentum & Return Features
    df['Log_returns'] = np.log(df["Close"]/df["Close"].shift(1))
    df["Sma_20"] = df["Close"].rolling(window=20).mean()
    df["Dist_from_sma20"] = (df["Close"] / df["Sma_20"]) - 1
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    # Volatility features
    df['Parkinson_Vol_Daily'] = np.sqrt(0.36067 * (np.log(df['High'] / df['Low']))**2)
    df['Rolling_Vol_20d'] = df['Log_returns'].rolling(window=20).std() * np.sqrt(252)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low'] - df['Close'].shift(1)).abs()
    df['True_Range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR_14'] = df['True_Range'].rolling(window=14).mean()
    # Liquidity & Conviction Features
    if df['Turnover (₹ Cr)'].dtype == 'O':
        df['Turnover (₹ Cr)'] = df['Turnover (₹ Cr)'].astype(str).str.replace(',', '').astype(float)
    turnover_roll_mean = df['Turnover (₹ Cr)'].rolling(window=20).mean()
    turnover_roll_std = df['Turnover (₹ Cr)'].rolling(window=20).std()
    df['Turnover_Z_Score'] = (df['Turnover (₹ Cr)'] - turnover_roll_mean) / turnover_roll_std
    epsilon = 1e-8 
    df['Turnover_to_Vol_Ratio'] = df['Turnover_Z_Score'] / (df['Rolling_Vol_20d'] + epsilon)
    # dropping the first 20 rows
    df = df.dropna().reset_index(drop=True)
    return df

class TFTRegimeEncoder(nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.tft = TemporalFusionTransformer.from_dataset(
            dataset,
            learning_rate=1e-3,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            output_size=7,
            loss=QuantileLoss(),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        # Register hook to capture encoder LSTM hidden states
        self._hidden_state = None
        self.tft.lstm_encoder.register_forward_hook(self._capture_hidden)
    
    def _capture_hidden(self, module, input, output):
        self._hidden_state = output[0]
    
    def forward(self, x):
        _ = self.tft(x)
        context_vector = self._hidden_state[:, -1, :]
        regime_embedding = self.regime_head(context_vector)
        return nn.functional.normalize(regime_embedding, p=2, dim=1)

def contrastive_loss(embedding_1, embedding_2, temperature=0.5):
    batch_size = embedding_1.shape[0]
    out = torch.cat([embedding_1, embedding_2], dim=0)
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    positives = torch.exp(torch.sum(embedding_1 * embedding_2, dim=-1) / temperature)
    positives = torch.cat([positives, positives], dim=0)
    loss = -torch.log(positives / sim_matrix.sum(dim=-1))
    return loss.mean()

def move_batch_to_device(x, device):
    # Move all tensors in the batch dict to the given device.
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: move_batch_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_batch_to_device(v, device) for v in x]
    elif isinstance(x, tuple):
        return tuple(move_batch_to_device(v, device) for v in x)
    return x

def clone_batch(x):
    # Deep clone all tensors in the batch dict.
    if isinstance(x, torch.Tensor):
        return x.clone()
    elif isinstance(x, dict):
        return {k: clone_batch(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [clone_batch(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(clone_batch(v) for v in x)
    return x

def get_embedding_rows(split_df, max_encoder_length, max_prediction_length):
    # Align each extracted embedding with the row it predicts.
    # For encoder length E and prediction length P, the first sample maps to
    # row index E + P - 1 in the split DataFrame.
    start_idx = max_encoder_length + max_prediction_length - 1
    if len(split_df) < start_idx + 1:
        return split_df.iloc[0:0].copy()
    return split_df.iloc[start_idx:].copy().reset_index(drop=True)

def extract_embeddings(model, loader, device):
    # Run the encoder over a dataloader and collect embeddings.
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x, _ = batch
            x = move_batch_to_device(x, device)
            embeddings.append(model(x).cpu().numpy())
    if not embeddings:
        return np.empty((0, 16), dtype=np.float32)
    return np.concatenate(embeddings, axis=0)

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "..", "data")
    cache_file = os.path.join(script_dir, "combined_data.csv")
    df = combine_all_data(data_folder, cache_path=cache_file)
    print(df.head())

    # Feature engineering
    df = feature_engineer(df)
    print(df.head())
    print(df.shape)

    features = ["Log_returns", "Dist_from_sma20", "RSI_14", "Parkinson_Vol_Daily", "Rolling_Vol_20d", "ATR_14", "Turnover_Z_Score", "Turnover_to_Vol_Ratio"]

    # Add required columns for pytorch_forecasting
    df["time_idx"] = range(len(df))
    df["group_id"] = "NIFTY50"

    # Chronological train/val/test split
    n = len(df)
    train_cutoff = int(0.70 * n)
    val_cutoff = int(0.85 * n)

    training_df = df[df["time_idx"] < train_cutoff].copy()
    val_df = df[(df["time_idx"] >= train_cutoff) & (df["time_idx"] < val_cutoff)].copy()
    test_df = df[df["time_idx"] >= val_cutoff].copy()

    print(f"Train: {len(training_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    max_encoder_length = 20
    max_prediction_length = 1

    # Create TimeSeriesDataSet for training
    training_tsds = TimeSeriesDataSet(
        training_df,
        time_idx="time_idx",
        target="Log_returns",
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=features,
        target_normalizer=GroupNormalizer(groups=["group_id"]),
        allow_missing_timesteps=True,
    )

    # Validation and test datasets from the training dataset config
    val_tsds = TimeSeriesDataSet.from_dataset(training_tsds, val_df, predict=False, stop_randomization=True)
    test_tsds = TimeSeriesDataSet.from_dataset(training_tsds, test_df, predict=False, stop_randomization=True)

    # Create DataLoaders
    batch_size = 64
    train_loader = training_tsds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    train_eval_loader = training_tsds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    val_loader = val_tsds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    test_loader = test_tsds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    # Verify batch structure
    for batch in train_loader:
        x, y = batch
        print("Batch x keys:", list(x.keys()))
        print("encoder_cont shape:", x["encoder_cont"].shape)
        break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TFTRegimeEncoder(training_tsds).to(device)
    model.tft.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            x, y = batch
            x = move_batch_to_device(x, device)
            
            # Create noisy version by adding noise to continuous encoder features
            x_noisy = clone_batch(x)
            x_noisy["encoder_cont"] = x_noisy["encoder_cont"] + torch.randn_like(x_noisy["encoder_cont"]) * 0.05
            
            optimizer.zero_grad()
            embeddings_original = model(x)
            embeddings_noisy = model(x_noisy)
            
            loss = contrastive_loss(embeddings_original, embeddings_noisy)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    print("Training Complete.")

    train_embeddings = extract_embeddings(model, train_eval_loader, device)
    val_embeddings = extract_embeddings(model, val_loader, device)
    test_embeddings = extract_embeddings(model, test_loader, device)
    latent_space = np.concatenate([train_embeddings, val_embeddings, test_embeddings], axis=0)

    num_regimes = 4

    gmm = GaussianMixture(n_components=num_regimes, covariance_type="full", random_state=42)

    # Fit GMM on train embeddings only
    if len(train_embeddings) == 0:
        raise ValueError("No training embeddings were extracted. Check the training split size and window lengths.")
    gmm.fit(train_embeddings)

    predicted_regimes = gmm.predict(latent_space)
    regime_probabilities = gmm.predict_proba(latent_space)

    # Align results with the original DataFrame
    train_results_df = get_embedding_rows(training_df, max_encoder_length, max_prediction_length)
    val_results_df = get_embedding_rows(val_df, max_encoder_length, max_prediction_length)
    test_results_df = get_embedding_rows(test_df, max_encoder_length, max_prediction_length)
    results_df = pd.concat([train_results_df, val_results_df, test_results_df], ignore_index=True)

    if len(results_df) != len(latent_space):
        raise ValueError(
            f"Embedding/data alignment mismatch: {len(latent_space)} embeddings vs {len(results_df)} aligned rows."
        )

    results_df['Regime_ID'] = predicted_regimes
    results_df['Confidence'] = regime_probabilities.max(axis=1)

    stats = results_df.groupby("Regime_ID")["Log_returns"].agg(["count", "mean", "std"])
    stats["Ann_Return"] = stats["mean"] * 252
    stats["Ann_Volatility"] = stats["std"] * np.sqrt(252)
    print("\n--- RAW CLUSTER STATISTICS ---")
    print(stats[['count', 'Ann_Return', 'Ann_Volatility']])

    high_vol_id = stats['Ann_Volatility'].idxmax()
    remaining_stats = stats.drop(high_vol_id)
    bull_id = remaining_stats['Ann_Return'].idxmax()
    remaining_stats = remaining_stats.drop(bull_id)
    bear_id = remaining_stats['Ann_Return'].idxmin()
    sideways_id = remaining_stats.drop(bear_id).index[0]
    dynamic_mapping = {
        high_vol_id: "High-Vol",
        bull_id: "Bull",
        bear_id: "Bear",
        sideways_id: "Sideways"
    }
    results_df['Regime'] = results_df['Regime_ID'].map(dynamic_mapping)

    sil_score = silhouette_score(latent_space, predicted_regimes)
    print(f"1. Silhouette Score: {sil_score:.4f}")

    print("\n2. Regime Counts & Average Confidence:")
    for regime_name in ["Bull", "Bear", "Sideways", "High-Vol"]:
        subset = results_df[results_df['Regime'] == regime_name]
        count = len(subset)
        avg_prob = subset['Confidence'].mean()
        print(f"   - {regime_name}: {count} days (Avg Probability: {avg_prob:.2%})")

    print("\n3. Per Regime Statistics (Annualized):")
    for regime_name in ["Bull", "Bear", "Sideways", "High-Vol"]:
        subset = results_df[results_df['Regime'] == regime_name]
        ann_ret = subset['Log_returns'].mean() * 252
        ann_vol = subset['Log_returns'].std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        print(f"   - {regime_name}: Return = {ann_ret:.2%}, Volatility = {ann_vol:.2%}, Pseudo-Sharpe = {sharpe:.2f}")

    print("\n4. Regime Transition Matrix (Probabilities):")
    results_df['Next_Regime'] = results_df['Regime'].shift(-1)
    transition_counts = pd.crosstab(results_df['Regime'], results_df['Next_Regime'])
    transition_matrix = transition_counts.div(transition_counts.sum(axis=1), axis=0)
    print(transition_matrix.round(4))

    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    palette = {
        "Bull": "#2ca02c",       # Green
        "Bear": "#d62728",       # Red
        "Sideways": "#7f7f7f",   # Gray
        "High-Vol": "#ff7f0e"    # Orange
    }

    # Regime Transition Matrix Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, cmap="Blues", fmt=".2%", cbar=False)
    plt.title("Regime Transition Matrix")
    plt.ylabel("Current Regime")
    plt.xlabel("Next Day Regime")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "1_transition_matrix.png"), dpi=300)
    plt.close()

    # Training Loss
    if 'training_losses' in locals():
        plt.figure(figsize=(8, 5))
        plt.plot(training_losses, color='purple', linewidth=2)
        plt.title("TFT Contrastive Loss Convergence")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "2_training_loss.png"), dpi=300)
        plt.close()

    # Return vs Volatility per Regime
    plt.figure(figsize=(8, 6))
    plot_stats = results_df.groupby('Regime')['Log_returns'].agg(['mean', 'std'])
    plot_stats['Ann_Ret'] = plot_stats['mean'] * 252
    plot_stats['Ann_Vol'] = plot_stats['std'] * np.sqrt(252)

    sns.scatterplot(data=plot_stats, x='Ann_Vol', y='Ann_Ret', hue=plot_stats.index, 
                    palette=palette, s=200, edgecolor='black')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Risk vs. Reward per Regime")
    plt.xlabel("Annualized Volatility (Risk)")
    plt.ylabel("Annualized Return (Reward)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "3_risk_reward_scatter.png"), dpi=300)
    plt.close()

    # Regime Label per Trading Day (Timeline)
    plt.figure(figsize=(15, 3))
    sns.scatterplot(data=results_df, x='Date', y='Regime', hue='Regime', 
                    palette=palette, marker='|', s=100, legend=False)
    plt.title("Regime Classification Over Time")
    plt.xlabel("Year")
    plt.ylabel("Regime")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "4_regime_timeline.png"), dpi=300)
    plt.close()

    # Close Price with Regime Overlay
    plt.figure(figsize=(15, 7))
    plt.plot(results_df['Date'], results_df['Close'], color='black', linewidth=1)

    current_regime = results_df['Regime'].iloc[0]
    start_date = results_df['Date'].iloc[0]

    for i in range(1, len(results_df)):
        if results_df['Regime'].iloc[i] != current_regime or i == len(results_df) - 1:
            end_date = results_df['Date'].iloc[i]
            plt.axvspan(start_date, end_date, color=palette[current_regime], alpha=0.2, lw=0)
            current_regime = results_df['Regime'].iloc[i]
            start_date = end_date

    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=palette[r], alpha=0.4, label=r) for r in palette.keys()]
    plt.legend(handles=handles, loc='upper left')

    plt.title("Nifty 50 Close Price with TFT Regime Overlay")
    plt.xlabel("Year")
    plt.ylabel("Price")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "5_price_overlay.png"), dpi=300)
    plt.close()

    # Reduces the 16-dimensional embeddings to 2D to verify cluster separation
    pca = PCA(n_components=2)
    latent_2d = pca.fit_transform(latent_space)
    results_df['PCA_1'] = latent_2d[:, 0]
    results_df['PCA_2'] = latent_2d[:, 1]

    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=results_df, x='PCA_1', y='PCA_2', hue='Regime', 
                    palette=palette, alpha=0.6, s=15, edgecolor=None)
    plt.title("TFT Latent Space (2D PCA Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "6_latent_space_pca.png"), dpi=300)
    plt.close()
        
if __name__ == "__main__":
    main()
