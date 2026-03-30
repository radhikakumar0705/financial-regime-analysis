"""
Deep Temporal Clustering (DTC) for NSE Stock Market Regime Detection
=====================================================================
Based on: Madiraju et al. (2018) "Deep Temporal Clustering: Fully Unsupervised
          Learning of Time-Domain Features"

Architecture:
  - Conv1D feature extractor  →  BiLSTM encoder  →  latent vector z
  - Mirrored decoder for autoencoder pretraining
  - Soft clustering layer using Student-t similarity
  - Joint optimisation: MSE reconstruction loss + KL divergence clustering loss

Usage:
  python dtc_nse.py                            # cleans cleaned_nse_data.csv then runs DTC
  python dtc_nse.py --input my_raw.csv         # use a different raw NSE CSV
  python dtc_nse.py --k 5 --window 40          # 5 regimes, 40-day windows
  python dtc_nse.py --joint_ep 200             # more training epochs

  Requires clean_nse_data.py in the same folder.
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import mode

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ── Optional: yfinance for auto-download ─────────────────────────────────────
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# ── Import cleaner from sibling module ───────────────────────────────────────
try:
    from data_cleaning import clean as clean_nse
except ImportError:
    raise ImportError(
        "clean_nse_data.py not found. Make sure it is in the same folder as dtc_nse.py."
    )

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA  —  download / load, feature engineering, windowing
# ─────────────────────────────────────────────────────────────────────────────

REGIME_COLOURS = {
    0: "#E74C3C",   # red    — bear
    1: "#2ECC71",   # green  — bull
    2: "#F39C12",   # amber  — sideways
    3: "#3498DB",   # blue   — high-vol
    4: "#9B59B6",   # purple — regime 5
}

def download_nse(ticker: str = "^NSEI", start: str = "2010-01-01",
                 end: str = "2024-12-31") -> pd.DataFrame:
    if not HAS_YF:
        raise ImportError("yfinance not installed. Run: pip install yfinance")
    print(f"Downloading {ticker} …")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"  {len(df)} trading days  ({df.index[0].date()} → {df.index[-1].date()})")
    return df



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log-returns + rolling technical features."""
    f = pd.DataFrame(index=df.index)

    # Returns
    f["log_ret"]     = np.log(df["Close"] / df["Close"].shift(1))
    f["ret_5d"]      = df["Close"].pct_change(5)
    f["ret_20d"]     = df["Close"].pct_change(20)

    # Volatility
    f["vol_10d"]     = f["log_ret"].rolling(10).std()
    f["vol_20d"]     = f["log_ret"].rolling(20).std()
    f["vol_60d"]     = f["log_ret"].rolling(60).std()

    # Trend
    f["ma_ratio"]    = df["Close"].rolling(20).mean() / df["Close"].rolling(60).mean() - 1

    # Momentum / RSI (14-period)
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    f["rsi"]         = 100 - 100 / (1 + rs)

    # Bollinger Band width
    ma20  = df["Close"].rolling(20).mean()
    sd20  = df["Close"].rolling(20).std()
    f["bb_width"]    = (2 * sd20) / (ma20 + 1e-9)

    # ATR (14-period)
    hl    = df["High"] - df["Low"]
    hc    = (df["High"] - df["Close"].shift(1)).abs()
    lc    = (df["Low"]  - df["Close"].shift(1)).abs()
    tr    = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    f["atr_norm"]    = tr.rolling(14).mean() / df["Close"]

    # Volume z-score
    if df["Volume"].std() > 0:
        f["vol_zscore"] = (df["Volume"] - df["Volume"].rolling(20).mean()) \
                          / (df["Volume"].rolling(20).std() + 1e-9)
    else:
        f["vol_zscore"] = 0.0

    return f.dropna()


def make_windows(features: pd.DataFrame, window: int = 30,
                 stride: int = 1) -> tuple:
    """
    Slide a fixed window across the feature matrix.
    Returns:
        X      : (N, window, n_features)  float32 tensor
        dates  : list of end-dates for each window
    """
    arr  = features.values.astype(np.float32)
    idx  = features.index
    X, dates = [], []

    for i in range(0, len(arr) - window + 1, stride):
        win = arr[i : i + window]
        # Z-score normalise within each window
        mu, sd = win.mean(axis=0), win.std(axis=0) + 1e-9
        X.append((win - mu) / sd)
        dates.append(idx[i + window - 1])

    return np.stack(X).astype(np.float32), dates


# ─────────────────────────────────────────────────────────────────────────────
# 2.  MODEL  —  encoder, decoder, clustering layer
# ─────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """Conv1D  →  BiLSTM  →  latent vector."""

    def __init__(self, n_features: int, latent_dim: int = 32,
                 conv_filters: int = 32, lstm_hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv_filters, conv_filters, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(conv_filters, lstm_hidden, batch_first=True,
                              bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, latent_dim)

    def forward(self, x):
        # x : (B, T, F)
        h = self.conv(x.permute(0, 2, 1))   # (B, conv_filters, T)
        h = h.permute(0, 2, 1)               # (B, T, conv_filters)
        _, (hn, _) = self.bilstm(h)          # hn: (2, B, lstm_hidden)
        z = torch.cat([hn[0], hn[1]], dim=1) # (B, lstm_hidden*2)
        return self.fc(z)                    # (B, latent_dim)


class TemporalDecoder(nn.Module):
    """Latent vector  →  reconstructed time-series window."""

    def __init__(self, latent_dim: int, window: int, n_features: int,
                 lstm_hidden: int = 64):
        super().__init__()
        self.window     = window
        self.n_features = n_features
        self.fc         = nn.Linear(latent_dim, lstm_hidden * 2)
        self.lstm       = nn.LSTM(lstm_hidden * 2, lstm_hidden,
                                  batch_first=True)
        self.out        = nn.Linear(lstm_hidden, n_features)

    def forward(self, z):
        h = self.fc(z).unsqueeze(1).repeat(1, self.window, 1)  # (B, T, H*2)
        h, _ = self.lstm(h)
        return self.out(h)                                      # (B, T, F)


class ClusteringLayer(nn.Module):
    """
    Soft cluster assignments via Student-t kernel (DEC / IDEC style).
    q_ij  ∝  (1 + ||z_i - µ_j||² / α)^{-(α+1)/2}
    """

    def __init__(self, n_clusters: int, latent_dim: int, alpha: float = 1.0):
        super().__init__()
        self.alpha    = alpha
        self.centroids = nn.Parameter(torch.randn(n_clusters, latent_dim))

    def forward(self, z):
        # z : (B, latent_dim)
        dist = torch.cdist(z, self.centroids)   # (B, K)
        q    = (1 + dist ** 2 / self.alpha) ** (-(self.alpha + 1) / 2)
        return q / q.sum(dim=1, keepdim=True)   # (B, K)  soft assignments


class DTC(nn.Module):
    """Full Deep Temporal Clustering model."""

    def __init__(self, n_features: int, window: int, n_clusters: int,
                 latent_dim: int = 32, conv_filters: int = 32,
                 lstm_hidden: int = 64):
        super().__init__()
        self.encoder  = TemporalEncoder(n_features, latent_dim,
                                        conv_filters, lstm_hidden)
        self.decoder  = TemporalDecoder(latent_dim, window, n_features,
                                        lstm_hidden)
        self.cluster  = ClusteringLayer(n_clusters, latent_dim)

    def forward(self, x):
        z    = self.encoder(x)
        xhat = self.decoder(z)
        q    = self.cluster(z)
        return z, xhat, q


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TRAINING  —  phase 1 AE pretraining, phase 2 k-means init, phase 3 joint
# ─────────────────────────────────────────────────────────────────────────────

def target_distribution(q: torch.Tensor) -> torch.Tensor:
    """Sharpen soft assignments into target distribution p."""
    f = q.sum(dim=0, keepdim=True)
    p = q ** 2 / (f + 1e-9)
    return p / p.sum(dim=1, keepdim=True)


def pretrain_autoencoder(model: DTC, loader: DataLoader,
                         epochs: int = 50, lr: float = 1e-3,
                         device: str = "cpu") -> list:
    model.train()
    opt = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder.parameters()),
        lr=lr)
    losses = []
    print("\n── Phase 1: Autoencoder pretraining ──────────────────────────")
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            z, xhat, _ = model(xb)
            loss = F.mse_loss(xhat, xb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(loader)
        losses.append(avg)
        if ep % 10 == 0:
            print(f"  Epoch {ep:3d}/{epochs}  |  recon loss: {avg:.6f}")
    return losses


def init_cluster_centroids(model: DTC, loader: DataLoader,
                           n_clusters: int, device: str = "cpu"):
    """Run k-means on frozen encoder embeddings to initialise µ."""
    print("\n── Phase 2: k-means centroid initialisation ──────────────────")
    model.eval()
    zs = []
    with torch.no_grad():
        for (xb,) in loader:
            zs.append(model.encoder(xb.to(device)).cpu().numpy())
    Z = np.concatenate(zs)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    km.fit(Z)
    model.cluster.centroids.data = torch.tensor(
        km.cluster_centers_, dtype=torch.float32).to(device)
    print(f"  k-means inertia: {km.inertia_:.4f}")
    return km.labels_


def joint_train(model: DTC, loader: DataLoader,
                epochs: int = 100, lr: float = 1e-4,
                alpha_recon: float = 1.0, beta_clust: float = 0.1,
                update_interval: int = 5, device: str = "cpu") -> list:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    print("\n── Phase 3: Joint fine-tuning ────────────────────────────────")
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            z, xhat, q = model(xb)
            # Update target distribution every update_interval epochs
            if ep % update_interval == 0:
                with torch.no_grad():
                    p = target_distribution(q).detach()
            else:
                with torch.no_grad():
                    p = target_distribution(q).detach()

            loss_recon = F.mse_loss(xhat, xb)
            loss_clust = F.kl_div(q.log(), p, reduction="batchmean")
            loss = alpha_recon * loss_recon + beta_clust * loss_clust
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        avg = ep_loss / len(loader)
        losses.append(avg)
        if ep % 10 == 0:
            print(f"  Epoch {ep:3d}/{epochs}  |  total loss: {avg:.6f}")
    return losses


# ─────────────────────────────────────────────────────────────────────────────
# 4.  EVALUATION  —  silhouette, regime stats, transition matrix
# ─────────────────────────────────────────────────────────────────────────────

def get_assignments(model: DTC, loader: DataLoader,
                    device: str = "cpu") -> tuple:
    model.eval()
    zs, qs = [], []
    with torch.no_grad():
        for (xb,) in loader:
            z, _, q = model(xb.to(device))
            zs.append(z.cpu().numpy())
            qs.append(q.cpu().numpy())
    Z = np.concatenate(zs)
    Q = np.concatenate(qs)
    labels = Q.argmax(axis=1)
    return Z, Q, labels


def regime_stats(prices: pd.Series, labels: np.ndarray,
                 dates: list, n_clusters: int) -> pd.DataFrame:
    """Per-regime: mean daily return, volatility, Sharpe, hit count."""
    df = pd.DataFrame({"date": dates, "regime": labels})
    df = df.set_index("date")
    log_ret = np.log(prices / prices.shift(1)).dropna()
    df["log_ret"] = log_ret.reindex(df.index)

    rows = []
    for k in range(n_clusters):
        r = df[df["regime"] == k]["log_ret"].dropna()
        rows.append({
            "regime":   k,
            "count":    len(r),
            "mean_ret": r.mean() * 252,          # annualised
            "vol":      r.std() * np.sqrt(252),
            "sharpe":   (r.mean() / (r.std() + 1e-9)) * np.sqrt(252),
        })
    return pd.DataFrame(rows).set_index("regime")


def transition_matrix(labels: np.ndarray, n_clusters: int) -> pd.DataFrame:
    mat = np.zeros((n_clusters, n_clusters))
    for i in range(len(labels) - 1):
        mat[labels[i], labels[i + 1]] += 1
    row_sums = mat.sum(axis=1, keepdims=True) + 1e-9
    return pd.DataFrame(mat / row_sums,
                        index=[f"R{k}" for k in range(n_clusters)],
                        columns=[f"R{k}" for k in range(n_clusters)])


# ─────────────────────────────────────────────────────────────────────────────
# 5.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

REGIME_NAMES = ["Bear", "Bull", "Sideways", "High-Vol", "R4", "R5"]

def assign_regime_names(stats: pd.DataFrame) -> dict:
    """Heuristically name regimes by return/vol profile."""
    names = {}
    for k, row in stats.iterrows():
        if row["mean_ret"] > 0.10 and row["vol"] < 0.20:
            names[k] = "Bull"
        elif row["mean_ret"] < -0.05:
            names[k] = "Bear"
        elif row["vol"] > 0.25:
            names[k] = "High-Vol"
        else:
            names[k] = "Sideways"
    # De-duplicate
    seen = {}
    for k, v in names.items():
        if v in seen.values():
            names[k] = f"{v}-{k}"
        seen[k] = names[k]
    return names


def plot_results(prices: pd.Series, labels: np.ndarray, dates: list,
                 stats: pd.DataFrame, trans: pd.DataFrame,
                 losses_pre: list, losses_joint: list,
                 n_clusters: int, save_path: str = "dtc_results.png"):

    regime_map = assign_regime_names(stats)
    colours    = [REGIME_COLOURS.get(k, "#95A5A6") for k in range(n_clusters)]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Deep Temporal Clustering — NSE Regime Detection",
                 fontsize=15, fontweight="bold", y=0.98)

    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.35)
    ax_price  = fig.add_subplot(gs[0, :])
    ax_regime = fig.add_subplot(gs[1, :])
    ax_stats  = fig.add_subplot(gs[2, 0])
    ax_trans  = fig.add_subplot(gs[2, 1])
    ax_loss   = fig.add_subplot(gs[2, 2])

    # ── Price + regime shading ──────────────────────────────────────────────
    ax_price.plot(prices.index, prices.values, color="#2C3E50", lw=0.8)
    ax_price.set_title("Nifty 50 — close price with regime overlay", fontsize=11)
    ax_price.set_ylabel("Index level")

    # Drop NaT dates before plotting
    dates_pd  = pd.to_datetime(dates)
    valid     = ~dates_pd.isna()
    date_arr  = dates_pd[valid]
    labels_v  = labels[valid]

    ymin, ymax = prices.min() * 0.98, prices.max() * 1.02
    for k in range(n_clusters):
        mask = labels_v == k
        days = date_arr[mask]
        if len(days) == 0:
            continue
        ax_price.fill_between(days, ymin, ymax,
                              alpha=0.18, color=colours[k], lw=0)

    # ── Regime strip ────────────────────────────────────────────────────────
    regime_series = pd.Series(labels_v, index=date_arr)
    ax_regime.set_ylim(-0.5, 0.5)
    ax_regime.set_yticks([])
    ax_regime.set_title("Regime label per trading day", fontsize=11)
    for k in range(n_clusters):
        mask = regime_series == k
        days = regime_series.index[mask]
        if len(days) == 0:
            continue
        ax_regime.scatter(days, [0] * mask.sum(),
                          c=colours[k], s=4, marker="|", linewidths=1.5)

    patches = [mpatches.Patch(color=colours[k],
                              label=f"R{k}: {regime_map.get(k, '')}")
               for k in range(n_clusters)]
    ax_regime.legend(handles=patches, loc="upper right",
                     fontsize=7, ncol=n_clusters)

    # ── Per-regime stats bar chart ───────────────────────────────────────────
    x = np.arange(n_clusters)
    w = 0.35
    ax_stats.bar(x - w/2, stats["mean_ret"] * 100, w,
                 color=colours, alpha=0.85, label="Ann. return %")
    ax_stats.bar(x + w/2, stats["vol"] * 100, w,
                 color=colours, alpha=0.45, hatch="//", label="Ann. vol %")
    ax_stats.axhline(0, color="black", lw=0.6)
    ax_stats.set_xticks(x)
    ax_stats.set_xticklabels([f"R{k}" for k in range(n_clusters)], fontsize=8)
    ax_stats.set_title("Return & vol per regime", fontsize=10)
    ax_stats.set_ylabel("% (annualised)")
    ax_stats.legend(fontsize=7)

    # ── Transition matrix heatmap ────────────────────────────────────────────
    im = ax_trans.imshow(trans.values, cmap="Blues", vmin=0, vmax=1)
    ax_trans.set_xticks(range(n_clusters))
    ax_trans.set_yticks(range(n_clusters))
    ax_trans.set_xticklabels([f"R{k}" for k in range(n_clusters)], fontsize=8)
    ax_trans.set_yticklabels([f"R{k}" for k in range(n_clusters)], fontsize=8)
    ax_trans.set_title("Regime transition matrix", fontsize=10)
    ax_trans.set_xlabel("To"); ax_trans.set_ylabel("From")
    for i in range(n_clusters):
        for j in range(n_clusters):
            ax_trans.text(j, i, f"{trans.values[i,j]:.2f}",
                          ha="center", va="center", fontsize=7,
                          color="white" if trans.values[i,j] > 0.6 else "black")
    fig.colorbar(im, ax=ax_trans, fraction=0.046, pad=0.04)

    # ── Training loss curves ─────────────────────────────────────────────────
    ax_loss.plot(losses_pre,   color="#E74C3C", lw=1.2, label="AE pretraining")
    ax_loss.plot(range(len(losses_pre),
                       len(losses_pre) + len(losses_joint)),
                 losses_joint, color="#3498DB", lw=1.2, label="Joint fine-tune")
    ax_loss.set_title("Training loss", fontsize=10)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.legend(fontsize=7)
    ax_loss.set_yscale("log")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  MAIN  —  pipeline orchestration
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DTC for NSE regime detection")
    p.add_argument("--ticker",       default="^NSEI",       help="Yahoo Finance ticker (fallback if input file not found)")
    p.add_argument("--start",        default="2010-01-01",  help="Start date for yfinance fallback")
    p.add_argument("--end",          default="2024-12-31",  help="End date for yfinance fallback")
    p.add_argument("--input",        default="cleaned_nse_data.csv", help="Path to raw NSE CSV (will be cleaned automatically)")
    p.add_argument("--window",       type=int,   default=30,  help="Sliding window length (trading days)")
    p.add_argument("--stride",       type=int,   default=1,   help="Window stride")
    p.add_argument("--k",            type=int,   default=4,   help="Number of regimes")
    p.add_argument("--latent_dim",   type=int,   default=32,  help="Latent space dimensionality")
    p.add_argument("--pretrain_ep",  type=int,   default=50,  help="AE pretraining epochs")
    p.add_argument("--joint_ep",     type=int,   default=100, help="Joint training epochs")
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--lr_pre",       type=float, default=1e-3)
    p.add_argument("--lr_joint",     type=float, default=1e-4)
    p.add_argument("--alpha_recon",  type=float, default=1.0,  help="Reconstruction loss weight")
    p.add_argument("--beta_clust",   type=float, default=0.1,  help="Clustering loss weight")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no_plot",      action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── 1. Load and clean data ───────────────────────────────────────────────
    import os
    if os.path.exists(args.input):
        print(f"\nCleaning raw data from: {args.input}")
        raw = clean_nse(args.input, verbose=True)
    else:
        print(f"  '{args.input}' not found — falling back to yfinance download.")
        raw = download_nse(args.ticker, args.start, args.end)

    # ── 2. Feature engineering ────────────────────────────────────────────────
    features = engineer_features(raw)
    n_features = features.shape[1]
    print(f"\nFeatures: {list(features.columns)}")
    print(f"Feature matrix shape: {features.shape}")

    # ── 3. Windowing ──────────────────────────────────────────────────────────
    X, dates = make_windows(features, window=args.window, stride=args.stride)
    print(f"\nWindows: {X.shape}  (N={X.shape[0]}, T={X.shape[1]}, F={X.shape[2]})")

    tensor_X = torch.tensor(X)
    loader   = DataLoader(TensorDataset(tensor_X),
                          batch_size=args.batch_size, shuffle=True,
                          num_workers=0)
    full_loader = DataLoader(TensorDataset(tensor_X),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=0)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    model = DTC(
        n_features  = n_features,
        window      = args.window,
        n_clusters  = args.k,
        latent_dim  = args.latent_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # ── 5. Phase 1: AE pretraining ────────────────────────────────────────────
    losses_pre = pretrain_autoencoder(
        model, loader, epochs=args.pretrain_ep,
        lr=args.lr_pre, device=device)

    # ── 6. Phase 2: k-means init ──────────────────────────────────────────────
    init_labels = init_cluster_centroids(
        model, full_loader, n_clusters=args.k, device=device)

    # ── 7. Phase 3: Joint fine-tuning ─────────────────────────────────────────
    losses_joint = joint_train(
        model, loader, epochs=args.joint_ep,
        lr=args.lr_joint, alpha_recon=args.alpha_recon,
        beta_clust=args.beta_clust, device=device)

    # ── 8. Assignments & evaluation ───────────────────────────────────────────
    Z, Q, labels = get_assignments(model, full_loader, device=device)

    sil = silhouette_score(Z, labels) if len(set(labels)) > 1 else float("nan")
    print(f"\n── Results ───────────────────────────────────────────────────")
    print(f"  Silhouette score : {sil:.4f}  (higher = better separated)")
    print(f"  Regime counts    : { {k: (labels==k).sum() for k in range(args.k)} }")

    # Align label dates with price series
    close = raw["Close"]
    if close.index.duplicated().any():
        close = close[~close.index.duplicated(keep="first")]
    price_aligned = close.reindex(dates)
    stats = regime_stats(close, labels, dates, args.k)
    trans = transition_matrix(labels, args.k)

    print("\n── Per-regime statistics (annualised) ────────────────────────")
    print(stats.to_string())
    print("\n── Transition matrix ─────────────────────────────────────────")
    print(trans.to_string())

    # ── 9. Save assignments CSV ───────────────────────────────────────────────
    out_df = pd.DataFrame({
        "date":   dates,
        "regime": labels,
        **{f"q_{k}": Q[:, k] for k in range(args.k)}
    }).set_index("date")
    out_df.to_csv("dtc_regime_assignments.csv")
    print("\n  Assignments saved → dtc_regime_assignments.csv")

    # Export regime performance for Power BI
    regime_map = assign_regime_names(stats)
    stats_export = stats.copy()
    stats_export['regime_name'] = [regime_map.get(k, f'R{k}') for k in stats_export.index]
    stats_export['Ann_Return'] = stats_export['mean_ret'] * 100
    stats_export['Ann_Vol'] = stats_export['vol'] * 100
    stats_export['Sharpe'] = stats_export['sharpe']
    stats_export[['regime_name', 'Ann_Return', 'Ann_Vol', 'Sharpe', 'count']].to_csv('regime_performance.csv')
    print("Performance saved → regime_performance.csv")

    # ── 10. Plot ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_results(
            prices     = raw["Close"],
            labels     = labels,
            dates      = dates,
            stats      = stats,
            trans      = trans,
            losses_pre = losses_pre,
            losses_joint = losses_joint,
            n_clusters = args.k,
        )


if __name__ == "__main__":
    main()