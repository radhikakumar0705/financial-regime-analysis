# Financial Market Regime Detection — NSE Nifty 50

> Unsupervised detection of market regimes in 20 years of Nifty 50 data using Deep Temporal Clustering (DTC), visualised through an interactive Power BI dashboard.

---

## Overview

This project applies **Deep Temporal Clustering (DTC)** — a deep learning approach combining Conv1D, BiLSTM, and soft clustering — to automatically detect distinct market regimes in the NSE Nifty 50 index from 2005 to 2025.

The detected regimes are visualised in a **5-page interactive Power BI dashboard**, covering price trends, volume analysis, event impact, and regime performance.

---

## Key Results

| Regime | Ann. Return | Ann. Volatility | Sharpe Ratio | Trading Days |
|--------|------------|-----------------|--------------|--------------|
| Bull | 23.3% | 18.0% | 1.29 | 1,193 |
| Bull-3 | 17.9% | 18.5% | 0.97 | 1,296 |
| Sideways | 5.2% | 19.1% | 0.27 | 1,198 |
| High-Vol | 5.3% | 25.9% | 0.21 | 1,432 |

**Key insight:** The Bull regime delivers the best risk-adjusted returns (Sharpe: 1.29) with relatively low volatility. The High-Vol regime dominates by day count (1,432 days) but offers the worst return per unit of risk.

---

## Model Architecture

```
Raw OHLCV Data
      |
      v
Feature Engineering
(log returns, RSI, Bollinger Bands, ATR, volatility, momentum)
      |
      v
Sliding Window (30-day)
      |
      v
+-----------------------------------+
|     Deep Temporal Clustering      |
|                                   |
|  Conv1D -> BiLSTM -> Latent z     |
|         |                         |
|    Clustering Layer               |
|  (Student-t soft assignments)     |
+-----------------------------------+
      |
      v
Regime Assignments + Probabilities
```

### Training Phases
1. **Phase 1 — Autoencoder Pretraining:** Conv1D + BiLSTM encoder/decoder trained with MSE reconstruction loss
2. **Phase 2 — K-means Initialisation:** Cluster centroids initialised from encoder embeddings
3. **Phase 3 — Joint Fine-tuning:** Combined MSE reconstruction loss + KL divergence clustering loss

---

## Power BI Dashboard

The dashboard consists of 5 pages:

| Page | Description |
|------|-------------|
| **Price Trend** | Nifty 50 close price 2005-2025, all-time high/low cards, date slicer |
| **Volume Analysis** | Average daily volume by year, volume trend, peak volume card |
| **Event Analysis** | Price chart with reference lines marking key market events (2008 crisis, demonetization, COVID crash, 2024 elections) |
| **Regime Overview** | Price line colored by regime, trading days per regime bar chart, regime slicer |
| **Regime Performance** | Return vs volatility bar chart, risk-return scatter plot, summary table |


---

## Repository Structure

```
financial-regime-analysis/
|
+-- data/
|   +-- dtc_regime_assignments.csv    # Per-day regime labels + soft probabilities
|   +-- regime_performance.csv        # Per-regime annualised stats
|
+-- models/
|   +-- dtc_nse.py                    # Main DTC model + training pipeline
|   +-- data_cleaning.py             # NSE data cleaning utilities
|
+-- powerbi/
|   +-- NSE_Dashboard.pbix           # Power BI dashboard file
|
+-- screenshots/
|   +-- page1_price_trend.png
|   +-- page2_volume.png
|   +-- page3_events.png
|   +-- page4_regime_overview.png
|   +-- page5_regime_performance.png
|
+-- dtc_results.png                   # Model output visualisation
+-- README.md
```

---

## How to Run

### Prerequisites
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

### Run the DTC model
```bash
# Default settings (4 regimes, 30-day window)
python models/dtc_nse.py

# Custom settings
python models/dtc_nse.py --k 5 --window 40 --joint_ep 200
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--k` | 4 | Number of regimes |
| `--window` | 30 | Sliding window length (days) |
| `--pretrain_ep` | 50 | Autoencoder pretraining epochs |
| `--joint_ep` | 100 | Joint training epochs |
| `--latent_dim` | 32 | Latent space dimensionality |

### View the Dashboard
1. Download and install [Power BI Desktop](https://powerbi.microsoft.com/desktop/) (free)
2. Open `powerbi/NSE_Dashboard.pbix`

---

## Features Engineered

| Feature | Description |
|---------|-------------|
| `log_ret` | Daily log return |
| `ret_5d`, `ret_20d` | 5-day and 20-day price change |
| `vol_10d`, `vol_20d`, `vol_60d` | Rolling volatility |
| `ma_ratio` | 20-day / 60-day MA ratio |
| `rsi` | 14-period Relative Strength Index |
| `bb_width` | Bollinger Band width |
| `atr_norm` | Normalised Average True Range |
| `vol_zscore` | Volume z-score |

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Model development |
| PyTorch | Deep learning (Conv1D, BiLSTM, clustering) |
| scikit-learn | K-means initialisation, silhouette scoring |
| pandas / numpy | Data processing |
| matplotlib | Model result visualisation |
| Power BI | Interactive dashboard |

---

## Reference

> Madiraju, N. S., Sadat, S. M., Fisher, D., & Karimanzira, D. (2018).
> *Deep Temporal Clustering: Fully Unsupervised Learning of Time-Domain Features.*
> arXiv:1802.01059


## License

This project is open source and available under the [MIT License](LICENSE).
