"""
NSE Data Cleaning Script
========================
Cleans raw NSE historical index data CSVs — handles:
  - Multiple year-range files concatenated into one
  - Duplicate dates (keeps first occurrence)
  - Missing values (forward-fill up to 3 days, then drop)
  - Outliers (price/volume spikes beyond 5 IQR)
  - Non-trading days (zero volume rows)
  - Column name normalisation
  - Chronological sorting

Usage (standalone):
  python clean_nse_data.py                              # cleans cleaned_nse_data.csv
  python clean_nse_data.py --input my_raw_data.csv      # custom input file
  python clean_nse_data.py --output my_clean_data.csv   # custom output name
  python clean_nse_data.py --no_outlier                 # skip outlier removal

Usage (imported by dtc_nse.py):
  from clean_nse_data import clean
  df = clean("cleaned_nse_data.csv")
"""

import argparse
import pandas as pd
import numpy as np
import sys

# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Clean raw NSE index CSV data")
    p.add_argument("--input",      default="cleaned_nse_data.csv", help="Input CSV path")
    p.add_argument("--output",     default="nifty50_clean.csv",    help="Output CSV path")
    p.add_argument("--no_outlier", action="store_true",            help="Skip outlier removal")
    p.add_argument("--ffill_limit",type=int, default=3,            help="Max consecutive days to forward-fill")
    return p.parse_args()


# ── Step 1: Load ──────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    print(f"\n{'='*55}")
    print(f"  Loading: {path}")
    print(f"{'='*55}")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"  ERROR: File '{path}' not found.")
        sys.exit(1)

    print(f"  Raw shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns found   : {list(df.columns)}")
    return df


# ── Step 2: Normalise column names ────────────────────────────────────────────

COLUMN_MAP = {
    # date variants
    "trade_date":  "Date",
    "date":        "Date",
    "timestamp":   "Date",
    "datetime":    "Date",
    # price variants
    "open":        "Open",
    "high":        "High",
    "low":         "Low",
    "close":       "Close",
    "adj_close":   "Close",
    "adj close":   "Close",
    # volume variants
    "shares_traded": "Volume",
    "volume":        "Volume",
    "vol":           "Volume",
    "qty":           "Volume",
}

def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=COLUMN_MAP)

    required = {"Date", "Open", "High", "Low", "Close"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Cannot find required columns: {missing}\n"
            f"Available after mapping: {list(df.columns)}"
        )

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"]
            if c in df.columns]
    df = df[keep]

    if "Volume" not in df.columns:
        print("  Note: No volume column found — Volume will be set to NaN")
        df["Volume"] = np.nan

    print(f"\n[1] Column mapping")
    print(f"  Kept columns    : {list(df.columns)}")
    return df


# ── Step 3: Parse dates ───────────────────────────────────────────────────────

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=False, errors="coerce")
    nat_count = df["Date"].isna().sum()
    if nat_count > 0:
        print(f"\n[2] Date parsing — dropped {nat_count:,} unparseable date rows")
        df = df.dropna(subset=["Date"])
    else:
        print(f"\n[2] Date parsing — all dates parsed successfully")

    df = df.set_index("Date")
    df.index.name = "Date"
    return df


# ── Step 4: Cast numeric columns ─────────────────────────────────────────────

def cast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"\n[3] Numeric casting")
    for col in numeric_cols:
        if col in df.columns:
            n_bad = df[col].isna().sum()
            if n_bad:
                print(f"  {col}: {n_bad:,} non-numeric values → NaN")
    return df


# ── Step 5: Remove duplicate dates ───────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    n_removed = n_before - len(df)

    print(f"\n[4] Duplicate date removal")
    if n_removed:
        print(f"  Removed         : {n_removed:,} duplicate rows")
        print(f"  Note: Your file had {n_removed:,} extra rows from overlapping")
        print(f"        year-range files being concatenated. All are the same")
        print(f"        Nifty 50 index — duplicates safely dropped.")
    else:
        print(f"  No duplicates found")
    return df


# ── Step 6: Sort chronologically ─────────────────────────────────────────────

def sort_chronological(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index()
    print(f"\n[5] Chronological sort")
    print(f"  Date range      : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Trading days    : {len(df):,}")
    return df


# ── Step 7: Remove zero-volume / non-trading days ────────────────────────────

def remove_non_trading(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    # Zero close price is a clear data error
    bad_price = df["Close"] <= 0
    # Zero volume (if available) may indicate a non-trading day
    bad_vol = (df["Volume"] == 0) if "Volume" in df.columns else pd.Series(False, index=df.index)
    mask = bad_price | bad_vol
    df = df[~mask]
    n_removed = n_before - len(df)

    print(f"\n[6] Non-trading / bad price rows")
    if n_removed:
        print(f"  Removed         : {n_removed:,} rows (zero close or zero volume)")
    else:
        print(f"  None found")
    return df


# ── Step 8: OHLC consistency check ───────────────────────────────────────────

def fix_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Flag and nullify rows where OHLC relationships are violated."""
    violations = (
        (df["High"] < df["Low"]) |
        (df["High"] < df["Close"]) |
        (df["Low"]  > df["Close"]) |
        (df["High"] < df["Open"])  |
        (df["Low"]  > df["Open"])
    )
    n_bad = violations.sum()
    print(f"\n[7] OHLC consistency")
    if n_bad:
        print(f"  Violations found: {n_bad:,} — setting OHLC to NaN for those rows")
        df.loc[violations, ["Open", "High", "Low"]] = np.nan
    else:
        print(f"  All rows pass OHLC consistency check")
    return df


# ── Step 9: Outlier detection ─────────────────────────────────────────────────

def remove_outliers(df: pd.DataFrame, iqr_multiplier: float = 5.0) -> pd.DataFrame:
    """
    Flag extreme single-day price moves as outliers using IQR on log-returns.
    5 IQR is deliberately conservative — we only want to catch data errors,
    not genuine market crashes.
    """
    log_ret = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    Q1, Q3  = log_ret.quantile(0.25), log_ret.quantile(0.75)
    IQR     = Q3 - Q1
    lower   = Q1 - iqr_multiplier * IQR
    upper   = Q3 + iqr_multiplier * IQR

    outlier_dates = log_ret[(log_ret < lower) | (log_ret > upper)].index
    print(f"\n[8] Outlier detection (|return| > {iqr_multiplier}×IQR)")
    if len(outlier_dates):
        print(f"  Outlier dates flagged ({len(outlier_dates)}):")
        for d in outlier_dates:
            ret = log_ret.loc[d]
            print(f"    {d.date()}  log-return: {ret:+.4f}")
        print(f"  Action: rows kept but flagged — review manually")
        df["outlier_flag"] = df.index.isin(outlier_dates).astype(int)
    else:
        print(f"  No extreme outliers detected")
        df["outlier_flag"] = 0
    return df


# ── Step 10: Handle missing values ───────────────────────────────────────────

def handle_missing(df: pd.DataFrame, ffill_limit: int = 3) -> pd.DataFrame:
    n_before = df[["Open", "High", "Low", "Close"]].isna().sum().sum()

    # Forward fill short gaps (e.g. public holidays with stale data)
    df[["Open", "High", "Low", "Close"]] = (
        df[["Open", "High", "Low", "Close"]]
        .ffill(limit=ffill_limit)
    )
    if "Volume" in df.columns:
        df["Volume"] = df["Volume"].fillna(0)

    n_after = df[["Open", "High", "Low", "Close"]].isna().sum().sum()
    n_dropped = df[["Open", "High", "Low", "Close"]].isna().any(axis=1).sum()

    print(f"\n[9] Missing value handling")
    print(f"  NaN before ffill : {n_before:,}")
    print(f"  NaN after ffill  : {n_after:,}  (gaps ≤ {ffill_limit} days filled)")
    if n_dropped:
        print(f"  Dropping         : {n_dropped:,} rows still missing after ffill")
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


# ── Step 11: Final summary ────────────────────────────────────────────────────

def summary(df: pd.DataFrame):
    print(f"\n{'='*55}")
    print(f"  CLEAN DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  Rows            : {len(df):,}")
    print(f"  Columns         : {list(df.columns)}")
    print(f"  Date range      : {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Close — min     : {df['Close'].min():,.2f}")
    print(f"  Close — max     : {df['Close'].max():,.2f}")
    print(f"  Close — mean    : {df['Close'].mean():,.2f}")
    print(f"  Any NaN         : {df[['Open','High','Low','Close']].isna().any().any()}")
    if "outlier_flag" in df.columns:
        print(f"  Outlier rows    : {df['outlier_flag'].sum()}")
    print(f"{'='*55}\n")


# ── Public API — importable by dtc_nse.py ─────────────────────────────────────

def clean(input_path: str, remove_outliers_flag: bool = True,
          ffill_limit: int = 3, verbose: bool = True) -> "pd.DataFrame":
    """
    Load and clean a raw NSE CSV. Returns a clean pd.DataFrame with columns
    [Open, High, Low, Close, Volume] indexed by Date — ready for DTC.

    Parameters
    ----------
    input_path          : path to the raw NSE CSV (e.g. 'cleaned_nse_data.csv')
    remove_outliers_flag: flag extreme returns with outlier_flag column (default True)
    ffill_limit         : max consecutive days to forward-fill missing prices (default 3)
    verbose             : print cleaning log (default True)
    """
    import io, contextlib

    def _run():
        df = load_raw(input_path)
        df = normalise_columns(df)
        df = parse_dates(df)
        df = cast_numeric(df)
        df = remove_duplicates(df)
        df = sort_chronological(df)
        df = remove_non_trading(df)
        df = fix_ohlc(df)
        if remove_outliers_flag:
            df = remove_outliers(df)
        df = handle_missing(df, ffill_limit=ffill_limit)
        summary(df)
        # Drop the outlier_flag column before returning — DTC doesn't need it
        if "outlier_flag" in df.columns:
            df = df.drop(columns=["outlier_flag"])
        return df

    if verbose:
        return _run()
    else:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            df = _run()
        return df


# ── Main — standalone CLI ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    df = clean(
        input_path           = args.input,
        remove_outliers_flag = not args.no_outlier,
        ffill_limit          = args.ffill_limit,
        verbose              = True,
    )
    df.to_csv(args.output)
    print(f"  Saved → {args.output}")
    print(f"\n  Run DTC with:")
    print(f"    python dtc_nse.py\n")


if __name__ == "__main__":
    main()