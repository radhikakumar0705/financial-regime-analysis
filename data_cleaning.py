import pandas as pd

def clean_nse_csv(filepath, symbol):
    df = pd.read_csv(filepath)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.strip("_")
    )

    df = df.rename(columns={'date': 'trade_date'})

    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df = df.dropna(subset=['trade_date'])

    df['symbol'] = symbol

    numeric_cols = [
        'open', 'high', 'low', 'close',
        'shares_traded', 'turnover_cr'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df[
        ['symbol', 'trade_date', 'open', 'high',
         'low', 'close', 'shares_traded', 'turnover_cr']
    ]

    df = df.sort_values('trade_date').reset_index(drop=True)

    return df
