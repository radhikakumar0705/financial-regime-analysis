import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data\cleaned_nse_data.xls"
OUTPUT_PATH = "hmm_features.csv"

VOL_WINDOW = 20
MEAN_WINDOW = 20
MOM_WINDOW = 10

df = pd.read_csv(DATA_PATH)

col_map = {}
for col in df.columns:
    lc = col.lower().strip()
    if lc in ['trade_date','date','datetime','timestamp']:
        col_map['date'] = col
    elif lc in ['close','close_price','adj_close']:
        col_map['close'] = col
    elif lc in ['volume','vol','traded_quantity']:
        col_map['volume'] = col

df = df.rename(columns={col_map['date']:'date', col_map['close']:'close'})
df['date'] = pd.to_datetime(df['date'])
df['close'] = pd.to_numeric(df['close'], errors='coerce')

df = df.sort_values('date').dropna(subset=['close']).reset_index(drop=True)

has_volume = 'volume' in col_map
if has_volume:
    df = df.rename(columns={col_map['volume']:'volume'})
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

# core signals
df['log_return'] = np.log(df['close'] / df['close'].shift(1))
df['rolling_vol'] = df['log_return'].rolling(VOL_WINDOW).std()
df['rolling_mean'] = df['log_return'].rolling(MEAN_WINDOW).mean()
df['momentum'] = df['close'].pct_change(MOM_WINDOW)

if has_volume:
    vol_ma = df['volume'].rolling(VOL_WINDOW).mean()
    df['volume_change'] = (df['volume'] - vol_ma) / vol_ma
    FEATS = ['log_return','rolling_vol','rolling_mean','momentum','volume_change']
else:
    FEATS = ['log_return','rolling_vol','rolling_mean','momentum']

df = df.dropna(subset=FEATS).reset_index(drop=True)

scaler = StandardScaler()
scaled = scaler.fit_transform(df[FEATS])

scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in FEATS])

out = pd.concat([df[['date','close']+FEATS], scaled_df], axis=1)

out.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print("Features:", FEATS)