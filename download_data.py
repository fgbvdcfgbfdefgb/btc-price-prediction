import os
import io
import zipfile
import requests
import pandas as pd
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

MONTHS = [
    "2025-09", "2025-10", "2025-11", "2025-12",
    "2026-01", "2026-02", "2026-03", "2026-04",
    "2026-05", "2026-06", "2026-07", "2026-08"
]

BINANCE_MONTHLY_URL = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{month}.zip"

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

def parse_timestamp_series(s):
    first_val = s.iloc[0]
    if first_val > 1e14:
        return pd.to_datetime(s.astype(np.int64), unit="us", utc=True)
    elif first_val > 1e11:
        return pd.to_datetime(s.astype(np.int64), unit="ms", utc=True)
    else:
        return pd.to_datetime(s.astype(np.int64), unit="s", utc=True)

def download_and_process_btc_1m():
    dfs = []
    print(f"Starting download of 1-year 1-minute Bitcoin (BTC/USDT) data across {len(MONTHS)} months...")
    
    for m in MONTHS:
        url = BINANCE_MONTHLY_URL.format(month=m)
        print(f"Fetching {m} from {url}...")
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    first_line = f.readline().decode('utf-8')
                    f.seek(0)
                    has_header = "open_time" in first_line.lower() or "open" in first_line.lower()
                    
                    df_m = pd.read_csv(
                        f,
                        header=0 if has_header else None,
                        names=COLUMNS if not has_header else None,
                        dtype=float,
                        usecols=range(len(COLUMNS))
                    )
                    if has_header:
                        df_m.columns = COLUMNS[:len(df_m.columns)]
                    
                    df_m["open_time"] = parse_timestamp_series(df_m["open_time"])
                    df_m["close_time"] = parse_timestamp_series(df_m["close_time"])
                    dfs.append(df_m)
            print(f"  -> Loaded {len(df_m):,} rows for {m} ({df_m['open_time'].min().strftime('%Y-%m-%d')} to {df_m['open_time'].max().strftime('%Y-%m-%d')})")
        else:
            print(f"  [!] Failed to download {m} (status {resp.status_code})")

    if not dfs:
        raise RuntimeError("No data downloaded successfully!")

    df = pd.concat(dfs, ignore_index=True)
    
    # Sort and remove duplicates
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    
    # Keep standard OHLCV and trading activity columns
    keep_cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume"
    ]
    df = df[keep_cols]
    
    print("\n" + "="*60)
    print("DATASET INGESTION SUMMARY")
    print("="*60)
    print(f"Total 1-Minute Bars: {len(df):,}")
    print(f"Earliest Timestamp:  {df['open_time'].min()}")
    print(f"Latest Timestamp:    {df['open_time'].max()}")
    print(f"Minimum Price:       ${df['low'].min():,.2f}")
    print(f"Maximum Price:       ${df['high'].max():,.2f}")
    print(f"Latest Close Price:  ${df['close'].iloc[-1]:,.2f}")
    print(f"Total Volume (BTC):  {df['volume'].sum():,.2f}")
    print(f"Total Quotes (USDT): ${df['quote_volume'].sum():,.2f}")
    print("="*60)
    
    # Save as Parquet
    parquet_path = os.path.join(DATA_DIR, "btc_1m_1year.parquet")
    csv_path = os.path.join(DATA_DIR, "btc_1m_sample.csv")
    
    df.to_parquet(parquet_path, index=False, compression="snappy")
    print(f"Saved complete 1-year dataset to {parquet_path} ({os.path.getsize(parquet_path)/1024/1024:.2f} MB)")
    
    # Save a lightweight sample CSV for quick inspection
    df.head(500).to_csv(csv_path, index=False)
    print(f"Saved preview sample to {csv_path}")
    
    return df

if __name__ == "__main__":
    download_and_process_btc_1m()
