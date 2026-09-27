import os
import gc
import pickle
import numpy as np
import pandas as pd

DATA_DIR = "data"
PARQUET_PATH = os.path.join(DATA_DIR, "btc_1m_1year.parquet")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "btc_features_targets.npz")
SCALER_PATH = os.path.join(DATA_DIR, "scaler_params.pkl")

HORIZONS = [1, 5, 15, 30, 60, 240, 720]
HORIZON_NAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "12h"]

def prepare_dataset(parquet_file=PARQUET_PATH, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15):
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet file {parquet_file} not found. Run download_data.py first.")

    print(f"Loading raw data from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    n_samples = len(df)
    
    close = df['close'].to_numpy(dtype=np.float32)
    high = df['high'].to_numpy(dtype=np.float32)
    low = df['low'].to_numpy(dtype=np.float32)
    open_p = df['open'].to_numpy(dtype=np.float32)
    vol = df['volume'].to_numpy(dtype=np.float32)
    q_vol = df['quote_volume'].to_numpy(dtype=np.float32)
    taker_vol = df['taker_buy_volume'].to_numpy(dtype=np.float32)
    open_time = df['open_time'].values

    del df
    gc.collect()

    print("Computing 48 multi-timeframe features & multi-horizon targets in float32...")
    feature_names = []
    X_mat = np.zeros((n_samples, 48), dtype=np.float32)
    col = 0

    # 1. Multi-Scale Log Returns
    close_s = pd.Series(close)
    for w in [1, 3, 5, 15, 30, 60, 120, 240, 720]:
        ret = np.log(close_s / close_s.shift(w)).to_numpy(dtype=np.float32)
        X_mat[:, col] = ret
        feature_names.append(f"return_{w}m")
        col += 1

    # 2. Geometry
    X_mat[:, col] = (high - low) / (close + 1e-8); feature_names.append("hl_ratio"); col += 1
    X_mat[:, col] = (close - open_p) / (open_p + 1e-8); feature_names.append("co_ratio"); col += 1
    X_mat[:, col] = (high - np.maximum(close, open_p)) / (close + 1e-8); feature_names.append("upper_shadow"); col += 1
    X_mat[:, col] = (np.minimum(close, open_p) - low) / (close + 1e-8); feature_names.append("lower_shadow"); col += 1

    # 3. EMAs
    for span in [5, 15, 30, 60, 120, 240, 720]:
        ema = close_s.ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float32)
        X_mat[:, col] = (close - ema) / (ema + 1e-8); feature_names.append(f"ema_dist_{span}"); col += 1
        ema_s = pd.Series(ema)
        X_mat[:, col] = ((ema_s - ema_s.shift(3)) / (ema_s.shift(3) + 1e-8)).to_numpy(dtype=np.float32)
        feature_names.append(f"ema_slope_{span}"); col += 1

    # 4. RSI
    def calc_rsi(s, p):
        delta = s.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=p).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=p).mean()
        rs = gain / (loss + 1e-9)
        return (100 - (100 / (1 + rs))).to_numpy(dtype=np.float32) / 100.0 - 0.5

    X_mat[:, col] = calc_rsi(close_s, 14); feature_names.append("rsi_14"); col += 1
    X_mat[:, col] = calc_rsi(close_s, 60); feature_names.append("rsi_60"); col += 1
    X_mat[:, col] = calc_rsi(close_s, 240); feature_names.append("rsi_240"); col += 1

    # 5. MACD
    ema12 = close_s.ewm(span=12, adjust=False).mean()
    ema26 = close_s.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26).to_numpy(dtype=np.float32)
    macd_s = pd.Series(macd)
    signal = macd_s.ewm(span=9, adjust=False).mean().to_numpy(dtype=np.float32)
    hist = macd - signal
    X_mat[:, col] = macd / (close + 1e-8); feature_names.append("macd_norm"); col += 1
    X_mat[:, col] = signal / (close + 1e-8); feature_names.append("macd_signal_norm"); col += 1
    X_mat[:, col] = hist / (close + 1e-8); feature_names.append("macd_hist_norm"); col += 1

    # 6. Bollinger Bands
    roll20_mean = close_s.rolling(20).mean().to_numpy(dtype=np.float32)
    roll20_std = close_s.rolling(20).std().to_numpy(dtype=np.float32)
    bb_upper = roll20_mean + 2 * roll20_std
    bb_lower = roll20_mean - 2 * roll20_std
    X_mat[:, col] = (close - bb_lower) / (bb_upper - bb_lower + 1e-8) - 0.5; feature_names.append("bb_percent_b"); col += 1
    X_mat[:, col] = (bb_upper - bb_lower) / (roll20_mean + 1e-8); feature_names.append("bb_bandwidth"); col += 1

    # 7. ATR & Volatility
    close_prev = close_s.shift(1).to_numpy(dtype=np.float32)
    tr = np.maximum.reduce([high - low, np.abs(high - close_prev), np.abs(low - close_prev)])
    tr_s = pd.Series(tr)
    X_mat[:, col] = (tr_s.rolling(14).mean().to_numpy(dtype=np.float32)) / (close + 1e-8); feature_names.append("natr_14"); col += 1
    X_mat[:, col] = (tr_s.rolling(60).mean().to_numpy(dtype=np.float32)) / (close + 1e-8); feature_names.append("natr_60"); col += 1
    pct_change = close_s.pct_change()
    X_mat[:, col] = pct_change.rolling(60).std().to_numpy(dtype=np.float32); feature_names.append("volatility_60"); col += 1
    X_mat[:, col] = pct_change.rolling(240).std().to_numpy(dtype=np.float32); feature_names.append("volatility_240"); col += 1

    # 8. Volume
    vol_s = pd.Series(vol)
    vol_ema30 = vol_s.ewm(span=30, adjust=False).mean().to_numpy(dtype=np.float32)
    X_mat[:, col] = vol / (vol_ema30 + 1e-8) - 1.0; feature_names.append("vol_ratio_30"); col += 1
    X_mat[:, col] = taker_vol / (vol + 1e-8) - 0.5; feature_names.append("taker_buy_ratio"); col += 1
    q_vol_s = pd.Series(np.log1p(q_vol))
    X_mat[:, col] = (q_vol_s - q_vol_s.rolling(60).mean()).to_numpy(dtype=np.float32); feature_names.append("quote_vol_log"); col += 1

    # 9. Time
    ts = pd.to_datetime(open_time)
    min_arr = ts.minute.values.astype(np.float32)
    hr_arr = ts.hour.values.astype(np.float32)
    dow_arr = ts.dayofweek.values.astype(np.float32)

    X_mat[:, col] = np.sin(2 * np.pi * min_arr / 60.0); feature_names.append("sin_minute"); col += 1
    X_mat[:, col] = np.cos(2 * np.pi * min_arr / 60.0); feature_names.append("cos_minute"); col += 1
    X_mat[:, col] = np.sin(2 * np.pi * hr_arr / 24.0); feature_names.append("sin_hour"); col += 1
    X_mat[:, col] = np.cos(2 * np.pi * hr_arr / 24.0); feature_names.append("cos_hour"); col += 1
    X_mat[:, col] = np.sin(2 * np.pi * dow_arr / 7.0); feature_names.append("sin_dayofweek"); col += 1
    X_mat[:, col] = np.cos(2 * np.pi * dow_arr / 7.0); feature_names.append("cos_dayofweek"); col += 1

    # 10. Targets
    Y_reg_mat = np.zeros((n_samples, len(HORIZONS)), dtype=np.float32)
    Y_dir_mat = np.zeros((n_samples, len(HORIZONS)), dtype=np.float32)

    for h_idx, h in enumerate(HORIZONS):
        ret = np.log(close_s.shift(-h) / close_s).to_numpy(dtype=np.float32)
        Y_reg_mat[:, h_idx] = ret
        Y_dir_mat[:, h_idx] = (ret > 0).astype(np.float32)

    # Trim invalid bounds (720m lag warm-up, 720m forward target)
    start_idx = 720
    end_idx = n_samples - 720

    X_valid = X_mat[start_idx:end_idx]
    Y_reg_valid = Y_reg_mat[start_idx:end_idx]
    Y_dir_valid = Y_dir_mat[start_idx:end_idx]
    prices_valid = close[start_idx:end_idx]
    timestamps_valid = open_time[start_idx:end_idx]

    del X_mat, Y_reg_mat, Y_dir_mat, close, high, low, open_p, vol, q_vol, taker_vol
    gc.collect()

    n_valid = len(X_valid)
    n_train = int(n_valid * train_ratio)
    n_val = int(n_valid * val_ratio)
    n_test = n_valid - n_train - n_val

    print(f"\nTotal Valid Samples: {n_valid:,}")
    print(f"Features Count:      {len(feature_names)}")
    print(f"Horizon Count:       {len(HORIZONS)} ({HORIZON_NAMES})")
    print(f"Splits -> Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    # Fit Scaler strictly on train set
    X_train_raw = X_valid[:n_train]
    mean = np.nanmean(X_train_raw, axis=0)
    std = np.nanstd(X_train_raw, axis=0)
    std[std < 1e-6] = 1.0

    X_scaled = np.clip((X_valid - mean) / std, -5.0, 5.0).astype(np.float32)

    # Save normalizer
    scaler = {"mean": mean, "std": std, "feature_names": feature_names}
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved feature normalizer to {SCALER_PATH}")

    np.savez_compressed(
        PROCESSED_DATA_PATH,
        X_train=X_scaled[:n_train],
        Y_train_reg=Y_reg_valid[:n_train],
        Y_train_dir=Y_dir_valid[:n_train],
        X_val=X_scaled[n_train:n_train+n_val],
        Y_val_reg=Y_reg_valid[n_train:n_train+n_val],
        Y_val_dir=Y_dir_valid[n_train:n_train+n_val],
        X_test=X_scaled[n_train+n_val:],
        Y_test_reg=Y_reg_valid[n_train+n_val:],
        Y_test_dir=Y_dir_valid[n_train+n_val:],
        feature_names=np.array(feature_names),
        horizon_names=np.array(HORIZON_NAMES),
        horizons=np.array(HORIZONS),
        prices_test=prices_valid[n_train+n_val:],
        timestamps_test=timestamps_valid[n_train+n_val:].astype(str)
    )
    print(f"Saved processed dataset to {PROCESSED_DATA_PATH} ({os.path.getsize(PROCESSED_DATA_PATH)/1024/1024:.2f} MB)")
    return PROCESSED_DATA_PATH

if __name__ == "__main__":
    prepare_dataset()
