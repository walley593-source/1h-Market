"""Offline dataset builder for the ML probability model.

Fetches Binance 1m history (resumable, cached in models/klines_1m.parquet) and
builds one sample per (15m window, elapsed minute 1..14) evaluated on CLOSED
candles only — the same no-leakage discipline as the live bot. Label =
window close > window open.

This dataset trains/validates probability CALIBRATION only. The edge-vs-market
question needs the live recorder data (logs/ml/) — no historical book exists.

Run:    python -m bot.ml.data [days]        (default 180)
Writes: models/klines_1m.parquet, models/dataset.parquet
"""
import math
import os
import sys
import time

import httpx
import numpy as np
import pandas as pd

from .features import time_features
from .. import indicators

SYMBOL = "BTCUSDT"
WMS = 15 * 60_000
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
KLINES_PATH = os.path.join(MODELS_DIR, "klines_1m.parquet")
DATASET_PATH = os.path.join(MODELS_DIR, "dataset.parquet")

# data-api.binance.vision is Binance's public market-data mirror — same /api/v3
# endpoints, no geo-restriction (api.binance.com can 451 in some regions).
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com",
         "https://api1.binance.com", "https://api2.binance.com"]


def fetch_klines(days: int) -> pd.DataFrame:
    """Fetch `days` of 1m klines ending now. Resumable: progress is saved to the
    parquet cache every ~25k candles, and an existing cache is extended, not refetched."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    end = int(time.time() * 1000)
    start = end - days * 86_400_000

    rows: list = []
    if os.path.exists(KLINES_PATH):
        cached = pd.read_parquet(KLINES_PATH)
        cached = cached[cached.openTime >= start]
        if len(cached):
            rows = list(cached.itertuples(index=False, name=None))
            start = int(cached.openTime.iloc[-1]) + 60_000
            print(f"resuming from cache: {len(rows)} candles, next start {start}")

    def save():
        df = pd.DataFrame(rows, columns=["openTime", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("openTime").sort_values("openTime").reset_index(drop=True)
        df.to_parquet(KLINES_PATH)
        return df

    host_i = 0
    last_save = len(rows)
    with httpx.Client(timeout=30) as client:
        cur = start
        while cur < end:
            batch = None
            for _ in range(len(HOSTS) * 3):
                host = HOSTS[host_i % len(HOSTS)]
                try:
                    r = client.get(f"{host}/api/v3/klines",
                                   params={"symbol": SYMBOL, "interval": "1m",
                                           "startTime": cur, "limit": 1000})
                    r.raise_for_status()
                    batch = r.json()
                    break
                except Exception as e:
                    print(f"  {host} failed ({type(e).__name__}); rotating host", flush=True)
                    host_i += 1
                    time.sleep(1.5)
            if batch is None:
                print("all hosts failing — saving progress and stopping")
                break
            if not batch:
                break
            rows.extend([(int(k[0]), float(k[1]), float(k[2]), float(k[3]),
                          float(k[4]), float(k[5])) for k in batch])
            cur = batch[-1][0] + 60_000
            if len(rows) - last_save >= 25_000:
                save()
                last_save = len(rows)
                print(f"  {len(rows)} candles (progress saved)...", flush=True)
    return save()


def build_dataset(kl: pd.DataFrame) -> pd.DataFrame:
    closes = kl["close"].to_numpy()
    highs, lows = kl["high"].to_numpy(), kl["low"].to_numpy()
    opens = kl["open"].to_numpy()
    open_times = kl["openTime"].to_numpy()

    logret = np.diff(np.log(closes), prepend=np.nan)
    sr = pd.Series(logret)
    sigma60 = sr.rolling(60).std().to_numpy()
    sigma15 = sr.rolling(15).std().to_numpy()

    ha = indicators.compute_heiken_ashi(kl.to_dict("records"))
    ha_green = np.array([h["isGreen"] for h in ha])
    ha_streak = np.ones(len(ha), dtype=int)
    for i in range(1, len(ha)):
        ha_streak[i] = ha_streak[i - 1] + 1 if ha_green[i] == ha_green[i - 1] else 1
    ha_signed = np.where(ha_green, ha_streak, -ha_streak).astype(float)

    from ta.momentum import AwesomeOscillatorIndicator, RSIIndicator
    ao = AwesomeOscillatorIndicator(high=pd.Series(highs), low=pd.Series(lows),
                                    window1=5, window2=34, fillna=False).awesome_oscillator()
    ao_diff = ao.diff().to_numpy()
    ao_green = ao_diff > 0
    ao_streak = np.zeros(len(kl), dtype=int)
    for i in range(1, len(kl)):
        if np.isnan(ao_diff[i]):
            continue
        ao_streak[i] = ao_streak[i - 1] + 1 if (not np.isnan(ao_diff[i - 1]) and ao_green[i] == ao_green[i - 1]) else 1
    ao_signed = np.where(ao_green, ao_streak, -ao_streak).astype(float)
    ao_signed[np.isnan(ao_diff)] = np.nan

    rsi = RSIIndicator(close=pd.Series(closes), window=14).rsi().to_numpy()

    win_ids = open_times // WMS
    samples = []
    n = len(kl)
    start_idx = 0
    while start_idx < n:
        wid = win_ids[start_idx]
        end_idx = start_idx
        while end_idx < n and win_ids[end_idx] == wid:
            end_idx += 1
        idxs = np.arange(start_idx, end_idx)
        start_idx = end_idx
        if len(idxs) != 15:
            continue
        first = idxs[0]
        w_open = opens[first]
        up_win = int(closes[idxs[-1]] > w_open)
        ts = pd.Timestamp(open_times[first], unit="ms", tz="UTC")
        tf_base = time_features(ts.to_pydatetime())

        w_closes, w_highs, w_lows = closes[idxs], highs[idxs], lows[idxs]
        for e in range(1, 15):                  # decide at candle (e-1)'s close
            i = first + e - 1
            s60 = sigma60[i]
            if not np.isfinite(s60) or s60 <= 0 or not np.isfinite(rsi[i]) or np.isnan(ao_signed[i]):
                continue
            px = closes[i]
            mins_left = 15 - e
            seen = w_closes[:e]
            signs = np.sign(seen - w_open)
            nz = signs[signs != 0]
            samples.append({
                "window_start": open_times[first],
                "elapsed_min": float(e),
                "mins_left": float(mins_left),
                "lead_bps": (px - w_open) / px * 10_000,
                "lead_z": math.log(px / w_open) / (s60 * math.sqrt(mins_left)),
                "sigma_1m": s60,
                "vol_ratio": (sigma15[i] / s60) if np.isfinite(sigma15[i]) else np.nan,
                "frac_above": float((seen > w_open).mean()),
                "crossings": float((np.diff(nz) != 0).sum()) if len(nz) > 1 else 0.0,
                "max_lead_bps": (w_highs[:e].max() - w_open) / px * 10_000,
                "min_lead_bps": (w_lows[:e].min() - w_open) / px * 10_000,
                "ha_signed": ha_signed[i],
                "ao_signed": ao_signed[i],
                "rsi": rsi[i],
                **tf_base,
                "up_win": up_win,
            })
    return pd.DataFrame(samples).dropna()


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    print(f"fetching/extending {days}d of {SYMBOL} 1m klines...")
    kl = fetch_klines(days)
    print(f"klines: {len(kl)} candles "
          f"({(kl.openTime.iloc[-1] - kl.openTime.iloc[0]) / 86_400_000:.0f}d)")
    print("building samples...")
    df = build_dataset(kl)
    df.to_parquet(DATASET_PATH)
    w = df.window_start.nunique()
    print(f"dataset: {len(df)} samples across {w} windows ({w / 96:.0f} days), "
          f"base rate P(up)={df.up_win.mean():.3f}")
    print(f"written to {DATASET_PATH}")


if __name__ == "__main__":
    main()
