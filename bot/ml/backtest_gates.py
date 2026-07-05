"""Win-rate backtest for the current purely-technical strategy.

Replays 30 days of Binance 1m candles through the exact entry stack
(1m HA colour + 1m AO colour + RSI(14)@50 + price-vs-15m-open), one fresh
entry per 15m window, held to expiry (outcome = window close > window open).
No odds data exists historically, so the max-price gate is not simulated.

Reports win rate sliced by:
  - entry minute (elapsed minutes into the window)
  - lead size (|price - open| in bps of price)
  - earliest-entry-minute gate x min-lead gate grid
Plus: how often a 3-bar HA+AO reversal-close would have exited a trade that
went on to WIN at expiry (is reversal-close saving or costing wins?).
"""
import sys, time, math
import httpx
import pandas as pd

sys.path.insert(0, r"c:\Users\abc\Desktop\WORKSPACE\15mins")
from bot import indicators

SYMBOL = "BTCUSDT"
DAYS = 30
WINDOW_MIN = 15

def fetch_klines_30d():
    end = int(time.time() // 60) * 60_000 * 1  # align to minute, ms
    end = int(time.time() * 1000)
    start = end - DAYS * 24 * 60 * 60 * 1000
    out = []
    cur = start
    with httpx.Client(timeout=30) as client:
        while cur < end:
            r = client.get("https://data-api.binance.vision/api/v3/klines",
                           params={"symbol": SYMBOL, "interval": "1m",
                                   "startTime": cur, "limit": 1000})
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            for k in batch:
                out.append({"openTime": int(k[0]), "open": float(k[1]),
                            "high": float(k[2]), "low": float(k[3]),
                            "close": float(k[4]), "closeTime": int(k[6])})
            cur = batch[-1][0] + 60_000
    return out

print("fetching 30d of 1m klines...", flush=True)
candles = fetch_klines_30d()
print(f"got {len(candles)} candles", flush=True)

# ---- precompute indicator series over the FULL history (closed candles) ----
ha = indicators.compute_heiken_ashi(candles)
ha_color = ["green" if h["isGreen"] else "red" for h in ha]
# HA streak length at each index
ha_streak = [1] * len(ha)
for i in range(1, len(ha)):
    ha_streak[i] = ha_streak[i - 1] + 1 if ha_color[i] == ha_color[i - 1] else 1

highs = pd.Series([c["high"] for c in candles])
lows = pd.Series([c["low"] for c in candles])
from ta.momentum import AwesomeOscillatorIndicator, RSIIndicator
ao = AwesomeOscillatorIndicator(high=highs, low=lows, window1=5, window2=34, fillna=False).awesome_oscillator()
ao_diff = ao.diff()
ao_color = [None if pd.isna(d) else ("green" if d > 0 else "red") for d in ao_diff]
ao_streak = [0] * len(candles)
for i in range(1, len(candles)):
    if ao_color[i] is None:
        continue
    ao_streak[i] = (ao_streak[i - 1] + 1) if ao_color[i] == ao_color[i - 1] else 1

closes = pd.Series([c["close"] for c in candles])
rsi = RSIIndicator(close=closes, window=14).rsi()

# ---- group into aligned 15m windows ----
WMS = WINDOW_MIN * 60_000
windows = {}  # window_start_ms -> list of candle indices
for i, c in enumerate(candles):
    windows.setdefault((c["openTime"] // WMS) * WMS, []).append(i)

trades = []           # dicts: entry_min, lead_bps, side, won, would_reverse_close_and_win
for wstart, idxs in sorted(windows.items()):
    if len(idxs) != WINDOW_MIN:
        continue  # partial window at either end
    w_open = candles[idxs[0]]["open"]
    w_close = candles[idxs[-1]]["close"]
    outcome = "UP" if w_close > w_open else "DOWN"

    entry = None
    for pos, i in enumerate(idxs):
        # evaluated at this candle's CLOSE => elapsed minutes = pos+1
        elapsed = pos + 1
        if elapsed >= WINDOW_MIN:
            break  # can't enter at the very end
        px = candles[i]["close"]
        if ao_color[i] is None or pd.isna(rsi.iloc[i]):
            continue
        hc = ha_color[i]
        side = "UP" if hc == "green" else "DOWN"
        if ao_color[i] != hc:
            continue
        r = rsi.iloc[i]
        if side == "UP" and r < 50:
            continue
        if side == "DOWN" and r >= 50:
            continue
        above = px > w_open
        if side == "UP" and not above:
            continue
        if side == "DOWN" and above:
            continue
        lead_bps = abs(px - w_open) / px * 10_000
        entry = {"entry_min": elapsed, "lead_bps": lead_bps, "side": side,
                 "won": side == outcome, "i": i, "pos": pos}
        break
    if entry is None:
        continue

    # would a 3-bar closed-candle HA+AO reversal have closed it before expiry?
    rev = "red" if entry["side"] == "UP" else "green"
    reversed_out = False
    for j in idxs[entry["pos"] + 1:]:
        if ha_color[j] == rev and ha_streak[j] >= 3 and ao_color[j] == rev and ao_streak[j] >= 3:
            reversed_out = True
            break
    entry["rev_closed"] = reversed_out
    trades.append(entry)

df = pd.DataFrame(trades)
n = len(df)
print(f"\nwindows with an entry: {n} of {len(windows)} ({n/len(windows)*100:.0f}%)  |  overall win rate: {df.won.mean()*100:.1f}%")

print("\n== win rate by ENTRY MINUTE ==")
df["min_bucket"] = pd.cut(df.entry_min, [0, 2, 4, 6, 8, 10, 12, 14], labels=["1-2", "3-4", "5-6", "7-8", "9-10", "11-12", "13-14"])
g = df.groupby("min_bucket", observed=True).agg(trades=("won", "size"), win_rate=("won", "mean"))
g["win_rate"] = (g.win_rate * 100).round(1)
print(g.to_string())

print("\n== win rate by LEAD SIZE (bps of price; BTC ~ $1XXk so 5bps = ~$50) ==")
df["lead_bucket"] = pd.cut(df.lead_bps, [0, 1, 2, 3, 5, 8, 12, 20, 1e9], labels=["0-1", "1-2", "2-3", "3-5", "5-8", "8-12", "12-20", "20+"])
g = df.groupby("lead_bucket", observed=True).agg(trades=("won", "size"), win_rate=("won", "mean"))
g["win_rate"] = (g.win_rate * 100).round(1)
print(g.to_string())

print("\n== GRID: earliest entry minute (rows) x min lead bps (cols) -> win% (trades/day) ==")
lead_grid = [0, 2, 3, 5, 8, 12]
min_grid = [1, 3, 5, 7, 9, 11]
header = "min\\lead " + "".join(f"{l:>16}" for l in lead_grid)
print(header)
for m in min_grid:
    row = f"{m:>8} "
    for l in lead_grid:
        sub = df[(df.entry_min >= m) & (df.lead_bps >= l)]
        if len(sub) < 20:
            row += f"{'--':>16}"
        else:
            row += f"{sub.won.mean()*100:>8.1f}% ({len(sub)/DAYS:>4.1f})"
    print(row)

print("\n== reversal-close autopsy (3-bar HA+AO against position, closed candles) ==")
rc = df[df.rev_closed]
print(f"trades a reversal-close would have exited early: {len(rc)} of {n}")
if len(rc):
    print(f"  ...of those, {rc.won.mean()*100:.1f}% would have WON at expiry anyway")
print(f"trades never reversed: {len(df[~df.rev_closed])}, win rate {df[~df.rev_closed].won.mean()*100:.1f}%")
