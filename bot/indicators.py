import math
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator, AwesomeOscillatorIndicator
from typing import List, Optional, Dict


def realized_drift_vol(candles: List[Dict], lookback: int = 240):
    """(mean, std) of per-candle log returns — the per-step drift & sigma for the fair-prob
    model. On the 1m timeframe each step is one minute. (None, None) if not enough data."""
    closes = [c["close"] for c in candles[-lookback:] if c.get("close")]
    if len(closes) < 20:
        return None, None
    arr = np.asarray(closes, dtype=float)
    rets = np.diff(np.log(arr))
    rets = rets[np.isfinite(rets)]
    if len(rets) < 10:
        return None, None
    return float(np.mean(rets)), float(np.std(rets))


def fair_prob_up(current_price: float, strike: float, steps: int,
                 sigma_per_step: Optional[float], drift_per_step: float = 0.0) -> float:
    """Closed-form GBM probability that price closes ABOVE `strike` after `steps` 1-minute
    intervals (steps = minutes left). "Is spot above the window open, given the volatility
    still to come?" Returns 0..1."""
    if not current_price or not strike or current_price <= 0 or strike <= 0:
        return 0.5
    n = max(1, int(steps))
    if sigma_per_step is None or sigma_per_step <= 0:
        return 1.0 if current_price > strike else 0.0
    mu = (drift_per_step - 0.5 * sigma_per_step ** 2) * n
    sd = sigma_per_step * math.sqrt(n)
    z = (math.log(strike / current_price) - mu) / sd
    prob = 0.5 * math.erfc(z / math.sqrt(2))
    return float(min(1.0, max(0.0, prob)))


def compute_rsi(closes: List[float], period: int) -> Optional[float]:
    """RSI(period). Used as trend confirmation at the 50 line (>=50 up, <50 down)."""
    if not closes or len(closes) < period:
        return None
    series = pd.Series(closes)
    rsi = RSIIndicator(close=series, window=period).rsi()
    if rsi.empty:
        return None
    val = rsi.iloc[-1]
    return float(val) if not pd.isna(val) else None


def compute_heiken_ashi(candles: List[Dict]) -> List[Dict]:
    """Heiken-Ashi candles. Used (via count_consecutive) for the 5m trend & 1m momentum."""
    if not candles:
        return []

    ha = []
    for i in range(len(candles)):
        c = candles[i]
        ha_close = (c["open"] + c["high"] + c["low"] + c["close"]) / 4

        if i > 0:
            prev = ha[i - 1]
            ha_open = (prev["open"] + prev["close"]) / 2
        else:
            ha_open = (c["open"] + c["close"]) / 2

        ha_high = max(c["high"], ha_open, ha_close)
        ha_low = min(c["low"], ha_open, ha_close)

        ha.append({
            "open": ha_open,
            "high": ha_high,
            "low": ha_low,
            "close": ha_close,
            "isGreen": ha_close >= ha_open,
            "body": abs(ha_close - ha_open)
        })
    return ha


def count_consecutive(ha_candles: List[Dict]) -> Dict:
    """Current same-colour Heiken-Ashi streak {color, count} (1m momentum: 1..6 = fresh)."""
    if not ha_candles or len(ha_candles) < 2:
        return {"color": None, "count": None}

    last = ha_candles[-1]
    target = "green" if last["isGreen"] else "red"

    count = 0
    for i in range(len(ha_candles) - 1, -1, -1):
        c = ha_candles[i]
        color = "green" if c["isGreen"] else "red"
        if color != target:
            break
        count += 1

    return {"color": target, "count": count}


def compute_awesome_oscillator(candles: List[Dict], fast: int = 5, slow: int = 34) -> Dict:
    """Awesome Oscillator (ta lib): SMA(median, 5) - SMA(median, 34), median=(high+low)/2.
    Bar COLOUR follows the standard AO histogram (TradingView/Pine): with
    `diff = ao - ao[1]`, the bar is GREEN when rising (diff > 0) and RED when falling or
    flat (diff <= 0). Returns {value, color, count}: latest AO value, its bar colour, and
    the consecutive same-colour streak length. The DECISION uses the colour; the streak is
    informational (displayed like the HA). All None if not enough candles (needs > slow)."""
    none = {"value": None, "color": None, "count": None}
    if not candles or len(candles) < slow + 1:
        return none
    highs = pd.Series([c["high"] for c in candles])
    lows = pd.Series([c["low"] for c in candles])
    ao = AwesomeOscillatorIndicator(high=highs, low=lows, window1=fast, window2=slow,
                                    fillna=False).awesome_oscillator().dropna()
    if len(ao) < 2:
        return none
    vals = ao.values
    diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]  # diff[i] = ao[i] - ao[i-1]
    last_green = diffs[-1] > 0                                    # diff <= 0 -> red (Pine)
    count = 0
    for d in diffs[::-1]:
        if (d > 0) == last_green:
            count += 1
        else:
            break
    return {"value": float(vals[-1]), "color": "green" if last_green else "red", "count": count}
