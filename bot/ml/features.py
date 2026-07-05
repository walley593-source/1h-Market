"""Shared ML feature definitions — single source of truth for the feature vector.

Used by: bot/ml/data.py (offline dataset), bot/ml/train.py (training),
bot/ml/model.py (live inference), bot/recorder.py (live feature computation).
The live and offline paths MUST produce the same features from the same state —
keep any change to this list in sync with both builders.
"""
import math
from datetime import datetime, timezone
from typing import Dict, Optional

# Order matters — the trained artifact stores this list and inference follows it.
FEATURES = [
    "lead_z",         # ln(price/open) / (sigma_1m * sqrt(mins_left)) — vol-normalized lead
    "lead_bps",       # signed lead over the window open, bps of price
    "elapsed_min",    # minutes into the 15m window
    "mins_left",      # minutes remaining
    "sigma_1m",       # std of 1m log returns (60 lookback)
    "vol_ratio",      # sigma(15) / sigma(60): vol regime, >1 = heating up
    "frac_above",     # fraction of the window so far spent above the open
    "crossings",      # number of times price crossed the open so far
    "max_lead_bps",   # best lead reached so far (signed, bps)
    "min_lead_bps",   # worst lead reached so far (signed, bps)
    "ha_signed",      # Heiken-Ashi streak, + = green run, - = red run (closed candles)
    "ao_signed",      # Awesome Oscillator streak, signed the same way
    "rsi",            # RSI(14)
    "hour_sin",       # time-of-day, cyclic
    "hour_cos",
    "dow",            # day of week (0=Mon)
]

# Monotonic constraints for gradient boosting (+1 = P(up) non-decreasing in feature).
# The physics of the barrier problem, injected as regularization.
MONOTONE = {"lead_z": 1, "lead_bps": 1, "frac_above": 1}


def time_features(dt: Optional[datetime] = None) -> Dict[str, float]:
    dt = dt or datetime.now(timezone.utc)
    frac = (dt.hour + dt.minute / 60.0) / 24.0
    return {
        "hour_sin": math.sin(2 * math.pi * frac),
        "hour_cos": math.cos(2 * math.pi * frac),
        "dow": float(dt.weekday()),
    }
