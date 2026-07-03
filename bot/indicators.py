from typing import List, Dict


def compute_heiken_ashi(candles: List[Dict]) -> List[Dict]:
    """Heiken-Ashi candles. Used (via count_consecutive) for the 15m trend signal."""
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
    """Current same-colour Heiken-Ashi streak {color, count}."""
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
