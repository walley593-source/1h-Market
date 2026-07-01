from typing import Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
#  1-hour market entry engine — simple.
#
#  BUY (UP)  when price is ABOVE the 1h open AND the 5m Heiken-Ashi is GREEN.
#  SELL(DOWN) when price is BELOW the 1h open AND the 5m Heiken-Ashi is RED.
#  Anything else = no trade. (Close-and-flip on the opposite signal is handled by
#  the caller — a new SELL closes a BUY, a new BUY closes a SELL.)
# ─────────────────────────────────────────────────────────────────────────────


def _no(reason: str) -> Dict[str, Any]:
    return {"action": "NO_TRADE", "side": None, "phase": "1H", "strength": "-", "reason": reason}


def decide_entry(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Price vs the 1h open (persistence) + 5m Heiken-Ashi colour.

    - above_open True  + HA(5m) green -> BUY (UP)
    - above_open False + HA(5m) red   -> SELL (DOWN)
    - otherwise -> NO_TRADE
    """
    above_open = inputs.get("aboveOpen")   # current price vs the 1h open
    ha5 = inputs.get("ha5Color")           # 5m Heiken-Ashi colour ("green"/"red"/None)
    price_up = inputs.get("priceUp")
    price_down = inputs.get("priceDown")

    if above_open is None:
        return _no("open_unavailable")
    if ha5 not in ("green", "red"):
        return _no("no_ha")
    if price_up is None or price_down is None:
        return _no("missing_prices")

    if above_open and ha5 == "green":
        return {"action": "ENTER", "side": "UP", "phase": "1H", "strength": "STRONG",
                "price": price_up, "reason": "above_open_ha_green"}
    if (not above_open) and ha5 == "red":
        return {"action": "ENTER", "side": "DOWN", "phase": "1H", "strength": "STRONG",
                "price": price_down, "reason": "below_open_ha_red"}

    # Price and HA disagree (e.g. above open but HA red) -> stand aside.
    return _no("no_signal")
