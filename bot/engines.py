from typing import Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
#  Entry engine — two selectable strategies (settings.STRATEGY_MODE):
#
#   "gates"  decide_entry — purely technical hand-gates: 1m HA direction + 1m AO +
#            RSI(50) + price-vs-open + min-entry-minute + min-lead + max-price cap.
#            All equal and mandatory. The proven, always-safe default logic.
#
#   "model"  decide_model — the calibrated ML win-probability drives the decision.
#            The five indicator/timing/lead gates above are SUBSUMED (they are model
#            features now); the fixed max-price cap is REPLACED by a dynamic EV gate
#            P(win) - ask >= margin. Only book liquidity + one-per-window remain
#            (enforced downstream in main.py).
#
#  decide() dispatches on mode. Both return the same shape so execution is identical.
# ─────────────────────────────────────────────────────────────────────────────


def _no_trade(reason: str) -> Dict[str, Any]:
    return {"action": "NO_TRADE", "side": None, "phase": "TREND", "strength": "-", "reason": reason}


def decide(mode: str, gate_inputs: Dict[str, Any], model_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch to the active strategy. `mode` = "model" or "gates"."""
    if (mode or "gates").lower() == "model":
        return decide_model(model_inputs)
    return decide_entry(gate_inputs)


def decide_model(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """ML-driven entry. The calibrated P(up) picks the side and its confidence; a
    dynamic EV gate (edge = P(win) - ask) replaces the fixed price cap.

    Inputs:
      pUp        calibrated P(window closes above open), 0..1 (None -> no model)
      priceUp    Polymarket UP ask (cost per share if it wins pays $1)
      priceDown  Polymarket DOWN ask
      minConf    minimum P(chosen side) to enter (default 0.55)
      evMargin   required edge = P(side) - ask, in probability/$ units (default 0.0)

    The indicator, timing and lead gates from decide_entry are intentionally ABSENT —
    the model already accounts for them via its features.
    """
    p_up = inputs.get("pUp")
    price_up = inputs.get("priceUp")
    price_down = inputs.get("priceDown")
    min_conf = inputs.get("minConf", 0.55)
    ev_margin = inputs.get("evMargin", 0.0)

    if p_up is None:
        return _no_trade("model_unavailable")
    if price_up is None or price_down is None:
        return _no_trade("missing_prices")

    # Side = whichever outcome the model favours; confidence = its probability.
    side = "UP" if p_up >= 0.5 else "DOWN"
    p_side = p_up if side == "UP" else (1.0 - p_up)
    ask = price_up if side == "UP" else price_down
    if ask is None:
        return _no_trade("no_price")

    if p_side < min_conf:
        return _no_trade(f"model_conf_{p_side:.2f}_below_{min_conf:.2f}")

    # EV gate: buying at `ask` a share that pays $1 with prob p_side has edge p_side - ask.
    edge = p_side - ask
    if edge < ev_margin:
        return _no_trade(f"model_edge_{edge:+.2f}_below_{ev_margin:.2f}")

    strength = "HIGH_CONVICTION" if p_side >= 0.80 else "STRONG"
    return {
        "action": "ENTER", "side": side, "phase": "MODEL", "strength": strength,
        "price": ask, "model_p": p_side, "edge": edge, "reason": "model_edge",
    }


def decide_exit(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """ML-driven EXIT for an open position — the mirror of decide_model, driven by the
    SAME calibrated probability. The entry model's P(up) already gives P(the held side
    wins) at any tick, so no separate model is needed: we compare that fair value to the
    price the market will pay us to sell back now.

    Two exit motives, both from the model probability:
      TAKE-PROFIT  the sell-back bid exceeds the model's fair value by >= tpMargin
                   (the market is overpaying for our position -> lock it in).
      STOP         the model's win prob has fallen below stopProb (it now expects a
                   loss -> salvage the remaining bid instead of riding to $0).

    Inputs:
      side       "UP"/"DOWN" — the held side
      pUp        calibrated P(up) (None -> HOLD, no data)
      heldBid    best bid for the held side = what selling now yields per share
      tpMargin   take-profit edge: sell if bid - p_side >= this (default 0.03)
      stopProb   stop level: sell if p_side < this (default 0.0 = stop disabled)

    Returns {"action": "SELL"/"HOLD", "reason", "p_side", "bid"}.
    """
    side = inputs.get("side")
    p_up = inputs.get("pUp")
    bid = inputs.get("heldBid")
    tp_margin = inputs.get("tpMargin", 0.03)
    stop_prob = inputs.get("stopProb", 0.0) or 0.0

    if p_up is None or side not in ("UP", "DOWN"):
        return {"action": "HOLD", "reason": "exit_no_model", "p_side": None, "bid": bid}
    p_side = p_up if side == "UP" else (1.0 - p_up)

    # STOP: model now expects this side to lose — salvage whatever the bid still offers.
    if stop_prob > 0.0 and p_side < stop_prob:
        return {"action": "SELL", "reason": f"stop_p_{p_side:.2f}_below_{stop_prob:.2f}",
                "p_side": p_side, "bid": bid}

    # TAKE-PROFIT: needs a sell-back price. Sell if the market overpays vs fair value.
    if bid is None or bid <= 0:
        return {"action": "HOLD", "reason": "exit_no_bid", "p_side": p_side, "bid": bid}
    over = bid - p_side
    if over >= tp_margin:
        return {"action": "SELL", "reason": f"take_profit_bid_over_fair_{over:+.2f}",
                "p_side": p_side, "bid": bid}

    return {"action": "HOLD", "reason": f"hold_p_{p_side:.2f}_bid_{bid:.2f}",
            "p_side": p_side, "bid": bid}


def decide_entry(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """1m HA direction + 1m AO confirm + RSI(50) confirm the DIRECTION; then price action
    (price vs the 15m window open) and the max-price cap must pass. Everything is
    1-minute — no 5m timeframe, no probability model.

    - 1m HA colour = the direction. Red -> only DOWN, green -> only UP.
    - 1m Awesome Oscillator must match by BAR COLOUR: green = rising bar (diff > 0),
      red = falling/flat (diff <= 0). UP needs AO green, DOWN needs AO red.
    - RSI(14) confirms at the 50 line: >= 50 = uptrend (UP), < 50 = downtrend (DOWN).
    - PRICE ACTION: current price must be on the side's side of the 15m window OPEN —
      UP needs price ABOVE the open, DOWN needs price BELOW (aboveOpen).
    - MIN ENTRY MINUTE: no entries before `minEntryMinute` elapsed minutes — early
      leads are coin-flips; the same signals win far more later in the window
      (30d backtest: 64% win in min 1-2 vs 78%+ from min 5).
    - MIN LEAD: the lead over the open (`leadBps`, in bps of price) must be at least
      `minLeadBps` — a price barely past the open is noise (30d backtest: <2 bps
      wins ~55%, 12+ bps wins 76-90%).
    - PRICE CAP: the chosen side's Polymarket ask must be BELOW `maxPrice` (default 0.60).

    All gates are EQUAL and MANDATORY; none overrides another.
    """
    ha1 = inputs.get("ha1Color")          # "green" / "red" / None  (1m direction)
    ao1 = inputs.get("ao1")               # "green" / "red" / None  (1m AO confirm)
    price_up = inputs.get("priceUp")
    price_down = inputs.get("priceDown")
    max_price = inputs.get("maxPrice", 0.60)

    if price_up is None or price_down is None:
        return _no_trade("missing_prices")

    # ── TIMING: let the window develop before trusting the lead ──
    min_minute = inputs.get("minEntryMinute") or 0
    if min_minute > 0:
        elapsed = inputs.get("elapsedMin")
        if elapsed is None:
            return _no_trade("elapsed_unavailable")
        if elapsed < min_minute:
            return _no_trade(f"too_early_{elapsed:.1f}m_of_{min_minute:.0f}m")

    # ── DIRECTION (1m HA) ──
    if ha1 not in ("green", "red"):
        return _no_trade("no_1m_trend")

    side = "UP" if ha1 == "green" else "DOWN"
    price = price_up if side == "UP" else price_down

    # ── 1m AWESOME OSCILLATOR confirmation by BAR COLOUR — REQUIRED ──
    # Standard AO histogram: green = rising bar (diff > 0), red = falling/flat (diff <= 0).
    if ao1 is None:
        return _no_trade("ao_unavailable")
    if side == "UP" and ao1 != "green":
        return _no_trade("ao1_not_green")
    if side == "DOWN" and ao1 != "red":
        return _no_trade("ao1_not_red")

    # ── RSI trend confirmation at the 50 line (>=50 up, <50 down) — REQUIRED ──
    rsi = inputs.get("rsi")
    if rsi is None:
        return _no_trade("rsi_unavailable")
    if side == "UP" and rsi < 50:
        return _no_trade(f"rsi_{rsi:.0f}_not_uptrend")
    if side == "DOWN" and rsi >= 50:
        return _no_trade(f"rsi_{rsi:.0f}_not_downtrend")

    # ── PRICE ACTION: price must be on the right side of the 15m window OPEN ──
    # aboveOpen = current price > the window's (Chainlink) open. UP needs it above,
    # DOWN needs it below — only trade the side that is currently "winning" the bet.
    above_open = inputs.get("aboveOpen")
    if above_open is None:
        return _no_trade("open_unavailable")
    if side == "UP" and not above_open:
        return _no_trade("price_below_open")
    if side == "DOWN" and above_open:
        return _no_trade("price_above_open")

    # ── LEAD SIZE: the move past the open must be big enough to mean something ──
    min_lead = inputs.get("minLeadBps") or 0
    if min_lead > 0:
        lead = inputs.get("leadBps")
        if lead is None:
            return _no_trade("lead_unavailable")
        if lead < min_lead:
            return _no_trade(f"lead_{lead:.1f}bps_below_{min_lead:.0f}")

    # ── PRICE CAP: only enter when the odds are below the cap ──
    if price is None:
        return _no_trade("no_price")
    if price >= max_price:
        return _no_trade(f"price_{price:.2f}_above_{max_price:.2f}")

    return {
        "action": "ENTER", "side": side, "phase": "TREND", "strength": "STRONG",
        "price": price, "reason": "trend_confirmed"
    }
