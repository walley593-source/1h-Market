"""Phase 4 — the go/no-go instrument for the live ML.

Reads the recorder's logs (logs/ml/) and answers the only questions that decide
whether the model is worth trading, none of which the offline trainer can:

  1. MODEL vs MARKET calibration — Brier(model P_up) vs Brier(market-implied P_up
     from the mid). This is THE test: a high win rate means nothing if the model
     only matches the price. The model must beat the market's own probability.
  2. ENTRY realism — of the ticks the model would ENTER on, realized win% and mean
     per-share edge = outcome − price_paid. Positive mean ⇒ profitable AFTER cost.
  3. EXIT shadow grading — for each would-sell in exits.csv, did selling at that bid
     beat holding to expiry? (Answers whether the ML exit should ever go live.)

Run: python -m bot.ml.evaluate      (reads ./logs/ml)
Shows little until a few days of live data exist — that's expected; run it as the
logs fill. It never trains or trades.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs", "ml")


def _brier(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(np.mean((p - np.asarray(y, float)) ** 2))


def load():
    ticks_files = sorted(glob.glob(os.path.join(LOG_DIR, "ticks-*.csv")))
    if not ticks_files:
        return None, None, None
    ticks = pd.concat([pd.read_csv(f) for f in ticks_files], ignore_index=True)
    opath = os.path.join(LOG_DIR, "outcomes.csv")
    outs = pd.read_csv(opath) if os.path.exists(opath) else pd.DataFrame()
    epath = os.path.join(LOG_DIR, "exits.csv")
    exits = pd.read_csv(epath) if os.path.exists(epath) else pd.DataFrame()
    return ticks, outs, exits


def main():
    ticks, outs, exits = load()
    if ticks is None:
        print(f"no tick logs in {LOG_DIR} yet — start the bot (paper) and let it record.")
        return
    print(f"loaded {len(ticks)} ticks"
          + (f", {len(outs)} settled windows" if len(outs) else ", 0 settled windows")
          + (f", {len(exits)} exit evals" if len(exits) else ""))

    if not len(outs):
        print("\nno settled windows yet — model-vs-market needs finished windows. Check back later.")
        return

    # join each tick to its window's outcome
    outs = outs.drop_duplicates("market_id", keep="last")[["market_id", "up_won"]]
    df = ticks.merge(outs, on="market_id", how="inner")
    df = df[df.ml_p_up.notna()]
    # market-implied P(up) from the mid of the UP side (fallback to ask)
    df["up_mid"] = np.where(df.up_bid.notna() & df.up_ask.notna(),
                            (df.up_bid + df.up_ask) / 2.0, df.up_ask)
    df = df[df.up_mid.notna()]
    print(f"\n{len(df)} scored ticks across {df.market_id.nunique()} settled windows")
    if len(df) < 200:
        print("(<200 — numbers are noisy; treat as a smoke test, not a verdict)")

    # ── 1. MODEL vs MARKET calibration ──────────────────────────────────────────
    bm = _brier(df.ml_p_up, df.up_won)
    bk = _brier(df.up_mid, df.up_won)
    print("\n== model vs market (Brier, lower=better) ==")
    print(f"  model P(up)   : {bm:.4f}")
    print(f"  market mid    : {bk:.4f}")
    verdict = ("MODEL BEATS MARKET — a real edge, keep validating" if bm < bk - 0.001
               else "MARKET BEATS/TIES MODEL — no statistical edge; do NOT trade it live"
               if bm > bk - 0.001 else "too close")
    print(f"  -> {verdict}")

    # ── 2. ENTRY realism (first model-entry per window) ─────────────────────────
    MIN_CONF, EV = 0.80, 0.02
    d = df.sort_values(["market_id", "elapsed_min"]).copy()
    d["conf"] = np.maximum(d.ml_p_up, 1 - d.ml_p_up)
    d["side_up"] = d.ml_p_up >= 0.5
    d["ask"] = np.where(d.side_up, d.up_ask, d.down_ask)
    d["p_side"] = np.where(d.side_up, d.ml_p_up, 1 - d.ml_p_up)
    d["enter"] = (d.conf >= MIN_CONF) & ((d.p_side - d.ask) >= EV) & d.ask.notna()
    picks = d[d.enter].groupby("market_id", sort=False).first()
    if len(picks):
        picks["won"] = np.where(picks.side_up, picks.up_won == 1, picks.up_won == 0)
        picks["pnl_per_share"] = picks.won.astype(float) - picks.ask
        print(f"\n== entry realism (conf>={MIN_CONF}, edge>={EV}) ==")
        print(f"  entries: {len(picks)}  |  win%: {100*picks.won.mean():.1f}"
              f"  |  mean P&L/share: {picks.pnl_per_share.mean():+.4f}"
              f"  ({'PROFITABLE' if picks.pnl_per_share.mean() > 0 else 'LOSING'} after price)")
    else:
        print(f"\n== entry realism ==\n  no ticks cleared conf>={MIN_CONF} & edge>={EV} yet")

    # ── 3. EXIT shadow grading ──────────────────────────────────────────────────
    if len(exits):
        ex = exits[exits.action == "SELL"].merge(outs, on="market_id", how="inner")
        if len(ex):
            ex["won_if_held"] = np.where(ex.side == "UP", ex.up_won == 1, ex.up_won == 0)
            ex["hold_val"] = ex.won_if_held.astype(float)     # $1 if the side wins
            ex["sell_val"] = ex.held_bid                       # what selling now yields
            ex["sell_minus_hold"] = ex.sell_val - ex.hold_val
            print("\n== exit shadow grading (would-sells vs holding to expiry) ==")
            print(f"  would-sell signals: {len(ex)}  |  mean(sell − hold): "
                  f"{ex.sell_minus_hold.mean():+.4f}/share "
                  f"({'EXIT HELPS' if ex.sell_minus_hold.mean() > 0 else 'HOLDING WINS — keep exit in shadow'})")
        else:
            print("\n== exit shadow grading ==\n  no settled would-sell signals yet")

    print("\nnote: the market-vs-model Brier is the decision. Win% alone is a trap — the\n"
          "market prices persistence too. Trade live only if the model beats the mid.")


if __name__ == "__main__":
    main()
