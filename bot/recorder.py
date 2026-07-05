"""ML data recorder + shadow scorer — Phase 0/4 of the ML plan.

Captures, every update-loop tick, the full feature snapshot PLUS the live Polymarket
bid/ask on both sides — the dataset that does not exist historically and that is the
only thing that can ever prove (or kill) a statistical edge over the market's price.
If a trained artifact exists (models/model.joblib), each tick is also scored in
SHADOW: the calibrated P(up) is logged alongside the odds so model-vs-market
calibration can be compared offline. The model makes no trading decisions.

Append-only CSVs under logs/ml/:
  ticks-YYYYMMDD.csv — one row per tick: features, odds, shadow P(up), decision
  outcomes.csv       — one row per finished window: chainlink open/close and winner

Purely additive: record() swallows its own errors and the caller wraps it in
try/except, so the recorder can never take down the trading loop.
"""
import math
import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np

from . import utils
from .ml.features import time_features
from .ml.model import ml_model

TICK_HEADER = [
    "ts", "market_id", "slug", "elapsed_min", "mins_left",
    "spot", "chainlink", "window_open", "open_source",
    "lead_bps", "lead_z", "sigma_1m", "vol_ratio",
    "frac_above", "crossings", "max_lead_bps", "min_lead_bps",
    "ha_signed", "ao_signed", "rsi",
    "up_bid", "up_ask", "down_bid", "down_ask",
    "up_ask_liq", "down_ask_liq",
    "ml_p_up",
    "decision_action", "decision_side", "decision_reason", "running",
]

OUTCOME_HEADER = ["ts", "market_id", "slug", "cl_open", "cl_close", "up_won"]

# One row per ML-exit evaluation on an OPEN position (shadow or live). Joins to
# outcomes.csv on market_id to grade "did selling beat holding to expiry".
EXIT_HEADER = [
    "ts", "market_id", "slug", "side", "elapsed_min", "mins_left",
    "entry_price", "p_side", "held_bid", "action", "reason", "exit_mode", "acted",
]


def _signed_streak(color: Optional[str], count: Optional[int]) -> Optional[float]:
    if color not in ("green", "red") or not count:
        return None
    return float(count) if color == "green" else -float(count)


class MlRecorder:
    def __init__(self, out_dir: str = "./logs/ml"):
        self.out_dir = out_dir
        self._cur_market_id: Optional[str] = None
        self._cur_slug: Optional[str] = None
        # per-window accumulators (reset when the live market rotates)
        self._above = 0
        self._below = 0
        self._crossings = 0
        self._last_sign = 0
        self._max_lead_bps: Optional[float] = None
        self._min_lead_bps: Optional[float] = None

    # ── per-window path accumulators ────────────────────────────────────────────
    def _reset_window(self):
        self._above = 0
        self._below = 0
        self._crossings = 0
        self._last_sign = 0
        self._max_lead_bps = None
        self._min_lead_bps = None

    def _update_path(self, lead_bps: Optional[float]):
        if lead_bps is None:
            return
        sign = 1 if lead_bps > 0 else (-1 if lead_bps < 0 else 0)
        if sign > 0:
            self._above += 1
        elif sign < 0:
            self._below += 1
        if sign != 0:
            if self._last_sign != 0 and sign != self._last_sign:
                self._crossings += 1
            self._last_sign = sign
        if self._max_lead_bps is None or lead_bps > self._max_lead_bps:
            self._max_lead_bps = lead_bps
        if self._min_lead_bps is None or lead_bps < self._min_lead_bps:
            self._min_lead_bps = lead_bps

    # ── outcome emission on market rotation ─────────────────────────────────────
    def _emit_outcome(self, prev_market_id: str, prev_slug: Optional[str], window_marks: Dict[str, Any]):
        wm = (window_marks or {}).get(str(prev_market_id)) or {}
        cl_open, cl_close = wm.get("open"), wm.get("last")
        if not cl_open or not cl_close:
            return
        utils.append_csv_row(
            os.path.join(self.out_dir, "outcomes.csv"), OUTCOME_HEADER,
            [datetime.now(timezone.utc).isoformat(), prev_market_id, prev_slug,
             cl_open, cl_close, int(cl_close > cl_open)])

    # ── feature assembly (must mirror bot/ml/data.py's offline builder) ─────────
    def _build_features(self, snap: Dict[str, Any], lead_bps: Optional[float]) -> Dict[str, Any]:
        timing = snap.get("timing") or {}
        window_open = snap.get("window_open")
        ref_px = snap.get("chainlink") if snap.get("open_source") == "chainlink" else snap.get("spot")

        sigma_1m = vol_ratio = lead_z = None
        closes: List[float] = snap.get("closed_closes") or []
        if len(closes) >= 61:
            rets = np.diff(np.log(np.asarray(closes[-61:], dtype=float)))
            s60 = float(np.std(rets))
            s15 = float(np.std(rets[-15:]))
            if s60 > 0:
                sigma_1m = s60
                vol_ratio = s15 / s60
                mins_left = max(0.25, float(timing.get("remainingMinutes") or 0))
                if window_open and ref_px:
                    lead_z = math.log(ref_px / window_open) / (s60 * math.sqrt(mins_left))

        total = self._above + self._below
        return {
            "lead_z": lead_z,
            "lead_bps": lead_bps,
            "elapsed_min": timing.get("elapsedMinutes"),
            "mins_left": timing.get("remainingMinutes"),
            "sigma_1m": sigma_1m,
            "vol_ratio": vol_ratio,
            "frac_above": (self._above / total) if total else None,
            "crossings": float(self._crossings),
            "max_lead_bps": self._max_lead_bps,
            "min_lead_bps": self._min_lead_bps,
            "ha_signed": _signed_streak(snap.get("ha_color"), snap.get("ha_count")),
            "ao_signed": _signed_streak(snap.get("ao_color"), snap.get("ao_count")),
            "rsi": snap.get("rsi"),
            **time_features(),
        }

    # ── two-phase per tick: score() BEFORE the decision, log() AFTER ────────────
    def score(self, snap: Dict[str, Any]) -> Dict[str, Any]:
        """Advance per-window path state, build the feature vector, and run the model.
        Called ONCE per tick, before the entry decision, so the decision can use P(up).
        Returns {"p_up", "feats", "lead_bps"}. Never raises for model/feature issues."""
        market_id = snap.get("market_id")
        if market_id is None:
            return {"p_up": None, "feats": None, "lead_bps": None}
        market_id = str(market_id)

        # window rotation: close out the previous window's outcome, reset path state
        if market_id != self._cur_market_id:
            if self._cur_market_id is not None:
                self._emit_outcome(self._cur_market_id, self._cur_slug, snap.get("window_marks") or {})
            self._cur_market_id = market_id
            self._cur_slug = snap.get("slug")
            self._reset_window()

        window_open = snap.get("window_open")
        ref_px = snap.get("chainlink") if snap.get("open_source") == "chainlink" else snap.get("spot")
        lead_bps = None
        if window_open and ref_px:
            lead_bps = (ref_px - window_open) / ref_px * 10_000
        self._update_path(lead_bps)

        feats = self._build_features(snap, lead_bps)
        p_up = ml_model.predict_p_up(feats)
        return {"p_up": p_up, "feats": feats, "lead_bps": lead_bps}

    def log(self, snap: Dict[str, Any], scored: Dict[str, Any], decision: Dict[str, Any]):
        """Write the tick row: features + live odds + P(up) + the decision that was made.
        Called AFTER the decision with the same `scored` dict returned by score()."""
        feats = (scored or {}).get("feats")
        if feats is None:
            return
        p_up = (scored or {}).get("p_up")
        ob = snap.get("orderbook") or {}
        up_ob, down_ob = ob.get("up") or {}, ob.get("down") or {}
        decision = decision or {}
        timing = snap.get("timing") or {}

        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        utils.append_csv_row(
            os.path.join(self.out_dir, f"ticks-{day}.csv"), TICK_HEADER,
            [
                datetime.now(timezone.utc).isoformat(), snap.get("market_id"), snap.get("slug"),
                timing.get("elapsedMinutes"), timing.get("remainingMinutes"),
                snap.get("spot"), snap.get("chainlink"), snap.get("window_open"), snap.get("open_source"),
                feats["lead_bps"], feats["lead_z"], feats["sigma_1m"], feats["vol_ratio"],
                feats["frac_above"], feats["crossings"], feats["max_lead_bps"], feats["min_lead_bps"],
                feats["ha_signed"], feats["ao_signed"], feats["rsi"],
                up_ob.get("bestBid"), up_ob.get("bestAsk"),
                down_ob.get("bestBid"), down_ob.get("bestAsk"),
                up_ob.get("askLiquidity"), down_ob.get("askLiquidity"),
                p_up,
                decision.get("action"), decision.get("side"), decision.get("reason"),
                int(bool(snap.get("running"))),
            ])


    def log_exit(self, trade: Dict[str, Any], exit_decision: Dict[str, Any],
                 timing: Dict[str, Any], exit_mode: str, acted: bool):
        """Record one ML-exit evaluation for an open position (shadow or live), so the
        exit model can be graded against hold-to-expiry offline. Never fatal."""
        try:
            utils.append_csv_row(
                os.path.join(self.out_dir, "exits.csv"), EXIT_HEADER,
                [
                    datetime.now(timezone.utc).isoformat(),
                    trade.get("market_id"), trade.get("market_slug"), trade.get("side"),
                    (timing or {}).get("elapsedMinutes"), (timing or {}).get("remainingMinutes"),
                    trade.get("entry_price"), exit_decision.get("p_side"),
                    exit_decision.get("bid"), exit_decision.get("action"),
                    exit_decision.get("reason"), exit_mode, int(bool(acted)),
                ])
        except Exception:
            pass


recorder = MlRecorder()
