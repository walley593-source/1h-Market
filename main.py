import asyncio
import time
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from bot.config import settings, normalize_private_key
import bot.data as data
import bot.ws_data as ws_data
import bot.chainlink as chainlink
import bot.indicators as indicators
import bot.engines as engines
import bot.utils as utils
from bot.clob_trader import clob_trader

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Silence aiohttp's harmless "Unclosed client session" noise at shutdown (sessions
    # are closed below; this is a safety net for any straggler collected during GC).
    try:
        _loop = asyncio.get_running_loop()
        _prev_handler = _loop.get_exception_handler()
        def _quiet_handler(loop, context):
            if context.get("message") == "Unclosed client session":
                return
            (_prev_handler(loop, context) if _prev_handler else loop.default_exception_handler(context))
        _loop.set_exception_handler(_quiet_handler)
    except Exception:
        pass

    # Load previous state
    load_state()

    # Initial seeding
    await seed_kline_buffers()

    # Start all background tasks
    tasks = [
        asyncio.create_task(binance_stream.start()),
        asyncio.create_task(binance_kline_15m.start()),
        asyncio.create_task(polymarket_ws_stream.start()),
        asyncio.create_task(chainlink_ws_stream.start()),
        asyncio.create_task(update_loop())
    ]

    yield

    # Shutdown: signal the streams to stop, cancel their tasks, then AWAIT each task so it
    # unwinds its `async with aiohttp.ClientSession()` and closes the session cleanly.
    # (Cancelling without awaiting leaves sessions open -> aiohttp's "Unclosed client
    # session" warning at garbage-collection.)
    binance_stream.close()
    binance_kline_15m.close()
    polymarket_ws_stream.close()
    chainlink_ws_stream.close()

    for task in tasks:
        task.cancel()
    for task in tasks:
        try:
            await asyncio.wait_for(task, timeout=5)
        except BaseException:
            pass  # CancelledError / timeout on shutdown — ignore

    # Close the web3 Chainlink RPC sessions (created by chainlink_fetcher).
    try:
        await chainlink.chainlink_fetcher.aclose()
    except Exception:
        pass

app = FastAPI(title="Polymarket BTC 1h Assistant", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

# Global state to store the latest data
state = {
    "latest_data": {},
    "last_update_ts": 0,
    "trading_mode": settings.MODE,
    "paper_balance": settings.PAPER_BALANCE_USD,
    "active_trades": [],
    "trade_history": [],
    "logs": [],
    "last_balance_refresh": 0,
    "running": False,  # trading is off until the user presses Start on the dashboard
    "window_marks": {},  # market_id -> {open, open_ts, last, last_ts}: Chainlink 1h open/close
    "entered_markets": []  # market_ids already traded — ONE fresh entry per 1h window (flips excepted)
}

def _mark_window_entered(market_id):
    """Record that a window (market) has had a position, so it can't be re-entered this
    1h window (one entry per window). Bounded to the most recent 100."""
    mid = str(market_id)
    if mid not in state["entered_markets"]:
        state["entered_markets"].append(mid)
        if len(state["entered_markets"]) > 100:
            state["entered_markets"].pop(0)

def save_state():
    try:
        data_to_save = {
            "paper_balance": state["paper_balance"],
            "active_trades": state["active_trades"],
            "trade_history": state["trade_history"],
            "window_marks": state["window_marks"],
            "entered_markets": state["entered_markets"]
        }
        with open("state_data.json", "w") as f:
            json.dump(data_to_save, f, indent=2)
            
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                cfg = json.load(f)
            cfg["paper_balance_usd"] = state["paper_balance"]
            with open("config.json", "w") as f:
                json.dump(cfg, f, indent=2)
    except Exception as e:
        print(f"Error saving state: {e}")

def load_state():
    try:
        if os.path.exists("state_data.json"):
            with open("state_data.json", "r") as f:
                loaded = json.load(f)
                state["paper_balance"] = loaded.get("paper_balance", settings.PAPER_BALANCE_USD)
                state["active_trades"] = loaded.get("active_trades", [])
                state["trade_history"] = loaded.get("trade_history", [])
                state["window_marks"] = loaded.get("window_marks", {})
                state["entered_markets"] = loaded.get("entered_markets", [])
                log_message("State loaded from state_data.json")
    except Exception as e:
        print(f"Error loading state: {e}")

def log_message(msg: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    state["logs"].append(formatted)
    if len(state["logs"]) > 100:
        state["logs"].pop(0)

def get_ws_symbol_filter(symbol: str) -> str:
    s = symbol.upper()
    if s.endswith("USDT"):
        return s[:-4].lower()
    return s.lower()

def merge_live_close(klines: List[Dict], spot: Optional[float]) -> List[Dict]:
    """Fold the live @trade websocket tick into the forming (last) candle so the
    indicators (HA, RSI, AO) update in real time between the slower kline-WS pushes.
    Returns a shallow copy with the last candle's close set to `spot` and its high/low
    extended; the shared WS buffer is NOT mutated."""
    if not klines or spot is None:
        return klines
    out = list(klines)
    last = dict(out[-1])
    last["close"] = spot
    if last.get("high") is not None:
        last["high"] = max(last["high"], spot)
    if last.get("low") is not None:
        last["low"] = min(last["low"], spot)
    out[-1] = last
    return out

# Background task instances
binance_stream = ws_data.BinanceTradeStream(symbol=settings.SYMBOL)
binance_kline_15m = ws_data.BinanceKlineStream(symbol=settings.SYMBOL, interval="15m", limit=200)

polymarket_ws_stream = ws_data.PolymarketChainlinkStream(
    ws_url=settings.POLYMARKET_LIVE_DATA_WS_URL,
    symbol_includes=get_ws_symbol_filter(settings.SYMBOL)
)
chainlink_ws_stream = ws_data.ChainlinkPriceStream(aggregator=settings.get_aggregator(settings.SYMBOL))

def get_candle_window_timing(window_minutes: int) -> Dict[str, float]:
    now_ms = time.time() * 1000
    window_ms = window_minutes * 60_000
    start_ms = (now_ms // window_ms) * window_ms
    end_ms = start_ms + window_ms
    elapsed_ms = now_ms - start_ms
    remaining_ms = end_ms - now_ms
    return {
        "startMs": start_ms,
        "endMs": end_ms,
        "elapsedMs": elapsed_ms,
        "remainingMs": remaining_ms,
        "elapsedMinutes": elapsed_ms / 60_000,
        "remainingMinutes": remaining_ms / 60_000
    }

async def fetch_polymarket_snapshot() -> Dict[str, Any]:
    market = None
    if settings.POLYMARKET_SLUG:
        market = await data.fetch_market_by_slug(settings.POLYMARKET_SLUG)
    elif settings.POLYMARKET_AUTO_SELECT_LATEST:
        events = await data.fetch_live_events_by_series_id(settings.POLYMARKET_SERIES_ID)
        markets = data.flatten_event_markets(events)

        now = time.time() * 1000
        live_markets = [m for m in markets if m.get("endDate") and datetime.fromisoformat(m["endDate"].replace('Z', '+00:00')).timestamp() * 1000 > now]
        if live_markets:
            live_markets.sort(key=lambda x: x["endDate"])
            market = live_markets[0]

    if not market:
        return {"ok": False, "reason": "market_not_found"}

    outcomes = market.get("outcomes", [])
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)

    clob_token_ids = market.get("clobTokenIds", [])
    if isinstance(clob_token_ids, str):
        clob_token_ids = json.loads(clob_token_ids)

    outcome_prices = market.get("outcomePrices", [])
    if isinstance(outcome_prices, str):
        outcome_prices = json.loads(outcome_prices)

    up_token_id = None
    down_token_id = None

    for i, outcome in enumerate(outcomes):
        token_id = clob_token_ids[i] if i < len(clob_token_ids) else None
        if not token_id: continue
        if outcome.lower() == settings.POLYMARKET_UP_LABEL.lower():
            up_token_id = token_id
        elif outcome.lower() == settings.POLYMARKET_DOWN_LABEL.lower():
            down_token_id = token_id

    up_index = next((i for i, x in enumerate(outcomes) if x.lower() == settings.POLYMARKET_UP_LABEL.lower()), -1)
    down_index = next((i for i, x in enumerate(outcomes) if x.lower() == settings.POLYMARKET_DOWN_LABEL.lower()), -1)

    gamma_yes = float(outcome_prices[up_index]) if up_index >= 0 and up_index < len(outcome_prices) else None
    gamma_no = float(outcome_prices[down_index]) if down_index >= 0 and down_index < len(outcome_prices) else None

    if not up_token_id or not down_token_id:
        return {"ok": False, "reason": "missing_token_ids"}

    try:
        up_buy, down_buy, up_book, down_book = await asyncio.gather(
            data.fetch_clob_price(up_token_id, "buy"),
            data.fetch_clob_price(down_token_id, "buy"),
            data.fetch_order_book(up_token_id),
            data.fetch_order_book(down_token_id)
        )
        up_book_summary = data.summarize_order_book(up_book)
        down_book_summary = data.summarize_order_book(down_book)
    except:
        up_buy = None
        down_buy = None
        up_book_summary = {"bestBid": None, "bestAsk": None, "spread": None, "bidLiquidity": None, "askLiquidity": None}
        down_book_summary = {"bestBid": None, "bestAsk": None, "spread": None, "bidLiquidity": None, "askLiquidity": None}

    return {
        "ok": True,
        "market": market,
        "prices": {
            "up": up_buy if up_buy is not None else gamma_yes,
            "down": down_buy if down_buy is not None else gamma_no
        },
        "token_ids": {
            "up": up_token_id,
            "down": down_token_id
        },
        "orderbook": {
            "up": up_book_summary,
            "down": down_book_summary
        }
    }

async def execute_trade(decision: Dict[str, Any], market_prices: Dict[str, Any], market: Dict[str, Any], target_open: float, token_ids: Dict[str, Any], orderbook: Optional[Dict[str, Any]] = None):
    # Entry from the decision engine. If a RUNNING position in THIS market is on the
    # OPPOSITE side of a new ENTER signal, FLIP it (close the held side, then open the
    # new one) so a fresh opposite signal is never blocked. Returns a short reason string.
    if decision["action"] != "ENTER":
        return decision.get("reason", "no_trade")
    side = decision["side"]
    now_ts = time.time()

    # The currently-RUNNING position in THIS market (if any).
    running_here = next((t for t in state["active_trades"]
                         if str(t.get("market_id")) == str(market.get("id"))
                         and now_ts < (t.get("end_ts") or float("inf"))), None)
    if running_here is not None:
        if running_here["side"] == side:
            return "already_in_position"            # already on the signalled side
        if not settings.FLIP_ON_SIGNAL_ENABLED:
            return "hold_opposite"                  # flip disabled — hold the position to expiry
        # Opposite signal -> close the held side, then open the new one (flip). This is
        # the ONE exception to "one entry per window" (it requires a held opposite side).
        if not await _close_position(running_here, market_prices, token_ids, orderbook, "signal_flip"):
            return "flip_close_failed"
        log_message(f"FLIP on new signal: {running_here['side']} -> {side}")
    else:
        # FRESH entry — ONE per 1h window. If this window already had a position,
        # block re-entry until the next market.
        if str(market.get("id")) in state["entered_markets"]:
            return "window_done"

    # CONSTRAINT: one *running* position at a time. A trade whose 1h window has already
    # ended (only awaiting Polymarket's resolution) no longer blocks a new entry — the next
    # market is already live, so we can trade it while the old one settles in the background.
    if any(now_ts < (t.get("end_ts") or float("inf")) for t in state["active_trades"]):
        return "slot_busy"
    # Never hold two positions in the same market (running or still settling).
    if any(str(t.get("market_id")) == str(market.get("id")) for t in state["active_trades"]):
        return "market_busy"

    return await _open_position(side, market_prices, market, target_open, token_ids, orderbook)


async def _close_position(trade: Dict[str, Any], market_prices: Optional[Dict[str, Any]], token_ids: Optional[Dict[str, Any]], orderbook: Optional[Dict[str, Any]], reason: str) -> bool:
    """Sell out of `trade`, book its P/L and move it to history. Returns True on success,
    False if it couldn't be closed (no exit price / live sell failed). Used by both the
    new-signal flip and the close-on-1m-reversal."""
    held_key = "up" if trade["side"] == "UP" else "down"
    ob = (orderbook or {}).get(held_key) or {}
    exit_price = ob.get("bestBid") or (market_prices or {}).get(held_key)
    if not exit_price or exit_price <= 0:
        log_message(f"Close aborted ({reason}): no exit price for {trade['side']}")
        return False

    if state["trading_mode"] == "live":
        token_id = (token_ids or {}).get(held_key)
        result = await asyncio.to_thread(clob_trader.place_market_sell, token_id, trade["shares"], exit_price)
        if not result.get("ok"):
            log_message(f"Close sell FAILED ({reason}, {trade['side']}): {result.get('error')}")
            return False
        # live balance is refreshed from chain elsewhere
    else:
        state["paper_balance"] += trade["shares"] * exit_price  # proceeds from selling out

    trade["status"] = "CLOSED"
    trade["exit_time"] = datetime.now().isoformat()
    trade["exit_reason"] = reason
    trade["settlement_price_at_expiry"] = exit_price
    trade["profit_loss"] = (trade["shares"] * exit_price) - trade["amount"]
    state["trade_history"].append(trade)
    state["active_trades"] = [t for t in state["active_trades"] if t is not trade]
    save_state()
    log_message(f"CLOSE ({reason}): {trade['side']} @ {exit_price:.2f} (P/L ${trade['profit_loss']:.2f})")
    return True


async def _open_position(side: str, market_prices: Dict[str, Any], market: Dict[str, Any], target_open: float, token_ids: Dict[str, Any], orderbook: Optional[Dict[str, Any]] = None):
    # Open a fresh position on `side`: sizing, liquidity trim, and paper/live execution.
    # Entry GATES (the decide_entry signal, slot/market-busy) are the caller's responsibility
    # — a flip reuses this to open the opposite side unconditionally.
    price = market_prices["up"] if side == "UP" else market_prices["down"]
    if price is None:
        return "no_price"

    # ── Risk per trade ──────────────────────────────────────────────────────────
    # RISK_TYPE selects how the stake (the dollars put at risk) is sized:
    #   "percent" -> RISK_VALUE% of the current balance
    #   "fixed"   -> RISK_VALUE dollars, flat
    balance = state["paper_balance"]
    risk_type = (settings.RISK_TYPE or "percent").lower()
    if risk_type == "fixed":
        amount_to_risk = float(settings.RISK_VALUE)
    else:  # "percent" (default)
        amount_to_risk = (float(settings.RISK_VALUE) / 100.0) * balance

    if amount_to_risk <= 0:
        return "stake_zero"

    # Liquidity: never outsize what the ask side of the book can absorb.
    ob = (orderbook or {}).get("up" if side == "UP" else "down") or {}
    ask_liq_shares = ob.get("askLiquidity")
    if ask_liq_shares is not None and price > 0:
        ask_liq_usd = ask_liq_shares * price
        if ask_liq_usd < settings.MIN_BOOK_LIQUIDITY_USD:
            log_message(f"Skip {side}: thin book (${ask_liq_usd:.2f} ask liquidity)")
            return "thin_book"
        amount_to_risk = min(amount_to_risk, ask_liq_usd)  # don't outsize the book

    if balance < amount_to_risk or amount_to_risk <= 0:
        print(f"Insufficient paper balance ({balance}) or invalid risk amount ({amount_to_risk})")
        return "insufficient_balance"

    end_date_str = market.get("endDate")
    end_ts = 0
    if end_date_str:
        try:
            end_ts = datetime.fromisoformat(end_date_str.replace('Z', '+00:00')).timestamp()
        except: pass
    # Fallback so a trade always has a definite expiry even if endDate is missing/unparseable
    if not end_ts:
        end_ts = time.time() + settings.CANDLE_WINDOW_MINUTES * 60

    trade = {
        "market_id": market["id"],
        "market_slug": market.get("slug"),
        "side": side,
        "entry_price": price,
        "amount": amount_to_risk,
        "shares": amount_to_risk / price,
        "entry_time": datetime.now().isoformat(),
        "status": "OPEN",
        "settlement_price": None,
        "profit_loss": None,
        "strike_price": target_open,
        "chainlink_open": state["window_marks"].get(str(market["id"]), {}).get("open"),
        "end_ts": end_ts,
        "mode": state["trading_mode"]
    }

    if state["trading_mode"] == "paper":
        state["paper_balance"] -= amount_to_risk
        state["active_trades"].append(trade)
        _mark_window_entered(market["id"])
        save_state()

        log_message(f"Executed PAPER trade: {side} @ {price} for {market.get('slug')} (Amount: ${amount_to_risk:.2f})")
        return "entered"
    else:
        # LIVE: place a real Fill-Or-Kill market BUY on the Polymarket CLOB
        token_id = token_ids.get("up") if side == "UP" else token_ids.get("down")
        if not token_id:
            log_message(f"LIVE trade aborted: missing token_id for side {side}")
            return "missing_token_id"

        result = await asyncio.to_thread(clob_trader.place_market_buy, token_id, amount_to_risk, price)
        if result.get("ok"):
            resp = result.get("response") or {}
            order_id = None
            if isinstance(resp, dict):
                order_id = resp.get("orderID") or resp.get("orderId") or resp.get("id")
            trade["order_id"] = order_id
            trade["order_response"] = resp
            state["active_trades"].append(trade)
            _mark_window_entered(market["id"])
            save_state()
            log_message(f"Executed LIVE trade: {side} ${amount_to_risk:.2f} on {market.get('slug')} (order {order_id})")
            return "entered"
        else:
            log_message(f"LIVE trade FAILED ({side}): {result.get('error')}")
            return "live_order_failed"

async def update_trades(current_prices: Dict[str, Any]):
    remaining_active = []
    trades_changed = False
    now_ts = time.time()

    # Settlement prefers Polymarket's authoritative resolution; if that isn't readable
    # by the time the window ends, we resolve the trade ourselves from the CHAINLINK
    # window open vs close (the same feed Polymarket settles on). cl_price is that
    # Chainlink price; spot is only a last-resort snapshot for the record.
    cl_price = current_prices.get("chainlink")
    cur_price = cl_price or current_prices.get("spot")
    SETTLEMENT_GRACE_SECONDS = 600  # only used if we have NO open/close marks to resolve with

    for trade in state["active_trades"]:
        # Keep a rolling price snapshot so settlement always has a recent value,
        # even if the feed drops out exactly at expiry.
        if cur_price:
            trade["last_price"] = cur_price

        # Effective window end. If endDate was missing at entry (end_ts == 0), derive
        # it from entry_time + window so a trade can never wait forever.
        end_ts = trade.get("end_ts", 0)
        if not end_ts:
            try:
                end_ts = datetime.fromisoformat(trade["entry_time"]).timestamp() + settings.CANDLE_WINDOW_MINUTES * 60
            except Exception:
                end_ts = now_ts
        expired = now_ts >= end_ts

        # Freeze the CHAINLINK close once the window ends (captured once, at the first
        # tick at/after expiry). Falls back to the last price seen while the window was
        # the live snapshot.
        if expired and trade.get("chainlink_close") is None:
            _wm0 = state["window_marks"].get(str(trade.get("market_id"))) or {}
            trade["chainlink_close"] = cl_price or _wm0.get("last")

        # Always poll the market (throttled ~15s) so we can read the AUTHORITATIVE
        # Polymarket resolution even after the local clock says the window expired.
        market = None
        if trade.get("last_api_check", 0) < now_ts - 15:
            try:
                market = await data.fetch_market_by_slug(trade["market_slug"])
            except Exception:
                market = None
            trade["last_api_check"] = now_ts
            if market is not None:
                trade["_market_closed"] = bool(market.get("closed"))
        market_closed = trade.get("_market_closed", False)

        # Still live: window running and market still open → keep waiting.
        if not expired and not market_closed:
            remaining_active.append(trade)
            continue

        # ---- Determine the winning outcome ----
        outcomes = []
        outcome_prices = []
        if market:
            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str): outcomes = json.loads(outcomes)
            outcome_prices = market.get("outcomePrices", [])
            if isinstance(outcome_prices, str): outcome_prices = json.loads(outcome_prices)
        if not outcomes:
            outcomes = [settings.POLYMARKET_UP_LABEL, settings.POLYMARKET_DOWN_LABEL]

        up_index = next((i for i, x in enumerate(outcomes) if x.lower() == settings.POLYMARKET_UP_LABEL.lower()), 0)
        down_index = next((i for i, x in enumerate(outcomes) if x.lower() == settings.POLYMARKET_DOWN_LABEL.lower()), 1)

        # Resolution: prefer Polymarket's AUTHORITATIVE result (the settled outcome
        # trades at ~$1 = the on-chain payout). If that isn't readable by expiry, fall
        # back to the bot's own CHAINLINK window open-vs-close (same feed Polymarket
        # settles on) — see step 2 below. Only VOID if neither is available.
        winning_index = -1
        resolved_by = None
        # 1) Authoritative: a settled Polymarket outcome trades at ~$1.
        for i, p in enumerate(outcome_prices):
            try:
                if float(p) > 0.9:
                    winning_index = i
                    resolved_by = "polymarket"
                    break
            except Exception:
                pass

        # 2) Self-contained fallback (window over): CHAINLINK open vs close. Mirrors
        # Polymarket's settlement — UP wins if close > open, else DOWN. A trade resolves
        # the instant its window ends instead of waiting on Polymarket to publish.
        _wm = state["window_marks"].get(str(trade.get("market_id"))) or {}
        open_px = trade.get("chainlink_open") or _wm.get("open")
        close_px = trade.get("chainlink_close") or _wm.get("last")
        if winning_index == -1 and expired and open_px and close_px:
            winning_index = up_index if close_px > open_px else down_index
            resolved_by = "chainlink_open_close"
            trade["chainlink_open"] = open_px
            trade["chainlink_close"] = close_px

        # Snapshot a settlement price for the record.
        settlement_price = (trade.get("settlement_price_at_expiry") or close_px
                            or trade.get("last_price") or cur_price)

        # ---- Could not resolve yet (no Polymarket result AND no open/close marks) ----
        if winning_index == -1:
            trade["status"] = "AWAITING"  # window over, waiting; does not block new entries
            first_seen = trade.get("unresolved_since")
            if first_seen is None:
                trade["unresolved_since"] = now_ts
                remaining_active.append(trade)
                continue
            if now_ts - first_seen < SETTLEMENT_GRACE_SECONDS:
                remaining_active.append(trade)
                continue
            # Grace exhausted and still no open/close marks — void so one stuck trade
            # can't block forever.
            trade["status"] = "VOID"
            trade["exit_time"] = datetime.now().isoformat()
            trade["profit_loss"] = 0.0
            if trade.get("mode", "paper") == "paper":
                state["paper_balance"] += trade["amount"]  # refund the simulated stake
                log_message(f"VOID: {trade['market_slug']} no open/close marks past grace; paper stake refunded.")
            else:
                log_message(f"VOID: {trade['market_slug']} unresolved within grace; live balance reflects on-chain settlement.")
            continue

        # ---- Settle WIN / LOSS ----
        won = ((trade["side"] == "UP" and winning_index == up_index) or
               (trade["side"] == "DOWN" and winning_index == down_index))

        if won:
            payout = trade["shares"] * 1.0
            # Paper credits the simulated balance; live balance comes from the
            # on-chain USDC refresh in the main loop, not credited here.
            if trade.get("mode", "paper") == "paper":
                state["paper_balance"] += payout
            trade["profit_loss"] = payout - trade["amount"]
            log_message(f"WIN [{resolved_by}]: {trade['market_slug']} settled (open {open_px} -> close {close_px}). Profit: ${trade['profit_loss']:.2f}")
        else:
            trade["profit_loss"] = -trade["amount"]
            log_message(f"LOSS [{resolved_by}]: {trade['market_slug']} settled (open {open_px} -> close {close_px}). Loss: ${trade['profit_loss']:.2f}")

        trade["status"] = "CLOSED"
        trade["exit_time"] = datetime.now().isoformat()
        trade["resolved_by"] = resolved_by
        trade["settlement_price_at_expiry"] = trade.get("settlement_price_at_expiry") or settlement_price
        trade["winning_outcome"] = outcomes[winning_index] if 0 <= winning_index < len(outcomes) else None
        state["trade_history"].append(trade)
        trades_changed = True

    state["active_trades"] = remaining_active
    if trades_changed:
        save_state()

async def seed_kline_buffers():
    try:
        k15m = await data.fetch_klines(settings.SYMBOL, "15m", 200)
        binance_kline_15m.set_candles(k15m)
        log_message(f"Seeded Binance 15m kline buffer for {settings.SYMBOL}")
    except Exception as e:
        log_message(f"Failed to seed kline buffers: {e}")

async def update_loop():
    csv_header = [
        "timestamp", "entry_minute", "time_left_min", "signal",
        "mkt_up", "mkt_down", "recommendation", "reason", "exec_result"
    ]

    while True:
        try:
            timing = get_candle_window_timing(settings.CANDLE_WINDOW_MINUTES)

            binance_ws = binance_stream.get_last()
            if not binance_ws.get("price"):
                poly_ws_last = polymarket_ws_stream.get_last()
                cl_ws_last = chainlink_ws_stream.get_last()
                binance_ws["price"] = poly_ws_last.get("price") or cl_ws_last.get("price")
            poly_ws = polymarket_ws_stream.get_last()
            cl_ws = chainlink_ws_stream.get_last()

            results = await asyncio.gather(
                data.fetch_last_price(settings.SYMBOL),
                chainlink.chainlink_fetcher.fetch_chainlink_btc_usd(),
                fetch_polymarket_snapshot(),
                return_exceptions=True
            )

            last_price = results[0] if not isinstance(results[0], Exception) else None
            chainlink_data = results[1] if not isinstance(results[1], Exception) else {}
            poly_snapshot = results[2] if not isinstance(results[2], Exception) else {"ok": False}

            spot_price = binance_ws.get("price") if binance_ws and binance_ws.get("price") else last_price

            # Fold the live @trade websocket tick into the forming 15m candle so the
            # Heiken-Ashi is computed in real time off live data, not just on the
            # slower kline-WS candle pushes.
            klines_15m = merge_live_close(binance_kline_15m.get_candles(), spot_price)

            # Window open price (the strike) — recorded on each trade for reference.
            # The 1h window start is also a 15m boundary, so take the 15m candle at it.
            target_open = spot_price
            if klines_15m:
                start_ms = timing["startMs"]
                for c in reversed(klines_15m):
                    if c["openTime"] <= start_ms:
                        target_open = c["open"]
                        break

            current_price = None
            price_source = None

            if cl_ws.get("price"):
                current_price = cl_ws["price"]
                price_source = "Chainlink RPC WS"
            elif poly_ws.get("price"):
                current_price = poly_ws["price"]
                price_source = "Polymarket WS"
            elif chainlink_data.get("price"):
                current_price = chainlink_data["price"]
                price_source = "Chainlink RPC REST"

            # ── Mark the 1h window's Chainlink OPEN/CLOSE ────────────────────────
            # Polymarket settles BTC Up/Down on Chainlink. A market becomes the live
            # window exactly at its start, so the FIRST time we see it we snapshot the
            # open; every later tick refreshes "last" (which freezes at ~window end,
            # once auto-select moves on to the next window). This lets us resolve a
            # trade ourselves from open-vs-close without waiting on Polymarket.
            if poly_snapshot.get("ok") and current_price:
                _mid = str(poly_snapshot["market"].get("id"))
                _wm = state["window_marks"].get(_mid)
                if _wm is None:
                    state["window_marks"][_mid] = {
                        "open": current_price, "open_ts": time.time(),
                        "last": current_price, "last_ts": time.time(),
                    }
                    if len(state["window_marks"]) > 50:  # prune oldest, bound the map
                        _oldest = min(state["window_marks"], key=lambda k: state["window_marks"][k].get("open_ts", 0))
                        state["window_marks"].pop(_oldest, None)
                else:
                    _wm["last"] = current_price
                    _wm["last_ts"] = time.time()

            settlement_ms = None
            if poly_snapshot["ok"] and poly_snapshot["market"].get("endDate"):
                settlement_ms = datetime.fromisoformat(poly_snapshot["market"]["endDate"].replace('Z', '+00:00')).timestamp() * 1000

            time_left_min = (settlement_ms - time.time() * 1000) / 60_000 if settlement_ms else timing["remainingMinutes"]

            # 15m Heiken-Ashi streak {color, count} — the direction signal.
            consec = indicators.count_consecutive(indicators.compute_heiken_ashi(klines_15m))

            market_up = poly_snapshot["prices"]["up"] if poly_snapshot["ok"] else None
            market_down = poly_snapshot["prices"]["down"] if poly_snapshot["ok"] else None

            # ── PERSISTENCE: current price vs the 1h window OPEN. above => UP, below => DOWN.
            # Prefer the CHAINLINK open (what Polymarket settles on) when it was snapshotted
            # at the window start (open_ts near it); else fall back to the BINANCE 15m hour
            # open (`target_open`, always available from the seeded klines). Same feed each
            # branch = no cross-feed offset.
            window_open = None
            above_open = None
            open_source = None
            if poly_snapshot.get("ok"):
                _wm = state["window_marks"].get(str(poly_snapshot["market"].get("id"))) or {}
                _start_sec = timing["startMs"] / 1000.0
                _cl_open, _cl_open_ts = _wm.get("open"), _wm.get("open_ts")
                if _cl_open and _cl_open_ts and _cl_open_ts <= _start_sec + 120 and current_price:
                    window_open, above_open, open_source = _cl_open, (current_price > _cl_open), "chainlink"
                elif target_open and spot_price:
                    window_open, above_open, open_source = target_open, (spot_price > target_open), "binance"

            # ── ENTRY (simple): BUY if price ABOVE the 1h open AND 15m HA green; SELL if
            # ── price BELOW AND 15m HA red. A new opposite signal flips the position.
            decision = engines.decide_entry({
                "aboveOpen": above_open,        # price vs the 1h open (persistence)
                "ha15Color": consec["color"],   # 15m Heiken-Ashi colour
                "priceUp": market_up,
                "priceDown": market_down,
            })

            current_prices_dict = {"spot": spot_price, "chainlink": current_price}

            # Trading actions only fire when the user has pressed Start. Data, prices
            # and the dashboard keep updating regardless so the balance view stays live.
            exec_result = None
            if poly_snapshot["ok"] and state["running"]:
                # A new opposite signal FLIPS the position (close + open opposite); a
                # same-side signal is already_in_position; otherwise hold to expiry.
                exec_result = await execute_trade(decision, poly_snapshot["prices"], poly_snapshot["market"], target_open, poly_snapshot.get("token_ids", {}), poly_snapshot.get("orderbook", {}))
            elif not state["running"]:
                exec_result = "stopped"

            # Always keep settling any already-open positions so they can't get stuck.
            await update_trades(current_prices_dict)

            # In live mode, reflect the real on-chain USDC balance in the dashboard
            if state["trading_mode"] == "live":
                now_ts = time.time()
                if now_ts - state.get("last_balance_refresh", 0) > 30:
                    real_bal = await asyncio.to_thread(clob_trader.get_usdc_balance)
                    if real_bal is not None:
                        state["paper_balance"] = real_bal
                    state["last_balance_refresh"] = now_ts

            signal_label = f"BUY {decision['side']}" if decision["action"] == "ENTER" else "NO TRADE"
            utils.append_csv_row("./logs/signals.csv", csv_header, [
                datetime.now().isoformat(), timing["elapsedMinutes"], time_left_min,
                signal_label, market_up, market_down,
                f"{decision['side']}:{decision['phase']}:{decision['strength']}" if decision["action"] == "ENTER" else "NO_TRADE",
                decision.get("reason", ""), exec_result or ""
            ])

            state["latest_data"] = {
                "timestamp": datetime.now().isoformat(),
                "timing": timing,
                "market": poly_snapshot.get("market") if poly_snapshot["ok"] else None,
                "trading_state": {
                    "mode": state["trading_mode"],
                    "balance": state["paper_balance"],
                    "running": state["running"],
                    "active_trades": state["active_trades"],
                    "history_count": len(state["trade_history"]),
                    "risk": {"type": settings.RISK_TYPE, "value": settings.RISK_VALUE},
                    "symbol": settings.SYMBOL
                },
                "prices": {
                    "spot": spot_price,
                    "chainlink": current_price,
                    "chainlink_source": price_source,
                    "poly_up": market_up,
                    "poly_down": market_down
                },
                "indicators": {
                    "heiken_15m": consec,
                    "vs_open": {"window_open": window_open, "above_open": above_open, "source": open_source}
                },
                "analysis": {
                    "decision": decision
                }
            }
            state["last_update_ts"] = time.time()

        except Exception as e:
            print(f"Error in update loop: {e}")

        await asyncio.sleep(settings.POLL_INTERVAL_MS / 1000)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/settings", response_class=HTMLResponse)
async def get_settings_page(request: Request):
    return templates.TemplateResponse("settings.html", {"request": request})

@app.get("/api/latest")
async def get_latest():
    return state["latest_data"]

@app.get("/api/logs")
async def get_logs():
    return state["logs"]

@app.get("/api/available-series")
async def get_available_series():
    return await data.fetch_available_hourly_series()

@app.get("/api/settings")
async def get_settings():
    pk = settings.PRIVATE_KEY
    masked_pk = pk[:6] + "..." + pk[-4:] if pk and len(pk) > 10 else pk

    return {
        "mode": settings.MODE,
        "paper_balance_usd": settings.PAPER_BALANCE_USD,
        "private_key": masked_pk,
        "chainlink": {
            "alchemy_api_key": settings.ALCHEMY_API_KEY
        },
        "relayer": {
            "api_key": settings.RELAYER_API_KEY
        },
        "polymarket": {
            "series_id": settings.POLYMARKET_SERIES_ID,
            "gamma_base_url": settings.GAMMA_BASE_URL,
            "clob_base_url": settings.CLOB_BASE_URL,
            "live_ws_url": settings.POLYMARKET_LIVE_DATA_WS_URL,
            "up_label": settings.POLYMARKET_UP_LABEL,
            "down_label": settings.POLYMARKET_DOWN_LABEL
        },
        "trading": {
            "symbol": settings.SYMBOL,
            "risk_type": settings.RISK_TYPE,
            "risk_value": settings.RISK_VALUE
        },
        "entry": {
            "flip_on_signal": settings.FLIP_ON_SIGNAL_ENABLED,
            "min_book_liquidity_usd": settings.MIN_BOOK_LIQUIDITY_USD
        }
    }

@app.post("/api/settings")
async def post_settings(new_settings: Dict[str, Any]):
    global binance_stream, polymarket_ws_stream, chainlink_ws_stream, binance_kline_15m
    old_symbol = settings.SYMBOL

    # Credential may be a hex private key OR a 12/24-word seed phrase. We persist
    # only the derived hex key (EOA) so the seed phrase is never written to disk.
    new_pk = new_settings.get("private_key")
    if new_pk and "..." in new_pk:
        new_settings["private_key"] = settings.PRIVATE_KEY  # masked value unchanged
    elif new_pk:
        try:
            resolved = normalize_private_key(new_pk)
        except Exception as e:
            log_message(f"Invalid private key / seed phrase: {e}")
            resolved = ""
        settings.PRIVATE_KEY = resolved
        new_settings["private_key"] = resolved

    # Deep-merge into the existing config so keys not present in the settings form
    # (chainlink, binance_base_url, poll_interval_ms, etc.) are preserved.
    existing_cfg = {}
    if os.path.exists("config.json"):
        try:
            with open("config.json", "r") as f:
                existing_cfg = json.load(f)
        except Exception:
            existing_cfg = {}

    def deep_merge(base, override):
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    merged_cfg = deep_merge(existing_cfg, new_settings)
    with open("config.json", "w") as f:
        json.dump(merged_cfg, f, indent=2)

    settings.MODE = new_settings.get("mode", settings.MODE)
    settings.PAPER_BALANCE_USD = float(new_settings.get("paper_balance_usd", settings.PAPER_BALANCE_USD))

    if "trading" in new_settings:
        t = new_settings["trading"]
        settings.SYMBOL = t.get("symbol", settings.SYMBOL)
        settings.RISK_TYPE = t.get("risk_type", settings.RISK_TYPE)
        settings.RISK_VALUE = float(t.get("risk_value", settings.RISK_VALUE))

    if "entry" in new_settings:
        en = new_settings["entry"]
        if "flip_on_signal" in en:
            settings.FLIP_ON_SIGNAL_ENABLED = bool(en["flip_on_signal"])
        settings.MIN_BOOK_LIQUIDITY_USD = float(en.get("min_book_liquidity_usd", settings.MIN_BOOK_LIQUIDITY_USD))

    if "polymarket" in new_settings:
        p = new_settings["polymarket"]
        settings.POLYMARKET_SERIES_ID = p.get("series_id", settings.POLYMARKET_SERIES_ID)
        settings.POLYMARKET_UP_LABEL = p.get("up_label", settings.POLYMARKET_UP_LABEL)
        settings.POLYMARKET_DOWN_LABEL = p.get("down_label", settings.POLYMARKET_DOWN_LABEL)

    if "chainlink" in new_settings:
        cl = new_settings["chainlink"]
        if "alchemy_api_key" in cl:
            settings.ALCHEMY_API_KEY = cl["alchemy_api_key"]

    if "relayer" in new_settings:
        rl = new_settings["relayer"]
        if "api_key" in rl:
            settings.RELAYER_API_KEY = rl["api_key"]
        if "api_key_address" in rl:
            settings.RELAYER_API_KEY_ADDRESS = rl["api_key_address"]

    # Credentials/signature may have changed — drop the cached CLOB client so the
    # next live order re-initialises with the new key/seed-derived wallet.
    clob_trader.reset()

    state["trading_mode"] = settings.MODE
    state["paper_balance"] = settings.PAPER_BALANCE_USD

    if settings.SYMBOL != old_symbol:
        binance_stream.close()
        binance_stream = ws_data.BinanceTradeStream(symbol=settings.SYMBOL)
        asyncio.create_task(binance_stream.start())

        binance_kline_15m.close()
        binance_kline_15m = ws_data.BinanceKlineStream(symbol=settings.SYMBOL, interval="15m", limit=200)
        asyncio.create_task(binance_kline_15m.start())

        await seed_kline_buffers()

        polymarket_ws_stream.close()
        polymarket_ws_stream = ws_data.PolymarketChainlinkStream(
            ws_url=settings.POLYMARKET_LIVE_DATA_WS_URL,
            symbol_includes=get_ws_symbol_filter(settings.SYMBOL)
        )
        asyncio.create_task(polymarket_ws_stream.start())

        chainlink_ws_stream.close()
        chainlink_ws_stream = ws_data.ChainlinkPriceStream(aggregator=settings.get_aggregator(settings.SYMBOL))
        asyncio.create_task(chainlink_ws_stream.start())

    return {"status": "ok"}

def _reflect_running_now():
    """Mirror the running flag into latest_data immediately so /api/latest is in sync
    on the very next poll (the update loop would otherwise lag ~1s, flickering the UI)."""
    ts = state["latest_data"].get("trading_state")
    if isinstance(ts, dict):
        ts["running"] = state["running"]

@app.post("/api/start")
async def start_trading():
    """Begin trading. Data/prices already stream continuously; this flips the gate so
    the engine may enter/flip trades."""
    state["running"] = True
    _reflect_running_now()
    log_message("Trading STARTED by user")
    return {"ok": True, "running": True}

@app.post("/api/stop")
async def stop_trading():
    """Stop all trading. New entries and flips are halted immediately; any already-open
    position keeps settling to expiry so it can't get stuck."""
    state["running"] = False
    _reflect_running_now()
    log_message("Trading STOPPED by user")
    return {"ok": True, "running": False}

@app.post("/api/test-credentials")
async def test_credentials():
    """Validate the saved private key / seed phrase: derive the EOA wallet address and
    confirm the CLOB client initialises. Returns the address (and USDC balance)."""
    result = await asyncio.to_thread(clob_trader.test_connection)
    if result.get("ok"):
        bal = result.get("usdc_balance")
        log_message(f"Credential test OK: wallet {result.get('address')}" + (f" (USDC ${bal:.2f})" if bal is not None else ""))
    else:
        log_message(f"Credential test failed: {result.get('error')}")
    return result

@app.get("/health")
async def health():
    return {"status": "ok", "last_update": state["last_update_ts"], "mode": state["trading_mode"], "running": state["running"]}

@app.get("/history")
async def get_history():
    return state["trade_history"]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
