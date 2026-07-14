import os
import json
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Dict, Any, Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    MODE: str = "paper"  # "paper" or "live"
    PAPER_BALANCE_USD: float = 1000.0
    PRIVATE_KEY: str = ""  # hex private key, or derived from a 12/24-word seed phrase

    # Live trading (Polymarket CLOB V2). The wallet is derived from PRIVATE_KEY
    # (hex key or seed phrase). New wallets trade via the gasless deposit-wallet
    # flow (POLY_1271); the relayer key sponsors on-chain setup (deploy/approvals).
    CLOB_MAX_SLIPPAGE: float = 0.02  # marketable-limit buffer above the quote (probability units)
    RELAYER_API_KEY: str = ""          # Polymarket relayer API key (gasless on-chain txs)
    # The relayer key's owner address is the EOA — derived from PRIVATE_KEY automatically.

    # Polymarket on-chain contracts (Polygon) — used for EOA allowance setup
    USDC_ADDRESS: str = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"
    CTF_ADDRESS: str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    CLOB_EXCHANGE_ADDRESS: str = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"
    CLOB_NEG_RISK_EXCHANGE_ADDRESS: str = "0xC5d563A36AE78145C45a50134d48A1215220f80a"
    CLOB_NEG_RISK_ADAPTER_ADDRESS: str = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

    SYMBOL: str = "BTCUSDT"
    # data-api.binance.vision = official market-data mirror; api.binance.com
    # geo-blocks (HTTP 451) in some regions and this bot only needs market data.
    BINANCE_BASE_URL: str = "https://data-api.binance.vision"
    GAMMA_BASE_URL: str = "https://gamma-api.polymarket.com"
    CLOB_BASE_URL: str = "https://clob.polymarket.com"

    POLL_INTERVAL_MS: int = 1000
    CANDLE_WINDOW_MINUTES: int = 15

    # Risk per trade: "percent" = RISK_VALUE% of balance; "fixed" = RISK_VALUE dollars.
    RISK_TYPE: str = "percent"
    RISK_VALUE: float = 10.0

    # ── Entry price gates ───────────────────────────────────────────────────────
    # After 1m HA + 1m AO + RSI(50) confirm the side and price action (price vs the 15m
    # open) agrees, the ask must be below the cap. Purely technical — no probability model.
    MAX_ENTRY_PRICE: float = 0.60       # skip if the side's odds (ask price) are >= this
    MIN_BOOK_LIQUIDITY_USD: float = 20.0  # skip if the ask side can't absorb the stake

    # ── Win-rate gates (from the 30d Binance 1m backtest, 2026-07-03) ───────────
    # Entering early or on a hair-thin lead is a coin flip; the same signals win far
    # more when the window has developed and the lead is real:
    #   entry minute 1-2 -> 64% win;  >=5 -> 78%;  >=9 -> 86%
    #   lead <2 bps -> ~55% win;  >=3 bps -> 62%+;  >=12 bps -> 76-90%
    # Defaults = the balanced tier: minute >=5 + lead >=3 bps ~ 82% win, ~18 trades/day.
    MIN_ENTRY_ELAPSED_MIN: float = 5.0  # no entries before this many elapsed minutes
    MIN_LEAD_BPS: float = 3.0           # |price - window open| must be >= this (bps of price)

    # ── Strategy selection ──────────────────────────────────────────────────────
    # "model" = the calibrated ML win-probability drives entries (indicator/timing/lead
    #           gates SUBSUMED into features; fixed price cap REPLACED by the EV gate).
    # "gates" = the purely-technical hand-gate stack (fallback / A-B).
    STRATEGY_MODE: str = "model"
    MODEL_MIN_CONF: float = 0.80        # min P(chosen side) to enter (backtest: ~86% win)
    MODEL_EV_MARGIN: float = 0.02       # required edge = P(side) - ask, covers spread/slippage
    # Sizing in model mode: fractional Kelly on (P, ask), capped at the RISK_VALUE stake.
    MODEL_KELLY_FRACTION: float = 0.25  # quarter-Kelly (0 = flat RISK_VALUE sizing)

    # ── ML EXIT (model mode) — decide_exit, driven by the SAME calibrated P(up) ──
    # The exit is the mirror of the entry: sell the held side when the market's bid
    # overpays vs the model's fair value (take-profit) or the model's win prob collapses
    # (stop). EXIT_MODE gates whether it TRADES:
    #   "off"    — never exit early; hold every position to expiry (backtest-preferred).
    #   "shadow" — LOG would-sell decisions (bid + prob + outcome) but DON'T act, so the
    #              exit can be validated against hold-to-expiry before trusting it.
    #   "live"   — actually sell when decide_exit says SELL.
    # Default "shadow": prior backtests showed early exits LOSE to holding, so the ML
    # exit must prove itself in shadow before it may act.
    EXIT_MODE: str = "shadow"
    EXIT_TAKE_PROFIT_MARGIN: float = 0.03  # sell if held bid - P(side win) >= this
    EXIT_STOP_PROB: float = 0.0            # sell if P(side win) < this (0 = stop disabled)

    # Entry gates (all mandatory): 1m HA direction (colour) + Awesome Oscillator(1m,
    # bar colour rising=green) + RSI(50) + price-vs-open (price action).

    # Close-on-reversal: CLOSE (do not reverse) a running position when the 1m HA AND the
    # 1m AO both flip against it for >= CLOSE_REVERSAL_BARS consecutive bars. Only closes
    # the position — never opens the opposite side. Also locks the window (one entry/window).
    CLOSE_ON_REVERSAL_ENABLED: bool = False
    CLOSE_REVERSAL_BARS: int = 3   # require the 1m HA & 1m AO reversal to hold >= this many bars

    # ── TAKE-PROFIT / STOP-LOSS — % of the AMOUNT STAKED on the trade ───────────
    # The open position is marked to market at the held side's best BID (what selling
    # back right now actually yields):
    #     pnl_pct = (shares * bid - amount) / amount * 100
    # TP closes the trade when pnl_pct >= +TAKE_PROFIT_PERCENT; SL closes it when
    # pnl_pct <= -STOP_LOSS_PERCENT. The two are INDEPENDENT toggles (either, both or
    # neither). Strategy-agnostic — they work in both "model" and "gates" mode.
    #
    # Default OFF: prior backtests on this bot showed early exits LOSE to holding to
    # expiry (held +$699 vs signal_flip -$209 / reversal_close -$62), so arming these
    # is a deliberate choice, not a default.
    TAKE_PROFIT_ENABLED: bool = False
    TAKE_PROFIT_PERCENT: float = 30.0   # close at +30% of the stake
    STOP_LOSS_ENABLED: bool = False
    STOP_LOSS_PERCENT: float = 30.0     # close at -30% of the stake

    RSI_PERIOD: int = 14

    # Polymarket
    POLYMARKET_SLUG: str = os.getenv("POLYMARKET_SLUG", "")
    POLYMARKET_SERIES_ID: str = os.getenv("POLYMARKET_SERIES_ID", "10192")
    POLYMARKET_SERIES_SLUG: str = os.getenv("POLYMARKET_SERIES_SLUG", "btc-up-or-down-15m")
    POLYMARKET_AUTO_SELECT_LATEST: bool = os.getenv("POLYMARKET_AUTO_SELECT_LATEST", "true").lower() == "true"
    POLYMARKET_LIVE_DATA_WS_URL: str = os.getenv("POLYMARKET_LIVE_WS_URL", "wss://ws-live-data.polymarket.com")
    POLYMARKET_UP_LABEL: str = os.getenv("POLYMARKET_UP_LABEL", "Up")
    POLYMARKET_DOWN_LABEL: str = os.getenv("POLYMARKET_DOWN_LABEL", "Down")

    # Chainlink
    POLYGON_RPC_URL: str = os.getenv("POLYGON_RPC_URL", "https://polygon.drpc.org")
    POLYGON_RPC_URLS: List[str] = [url.strip() for url in os.getenv("POLYGON_RPC_URLS", "").split(",") if url.strip()]
    POLYGON_WSS_URL: str = os.getenv("POLYGON_WSS_URL", "wss://polygon-bor-rpc.publicnode.com")
    POLYGON_WSS_URLS: List[str] = [url.strip() for url in os.getenv("POLYGON_WSS_URLS", "").split(",") if url.strip()]
    CHAINLINK_BTC_USD_AGGREGATOR: str = os.getenv("CHAINLINK_BTC_USD_AGGREGATOR", "0xc907E116054Ad103354f2D350FD2514433D57F6f")

    # Alchemy — preferred Polygon RPC/WSS when an API key is set (used first, with
    # the public RPCs kept as fallback). HTTP for reads/allowances, WSS for the feed.
    ALCHEMY_API_KEY: str = os.getenv("ALCHEMY_API_KEY", "")

    CHAINLINK_AGGREGATORS: Dict[str, str] = {
        "BTC": "0xc907E116054Ad103354f2D350FD2514433D57F6f",
        "ETH": "0xF9680D99D6C9589e2a93a78A04A279e509205945",
        "SOL": "0x39771505D18301D239916F4C88367A6010F7D2e3",
        "XRP": "0x3454796324D6469C3110996E2E10972688045F19",
        "DOGE": "0xbAf93Ba318f77363f82E8896a2E830206121D506",
        "BNB": "0x82a6C67606bdc0409f959f60608226064223A57c"
    }

    def get_aggregator(self, symbol: str) -> str:
        s = symbol.upper()
        if s.endswith("USDT"): s = s[:-4]
        return self.CHAINLINK_AGGREGATORS.get(s, self.CHAINLINK_BTC_USD_AGGREGATOR)

    def alchemy_rpc_url(self) -> str:
        # Polygon RPC used ONLY for the on-chain live-trading path (allowance approvals).
        return f"https://polygon-mainnet.g.alchemy.com/v2/{self.ALCHEMY_API_KEY}" if self.ALCHEMY_API_KEY else ""

    # Proxy
    HTTP_PROXY: str = os.getenv("HTTP_PROXY", os.getenv("http_proxy", ""))
    HTTPS_PROXY: str = os.getenv("HTTPS_PROXY", os.getenv("https_proxy", ""))
    ALL_PROXY: str = os.getenv("ALL_PROXY", os.getenv("all_proxy", ""))

def normalize_private_key(secret: str) -> str:
    """Accept either a raw hex private key or a 12/24-word seed phrase and return a
    hex private key. EOA only — the wallet is derived from the secret, nothing else.
    Returns "" for empty input. Raises if a seed phrase can't be parsed."""
    secret = (secret or "").strip()
    if not secret:
        return ""
    # A mnemonic is several space-separated words; a private key is a single token.
    if len(secret.split()) >= 12:
        from eth_account import Account
        Account.enable_unaudited_hdwallet_features()
        return Account.from_mnemonic(secret).key.hex()
    return secret


def load_settings():
    base_settings = Settings()
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_data = json.load(f)

            if "mode" in config_data: base_settings.MODE = config_data["mode"]
            if "paper_balance_usd" in config_data: base_settings.PAPER_BALANCE_USD = config_data["paper_balance_usd"]
            if "private_key" in config_data: base_settings.PRIVATE_KEY = normalize_private_key(config_data["private_key"])

            if "relayer" in config_data:
                rl = config_data["relayer"]
                if "api_key" in rl: base_settings.RELAYER_API_KEY = rl["api_key"]

            if "polymarket" in config_data:
                poly = config_data["polymarket"]
                if "gamma_base_url" in poly: base_settings.GAMMA_BASE_URL = poly["gamma_base_url"]
                if "clob_base_url" in poly: base_settings.CLOB_BASE_URL = poly["clob_base_url"]
                if "live_ws_url" in poly: base_settings.POLYMARKET_LIVE_DATA_WS_URL = poly["live_ws_url"]
                if "series_id" in poly: base_settings.POLYMARKET_SERIES_ID = poly["series_id"]
                if "series_slug" in poly: base_settings.POLYMARKET_SERIES_SLUG = poly["series_slug"]
                if "auto_select_latest" in poly: base_settings.POLYMARKET_AUTO_SELECT_LATEST = poly["auto_select_latest"]
                if "up_label" in poly: base_settings.POLYMARKET_UP_LABEL = poly["up_label"]
                if "down_label" in poly: base_settings.POLYMARKET_DOWN_LABEL = poly["down_label"]

            if "trading" in config_data:
                trading = config_data["trading"]
                if "symbol" in trading: base_settings.SYMBOL = trading["symbol"]
                if "binance_base_url" in trading: base_settings.BINANCE_BASE_URL = trading["binance_base_url"]
                if "candle_window_minutes" in trading: base_settings.CANDLE_WINDOW_MINUTES = trading["candle_window_minutes"]
                if "poll_interval_ms" in trading: base_settings.POLL_INTERVAL_MS = trading["poll_interval_ms"]
                if "risk_type" in trading: base_settings.RISK_TYPE = trading["risk_type"]
                if "risk_value" in trading: base_settings.RISK_VALUE = trading["risk_value"]

            if "entry" in config_data:
                en = config_data["entry"]
                if "max_price" in en: base_settings.MAX_ENTRY_PRICE = float(en["max_price"])
                if "min_book_liquidity_usd" in en: base_settings.MIN_BOOK_LIQUIDITY_USD = float(en["min_book_liquidity_usd"])
                if "min_entry_minute" in en: base_settings.MIN_ENTRY_ELAPSED_MIN = float(en["min_entry_minute"])
                if "min_lead_bps" in en: base_settings.MIN_LEAD_BPS = float(en["min_lead_bps"])

            if "strategy" in config_data:
                st = config_data["strategy"]
                if "mode" in st: base_settings.STRATEGY_MODE = st["mode"]
                if "model_min_conf" in st: base_settings.MODEL_MIN_CONF = float(st["model_min_conf"])
                if "model_ev_margin" in st: base_settings.MODEL_EV_MARGIN = float(st["model_ev_margin"])
                if "model_kelly_fraction" in st: base_settings.MODEL_KELLY_FRACTION = float(st["model_kelly_fraction"])
                if "exit_mode" in st: base_settings.EXIT_MODE = st["exit_mode"]
                if "exit_take_profit_margin" in st: base_settings.EXIT_TAKE_PROFIT_MARGIN = float(st["exit_take_profit_margin"])
                if "exit_stop_prob" in st: base_settings.EXIT_STOP_PROB = float(st["exit_stop_prob"])

            if "close_on_reversal" in config_data:
                cor = config_data["close_on_reversal"]
                if "enabled" in cor: base_settings.CLOSE_ON_REVERSAL_ENABLED = bool(cor["enabled"])
                if "bars" in cor: base_settings.CLOSE_REVERSAL_BARS = int(cor["bars"])

            if "take_profit" in config_data:
                tp = config_data["take_profit"]
                if "enabled" in tp: base_settings.TAKE_PROFIT_ENABLED = bool(tp["enabled"])
                if "percent" in tp: base_settings.TAKE_PROFIT_PERCENT = float(tp["percent"])

            if "stop_loss" in config_data:
                sl = config_data["stop_loss"]
                if "enabled" in sl: base_settings.STOP_LOSS_ENABLED = bool(sl["enabled"])
                if "percent" in sl: base_settings.STOP_LOSS_PERCENT = float(sl["percent"])

            if "chainlink" in config_data:
                cl = config_data["chainlink"]
                if "polygon_rpc_url" in cl: base_settings.POLYGON_RPC_URL = cl["polygon_rpc_url"]
                if "polygon_wss_url" in cl: base_settings.POLYGON_WSS_URL = cl["polygon_wss_url"]
                if "btc_usd_aggregator" in cl: base_settings.CHAINLINK_BTC_USD_AGGREGATOR = cl["btc_usd_aggregator"]
                if "alchemy_api_key" in cl: base_settings.ALCHEMY_API_KEY = cl["alchemy_api_key"]

        except Exception as e:
            print(f"Warning: Failed to load config.json: {e}")

    return base_settings

settings = load_settings()
