# Polymarket BTC 15m Assistant (Python FastAPI)

A real-time trading assistant for Polymarket **"Bitcoin Up or Down" 15-minute** markets, ported to Python and FastAPI.

It runs a **1-minute trend** strategy (no 5m): the **1m Heiken-Ashi** sets the direction,
and the **1m Awesome Oscillator** and **RSI(14)** confirm it. The only price gate is a
**max-price cap** — buy only when the side's Polymarket odds are below a configurable
limit (default 0.60). All gates must agree. Position size is a simple percent-of-balance
or fixed-dollar risk. See [`strategy.md`](strategy.md) for the full rationale.

## Features

- Real-time Web Dashboard (FastAPI + Jinja2 + Alpine.js)
- Entry engine (1m only): 1m HA direction + 1m Awesome Oscillator + RSI(50) + max-price cap (all mandatory)
- Close on 1m reversal (exit only — sells out when the 1m HA + 1m AO both flip against the position)
- Self-contained settlement: Polymarket authoritative result, with a Chainlink window open/close fallback
- Trade Execution: Paper Trading simulation vs Live Mode toggle
- Data Sources: Binance, Polymarket (Gamma/CLOB), Chainlink (WebSocket + RPC)
- Proxy Support: Global HTTP/HTTPS/SOCKS proxy configuration

## Requirements

- Python **3.12+** (required by `polymarket-apis`)
- pip (comes with Python)

## Local Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure `config.json`

Set your trading mode, risk preferences, and optional private key in `config.json`.
Position size is set by `trading.risk_type` (`"percent"` = `risk_value`% of balance,
or `"fixed"` = `risk_value` dollars) and `trading.risk_value`.

Direction is decided by the **1m HA + 1m Awesome Oscillator (bar colour: rising=green) +
RSI(14) at the 50 line** (all 1-minute, fixed in code). The only price gate is the
`entry` block:

```jsonc
"entry": {
  "max_price": 0.60,             // only buy when the side's Polymarket odds (ask) are BELOW this
  "min_book_liquidity_usd": 20.0 // skip if the ask side can't absorb the stake
}
```

Close-on-1m-reversal (exit only) is toggled with `close_on_reversal.enabled`. The `entry`
and `close_on_reversal` settings are also editable live on the **Settings** page.

### 3) Run

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Access the dashboard at `http://localhost:8000`.

## Docker

```bash
docker build -t polymarket-assistant .
docker run -p 8000:8000 polymarket-assistant
```

## Deployment on Render

If you are seeing errors related to Node.js or `npm run start`, it is because Render is auto-detecting the old environment. **You must manually set the runtime to Python.**

### Recommended: Use `render.yaml`
The repository includes a `render.yaml`. When creating a new blueprint on Render, it will automatically set the correct environment.

### Manual Setup
1. Create a **Web Service** on Render.
2. Under **Runtime**, explicitly select **Python 3**.
3. Set the following commands:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 8000`
4. Add any necessary environment variables (optional).

## Live Trading

Switching **Mode** to `live` (config or the Settings page) makes the bot place real
**Fill-Or-Kill market BUY** orders on the Polymarket **CLOB V2** (via `polymarket-apis`).
See [`SETUP.md`](SETUP.md) for the full one-time walkthrough. In short:

1. Set a **private key or 12/24-word seed phrase** (Settings → Credentials, or
   `config.json`). It only **signs** orders — it holds no funds or gas; funds live in
   your Polymarket **deposit wallet**.
2. Set a **Relayer API key** (polymarket.com → Settings → Relayer API keys). It sponsors
   the one-time on-chain setup (deposit-wallet deploy + token approvals) **gaslessly** —
   you never pay gas.
3. **Deposit USDC into your Polymarket account** and complete the website's "deposit
   wallet" migration (place one manual trade) so the API is allowed to trade.
4. Click **Test** on the Settings page to confirm the wallet is **Active ✓** and shows
   your balance. The gasless approvals run automatically before your first live order.

Orders are placed as **slippage-capped marketable Fill-Or-Kill** orders: the limit
price is the current market quote plus a small buffer (`CLOB_MAX_SLIPPAGE`, default
2¢), so if the book moves away the order is killed rather than filled at a bad
price. In live mode the dashboard balance reflects the real **deposit-wallet balance**
(refreshed periodically). Order failures are reported in the Console Log.

## Machine Learning (shadow mode)

The `bot/ml/` package adds a calibrated win-probability model that runs **in shadow**:
it is logged and displayed (dashboard "ML P(up) · shadow") but makes **no trading
decisions**. Promotion into the decision path requires it to beat the market's ask
on calibration over 3-4 weeks of recorded live odds.

- `python -m bot.ml.data [days]` — build the offline dataset from Binance 1m history
  (resumable; cached under `models/`)
- `python -m bot.ml.train` — walk-forward train/evaluate (logistic + monotonic
  gradient boosting vs the closed-form GBM baseline), then save `models/model.joblib`
- `python -m bot.ml.backtest_gates` — the gate-stack win-rate backtest
- `bot/recorder.py` — logs features + live Polymarket bid/ask + shadow P(up) every
  tick to `logs/ml/` (this accumulating dataset is the only way to ever validate a
  model against the market's price — keep the bot running to collect it)

## Safety

This is not financial advice. Use at your own risk; live mode trades real funds.

**The edge is unproven.** The HA/AO/RSI gates set and confirm the trend but do not, on
their own, beat the trivial "is spot above the open" baseline, and the max-price cap only
limits what you pay — it is not itself an edge. Run in **paper mode** and watch the trades
settle *before* risking capital.
