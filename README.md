# Telegram Crypto Signal Bot

A Telegram bot that connects to Binance Futures, Bybit Futures, and OKX (futures) to generate trading signals for the following pairs:

```
BTC/USDT, AAVE/USDT, XRP/USDT, POL/USDT, ADA/USDT, BNB/USDT, XLM/USDT, ETH/USDT, TON/USDT, LTC/USDT, ARB/USDT, TRX/USDT, DOGE/USDT, SOL/USDT, BCH/USDT, SUI/USDT
```

Signals use:

* EMA9 / EMA21 cross with EMA200 filter
* RSI, ATR, ADX confirmation
* Categorised as **Safe / Medium / Risky**
* Both **Long** and **Short** scenarios, with 5-step entries & exits.

The bot refreshes market data every **5 minutes** and evaluates strategies every **30 minutes**. Messages are sent only to the configured Telegram chat ID.

---

## Quick Start

1. **Clone & install deps**

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Create `.env`** based on `.env.example` and fill in:

```
TELEGRAM_TOKEN=8122351361:AAHQMwz1Ja3ljSYclVrW-4EEOZGj9hGEmZ8
TELEGRAM_CHAT_ID=37292924
BINANCE_API_KEY=...
BINANCE_SECRET=...
BYBIT_API_KEY=...
BYBIT_SECRET=...
OKX_API_KEY=...
OKX_SECRET=...
OKX_PASSPHRASE=...
```

3. **Run**

```bash
python main.py
```

---

## Project Structure

```
.
├── bot.py            # Telegram bot utilities
├── cache.py          # In-memory cache for candles
├── config.py         # Loads environment variables
├── exchanges.py      # Unified wrappers around ccxt exchanges
├── indicators.py     # Technical indicator helpers (EMA, RSI, ATR, ADX)
├── strategy.py       # Trade signal generation
├── scheduler.py      # APScheduler jobs (fetch, evaluate)
├── main.py           # Entry point
└── requirements.txt
```

---

## TODO

* Complete strategy risk classification & position sizing
* Persist cache to disk (optional)
* Unit tests
