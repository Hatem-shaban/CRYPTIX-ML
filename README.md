CRYPTIX-ML — Live Binance Setup Guide

Overview
This bot trades on Binance Spot using technical indicators and an adaptive strategy. It supports both Testnet and Live. Follow these steps to run on Live safely.

Environment variables (Render or local)
- API_KEY / API_SECRET (preferred) or BINANCE_API_KEY / BINANCE_API_SECRET: Your Binance Spot API credentials (with trading enabled and IP restrictions configured).
- USE_TESTNET: Set to false (or remove) for Live. You can also set BINANCE_TESTNET=false.
- Optional safety flag: AUTO_TRADING is controlled in config.py (defaults to True). Set to False for dry-runs.
- Telegram (optional): TELEGRAM settings are configured in config.py. You can disable notifications via TELEGRAM.enabled or TELEGRAM_SEND_SIGNALS.

Live vs Testnet
- Live: USE_TESTNET=false and do not set BINANCE_TESTNET, or set BINANCE_TESTNET=false.
- Testnet: USE_TESTNET=true or BINANCE_TESTNET=true. The client automatically uses https://testnet.binance.vision/api.

Risk and minimums
- The bot enforces a $10 minimum notional per trade.
- Position sizing uses RISK_PERCENTAGE in config.py (default 2%). Ensure your USDT balance is sufficient.

Quick checklist for Live
1) Set API_KEY and API_SECRET (or BINANCE_API_KEY/BINANCE_API_SECRET) to Live keys.
2) Ensure USE_TESTNET is false (default) or set BINANCE_TESTNET=false.
3) Fund your Spot wallet with USDT (> $10).
4) Review RISK_PERCENTAGE and AUTO_TRADING in config.py.
5) Deploy/run the app; watch logs for "Initializing Binance client for LIVE trading…" and account capability checks.

Notes
- The client initialization logs the detected base URL and whether it’s TESTNET or LIVE.
- Exchange info and Coinbase market data are cached to reduce API usage.
- Signals are logged with reduced HOLD noise; only BUY/SELL signals trigger optional Telegram alerts if enabled.

