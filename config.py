"""Configuration for Crypto Signal Bot."""
import os
from typing import List, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Bot Mode
TRADING_ENABLED = False  # Disable actual trading
SIGNAL_ONLY = True      # Only generate and send signals

# Telegram Bot Token
TELEGRAM_TOKEN: Optional[str] = os.getenv('TELEGRAM_TOKEN')

# List of allowed chat IDs (comma-separated)
ALLOWED_CHAT_IDS: List[int] = [37292924]

# Exchange API Keys (only for market data)
BINANCE_API_KEY: Optional[str] = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY: Optional[str] = os.getenv('BINANCE_SECRET_KEY')
YEX_API_KEY: Optional[str] = os.getenv('YEX_API_KEY')
YEX_SECRET_KEY: Optional[str] = os.getenv('YEX_SECRET_KEY')

# Signal parameters
RISK_REWARD_RATIO: float = 3.0  # 1:3 risk:reward ratio for signal calculation
DEFAULT_LEVERAGE: int = 5  # Default leverage (cross)
INITIAL_CAPITAL_USDT: float = 1000.0  # Base capital for position sizing
POSITION_SIZE_PERCENT: float = 10.0  # Virtual position size (percent of capital)
PROFIT_THRESHOLD_PERCENT: float = float(os.getenv('PROFIT_THRESHOLD_PERCENT', '1.0'))  # Warn if expected profit below this

# Technical Indicators Settings
RSI_PERIOD: int = 14
ADX_PERIOD: int = 14
ATR_PERIOD: int = 14

# Signal Generation
MIN_ADX_FOR_TREND: float = 25.0  # Minimum ADX value to consider a trend
RSI_OVERBOUGHT: float = 70.0
RSI_OVERSOLD: float = 30.0

# Bybit Futures
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY')
BYBIT_SECRET = os.getenv('BYBIT_SECRET')

# OKX Futures
OKX_API_KEY = os.getenv('OKX_API_KEY')
OKX_SECRET = os.getenv('OKX_SECRET')
OKX_PASSPHRASE = os.getenv('OKX_PASSPHRASE')

# YEX Exchange
YEX_API_KEY = os.getenv('YEX_API_KEY', '04f4fc4807382b859ced7b51b11d40d5')
YEX_SECRET_KEY = os.getenv('YEX_SECRET_KEY', '69a488b78c1cef9863fdf2eeff681278')
YEX_BASE_URL = os.getenv('YEX_BASE_URL', 'https://api.yex.io/v1')

# General
REFRESH_INTERVAL_MIN = int(os.getenv('REFRESH_INTERVAL_MIN', 5))  # in minutes
EVALUATE_INTERVAL_MIN = int(os.getenv('EVALUATE_INTERVAL_MIN', 30))  # in minutes
# Default trading pairs
SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
    'ADA/USDT', 'DOGE/USDT', 'TON/USDT', 'LINK/USDT', 'DOT/USDT',
    'MATIC/USDT', 'LTC/USDT', 'BCH/USDT', 'AVAX/USDT', 'XLM/USDT',
    'SUI/USDT',
    'ATOM/USDT', 'UNI/USDT', 'XMR/USDT', 'ETC/USDT', 'FIL/USDT'
]

# YEX specific symbols (will be populated from API)
YEX_FUTURES_SYMBOLS = []
class MissingConfig(Exception):
    """Raised when required configuration values are missing."""


async def ensure_config():
    """Ensure mandatory configs exist; raise MissingConfig otherwise."""
    missing = []
    if not TELEGRAM_TOKEN:
        missing.append('TELEGRAM_TOKEN')
    
    # Check YEX API credentials
    if not YEX_API_KEY or not YEX_SECRET_KEY:
        logger.warning(
            "YEX API credentials not found. Some features may be limited. "
            "Please set YEX_API_KEY and YEX_SECRET_KEY in your .env file."
        )
    if not TELEGRAM_CHAT_ID:
        missing.append('TELEGRAM_CHAT_ID')
    if missing:
        raise MissingConfig(f"Missing configuration values: {', '.join(missing)}")
