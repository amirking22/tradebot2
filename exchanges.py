"""Unified exchange clients for Binance, Bybit, and OKX."""
from __future__ import annotations
import ccxt
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
import pandas as pd
from datetime import datetime, timezone
import time

from config import (
    BINANCE_API_KEY, BINANCE_SECRET_KEY,
    BYBIT_API_KEY, BYBIT_SECRET,
    SYMBOLS
)

ExchangeName = Literal['binance', 'bybit', 'okx']
Timeframe = Literal['1m', '5m', '15m', '30m', '1h', '4h', '1d']

@dataclass
class OHLCV:
    """Standardized OHLCV data container."""
    symbol: str
    timeframe: Timeframe
    timestamp: pd.DatetimeIndex
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series
    exchange: ExchangeName
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }, index=self.timestamp)


def create_exchange(exchange_name: ExchangeName) -> ccxt.Exchange:
    """Initialize and return a configured exchange client."""
    config = {
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},  # For futures
    }
    
    if exchange_name == 'binance':
        config.update({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET,
        })
    elif exchange_name == 'bybit':
        config.update({
            'apiKey': BYBIT_API_KEY,
            'secret': BYBIT_SECRET,
        })
    elif exchange_name == 'okx':
        config.update({
            'apiKey': OKX_API_KEY,
            'secret': OKX_SECRET,
            'password': OKX_PASSPHRASE,
        })
    else:
        raise ValueError(f"Unsupported exchange: {exchange_name}")
    
    exchange_class = getattr(ccxt, exchange_name)
    return exchange_class(config)


def fetch_ohlcv(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: Timeframe = '30m',
    limit: int = 200
) -> OHLCV:
    """Fetch OHLCV data from exchange and standardize it."""
    # Convert symbol to exchange's format (e.g., BTC/USDT -> BTCUSDT for Binance)
    exchange_symbol = symbol.replace('/', '')
    
    # Fetch data
    ohlcv = exchange.fetch_ohlcv(exchange_symbol, timeframe, limit=limit)
    
    # Convert to DataFrame
    df = pd.DataFrame(
        ohlcv,
        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return OHLCV(
        symbol=symbol,
        timeframe=timeframe,
        timestamp=df['timestamp'],
        open=df['open'].astype(float),
        high=df['high'].astype(float),
        low=df['low'].astype(float),
        close=df['close'].astype(float),
        volume=df['volume'].astype(float),
        exchange=exchange.id
    )


def get_markets(exchange: ccxt.Exchange) -> pd.DataFrame:
    """Get available markets and their details."""
    markets = exchange.load_markets()
    return pd.DataFrame(markets).T


def test_connection(exchange: ccxt.Exchange) -> bool:
    """Test if the exchange connection works."""
    try:
        exchange.fetch_balance()
        return True
    except Exception as e:
        print(f"Connection test failed for {exchange.id}: {str(e)}")
        return False


if __name__ == "__main__":
    # Test connection to all configured exchanges
    for name in ['binance', 'bybit', 'okx']:
        try:
            ex = create_exchange(name)
            if test_connection(ex):
                print(f"✅ {name.upper()} connection successful")
                print(f"   Rate limit: {ex.rateLimit}ms")
            else:
                print(f"❌ {name.upper()} connection failed")
        except Exception as e:
            print(f"❌ {name.upper()} error: {e}")
        print()
