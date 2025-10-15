"""Fetch OHLCV data from multiple exchanges via REST API."""
from __future__ import annotations
import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
import json

# Timeframe mapping for different exchanges
TIMEFRAME_MAP = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}

@dataclass
class OHLCV:
    """Standardized OHLCV data container."""
    exchange: str
    symbol: str
    timeframe: str
    timestamp: pd.DatetimeIndex
    open: pd.Series
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame({
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }, index=self.timestamp)


class MarketDataFetcher:
    """Fetch OHLCV data from multiple exchanges."""
    
    def __init__(self):
        self.session = aiohttp.ClientSession()
        self.base_urls = {
            'binance': 'https://api.binance.com/api/v3/klines',
            'kucoin': 'https://api.kucoin.com/api/v1/market/candles',
            'phemex': 'https://api.phemex.com/md/kline',
            'mexc': 'https://www.mexc.com/open/api/v2/market/kline',
            'gateio': 'https://api.gateio.ws/api2/1/candlestick'
        }
    
    async def close(self):
        """Close the HTTP session."""
        await self.session.close()
    
    async def fetch_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 100
    ) -> Optional[OHLCV]:
        """Fetch OHLCV data from the specified exchange."""
        try:
            if exchange == 'binance':
                return await self._fetch_binance(symbol, timeframe, limit)
            elif exchange == 'kucoin':
                return await self._fetch_kucoin(symbol, timeframe, limit)
            elif exchange == 'phemex':
                return await self._fetch_phemex(symbol, timeframe, limit)
            elif exchange == 'mexc':
                return await self._fetch_mexc(symbol, timeframe, limit)
            elif exchange == 'gateio':
                return await self._fetch_gateio(symbol, timeframe, limit)
            else:
                print(f"Unsupported exchange: {exchange}")
                return None
        except Exception as e:
            print(f"Error fetching {exchange} {symbol} {timeframe}: {str(e)}")
            return None
    
    async def _fetch_binance(self, symbol: str, timeframe: str, limit: int) -> OHLCV:
        """Fetch OHLCV data from Binance."""
        params = {
            'symbol': symbol.replace('/', '').upper(),
            'interval': timeframe,
            'limit': limit
        }
        
        async with self.session.get(self.base_urls['binance'], params=params) as resp:
            data = await resp.json()
            
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        return self._create_ohlcv('binance', symbol, timeframe, df)
    
    async def _fetch_kucoin(self, symbol: str, timeframe: str, limit: int) -> OHLCV:
        """Fetch OHLCV data from KuCoin."""
        params = {
            'symbol': symbol.replace('/', '-').upper(),
            'type': timeframe,
            'limit': limit
        }
        
        async with self.session.get(self.base_urls['kucoin'], params=params) as resp:
            data = await resp.json()
            
        if data.get('code') != '200000':
            raise Exception(f"KuCoin API error: {data.get('msg', 'Unknown error')}")
            
        df = pd.DataFrame(data['data'], columns=[
            'timestamp', 'open', 'close', 'high', 'low', 'volume', 'amount'
        ])
        
        return self._create_ohlcv('kucoin', symbol, timeframe, df)
    
    async def _fetch_phemex(self, symbol: str, timeframe: str, limit: int) -> OHLCV:
        """Fetch OHLCV data from Phemex."""
        resolution_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        
        params = {
            'symbol': symbol.replace('/', '').upper(),
            'resolution': resolution_map.get(timeframe, 3600),
            'limit': limit
        }
        
        async with self.session.get(self.base_urls['phemex'], params=params) as resp:
            data = await resp.json()
            
        if data.get('result') != 0:
            raise Exception(f"Phemex API error: {data.get('msg', 'Unknown error')}")
            
        df = pd.DataFrame(data['data'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        
        return self._create_ohlcv('phemex', symbol, timeframe, df)
    
    async def _fetch_mexc(self, symbol: str, timeframe: str, limit: int) -> OHLCV:
        """Fetch OHLCV data from MEXC."""
        params = {
            'symbol': symbol.replace('/', '_').lower(),
            'interval': timeframe,
            'limit': limit
        }
        
        async with self.session.get(self.base_urls['mexc'], params=params) as resp:
            data = await resp.json()
            
        if data.get('code') != 200:
            raise Exception(f"MEXC API error: {data.get('msg', 'Unknown error')}")
            
        df = pd.DataFrame(data['data'], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount'
        ])
        
        return self._create_ohlcv('mexc', symbol, timeframe, df)
    
    async def _fetch_gateio(self, symbol: str, timeframe: str, limit: int) -> OHLCV:
        """Fetch OHLCV data from Gate.io."""
        timeframe_map = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }
        
        url = f"{self.base_urls['gateio']}/{symbol.replace('/', '_').lower()}?group_sec={timeframe_map.get(timeframe, 3600)}"
        
        async with self.session.get(url) as resp:
            data = await resp.json()
            
        if not data or 'result' in data and data['result'] != 'true':
            raise Exception("Gate.io API error")
            
        # Take the last 'limit' candles
        data = data[-limit:] if len(data) > limit else data
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume'
        ])
        
        return self._create_ohlcv('gateio', symbol, timeframe, df)
    
    def _create_ohlcv(
        self,
        exchange: str,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame
    ) -> OHLCV:
        """Create an OHLCV object from a DataFrame."""
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
        
        # Convert numeric columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        return OHLCV(
            exchange=exchange,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        )


# Example usage
async def main():
    fetcher = MarketDataFetcher()
    
    # Example: Fetch BTC/USDT 1h data from multiple exchanges
    exchanges = ['binance', 'kucoin', 'phemex', 'mexc', 'gateio']
    symbol = 'BTC/USDT'
    timeframe = '1h'
    limit = 100
    
    tasks = [
        fetcher.fetch_ohlcv(exchange, symbol, timeframe, limit)
        for exchange in exchanges
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for exchange, result in zip(exchanges, results):
        if isinstance(result, Exception):
            print(f"{exchange}: Error - {result}")
        elif result is not None:
            print(f"{exchange}: Got {len(result.close)} candles, latest close: {result.close.iloc[-1]}")
    
    await fetcher.close()


if __name__ == "__main__":
    asyncio.run(main())
