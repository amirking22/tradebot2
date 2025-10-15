"""
Multi-exchange data aggregator with price validation.
"""
import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MultiExchangeAggregator:
    """Aggregates data from multiple exchanges for price validation."""
    
    def __init__(self):
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created lazily."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def get_binance_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker from Binance."""
        try:
            # Make sure session exists
            await self._ensure_session()
            # Convert symbol format (e.g., BTCUSDT -> BTCUSDT)
            binance_symbol = symbol.replace('/', '').upper()
            url = f"https://fapi.binance.com/fapi/v1/ticker/24hr?symbol={binance_symbol}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'exchange': 'binance',
                        'symbol': symbol,
                        'price': float(data['lastPrice']),
                        'change_24h': float(data['priceChangePercent']),
                        'volume': float(data['volume']),
                        'high_24h': float(data['highPrice']),
                        'low_24h': float(data['lowPrice'])
                    }
        except Exception as e:
            logger.error(f"Error getting Binance ticker for {symbol}: {e}")
        return None
    
    async def get_bybit_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker from Bybit."""
        try:
            # Convert symbol format
            bybit_symbol = symbol.replace('/', '').upper()
            url = f"https://api.bybit.com/v5/market/tickers?category=linear&symbol={bybit_symbol}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['retCode'] == 0 and data['result']['list']:
                        ticker = data['result']['list'][0]
                        return {
                            'exchange': 'bybit',
                            'symbol': symbol,
                            'price': float(ticker['lastPrice']),
                            'change_24h': float(ticker['price24hPcnt']) * 100,
                            'volume': float(ticker['volume24h']),
                            'high_24h': float(ticker['highPrice24h']),
                            'low_24h': float(ticker['lowPrice24h'])
                        }
        except Exception as e:
            logger.error(f"Error getting Bybit ticker for {symbol}: {e}")
        return None
    
    async def get_okx_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker from OKX."""
        try:
            # Convert symbol format (e.g., BTC/USDT -> BTC-USDT-SWAP)
            okx_symbol = symbol.replace('/', '-') + '-SWAP'
            url = f"https://www.okx.com/api/v5/market/ticker?instId={okx_symbol}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['code'] == '0' and data['data']:
                        ticker = data['data'][0]
                        return {
                            'exchange': 'okx',
                            'symbol': symbol,
                            'price': float(ticker['last']),
                            'change_24h': float(ticker['sodUtc8']) * 100,
                            'volume': float(ticker['vol24h']),
                            'high_24h': float(ticker['high24h']),
                            'low_24h': float(ticker['low24h'])
                        }
        except Exception as e:
            logger.error(f"Error getting OKX ticker for {symbol}: {e}")
        return None
    
    async def get_hyperliquid_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker from HyperLiquid."""
        try:
            url = "https://api.hyperliquid.xyz/info"
            payload = {"type": "allMids"}
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    # HyperLiquid uses different symbol format
                    hl_symbol = symbol.replace('/USDT', '').replace('USDT', '')
                    
                    if hl_symbol in data:
                        price = float(data[hl_symbol])
                        
                        # Get 24h stats
                        stats_payload = {"type": "meta"}
                        async with self.session.post(url, json=stats_payload) as stats_response:
                            if stats_response.status == 200:
                                stats_data = await stats_response.json()
                                for asset in stats_data.get('universe', []):
                                    if asset['name'] == hl_symbol:
                                        return {
                                            'exchange': 'hyperliquid',
                                            'symbol': symbol,
                                            'price': price,
                                            'change_24h': 0.0,  # HyperLiquid doesn't provide 24h change in this endpoint
                                            'volume': 0.0,
                                            'high_24h': price,
                                            'low_24h': price
                                        }
        except Exception as e:
            logger.error(f"Error getting HyperLiquid ticker for {symbol}: {e}")
        return None
    
    async def get_yex_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker from YEX."""
        try:
            url = f"https://api.yex.io/futures/ticker?symbol={symbol}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'exchange': 'yex',
                        'symbol': symbol,
                        'price': float(data.get('price', 0)),
                        'change_24h': float(data.get('priceChangePercent', 0)),
                        'volume': float(data.get('volume', 0)),
                        'high_24h': float(data.get('high', 0)),
                        'low_24h': float(data.get('low', 0))
                    }
        except Exception as e:
            logger.error(f"Error getting YEX ticker for {symbol}: {e}")
        return None
    
    async def get_aggregated_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get ticker data from multiple exchanges and validate."""
        tasks = [
            self.get_binance_ticker(symbol),
            self.get_bybit_ticker(symbol),
            self.get_okx_ticker(symbol),
            self.get_hyperliquid_ticker(symbol),
            self.get_yex_ticker(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        valid_tickers = [r for r in results if r and not isinstance(r, Exception)]
        
        if not valid_tickers:
            return {'error': 'No valid data from any exchange'}
        
        # Calculate average price and validate consistency
        prices = [t['price'] for t in valid_tickers if t['price'] > 0]
        if not prices:
            return {'error': 'No valid prices found'}
        
        avg_price = sum(prices) / len(prices)
        price_deviation = max(abs(p - avg_price) / avg_price * 100 for p in prices)
        
        # Find most reliable ticker (closest to average)
        best_ticker = min(valid_tickers, key=lambda t: abs(t['price'] - avg_price) if t['price'] > 0 else float('inf'))
        
        return {
            'symbol': symbol,
            'price': best_ticker['price'],
            'avg_price': avg_price,
            'price_deviation': price_deviation,
            'change_24h': best_ticker['change_24h'],
            'volume': best_ticker['volume'],
            'high_24h': best_ticker['high_24h'],
            'low_24h': best_ticker['low_24h'],
            'source_exchange': best_ticker['exchange'],
            'total_sources': len(valid_tickers),
            'all_tickers': valid_tickers,
            'reliability': 'high' if price_deviation < 1.0 else 'medium' if price_deviation < 3.0 else 'low'
        }
    
    async def get_top_markets(self, limit: int = 20) -> List[str]:
        """Get top crypto markets by volume from Binance."""
        try:
            url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    # Filter USDT pairs and sort by volume
                    usdt_pairs = [item for item in data if item['symbol'].endswith('USDT')]
                    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
                    
                    # Convert to standard format
                    symbols = []
                    for pair in sorted_pairs[:limit]:
                        symbol = pair['symbol']
                        formatted = f"{symbol[:-4]}/USDT"  # Remove USDT and add /
                        symbols.append(formatted)
                    
                    return symbols
        except Exception as e:
            logger.error(f"Error getting top markets: {e}")
        
        # Fallback list
        return [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'LTC/USDT',
            'AVAX/USDT', 'UNI/USDT', 'LINK/USDT', 'ATOM/USDT', 'FTM/USDT',
            'NEAR/USDT', 'ALGO/USDT', 'VET/USDT', 'ICP/USDT', 'SAND/USDT'
        ]
    
    async def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Get ticker data for a symbol (compatible with existing code)."""
        try:
            result = await self.get_aggregated_ticker(symbol)
            if 'error' in result:
                return None
            
            # Convert to format expected by telegram bot
            return {
                'last': result['price'],
                'percentage': result['change_24h'],
                'quoteVolume': result['volume'],
                'high': result['high_24h'],
                'low': result['low_24h']
            }
        except Exception as e:
            logger.error(f"Error in get_ticker for {symbol}: {e}")
            return None
    
    async def get_klines(self, symbol: str, interval: str = '5m', limit: int = 200) -> Optional[List]:
        """Get klines/candlestick data from Binance (primary source)."""
        try:
            # Make sure session exists
            await self._ensure_session()
            # Convert symbol format for Binance
            binance_symbol = symbol.replace('/', '').upper()
            url = f"https://fapi.binance.com/fapi/v1/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit
            }
            
            session = await self._ensure_session()
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    text = await response.text()
                    logger.error(f"Binance klines API returned status {response.status}: {text[:200]}")
                    return None
                    
        except Exception as e:
            logger.error(f"Exception getting klines for {symbol}: {e}", exc_info=True)
            return None
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 200) -> Optional[List]:
        """Get OHLCV data - alias for get_klines for compatibility."""
        return await self.get_klines(symbol, timeframe, limit)
