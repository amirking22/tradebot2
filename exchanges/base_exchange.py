"""Base class for exchange interfaces."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import pandas as pd
import aiohttp
import asyncio

class BaseExchange(ABC):
    """Abstract base class for exchange interfaces."""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.name = "BaseExchange"
    
    @abstractmethod
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get all available trading pairs."""
        pass
    
    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str = '30m', limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data for a symbol."""
        pass
    
    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for a symbol."""
        pass
    
    @abstractmethod
    async def get_open_interest(self, symbol: str) -> Dict[str, float]:
        """Get open interest data for a symbol."""
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Get order book for a symbol."""
        pass
    
    async def get_balance(self) -> Dict[str, float]:
        """Get account balance (if authenticated)."""
        return {}
    
    async def place_order(self, symbol: str, side: str, order_type: str, **kwargs) -> Dict[str, Any]:
        """Place an order (if authenticated)."""
        raise NotImplementedError("Order placement not implemented")

class YEXExchange(BaseExchange):
    """YEX Futures exchange implementation (uses REST endpoints provided by the user)."""

    BASE_URL = "https://api.yex.com"

    def __init__(self, api_key: str = "", api_secret: str = ""):
        super().__init__(api_key, api_secret)
        self.name = "YEX Futures"

    async def _get_json(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Helper to perform an async HTTP GET and return JSON."""
        url = f"{self.BASE_URL}{endpoint}"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                resp.raise_for_status()
                return await resp.json()

    # --------------------------- public API --------------------------- #
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Return list of available futures markets."""
        return await self._get_json("/futures/markets")

    async def get_ohlcv(self, symbol: str, timeframe: str = "30m", limit: int = 200) -> pd.DataFrame:
        """Return OHLCV dataframe for the requested symbol/timeframe."""
        raw = await self._get_json(
            "/futures/klines",
            params={"symbol": symbol, "interval": timeframe, "limit": limit},
        )
        # raw is expected to be list of [ts, open, high, low, close, volume]
        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        df = pd.DataFrame(raw, columns=columns)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        return df

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Return ticker information for symbol (price, change, high/low, volume)."""
        return await self._get_json("/futures/ticker", params={"symbol": symbol})

    async def get_funding_rate(self, symbol: str) -> float:
        # YEX endpoint not provided; return 0 for now.
        return 0.0

    async def get_open_interest(self, symbol: str) -> Dict[str, float]:
        # YEX endpoint not provided
        return {}

    async def get_order_book(self, symbol: str) -> Dict[str, Any]:
        # YEX endpoint not provided
        return {}
    """YEX Exchange implementation."""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        super().__init__(api_key, api_secret)
        self.name = "YEX"
        self.base_url = "https://api.yex.com"
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get all available trading pairs on YEX."""
        # Implementation for YEX API
        return []
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '30m', limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data from YEX."""
        # Implementation for YEX API
        return pd.DataFrame()
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for a symbol."""
        # Implementation for YEX
        return 0.0
    
    async def get_open_interest(self, symbol: str) -> Dict[str, float]:
        """Get open interest data for a symbol."""
        # Implementation for YEX
        return {}
    
    async def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Get order book for a symbol."""
        # Implementation for YEX
        return {}

class BinanceFuturesExchange(BaseExchange):
    """Binance Futures implementation."""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        super().__init__(api_key, api_secret)
        self.name = "Binance Futures"
        self.base_url = "https://fapi.binance.com"
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get all available trading pairs on Binance Futures."""
        # Implementation for Binance Futures API
        return []
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '30m', limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data from Binance Futures."""
        # Implementation for Binance Futures API
        return pd.DataFrame()
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for a symbol."""
        # Implementation for Binance Futures
        return 0.0
    
    async def get_open_interest(self, symbol: str) -> Dict[str, float]:
        """Get open interest data for a symbol."""
        # Implementation for Binance Futures
        return {}
    
    async def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Get order book for a symbol."""
        # Implementation for Binance Futures
        return {}

class BybitFuturesExchange(BaseExchange):
    """Bybit Futures implementation."""
    
    def __init__(self, api_key: str = "", api_secret: str = ""):
        super().__init__(api_key, api_secret)
        self.name = "Bybit Futures"
        self.base_url = "https://api.bybit.com"
    
    async def get_markets(self) -> List[Dict[str, Any]]:
        """Get all available trading pairs on Bybit Futures."""
        # Implementation for Bybit Futures API
        return []
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '30m', limit: int = 100) -> pd.DataFrame:
        """Get OHLCV data from Bybit Futures."""
        # Implementation for Bybit Futures API
        return pd.DataFrame()
    
    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate for a symbol."""
        # Implementation for Bybit Futures
        return 0.0
    
    async def get_open_interest(self, symbol: str) -> Dict[str, float]:
        """Get open interest data for a symbol."""
        # Implementation for Bybit Futures
        return {}
    
    async def get_order_book(self, symbol: str) -> Dict[str, Any]:
        """Get order book for a symbol."""
        # Implementation for Bybit Futures
        return {}
