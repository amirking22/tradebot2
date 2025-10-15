"""YEX Exchange API wrapper."""
import hmac
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
import logging
import aiohttp
import pandas as pd
from urllib.parse import urlencode
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class YEXClient:
    """YEX Exchange API client."""
    
    BASE_URL = "https://api.yex.io/v1"
    
    def __init__(self, api_key: str, api_secret: str):
        """Initialize the YEX client."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = aiohttp.ClientSession()
    
    async def close(self):
        """Close the HTTP session."""
        await self.session.close()
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature."""
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    async def _request(self, method: str, endpoint: str, params: Optional[Dict] = None, signed: bool = False) -> Dict:
        """Send a request to the YEX API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            params: Request parameters
            signed: Whether to sign the request
            
        Returns:
            JSON response as dictionary
        """
        url = f"{self.BASE_URL}/{endpoint}"
        headers = {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        if params is None:
            params = {}
        
        # Add timestamp for signed requests
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._generate_signature(params)
        
        try:
            async with self.session.request(method, url, params=params, headers=headers) as response:
                data = await response.json()
                
                if response.status != 200:
                    error_msg = data.get('msg', 'Unknown error')
                    raise Exception(f"YEX API error: {error_msg}")
                
                return data
                
        except Exception as e:
            logger.error(f"YEX API request failed: {e}")
            raise
    
    # Public API endpoints
    
    # Market Data Endpoints
    
    async def get_exchange_info(self) -> Dict:
        """Get exchange trading rules and symbol information.
        
        Returns:
            Dict containing exchange information including all symbols
        """
        return await self._request('GET', 'exchangeInfo')
    
    async def get_futures_symbols(self) -> List[Dict]:
        """Get all available futures symbols.
        
        Returns:
            List of dictionaries containing futures symbol information
        """
        try:
            data = await self.get_exchange_info()
            # Filter for futures symbols (assuming they have 'futures' in the symbol name)
            futures_symbols = [
                s for s in data.get('symbols', [])
                if 'futures' in s.get('symbol', '').lower()
            ]
            return futures_symbols
        except Exception as e:
            logger.error(f"Error fetching futures symbols: {e}")
            return []
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = '1h',
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> List[List[Any]]:
        """Get kline/candlestick data.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            limit: Number of candles to return (max 1000)
            
        Returns:
            List of klines: [
                [open_time, open, high, low, close, volume, close_time, ...],
                ...
            ]
        """
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000)  # API max limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return await self._request('GET', 'klines', params)
    
    async def get_historical_klines(
        self,
        symbol: str,
        interval: str = '1h',
        start_str: str = '30 days ago UTC',
        end_str: str = 'now UTC',
        limit: int = 1000
    ) -> pd.DataFrame:
        """Get historical klines as a pandas DataFrame.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval
            start_str: Start date string (parsed by pandas)
            end_str: End date string (parsed by pandas)
            limit: Max number of candles to return
            
        Returns:
            DataFrame with OHLCV data and datetime index
        """
        # Convert start and end times to timestamps
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)
        
        # Initialize empty DataFrame
        all_klines = []
        
        # Fetch data in chunks to respect API limits
        current_dt = start_dt
        while current_dt < end_dt and len(all_klines) < limit:
            # Calculate end time for this chunk
            chunk_end_dt = min(
                current_dt + timedelta(days=30),  # 30 days at a time
                end_dt
            )
            
            # Convert to timestamps in milliseconds
            start_ts = int(current_dt.timestamp() * 1000)
            end_ts = int(chunk_end_dt.timestamp() * 1000)
            
            # Fetch klines for this chunk
            klines = await self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_ts,
                end_time=end_ts,
                limit=limit - len(all_klines)
            )
            
            if not klines:
                break
                
            all_klines.extend(klines)
            
            # Move to next chunk
            current_dt = chunk_end_dt
            
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        # Convert to DataFrame
        if not all_klines:
            return pd.DataFrame()
            
        df = pd.DataFrame(
            all_klines,
            columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ]
        )
        
        # Convert types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Set index
        df.set_index('open_time', inplace=True)
        
        return df
    
    async def get_ticker_24h(self, symbol: Optional[str] = None) -> Dict:
        """Get 24 hour price change statistics.
        
        Args:
            symbol: Trading pair (if None, returns all symbols)
            
        Returns:
            Dictionary with 24h price change statistics
        """
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
            
        return await self._request('GET', 'ticker/24hr', params)
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book for a symbol.
        
        Args:
            symbol: Trading pair
            limit: Number of orders to return (5, 10, 20, 50, 100, 500, 1000, 5000)
            
        Returns:
            Dictionary with order book data
        """
        params = {
            'symbol': symbol.upper(),
            'limit': limit
        }
        return await self._request('GET', 'depth', params)
    
    # Account Endpoints (signed)
    
    async def get_account_info(self) -> Dict:
        """Get current account information.
        
        Requires authentication.
        """
        return await self._request('GET', 'account', signed=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders.
        
        Args:
            symbol: Optional trading pair
            
        Returns:
            List of open orders
        """
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
            
        return await self._request('GET', 'openOrders', params, signed=True)
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        time_in_force: str = 'GTC',
        stop_price: Optional[float] = None,
        close_position: bool = False
    ) -> Dict:
        """Place a new order.
        
        Args:
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            order_type: 'LIMIT', 'MARKET', 'STOP_LOSS', 'TAKE_PROFIT', etc.
            quantity: Order quantity
            price: Order price (required for limit orders)
            time_in_force: 'GTC' (Good Till Cancel), 'IOC' (Immediate or Cancel), etc.
            stop_price: Required for stop loss/take profit orders
            close_position: Whether to close the position (for hedge mode)
            
        Returns:
            Order details
        """
        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': quantity,
            'timeInForce': time_in_force
        }
        
        if price is not None:
            params['price'] = price
            
        if stop_price is not None:
            params['stopPrice'] = stop_price
            
        if close_position:
            params['closePosition'] = 'true'
            
        return await self._request('POST', 'order', params, signed=True)
    
    async def get_ticker_price(self, symbol: str) -> Dict:
        """Get latest price for a symbol."""
        return await self._request('GET', 'ticker/price', {'symbol': symbol})
    
    async def get_klines(
        self, 
        symbol: str, 
        interval: str = '1m', 
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List[Any]]:
        """Get kline/candlestick data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles to return (default 500, max 1000)
            start_time: Start time in milliseconds since epoch
            end_time: End time in milliseconds since epoch
            
        Returns:
            List of klines in the format:
            [
                [
                    open_time,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    close_time,
                    quote_asset_volume,
                    number_of_trades,
                    taker_buy_base_asset_volume,
                    taker_buy_quote_asset_volume,
                    ignore
                ],
                ...
            ]
        """
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': min(limit, 1000)  # Enforce max limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return await self._request('GET', 'klines', params)
    
    # Private API endpoints (require authentication)
    
    async def get_account_info(self) -> Dict:
        """Get current account information."""
        return await self._request('GET', 'account', {}, signed=True)
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = 'GTC',
        reduce_only: bool = False,
        close_position: bool = False,
        position_side: str = 'BOTH',
        new_client_order_id: Optional[str] = None,
        working_type: str = 'CONTRACT_PRICE',
        price_protect: bool = False,
        new_order_resp_type: str = 'RESULT'
    ) -> Dict:
        """Place a new order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            side: 'BUY' or 'SELL'
            order_type: 'LIMIT', 'MARKET', 'STOP', 'TAKE_PROFIT', etc.
            quantity: Order quantity
            price: Order price (required for limit orders)
            stop_price: Used with STOP/STOP_MARKET or TAKE_PROFIT/TAKE_PROFIT_MARKET orders
            time_in_force: 'GTC' (default), 'IOC', 'FOK', 'GTX'
            reduce_only: If true, only reduce position, not increase it
            close_position: If true, close position with a market order
            position_side: 'BOTH', 'LONG', or 'SHORT'
            new_client_order_id: A unique ID for the order
            working_type: 'MARK_PRICE' or 'CONTRACT_PRICE'
            price_protect: If true, enables price protection
            new_order_resp_type: 'ACK', 'RESULT', or 'FULL'
            
        Returns:
            Order details
        """
        params = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'type': order_type.upper(),
            'timeInForce': time_in_force,
            'reduceOnly': str(reduce_only).lower(),
            'closePosition': str(close_position).lower(),
            'positionSide': position_side.upper(),
            'workingType': working_type.upper(),
            'priceProtect': str(price_protect).lower(),
            'newOrderRespType': new_order_resp_type.upper()
        }
        
        if price is not None:
            params['price'] = str(price)
        if stop_price is not None:
            params['stopPrice'] = str(stop_price)
        if new_client_order_id is not None:
            params['newClientOrderId'] = new_client_order_id
        if not close_position:
            params['quantity'] = str(quantity)
        
        return await self._request('POST', 'order', params, signed=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get all open orders."""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return await self._request('GET', 'openOrders', params, signed=True)
    
    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an active order."""
        params = {
            'symbol': symbol.upper(),
            'orderId': order_id
        }
        return await self._request('DELETE', 'order', params, signed=True)
    
    async def get_position_risk(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get position risk information."""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return await self._request('GET', 'positionRisk', params, signed=True)
    
    async def get_account_trades(self, symbol: str, limit: int = 500) -> List[Dict]:
        """Get account trade list."""
        params = {
            'symbol': symbol.upper(),
            'limit': min(limit, 1000)
        }
        return await self._request('GET', 'userTrades', params, signed=True)
    
    async def get_income_history(
        self, 
        symbol: Optional[str] = None, 
        income_type: Optional[str] = None,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 500
    ) -> List[Dict]:
        """Get income history."""
        params = {'limit': min(limit, 1000)}
        if symbol:
            params['symbol'] = symbol.upper()
        if income_type:
            params['incomeType'] = income_type
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        return await self._request('GET', 'income', params, signed=True)
