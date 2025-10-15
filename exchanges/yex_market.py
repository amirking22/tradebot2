"""YEX Futures Market Data and Analysis."""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import mplfinance as mpf
from datetime import datetime, timedelta
import os

from .yex import YEXClient
from ..config import YEX_API_KEY, YEX_SECRET_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('yex_market.log')
    ]
)
logger = logging.getLogger(__name__)

class YEXFuturesMarket:
    """YEX Futures Market Data and Analysis."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize the YEX Futures client."""
        self.api_key = api_key or YEX_API_KEY
        self.api_secret = api_secret or YEX_SECRET_KEY
        self.client = YEXClient(self.api_key, self.api_secret)
        self.symbols = []
        self.market_data = {}
        
    async def initialize(self):
        """Initialize the market data by fetching available symbols."""
        try:
            self.symbols = await self.get_futures_symbols()
            logger.info(f"Initialized YEX Futures with {len(self.symbols)} symbols")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize YEX Futures: {e}")
            return False
    
    async def get_futures_symbols(self) -> List[Dict]:
        """Get all available futures symbols."""
        try:
            symbols = await self.client.get_futures_symbols()
            # Filter for USDT pairs and format symbol names
            formatted_symbols = []
            for symbol in symbols:
                if 'USDT' in symbol['symbol']:
                    formatted_symbols.append({
                        'symbol': symbol['symbol'],
                        'base_asset': symbol['symbol'].replace('USDT', ''),
                        'quote_asset': 'USDT',
                        'status': symbol.get('status', 'TRADING'),
                        'filters': symbol.get('filters', [])
                    })
            return formatted_symbols
        except Exception as e:
            logger.error(f"Error fetching futures symbols: {e}")
            return []
    
    async def get_historical_data(
        self,
        symbol: str,
        interval: str = '1h',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500
    ) -> pd.DataFrame:
        """Get historical klines data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
            start_time: Start time (default: 30 days ago)
            end_time: End time (default: now)
            limit: Maximum number of candles to return
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Set default time range if not provided
            if end_time is None:
                end_time = datetime.utcnow()
            if start_time is None:
                start_time = end_time - timedelta(days=30)
                
            # Convert to timestamp strings
            start_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Fetch data
            df = await self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )
            
            # Cache the data
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
            self.market_data[symbol][interval] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_market_overview(self, top_n: int = 10) -> List[Dict]:
        """Get market overview with top gainers and losers.
        
        Args:
            top_n: Number of top gainers/losers to return
            
        Returns:
            List of dictionaries with market data
        """
        try:
            # Get 24h ticker for all symbols
            tickers = await self.client.get_ticker_24h()
            
            # Convert to DataFrame and process
            df = pd.DataFrame(tickers)
            
            # Filter for USDT pairs
            df = df[df['symbol'].str.endswith('USDT')]
            
            # Convert string values to float
            numeric_cols = ['priceChange', 'priceChangePercent', 'lastPrice', 'volume', 'quoteVolume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by price change percent
            df = df.sort_values('priceChangePercent', ascending=False)
            
            # Get top gainers and losers
            top_gainers = df.head(top_n).to_dict('records')
            top_losers = df.tail(top_n).to_dict('records')
            
            return {
                'top_gainers': top_gainers,
                'top_losers': top_losers,
                'total_volume': df['quoteVolume'].sum(),
                'total_pairs': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {'top_gainers': [], 'top_losers': [], 'total_volume': 0, 'total_pairs': 0}
    
    async def analyze_symbol(
        self, 
        symbol: str, 
        interval: str = '1h',
        lookback: str = '30d'
    ) -> Dict:
        """Perform technical analysis on a symbol.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            lookback: Lookback period (e.g., '30d', '7d', '1y')
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Calculate start time based on lookback
            end_time = datetime.utcnow()
            if lookback.endswith('d'):
                days = int(lookback[:-1])
                start_time = end_time - timedelta(days=days)
            elif lookback.endswith('h'):
                hours = int(lookback[:-1])
                start_time = end_time - timedelta(hours=hours)
            else:
                start_time = end_time - timedelta(days=30)  # Default to 30 days
            
            # Get historical data
            df = await self.get_historical_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            
            if df.empty:
                return {
                    'symbol': symbol,
                    'error': 'No data available',
                    'success': False
                }
            
            # Calculate technical indicators
            analysis = {
                'symbol': symbol,
                'interval': interval,
                'lookback': lookback,
                'success': True,
                'current_price': df['close'].iloc[-1],
                'price_change_24h': self._calculate_price_change(df, periods=24 if interval == '1h' else 24*7),
                'volume_24h': df['volume'].tail(24).sum() if interval == '1h' else df['volume'].mean(),
                'support_levels': self._find_support_levels(df),
                'resistance_levels': self._find_resistance_levels(df),
                'rsi': self._calculate_rsi(df),
                'macd': self._calculate_macd(df),
                'bollinger_bands': self._calculate_bollinger_bands(df),
                'trend': self._determine_trend(df)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e),
                'success': False
            }
    
    def _calculate_price_change(self, df: pd.DataFrame, periods: int = 24) -> float:
        """Calculate price change over the specified number of periods."""
        if len(df) < periods + 1:
            return 0.0
        
        start_price = df['close'].iloc[-periods-1]
        end_price = df['close'].iloc[-1]
        
        return ((end_price - start_price) / start_price) * 100
    
    def _find_support_levels(self, df: pd.DataFrame, window: int = 20) -> List[float]:
        """Find support levels using local minima."""
        df = df.copy()
        df['min'] = df['low'].rolling(window=window, center=True).min()
        
        # Find local minima
        support_levels = []
        for i in range(1, len(df)-1):
            if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                support_levels.append(df['low'].iloc[i])
        
        # Get unique levels and sort
        support_levels = sorted(list(set(support_levels)))
        
        # Return top 3 most recent support levels
        return support_levels[-3:] if len(support_levels) > 3 else support_levels
    
    def _find_resistance_levels(self, df: pd.DataFrame, window: int = 20) -> List[float]:
        """Find resistance levels using local maxima."""
        df = df.copy()
        df['max'] = df['high'].rolling(window=window, center=True).max()
        
        # Find local maxima
        resistance_levels = []
        for i in range(1, len(df)-1):
            if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                resistance_levels.append(df['high'].iloc[i])
        
        # Get unique levels and sort
        resistance_levels = sorted(list(set(resistance_levels)))
        
        # Return top 3 most recent resistance levels
        return resistance_levels[-3:] if len(resistance_levels) > 3 else resistance_levels
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Relative Strength Index (RSI)."""
        if len(df) < period + 1:
            return 50.0  # Neutral RSI if not enough data
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicator."""
        if len(df) < slow + signal:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': (macd - signal_line).iloc[-1]
        }
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: int = 2) -> Dict:
        """Calculate Bollinger Bands."""
        if len(df) < window:
            return {'upper': 0, 'middle': 0, 'lower': 0}
        
        middle = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        
        return {
            'upper': (middle + (std * num_std)).iloc[-1],
            'middle': middle.iloc[-1],
            'lower': (middle - (std * num_std)).iloc[-1]
        }
    
    def _determine_trend(self, df: pd.DataFrame, short_window: int = 20, long_window: int = 50) -> str:
        """Determine the current trend using moving averages."""
        if len(df) < long_window:
            return 'neutral'
        
        df = df.copy()
        df['sma_short'] = df['close'].rolling(window=short_window).mean()
        df['sma_long'] = df['close'].rolling(window=long_window).mean()
        
        # Check if short MA is above long MA (uptrend) or below (downtrend)
        if df['sma_short'].iloc[-1] > df['sma_long'].iloc[-1]:
            return 'uptrend'
        elif df['sma_short'].iloc[-1] < df['sma_long'].iloc[-1]:
            return 'downtrend'
        else:
            return 'neutral'
    
    async def plot_analysis(
        self,
        symbol: str,
        interval: str = '1h',
        lookback: str = '30d',
        output_dir: str = 'charts'
    ) -> Optional[str]:
        """Generate and save a chart with technical analysis.
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval
            lookback: Lookback period
            output_dir: Directory to save the chart
            
        Returns:
            Path to the saved chart or None if failed
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Calculate start time based on lookback
            end_time = datetime.utcnow()
            if lookback.endswith('d'):
                days = int(lookback[:-1])
                start_time = end_time - timedelta(days=days)
            elif lookback.endswith('h'):
                hours = int(lookback[:-1])
                start_time = end_time - timedelta(hours=hours)
            else:
                start_time = end_time - timedelta(days=30)  # Default to 30 days
            
            # Get historical data
            df = await self.get_historical_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
            
            if df.empty:
                logger.error(f"No data available for {symbol}")
                return None
            
            # Calculate indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            
            # Prepare data for mplfinance
            plot_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            plot_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Create the plot
            mc = mpf.make_marketcolors(
                up='green',
                down='red',
                edge='inherit',
                wick='inherit',
                volume='in',
                ohlc='i'
            )
            
            style = mpf.make_mpf_style(
                marketcolors=mc,
                gridstyle=':',
                gridcolor='#e0e0e0',
                facecolor='white',
                edgecolor='black',
                figcolor='white',
                rc={
                    'font.size': 10,
                    'axes.titlesize': 12,
                    'axes.labelsize': 10,
                    'xtick.labelsize': 8,
                    'ytick.labelsize': 8,
                }
            )
            
            # Add moving averages
            apds = [
                mpf.make_addplot(df['sma_20'], color='blue', width=1.5, panel=0),
                mpf.make_addplot(df['sma_50'], color='orange', width=1.5, panel=0)
            ]
            
            # Generate filename
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"{symbol.replace('/', '')}_{interval}_{lookback}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Create the plot
            fig, axes = mpf.plot(
                plot_df,
                type='candle',
                style=style,
                title=f"{symbol} {interval} - {lookback} lookback",
                ylabel='Price',
                volume=True,
                addplot=apds,
                returnfig=True,
                figratio=(12, 8),
                figscale=1.2
            )
            
            # Save the figure
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating chart for {symbol}: {e}")
            return None
    
    async def close(self):
        """Close the client connection."""
        await self.client.close()


# Singleton instance
yex_market = YEXFuturesMarket()


async def main():
    """Example usage."""
    market = YEXFuturesMarket()
    
    try:
        # Initialize and fetch symbols
        await market.initialize()
        
        # Get market overview
        overview = await market.get_market_overview()
        print(f"Top Gainers: {[s['symbol'] for s in overview['top_gainers']]}")
        print(f"Top Losers: {[s['symbol'] for s in overview['top_losers']]}")
        
        # Analyze a symbol
        symbol = 'BTCUSDT'
        analysis = await market.analyze_symbol(symbol)
        print(f"\nAnalysis for {symbol}:")
        print(f"Current Price: {analysis['current_price']}")
        print(f"24h Change: {analysis['price_change_24h']:.2f}%")
        print(f"RSI: {analysis['rsi']:.2f}")
        print(f"Trend: {analysis['trend']}")
        
        # Generate and save chart
        chart_path = await market.plot_analysis(symbol)
        if chart_path:
            print(f"\nChart saved to: {chart_path}")
        
    finally:
        await market.close()


if __name__ == "__main__":
    asyncio.run(main())
