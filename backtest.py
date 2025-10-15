"""Backtesting module for evaluating trading strategies."""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from tqdm import tqdm

from market_data import get_historical_data, OHLCV
from indicators import generate_signals, SignalDirection
from cache import signal_tracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest.log')
    ]
)
logger = logging.getLogger(__name__)

class BacktestResult:
    """Holds the results of a backtest."""
    
    def __init__(self):
        self.trades = []
        self.equity_curve = []
        self.metrics = {}
    
    def add_trade(self, trade: Dict[str, Any]) -> None:
        """Add a trade to the results."""
        self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.trades:
            return {}
        
        # Convert to DataFrame for easier calculations
        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df)
        winning_trades = len(df[df['pnl_pct'] > 0])
        losing_trades = len(df[df['pnl_pct'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = df['pnl_pct'].sum()
        avg_win = df[df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = abs(df[df['pnl_pct'] < 0]['pnl_pct'].mean()) if losing_trades > 0 else 0
        profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(df)
        sharpe_ratio = self._calculate_sharpe_ratio(df)
        
        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'expectancy': (win_rate * avg_win - (100 - win_rate) * avg_loss) / 100
        }
        
        return self.metrics
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        df['cum_pnl'] = df['pnl_pct'].cumsum()
        df['cum_max'] = df['cum_pnl'].cummax()
        df['drawdown'] = df['cum_max'] - df['cum_pnl']
        return df['drawdown'].max() if not df.empty else 0
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(df) < 2:
            return 0
            
        returns = df['pnl_pct'].pct_change().dropna()
        if returns.empty:
            return 0
            
        excess_returns = returns - (risk_free_rate / 252)  # Annualized risk-free rate
        sharpe = (excess_returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)  # Annualized
        return sharpe
    
    def generate_report(self) -> str:
        """Generate a formatted report of the backtest results."""
        if not self.metrics:
            self.calculate_metrics()
            
        report = [
            "📊 *Backtest Results*\n",
            f"• Total Trades: {self.metrics['total_trades']}",
            f"• Win Rate: {self.metrics['win_rate']:.2f}%",
            f"• Total PnL: {self.metrics['total_pnl']:.2f}%",
            f"• Avg Win: {self.metrics['avg_win_pct']:.2f}%",
            f"• Avg Loss: {self.metrics['avg_loss_pct']:.2f}%",
            f"• Profit Factor: {self.metrics['profit_factor']:.2f}",
            f"• Max Drawdown: {self.metrics['max_drawdown']:.2f}%",
            f"• Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}",
            f"• Expectancy: {self.metrics['expectancy']:.2f}% per trade"
        ]
        
        return "\n".join(report)


class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str = '1h',
        initial_balance: float = 10000.0,
        risk_per_trade: float = 1.0,
        leverage: int = 1,
        exchange: str = 'binance'
    ):
        """Initialize the backtester.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: OHLCV timeframe (e.g., '1h', '4h', '1d')
            initial_balance: Starting balance in quote currency
            risk_per_trade: Percentage of capital to risk per trade
            leverage: Leverage to use (1-100)
            exchange: Exchange to use for data
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.exchange = exchange
        self.position = None
        self.trades = []
        self.equity_curve = []
        
    async def run(self, start_date: str, end_date: str = None) -> BacktestResult:
        """Run the backtest.
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: now)
            
        Returns:
            BacktestResult containing performance metrics
        """
        logger.info(f"Starting backtest for {self.symbol} ({self.timeframe}) from {start_date} to {end_date or 'now'}")
        
        # Load historical data
        data = await get_historical_data(
            self.exchange,
            self.symbol,
            self.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if data is None or data.empty:
            logger.error("No data available for the specified period")
            return None
            
        # Initialize result object
        result = BacktestResult()
        
        # Main backtest loop
        for i in tqdm(range(1, len(data)), desc="Backtesting"):
            current_data = data.iloc[:i+1].copy()
            current_candle = current_data.iloc[-1]
            
            # Generate signals
            signals = generate_signals(current_data)
            
            # Process signals
            for signal in signals:
                await self._process_signal(signal, current_candle, result)
            
            # Update running PnL for open positions
            if self.position:
                self._update_running_pnl(current_candle, result)
            
            # Record equity curve
            self.equity_curve.append({
                'timestamp': current_candle.name,
                'equity': self.balance,
                'price': current_candle['close']
            })
        
        # Close any open position at the end
        if self.position:
            self._close_position(current_candle, result, reason="End of backtest")
        
        # Calculate final metrics
        result.calculate_metrics()
        
        return result
    
    async def _process_signal(self, signal: Dict, current_candle: pd.Series, result: BacktestResult) -> None:
        """Process a trading signal."""
        if signal['direction'] == SignalDirection.LONG:
            if not self.position or self.position['direction'] == 'SHORT':
                if self.position:  # Close opposite position first
                    self._close_position(current_candle, result, "Signal reversal")
                
                # Open long position
                entry_price = current_candle['close']
                stop_loss = signal['stop_loss']
                take_profit = signal['take_profit']
                
                self._open_position(
                    direction='LONG',
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    current_candle=current_candle,
                    result=result
                )
        
        elif signal['direction'] == SignalDirection.SHORT:
            if not self.position or self.position['direction'] == 'LONG':
                if self.position:  # Close opposite position first
                    self._close_position(current_candle, result, "Signal reversal")
                
                # Open short position
                entry_price = current_candle['close']
                stop_loss = signal['stop_loss']
                take_profit = signal['take_profit']
                
                self._open_position(
                    direction='SHORT',
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    current_candle=current_candle,
                    result=result
                )
    
    def _open_position(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        current_candle: pd.Series,
        result: BacktestResult
    ) -> None:
        """Open a new position."""
        # Calculate position size based on risk
        risk_amount = self.balance * (self.risk_per_trade / 100)
        
        if direction == 'LONG':
            risk_per_share = entry_price - stop_loss
            position_size = (risk_amount * self.leverage) / (risk_per_share * entry_price)
        else:  # SHORT
            risk_per_share = stop_loss - entry_price
            position_size = (risk_amount * self.leverage) / (risk_per_share * entry_price)
        
        # Update position
        self.position = {
            'direction': direction,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': position_size,
            'entry_time': current_candle.name,
            'max_drawdown': 0,
            'max_profit': 0
        }
        
        logger.info(
            f"Opened {direction} position at {entry_price:.8f} "
            f"(SL: {stop_loss:.8f}, TP: {take_profit:.8f})"
        )
    
    def _close_position(
        self,
        current_candle: pd.Series,
        result: BacktestResult,
        reason: str = ""
    ) -> None:
        """Close the current position."""
        if not self.position:
            return
            
        exit_price = current_candle['close']
        entry_price = self.position['entry_price']
        
        # Calculate PnL
        if self.position['direction'] == 'LONG':
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100 * self.leverage
        else:  # SHORT
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100 * self.leverage
        
        # Update balance
        self.balance *= (1 + pnl_pct / 100)
        
        # Record trade
        trade = {
            'entry_time': self.position['entry_time'],
            'exit_time': current_candle.name,
            'direction': self.position['direction'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl_pct': pnl_pct,
            'balance': self.balance,
            'stop_loss': self.position['stop_loss'],
            'take_profit': self.position['take_profit'],
            'reason': reason
        }
        
        result.add_trade(trade)
        
        logger.info(
            f"Closed {self.position['direction']} position: "
            f"{pnl_pct:+.2f}% (Balance: {self.balance:.2f}) - {reason}"
        )
        
        # Clear position
        self.position = None
    
    def _update_running_pnl(self, current_candle: pd.Series, result: BacktestResult) -> None:
        """Update running PnL and check for stop loss/take profit."""
        if not self.position:
            return
            
        current_price = current_candle['close']
        entry_price = self.position['entry_price']
        
        # Calculate current PnL
        if self.position['direction'] == 'LONG':
            pnl_pct = ((current_price - entry_price) / entry_price) * 100 * self.leverage
            
            # Check for stop loss
            if current_price <= self.position['stop_loss']:
                self._close_position(current_candle, result, "Stop loss hit")
                return
                
            # Check for take profit
            if current_price >= self.position['take_profit']:
                self._close_position(current_candle, result, "Take profit hit")
                return
                
        else:  # SHORT
            pnl_pct = ((entry_price - current_price) / entry_price) * 100 * self.leverage
            
            # Check for stop loss
            if current_price >= self.position['stop_loss']:
                self._close_position(current_candle, result, "Stop loss hit")
                return
                
            # Check for take profit
            if current_price <= self.position['take_profit']:
                self._close_position(current_candle, result, "Take profit hit")
                return
        
        # Update running stats
        self.position['max_drawdown'] = min(self.position['max_drawdown'], pnl_pct)
        self.position['max_profit'] = max(self.position['max_profit'], pnl_pct)


async def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str = None,
    timeframe: str = '1h',
    initial_balance: float = 10000.0,
    risk_per_trade: float = 1.0,
    leverage: int = 1,
    exchange: str = 'binance'
) -> BacktestResult:
    """Run a backtest with the given parameters.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (default: now)
        timeframe: OHLCV timeframe (e.g., '1h', '4h', '1d')
        initial_balance: Starting balance in quote currency
        risk_per_trade: Percentage of capital to risk per trade
        leverage: Leverage to use (1-100)
        exchange: Exchange to use for data
        
    Returns:
        BacktestResult containing performance metrics
    """
    backtester = Backtester(
        symbol=symbol,
        timeframe=timeframe,
        initial_balance=initial_balance,
        risk_per_trade=risk_per_trade,
        leverage=leverage,
        exchange=exchange
    )
    
    return await backtester.run(start_date, end_date)


if __name__ == "__main__":
    # Example usage
    async def main():
        result = await run_backtest(
            symbol="BTC/USDT",
            start_date="2023-01-01",
            end_date="2023-12-31",
            timeframe="4h",
            initial_balance=10000.0,
            risk_per_trade=1.0,
            leverage=5
        )
        
        if result:
            print("\n" + "="*50)
            print(result.generate_report())
            print("="*50)
    
    asyncio.run(main())
