"""Signal monitoring system for tracking active trades and sending notifications."""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext

from cache import signal_tracker, SignalStatus, SignalDirection, ActiveSignal
from market_data import get_market_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('signal_monitor.log')
    ]
)
logger = logging.getLogger(__name__)

class SignalMonitor:
    """Monitors active signals and sends notifications for price targets."""
    
    def __init__(self, bot: Bot, update_interval: int = 30):
        """Initialize the signal monitor.
        
        Args:
            bot: Telegram bot instance
            update_interval: How often to check prices (in seconds)
        """
        self.bot = bot
        self.update_interval = update_interval
        self.running = False
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
        self.chat_id = None  # Will be set when monitoring starts
        
    async def start_monitoring(self, chat_id: int):
        """Start monitoring active signals."""
        if self.running:
            logger.warning("Monitoring is already running")
            return
            
        self.running = True
        self.chat_id = chat_id
        logger.info(f"Started monitoring signals for chat {chat_id}")
        
        # Start background tasks for each symbol with active signals
        while self.running:
            try:
                # Update all active signals
                active_signals = signal_tracker.get_active_signals()
                symbols = {s.symbol for s in active_signals}
                
                # Start monitoring for new symbols
                for symbol in symbols:
                    if symbol not in self.monitoring_tasks:
                        self.monitoring_tasks[symbol] = asyncio.create_task(
                            self._monitor_symbol(symbol)
                        )
                
                # Clean up completed tasks
                for symbol in list(self.monitoring_tasks.keys()):
                    if symbol not in symbols:
                        task = self.monitoring_tasks.pop(symbol, None)
                        if task:
                            task.cancel()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Prevent tight loop on errors
    
    async def stop_monitoring(self):
        """Stop monitoring signals."""
        self.running = False
        for task in self.monitoring_tasks.values():
            task.cancel()
        self.monitoring_tasks.clear()
        logger.info("Stopped monitoring signals")
    
    async def _monitor_symbol(self, symbol: str):
        """Monitor a single symbol for price updates."""
        logger.info(f"Starting to monitor {symbol}")
        
        try:
            while self.running:
                try:
                    # Get current price (simplified - in production, use websocket or API)
                    ohlcv = await get_market_data('binance', symbol, '1m', limit=1)
                    if ohlcv is None or ohlcv.empty:
                        logger.warning(f"No data for {symbol}, retrying...")
                        await asyncio.sleep(5)
                        continue
                        
                    current_price = ohlcv['close'].iloc[-1]
                    
                    # Update signals and get any triggers
                    triggers = signal_tracker.update_price(symbol, current_price)
                    
                    # Process triggers (e.g., send notifications)
                    for trigger in triggers:
                        await self._handle_trigger(trigger)
                    
                    await asyncio.sleep(self.update_interval)
                    
                except Exception as e:
                    logger.error(f"Error monitoring {symbol}: {e}", exc_info=True)
                    await asyncio.sleep(5)  # Prevent tight loop on errors
                    
        except asyncio.CancelledError:
            logger.info(f"Stopped monitoring {symbol}")
        except Exception as e:
            logger.error(f"Fatal error monitoring {symbol}: {e}", exc_info=True)
    
    async def _handle_trigger(self, trigger: Dict):
        """Handle a trigger (target hit, stop loss, etc.)."""
        try:
            signal_id = trigger['signal_id']
            signal = signal_tracker.get_signal(signal_id)
            
            if not signal:
                logger.warning(f"Signal {signal_id} not found for trigger {trigger}")
                return
            
            message = f"🔔 *{signal.symbol}*\n"
            
            if trigger['type'] == 'TAKE_PROFIT':
                message += (
                    f"🎯 *Target {trigger['level']} Reached!*\n"
                    f"Entry: {signal.entry_price:.8f}\n"
                    f"Target: {trigger['price']:.8f} "
                    f"({self._calculate_percent_change(signal.entry_price, trigger['price'], signal.direction):.2f}%)"
                )
                
                # If this was the last target, add PNL info
                if len(signal.hit_targets) == len(signal.take_profits):
                    pnl_pct = self._calculate_percent_change(
                        signal.entry_price, trigger['price'], signal.direction
                    )
                    pnl_leveraged = pnl_pct * signal.leverage
                    message += (
                        f"\n\n✅ *All Targets Reached!*\n"
                        f"Total PnL: {pnl_pct:.2f}% "
                        f"(Leveraged: {pnl_leveraged:.2f}%)"
                    )
            
            elif trigger['type'] == 'STOP_LOSS':
                pnl_pct = self._calculate_percent_change(
                    signal.entry_price, trigger['price'], signal.direction
                )
                pnl_leveraged = pnl_pct * signal.leverage
                
                message += (
                    f"🛑 *Stop Loss Hit!*\n"
                    f"Entry: {signal.entry_price:.8f}\n"
                    f"Exit: {trigger['price']:.8f} "
                    f"({pnl_pct:.2f}% / {pnl_leveraged:.2f}% with {signal.leverage}x)\n"
                    f"Max Drawdown: {self._calculate_max_drawdown(signal):.2f}%"
                )
            
            # Add signal info
            message += f"\n\n📊 *Signal Info*\n"
            message += f"Direction: {'🟢 LONG' if signal.direction == SignalDirection.LONG else '🔴 SHORT'}\n"
            message += f"Leverage: {signal.leverage}x\n"
            message += f"Risk: {signal.risk_percentage}%\n"
            
            # Add progress bar for targets
            message += "\n🎯 *Progress*\n"
            total_targets = len(signal.take_profits)
            hit_targets = len(signal.hit_targets)
            
            # Create a simple progress bar
            progress = min(hit_targets / total_targets, 1.0)
            bar_length = 10
            filled = int(progress * bar_length)
            progress_bar = '🟩' * filled + '⬜' * (bar_length - filled)
            
            message += f"{progress_bar} {hit_targets}/{total_targets} targets hit\n"
            
            # Reply to the original signal message if possible
            reply_to_message_id = signal.message_id
            
            # Send the message
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown',
                reply_to_message_id=reply_to_message_id if reply_to_message_id else None
            )
            
            logger.info(f"Sent notification for {signal.symbol}: {trigger['type']}")
            
        except Exception as e:
            logger.error(f"Error handling trigger {trigger}: {e}", exc_info=True)
    
    def _calculate_percent_change(self, entry: float, current: float, direction: SignalDirection) -> float:
        """Calculate percentage change based on trade direction."""
        if direction == SignalDirection.LONG:
            return ((current - entry) / entry) * 100
        else:  # SHORT
            return ((entry - current) / entry) * 100
    
    def _calculate_max_drawdown(self, signal: ActiveSignal) -> float:
        """Calculate maximum drawdown for a signal."""
        if signal.direction == SignalDirection.LONG:
            max_price = max(tp for i, tp in enumerate(signal.take_profits) if (i+1) in signal.hit_targets)
            max_drawdown = ((max_price - signal.entry_price) / signal.entry_price) * 100
        else:  # SHORT
            min_price = min(tp for i, tp in enumerate(signal.take_profits) if (i+1) in signal.hit_targets)
            max_drawdown = ((signal.entry_price - min_price) / signal.entry_price) * 100
        
        return max_drawdown


# Helper functions for signal creation
def create_signal(
    symbol: str,
    direction: SignalDirection,
    entry_price: float,
    stop_loss: float,
    take_profits: List[float],
    message_id: int = None,
    exchange: str = "binance",
    leverage: int = 1,
    risk_percentage: float = 1.0,
    indicators: dict = None
) -> ActiveSignal:
    """Helper function to create a new signal."""
    signal = ActiveSignal(
        signal_id=None,  # Will be auto-generated
        symbol=symbol.upper(),
        direction=direction,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profits=take_profits,
        message_id=message_id,
        exchange=exchange,
        leverage=leverage,
        risk_percentage=risk_percentage
    )
    
    if indicators:
        signal.indicators = indicators
    
    # Add to tracker
    signal_id = signal_tracker.add_signal(signal)
    signal.signal_id = signal_id
    
    return signal
