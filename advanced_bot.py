"""
Advanced Crypto Trading Bot with Futures Signals
Features:
- Real-time market data
- 5-level TP strategy
- Technical indicators (RSI, ADX, ATR)
- Signal generation and notifications
- Restricted access by chat ID
"""
import logging
import json
import time
import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load configuration
from config import (
    TELEGRAM_TOKEN,
    ALLOWED_CHAT_IDS,  # List of allowed chat IDs
    BINANCE_API_KEY,    # For real market data
    BINANCE_SECRET_KEY  # For real market data
)

class TechnicalIndicators:
    """Class for calculating technical indicators"""
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0  # Default neutral value
            
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else 1.0
        rsi = 100. - (100./(1. + rs))

        for i in range(period+1, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up/down if down != 0 else 1.0
            rsi = 100. - (100./(1. + rs))
            
        return round(rsi, 2)

    @staticmethod
    def calculate_atr(high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        if len(high) < period or len(low) < period or len(close) < period:
            return 0.0
            
        tr = [high[0] - low[0]]
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr.append(max(tr1, tr2, tr3))
            
        atr = sum(tr[:period]) / period
        return round(atr, 4)

    @staticmethod
    def calculate_adx(high: List[float], low: List[float], close: List[float], period: int = 14) -> float:
        """Calculate ADX"""
        if len(high) < period * 2:
            return 25.0  # Default neutral value
            
        # Calculate +DM and -DM
        plus_dm = [0.0]
        minus_dm = [0.0]
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0.0)
                
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0.0)
        
        # Calculate TR, +DI, -DI
        tr = [high[0] - low[0]]
        plus_di = [0.0]
        minus_di = [0.0]
        
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_range = max(tr1, tr2, tr3)
            tr.append(true_range)
            
            plus_di_val = (100 * plus_dm[i] / true_range) if true_range != 0 else 0
            minus_di_val = (100 * minus_dm[i] / true_range) if true_range != 0 else 0
            plus_di.append(plus_di_val)
            minus_di.append(minus_di_val)
        
        # Calculate ADX
        dx = [0.0] * len(high)
        for i in range(1, len(high)):
            if (plus_di[i] + minus_di[i]) != 0:
                dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = sum(dx[-period:]) / period
        return round(adx, 2)

class BinanceAPI:
    """Class for interacting with Binance API"""
    
    def __init__(self, api_key: str, api_secret: str):
        self.base_url = "https://api.binance.com"
        self.api_key = api_key
        self.api_secret = api_secret
    
    def get_klines(self, symbol: str, interval: str = '30m', limit: int = 100) -> List[list]:
        """Get kline/candlestick data"""
        endpoint = f"{self.base_url}/api/v3/klines"
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        try:
            response = requests.get(endpoint, params=params)
            data = response.json()
            return data
        except Exception as e:
            logger.error(f"Error getting klines: {e}")
            return []
    
    def get_ticker_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        endpoint = f"{self.base_url}/api/v3/ticker/price"
        params = {'symbol': symbol.upper()}
        
        try:
            response = requests.get(endpoint, params=params)
            data = response.json()
            return float(data['price'])
        except Exception as e:
            logger.error(f"Error getting ticker price: {e}")
            return None

class TradingSignal:
    """Class for generating trading signals"""
    
    def __init__(self, symbol: str, current_price: float, rsi: float, adx: float, atr: float):
        self.symbol = symbol
        self.current_price = current_price
        self.rsi = rsi
        self.adx = adx
        self.atr = atr
        self.signal_type = self._determine_signal()
        self.entry_price = self._calculate_entry()
        self.stop_loss = self._calculate_stop_loss()
        self.take_profits = self._calculate_take_profits()
    
    def _determine_signal(self) -> str:
        """Determine if the signal is LONG or SHORT"""
        if self.rsi > 50 and self.adx > 25:
            return "LONG"
        elif self.rsi < 50 and self.adx > 25:
            return "SHORT"
        return "NEUTRAL"
    
    def _calculate_entry(self) -> float:
        """Calculate entry price"""
        if self.signal_type == "LONG":
            return self.current_price * 1.001  # Slightly above current for LONG
        elif self.signal_type == "SHORT":
            return self.current_price * 0.999  # Slightly below current for SHORT
        return self.current_price
    
    def _calculate_stop_loss(self) -> float:
        """Calculate stop loss price"""
        if self.signal_type == "LONG":
            return self.entry_price - (self.atr * 2)
        elif self.signal_type == "SHORT":
            return self.entry_price + (self.atr * 2)
        return 0.0
    
    def _calculate_take_profits(self) -> List[float]:
        """Calculate 5 take profit levels"""
        if self.signal_type == "NEUTRAL":
            return []
            
        tp_levels = []
        tp_pct = [0.2, 0.2, 0.2, 0.2, 0.2]  # 5 levels, 20% each
        total_pips = abs(self.entry_price - self.stop_loss) * 3  # 1:3 risk:reward
        
        for i in range(5):
            if self.signal_type == "LONG":
                tp = self.entry_price + (total_pips * (i + 1) * 0.2)
            else:  # SHORT
                tp = self.entry_price - (total_pips * (i + 1) * 0.2)
            tp_levels.append(round(tp, 4))
            
        return tp_levels
    
    def get_signal_message(self) -> str:
        """Generate signal message"""
        if self.signal_type == "NEUTRAL":
            return f"🔍 {self.symbol} - No clear signal (RSI: {self.rsi}, ADX: {self.adx})"
        
        signal_emoji = "📈" if self.signal_type == "LONG" else "📉"
        message = (
            f"{signal_emoji} *سیگنال {self.signal_type} — {self.symbol}*\n"
            f"⏱️ تایم‌فریم: 30m\n"
            f"🔹 شروع (Entry): {self.entry_price:.4f}\n"
            f"🛑 حدضرر: {self.stop_loss:.4f}\n"
        )
        
        for i, tp in enumerate(self.take_profits, 1):
            message += f"🎯 TP{i}: {tp:.4f}\n"
            
        message += (
            f"📏 ATR%: {(self.atr / self.current_price * 100):.2f}% | "
            f"RSI: {self.rsi:.1f} | "
            f"ADX: {self.adx:.1f}\n"
            f"🧾 سایز کل پوزیشن پیشنهادی: 100% (تقسیم در 5 پله)\n"
            f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}"
        )
        
        return message

class AdvancedCryptoBot:
    """Advanced Crypto Trading Bot with Futures Signals"""
    
    def __init__(self, token: str, allowed_chat_ids: List[int], binance_api_key: str = "", binance_secret_key: str = ""):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.offset = 0
        self.allowed_chat_ids = allowed_chat_ids
        self.active_trades = {}  # Track active trades for TP notifications
        self.binance = BinanceAPI(binance_api_key, binance_secret_key) if binance_api_key and binance_secret_key else None
    
    def _send_message(self, chat_id: int, text: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to a chat"""
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode
        }
        
        try:
            response = requests.post(url, json=data)
            return response.json().get("ok", False)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def _is_authorized(self, chat_id: int) -> bool:
        """Check if chat ID is authorized"""
        return chat_id in self.allowed_chat_ids
    
    def analyze_market(self, symbol: str) -> Optional[str]:
        """Analyze market and generate signal"""
        if not self.binance:
            return "❌ Binance API not configured. Please set BINANCE_API_KEY and BINANCE_SECRET_KEY in config.py"
        
        # Get historical data
        klines = self.binance.get_klines(symbol, interval='30m', limit=100)
        if not klines:
            return f"❌ Could not fetch data for {symbol}"
        
        # Extract OHLCV data
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]
        
        # Calculate indicators
        rsi = TechnicalIndicators.calculate_rsi(closes)
        adx = TechnicalIndicators.calculate_adx(highs, lows, closes)
        atr = TechnicalIndicators.calculate_atr(highs, lows, closes)
        current_price = closes[-1]
        
        # Generate signal
        signal = TradingSignal(symbol, current_price, rsi, adx, atr)
        return signal.get_signal_message()
    
    def handle_message(self, chat_id: int, text: str) -> None:
        """Handle incoming messages"""
        if not self._is_authorized(chat_id):
            self._send_message(chat_id, "❌ دسترسی غیرمجاز. لطفاً با ادمین تماس بگیرید.")
            return
        
        if text.startswith('/'):
            command = text.split('@')[0].lower()
            if command == "/start":
                self._send_message(chat_id, "🤖 *ربات تحلیل و سیگنال‌دهی ارزهای دیجیتال*\n\n"
                                 "برای دریافت تحلیل یک نماد، آن را ارسال کنید (مثلاً BTCUSDT)")
            elif command == "/help":
                self._send_message(chat_id, "ℹ️ *راهنما*\n\n"
                                 "• برای تحلیل یک نماد، آن را مستقیماً ارسال کنید (مثلاً BTCUSDT)\n"
                                 "• برای دریافت سیگنال‌های معاملاتی، از منوی زیر استفاده کنید")
        else:
            # Assume it's a symbol to analyze
            symbol = text.upper()
            if not symbol.endswith('USDT'):
                symbol += 'USDT'
                
            self._send_message(chat_id, f"🔍 در حال تحلیل {symbol}...")
            analysis = self.analyze_market(symbol)
            self._send_message(chat_id, analysis)
    
    def run(self) -> None:
        """Run the bot"""
        logger.info("Starting Advanced Crypto Bot...")
        
        while True:
            try:
                # Get updates
                url = f"{self.base_url}/getUpdates"
                params = {"offset": self.offset + 1, "timeout": 30}
                
                response = requests.get(url, params=params, timeout=35)
                updates = response.json()
                
                if not updates.get("ok"):
                    logger.error("Failed to get updates")
                    time.sleep(5)
                    continue
                
                # Process updates
                for update in updates.get("result", []):
                    self.offset = max(self.offset, update["update_id"])
                    
                    if "message" in update and "text" in update["message"]:
                        chat_id = update["message"]["chat"]["id"]
                        text = update["message"]["text"]
                        self.handle_message(chat_id, text)
                
                # Check for TP hits
                self._check_tp_levels()
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)
    
    def _check_tp_levels(self) -> None:
        """Check if any TP levels have been hit"""
        if not self.active_trades or not self.binance:
            return
            
        for trade_id, trade in list(self.active_trades.items()):
            symbol = trade["symbol"]
            current_price = self.binance.get_ticker_price(symbol)
            
            if not current_price:
                continue
                
            for i, tp_level in enumerate(trade["tp_levels"], 1):
                if trade["type"] == "LONG" and current_price >= tp_level and f"tp{i}" not in trade["hit_levels"]:
                    self._send_message(
                        trade["chat_id"],
                        f"✅ {symbol} — پله {i} از 5 برای لانگ رسید.\n"
                        f"قیمت فعلی: {current_price:.4f}\n"
                        f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}"
                    )
                    self.active_trades[trade_id]["hit_levels"].append(f"tp{i}")
                    
                elif trade["type"] == "SHORT" and current_price <= tp_level and f"tp{i}" not in trade["hit_levels"]:
                    self._send_message(
                        trade["chat_id"],
                        f"✅ {symbol} — پله {i} از 5 برای شورت رسید.\n"
                        f"قیمت فعلی: {current_price:.4f}\n"
                        f"🕒 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}"
                    )
                    self.active_trades[trade_id]["hit_levels"].append(f"tp{i}")

def main():
    """Main function"""
    # Load configuration
    from config import (
        TELEGRAM_TOKEN,
        ALLOWED_CHAT_IDS,
        BINANCE_API_KEY,
        BINANCE_SECRET_KEY
    )
    
    if not TELEGRAM_TOKEN:
        logger.error("No TELEGRAM_TOKEN found. Please set it in config.py")
        return
    
    if not ALLOWED_CHAT_IDS:
        logger.error("No ALLOWED_CHAT_IDS found. Please set it in config.py")
        return
    
    # Initialize and run the bot
    bot = AdvancedCryptoBot(
        token=TELEGRAM_TOKEN,
        allowed_chat_ids=ALLOWED_CHAT_IDS,
        binance_api_key=BINANCE_API_KEY,
        binance_secret_key=BINANCE_SECRET_KEY
    )
    
    bot.run()

if __name__ == "__main__":
    main()
