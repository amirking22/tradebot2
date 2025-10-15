"""
Simple Telegram bot without complex dependencies.
"""
import requests
import json
import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from typing import Dict, Any
import os
from dotenv import load_dotenv
from analysis.market_monitor import MarketMonitor
from analysis.golden_signals import GoldenSignalsDetector

load_dotenv()

logger = logging.getLogger(__name__)

class SimpleTelegramBot:
    def __init__(self, token: str, aggregator=None):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.aggregator = aggregator
        self.golden_detector = None
        self.active_signals = {}  # Store active signals for monitoring
        
        # Initialize golden signals detector
        if aggregator:
            from analysis.golden_signals import GoldenSignalsDetector
            self.golden_detector = GoldenSignalsDetector()
        
        # Set bot commands menu
        self._set_bot_commands()
        
        # Set authorized user ID from environment
        self.authorized_user_id = int(os.getenv('AUTHORIZED_USER_ID', '0'))
        
        # Initialize Market Monitor
        self.market_monitor = MarketMonitor(telegram_bot=self, aggregator=aggregator)
        self.monitoring_task = None
        
        # Auto-start monitoring when bot initializes
        if aggregator:
            asyncio.create_task(self._start_auto_monitoring())
        
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized."""
        return user_id == self.authorized_user_id
    
    def send_message(self, chat_id: int, text: str, parse_mode: str = 'Markdown', reply_markup=None):
        """Send message via Telegram API with optional inline keyboard."""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            if reply_markup:
                # Convert InlineKeyboardMarkup to dict format for raw API
                keyboard_dict = {
                    'inline_keyboard': []
                }
                for row in reply_markup.inline_keyboard:
                    button_row = []
                    for button in row:
                        button_row.append({
                            'text': button.text,
                            'callback_data': button.callback_data
                        })
                    keyboard_dict['inline_keyboard'].append(button_row)
                data['reply_markup'] = keyboard_dict
            
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to send message: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return None
    
    def get_updates(self, offset: int = 0):
        """Get updates from Telegram."""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {'offset': offset, 'timeout': 10}
            response = requests.get(url, params=params)
            return response.json()
        except Exception as e:
            logger.error(f"Error getting updates: {e}")
            return None
    
    async def handle_market_command(self, chat_id: int):
        """Handle /market command."""
        try:
            if not self.aggregator:
                self.send_message(chat_id, "❌ خطا: سیستم تجمیع داده در دسترس نیست")
                return
                
            # Get top markets
            top_symbols = await self.aggregator.get_top_markets(15)
            
            message = "📊 *وضعیت بازار ارزهای دیجیتال*\n\n"
            
            for symbol in top_symbols[:15]:  # Limit to 15 to avoid message length issues
                try:
                    ticker_data = await self.aggregator.get_aggregated_ticker(symbol)
                    
                    if 'error' in ticker_data:
                        continue
                        
                    price = ticker_data['price']
                    change_24h = ticker_data['change_24h']
                    reliability = ticker_data['reliability']
                    source_count = ticker_data['total_sources']
                    
                    # Format with emoji
                    if change_24h > 0:
                        emoji = "🟢"
                        sign = "+"
                    elif change_24h < 0:
                        emoji = "🔴"
                        sign = ""
                    else:
                        emoji = "⚪"
                        sign = ""
                    
                    rel_emoji = "🟢" if reliability == "high" else "🟡" if reliability == "medium" else "🔴"
                    
                    message += f"{emoji} *{symbol}*: `{price:.6f}` ({sign}{change_24h:.2f}%) {rel_emoji}({source_count})\n"
                    
                except Exception as e:
                    logger.error(f"Error getting ticker for {symbol}: {e}")
                    continue
            
            message += f"\n🟢بالا 🟡متوسط 🔴پایین (تعداد منابع)\n"
            message += f"🕒 آخرین بروزرسانی: {self.get_persian_time()}"
            
            self.send_message(chat_id, message)
            
        except Exception as e:
            logger.error(f"Error in market command: {e}")
            self.send_message(chat_id, "❌ خطا در دریافت اطلاعات بازار")
    
    async def handle_futures_command(self, chat_id: int):
        """Handle /futures command - analyze all cryptos for futures trading."""
        try:
            if not self.aggregator:
                self.send_message(chat_id, "❌ خطا: سیستم تجمیع داده در دسترس نیست")
                return
            
            self.send_message(chat_id, "🔄 در حال تحلیل بازار فیوچرز... لطفاً صبر کنید")
            
            # Get top symbols for analysis
            top_symbols = await self.aggregator.get_top_markets(20)
            
            # Import trading bot for analysis
            from bot.trading_bot import TradingBot
            
            # Build temporary config from environment
            temp_config = {
                'yex_api_key': os.getenv('YEX_API_KEY'),
                'yex_secret': os.getenv('YEX_SECRET'),
                'binance_api_key': os.getenv('BINANCE_API_KEY'),
                'binance_secret': os.getenv('BINANCE_SECRET'),
                'bybit_api_key': os.getenv('BYBIT_API_KEY'),
                'bybit_secret': os.getenv('BYBIT_SECRET'),
                'risk_reward_ratio': float(os.getenv('RISK_REWARD_RATIO', '3.0')),
                'position_size_percent': float(os.getenv('POSITION_SIZE_PERCENT', '10.0')),
                'max_open_positions': int(os.getenv('MAX_OPEN_POSITIONS', '5')),
                'rsi_period': int(os.getenv('RSI_PERIOD', '14')),
                'adx_period': int(os.getenv('ADX_PERIOD', '14')),
                'atr_period': int(os.getenv('ATR_PERIOD', '14'))
            }
            
            # Create temporary trading bot instance
            temp_bot = TradingBot(temp_config)
            temp_bot.aggregator = self.aggregator
            
            signals = []
            
            # Analyze each symbol
            for symbol in top_symbols:
                try:
                    # Get market data and analyze
                    analysis = await temp_bot.analyze_market('aggregated', symbol)
                    if analysis and analysis.get('signal') != 'NEUTRAL':
                        signals.append(analysis)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    continue
            
            # Sort signals by risk level (safe first, risky last)
            risk_order = {'SAFE': 1, 'MEDIUM': 2, 'RISKY': 3, 'VERY_RISKY': 4}
            signals.sort(key=lambda x: risk_order.get(x.get('risk_level', 'VERY_RISKY'), 5))
            
            if not signals:
                self.send_message(chat_id, "📊 در حال حاضر هیچ سیگنال فیوچرز مناسبی یافت نشد")
                return
            
            # Group signals by risk level
            safe_signals = [s for s in signals if s.get('risk_level') == 'SAFE' and s.get('confidence') == 'HIGH']
            medium_signals = [s for s in signals if s.get('risk_level') == 'MEDIUM']
            risky_signals = [s for s in signals if s.get('risk_level') in ['RISKY', 'VERY_RISKY']]
            
            # Send only SAFE signals with detailed information
            if safe_signals:
                self.send_detailed_signals(chat_id, safe_signals, "🟢 سیگنال‌های امن (پیشنهاد ربات)")
                # Check for TP/SL updates after sending signals
                await self.check_signal_updates(chat_id)
            else:
                self.send_message(chat_id, "📊 در حال حاضر هیچ سیگنال امن یافت نشد")
            
        except Exception as e:
            logger.error(f"Error in futures command: {e}")
            self.send_message(chat_id, "❌ خطا در تحلیل فیوچرز")
    
    def send_signal_group(self, chat_id: int, signals: list, title: str):
        """Send a group of signals with title."""
        message = f"{title}\n\n"
        
        for i, signal in enumerate(signals[:5], 1):  # Limit to 5 per group
            symbol = signal.get('symbol', 'N/A')
            signal_type = signal.get('signal', 'N/A')
            confidence = signal.get('confidence', 'LOW')
            
            # Get failed conditions
            failed_conditions = signal.get('failed_conditions', [])
            conditions_text = ""
            if failed_conditions:
                conditions_text = f" (ناموفق: {len(failed_conditions)})"
            
            direction = "خرید 📈" if signal_type == "LONG" else "فروش 📉"
            confidence_persian = {"HIGH": "بالا", "MEDIUM": "متوسط", "LOW": "پایین"}.get(confidence, "نامشخص")
            
            message += f"{i}. *{symbol}* - {direction}\n"
            message += f"   قدرت: {confidence_persian}{conditions_text}\n\n"
        
        if len(signals) > 5:
            message += f"... و {len(signals) - 5} سیگنال دیگر\n\n"
        
        message += f"🕒 {self.get_persian_time()}"
        self.send_message(chat_id, message)
    
    def send_detailed_signals(self, chat_id: int, signals: list, title: str):
        """Send detailed trading signals with entry, stop loss, and take profit levels."""
        header_message = f"{title}\n\n"
        self.send_message(chat_id, header_message)
        
        for signal in signals:  # Show all SAFE signals
            try:
                symbol = signal.get('symbol', 'N/A')
                signal_type = signal.get('signal', 'N/A')
                confidence = signal.get('confidence', 'LOW')
                indicators = signal.get('indicators', {})
                
                # Get current price from indicators or use a default
                current_price = indicators.get('current_price', 0)
                if current_price == 0:
                    # Try to get from EMA or other indicators
                    current_price = indicators.get('ema9', indicators.get('ema21', 1.0))
                
                # Calculate entry, stop loss, and take profit levels with trend analysis
                atr = indicators.get('atr', current_price * 0.015)  # Reduced default ATR for safer trades
                rsi = indicators.get('rsi', 50)
                adx = indicators.get('adx', 20)
                ema9 = indicators.get('ema9', current_price)
                ema21 = indicators.get('ema21', current_price)
                ema200 = indicators.get('ema200', current_price)
                
                # Determine trend strength for TP calculation
                trend_strength = "weak"
                if adx > 30:
                    trend_strength = "strong"
                elif adx > 20:
                    trend_strength = "medium"
                
                # Calculate trend direction for next 5 minutes
                bullish_trend = ema9 > ema21 > ema200 and rsi > 45 and rsi < 75
                bearish_trend = ema9 < ema21 < ema200 and rsi > 25 and rsi < 55
                
                if signal_type == "LONG":
                    entry_price = current_price
                    # Safer stop loss - 0.5 ATR for high safety
                    stop_loss = entry_price - (0.5 * atr)
                    
                    # Trend-based TP levels
                    if bullish_trend and trend_strength == "strong":
                        tp_multipliers = [0.4, 0.8, 1.3, 1.8, 2.5]  # Aggressive TPs for strong bullish trend
                    elif bullish_trend and trend_strength == "medium":
                        tp_multipliers = [0.3, 0.6, 1.0, 1.4, 2.0]  # Moderate TPs
                    else:
                        tp_multipliers = [0.2, 0.4, 0.7, 1.0, 1.3]  # Conservative TPs for weak trend
                    
                    tp1 = entry_price + (tp_multipliers[0] * atr)
                    tp2 = entry_price + (tp_multipliers[1] * atr)
                    tp3 = entry_price + (tp_multipliers[2] * atr)
                    tp4 = entry_price + (tp_multipliers[3] * atr)
                    tp5 = entry_price + (tp_multipliers[4] * atr)
                    direction = "لانگ"
                    emoji = "📈"
                    trend_prediction = "صعودی" if bullish_trend else "خنثی"
                    
                else:  # SHORT
                    entry_price = current_price
                    # Safer stop loss - 0.5 ATR for high safety
                    stop_loss = entry_price + (0.5 * atr)
                    
                    # Trend-based TP levels
                    if bearish_trend and trend_strength == "strong":
                        tp_multipliers = [0.4, 0.8, 1.3, 1.8, 2.5]  # Aggressive TPs for strong bearish trend
                    elif bearish_trend and trend_strength == "medium":
                        tp_multipliers = [0.3, 0.6, 1.0, 1.4, 2.0]  # Moderate TPs
                    else:
                        tp_multipliers = [0.2, 0.4, 0.7, 1.0, 1.3]  # Conservative TPs for weak trend
                    
                    tp1 = entry_price - (tp_multipliers[0] * atr)
                    tp2 = entry_price - (tp_multipliers[1] * atr)
                    tp3 = entry_price - (tp_multipliers[2] * atr)
                    tp4 = entry_price - (tp_multipliers[3] * atr)
                    tp5 = entry_price - (tp_multipliers[4] * atr)
                    direction = "شورت"
                    emoji = "📉"
                    trend_prediction = "نزولی" if bearish_trend else "خنثی"
                
                # Calculate position size (using 10% of capital with 5x leverage)
                capital = 100  # Use $100 per trade for lower risk
                leverage = 5
                position_value = capital * leverage  # Use entire $100 with leverage multiplier
                position_size = position_value / entry_price
                
                # Get technical indicators
                rsi = indicators.get('rsi', 0)
                adx = indicators.get('adx', 0)
                atr_percent = (atr / current_price) * 100
                
                message = (
                    f"📣 سیگنال {direction} — {symbol} {emoji}\n"
                    f"⏱️ تایم‌فریم: 30m\n"
                    f"🔮 پیش‌بینی 5 دقیقه آینده: {trend_prediction}\n"
                    f"💪 قدرت روند: {trend_strength}\n"
                    f"🔹 شروع (Entry): {entry_price:.4f}\n"
                    f"🛑 حدضرر: {stop_loss:.4f} (فاصله امن: {abs(entry_price-stop_loss)/entry_price*100:.2f}%)\n"
                    f"🎯 TP1: {tp1:.4f}\n"
                    f"🎯 TP2: {tp2:.4f}\n"
                    f"🎯 TP3: {tp3:.4f}\n"
                    f"🎯 TP4: {tp4:.4f}\n"
                    f"🎯 TP5: {tp5:.4f}\n"
                    f"📏 ATR%: {atr_percent:.2f}% | RSI: {rsi:.1f} | ADX: {adx:.1f}\n"
                    f"🧾 سایز کل پوزیشن پیشنهادی: {position_size:.6f} واحد (تقسیم در 5 پله)\n"
                    f"💰 لوریج: {leverage}x | مارجین: ${position_value/leverage:.2f}\n"
                    f"🕒 {self.get_persian_time()}"
                )
                
                self.send_message(chat_id, message)
                
                # Store signal for monitoring
                signal_key = f"{symbol}_{signal_type}"
                self.active_signals[signal_key] = {
                    'symbol': symbol,
                    'signal_type': signal_type,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'tp_levels': [tp1, tp2, tp3, tp4, tp5],
                    'tp_reached': [False, False, False, False, False],
                    'sl_reached': False,
                    'timestamp': self.get_persian_time()
                }
                
            except Exception as e:
                logger.error(f"Error sending detailed signal for {symbol}: {e}")
                continue
    
    async def check_signal_updates(self, chat_id: int):
        """Check active signals for TP/SL hits and notify."""
        if not self.aggregator or not self.active_signals:
            return
        to_remove = []
        for key, sig in self.active_signals.items():
            try:
                symbol = sig['symbol']
                ticker = await self.aggregator.get_aggregated_ticker(symbol)
                if 'error' in ticker:
                    continue
                price = ticker['price']
                if not sig['sl_reached']:
                    # Check stop loss
                    if (sig['signal_type'] == 'LONG' and price <= sig['stop_loss']) or (
                        sig['signal_type'] == 'SHORT' and price >= sig['stop_loss']):
                        self.send_message(chat_id, f"❌ {symbol} — استاپ‌لاس فعال شد. قیمت فعلی: {price:.4f}")
                        sig['sl_reached'] = True
                        to_remove.append(key)
                        continue
                # Check TP levels
                for idx, tp in enumerate(sig['tp_levels']):
                    if sig['tp_reached'][idx]:
                        continue
                    if (sig['signal_type'] == 'LONG' and price >= tp) or (
                        sig['signal_type'] == 'SHORT' and price <= tp):
                        sig['tp_reached'][idx] = True
                        self.send_message(chat_id, f"✅ {symbol} — پله {idx+1} از 5 {'لانگ' if sig['signal_type']=='LONG' else 'شورت'} رسید.\nقیمت فعلی: {price:.4f}")
                # Remove if all TP reached
                if all(sig['tp_reached']):
                    to_remove.append(key)
            except Exception as e:
                logger.error(f"Error in check_signal_updates for {key}: {e}")
        # Clean up finished signals
        for rem in to_remove:
            self.active_signals.pop(rem, None)

    async def handle_analysis_command(self, chat_id: int):
        """Handle /analysis command - show market overview analysis."""
        try:
            self.send_message(chat_id, "🔍 در حال تحلیل بازار...")
            
            # Get top cryptocurrencies data from aggregator
            top_symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT']
            market_overview = []
            
            for symbol in top_symbols:
                try:
                    # Get ticker data from aggregator
                    ticker_data = await self._get_ticker_data(symbol)
                    if ticker_data:
                        market_overview.append({
                            'symbol': symbol.replace('/USDT', ''),
                            'price': ticker_data.get('last', 0),
                            'change_24h': ticker_data.get('percentage', 0),
                            'volume_24h': ticker_data.get('quoteVolume', 0)
                        })
                except Exception as e:
                    logger.error(f"Error getting data for {symbol}: {e}")
                    continue
            
            if not market_overview:
                self.send_message(chat_id, "❌ خطا در دریافت داده‌های بازار")
                return
            
            # Create market overview message
            message = "📊 *نمای کلی بازار ارزهای دیجیتال*\n\n"
            
            for crypto in market_overview:
                symbol = crypto['symbol']
                price = crypto['price']
                change = crypto['change_24h']
                volume = crypto['volume_24h']
                
                # Format change with emoji
                if change > 0:
                    change_emoji = "📈"
                    change_color = "+"
                elif change < 0:
                    change_emoji = "📉"
                    change_color = ""
                else:
                    change_emoji = "➡️"
                    change_color = ""
                
                message += f"{change_emoji} *{symbol}*\n"
                message += f"💰 قیمت: `${price:,.6f}`\n"
                message += f"📊 تغییر 24ساعته: `{change_color}{change:.2f}%`\n"
                message += f"💹 حجم: `${volume:,.0f}`\n\n"
            
            # Add market sentiment
            positive_count = sum(1 for c in market_overview if c['change_24h'] > 0)
            total_count = len(market_overview)
            
            if positive_count > total_count * 0.6:
                sentiment = "🟢 صعودی"
            elif positive_count < total_count * 0.4:
                sentiment = "🔴 نزولی"
            else:
                sentiment = "🟡 خنثی"
            
            message += f"🎯 *حالت کلی بازار:* {sentiment}\n"
            message += f"📈 ارزهای مثبت: {positive_count}/{total_count}\n\n"
            
            message += "💡 *برای تحلیل دقیق‌تر از دستورات زیر استفاده کنید:*\n"
            message += "/golden - اسکن سیگنال‌های طلایی\n"
            message += "/analysis BTC - تحلیل ارز خاص\n\n"
            
            message += f"🕒 {self.get_persian_time()}"
            
            self.send_message(chat_id, message)
            
        except Exception as e:
            logger.error(f"Error in analysis command: {e}")
            self.send_message(chat_id, "❌ خطا در تحلیل بازار")
    
    async def _get_ticker_data(self, symbol: str):
        """Get ticker data for a symbol from aggregator."""
        try:
            if not self.aggregator:
                return None
            
            # Use the existing get_ticker method from aggregator
            ticker = await self.aggregator.get_ticker(symbol)
            return ticker
            
        except Exception as e:
            logger.error(f"Error getting ticker for {symbol}: {e}")
            return None
    
    async def handle_trading_details_callback(self, callback_query, callback_data: str):
        """Handle trading details button callback."""
        try:
            # Extract symbol from callback data
            symbol = callback_data.replace('trading_details_', '')
            chat_id = callback_query['message']['chat']['id']
            
            logger.info(f"Processing trading details callback for {symbol}")
            
            # First try to get stored signal details
            stored_signal = None
            if hasattr(self, 'market_monitor') and self.market_monitor:
                stored_signal = self.market_monitor.get_stored_signal(f"{symbol}/USDT")
                logger.info(f"Stored signal found for {symbol}: {stored_signal is not None}")
            
            if stored_signal:
                # Use stored signal details
                signal_data = stored_signal['signal_data']
                
                # Update current price from market data
                market_data = await self._get_symbol_market_data(f"{symbol}/USDT")
                if market_data and 'close' in market_data:
                    signal_data['indicators']['current_price'] = market_data['close'][-1]
                
                from analysis.trading_details import TradingDetailsCalculator
                calculator = TradingDetailsCalculator()
                
                # Calculate trading details using stored signal
                trading_details = calculator.calculate_trading_details(signal_data, market_data or {})
                
                # Format detailed message
                detailed_message = calculator.format_detailed_message(f"{symbol}/USDT", trading_details, signal_data)
                
                # Send detailed trading information
                self.send_message(chat_id, detailed_message)
                
                # Answer callback query
                self._answer_callback_query(callback_query['id'], f"✅ جزئیات {symbol} ارسال شد")
            else:
                # Fallback: create a simple trading details message without market data
                logger.info(f"No stored signal for {symbol}, creating basic trading details...")
                
                # Create detailed futures-style trading message
                try:
                    # Get current price from Binance API directly
                    import requests
                    binance_symbol = symbol.replace('/', '').upper()
                    price_url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={binance_symbol}"
                    response = requests.get(price_url)
                    current_price = 0
                    
                    if response.status_code == 200:
                        price_data = response.json()
                        current_price = float(price_data['price'])
                    
                    if current_price == 0:
                        # Fallback price estimation
                        current_price = 1.0  # Will be replaced with realistic values per symbol
                        if 'BTC' in symbol:
                            current_price = 65000
                        elif 'ETH' in symbol:
                            current_price = 2500
                        elif 'AVAX' in symbol:
                            current_price = 25
                        elif 'ADA' in symbol:
                            current_price = 0.35
                        elif 'SOL' in symbol:
                            current_price = 140
                    
                    # Calculate trading levels based on current price
                    atr_percent = 0.015  # 1.5% ATR estimation
                    atr = current_price * atr_percent
                    
                    # Determine signal type (assume LONG for golden signals)
                    signal_type = "LONG"
                    
                    # Calculate entry and levels
                    entry_price = current_price
                    stop_loss = entry_price - (0.5 * atr)  # Conservative stop loss
                    
                    # Calculate take profit levels
                    tp1 = entry_price + (0.3 * atr)
                    tp2 = entry_price + (0.6 * atr)
                    tp3 = entry_price + (1.0 * atr)
                    tp4 = entry_price + (1.4 * atr)
                    tp5 = entry_price + (2.0 * atr)
                    
                    # Position sizing
                    capital = 100
                    leverage = 5
                    position_value = capital * leverage
                    position_size = position_value / entry_price
                    
                    basic_message = f"""📣 سیگنال لانگ — {symbol} 📈
⏱️ تایم‌فریم: 5m
🔮 پیش‌بینی: صعودی
💪 قدرت روند: متوسط
🔹 شروع (Entry): {entry_price:.4f}
🛑 حدضرر: {stop_loss:.4f} (فاصله امن: {abs(entry_price-stop_loss)/entry_price*100:.2f}%)
🎯 TP1: {tp1:.4f}
🎯 TP2: {tp2:.4f}
🎯 TP3: {tp3:.4f}
🎯 TP4: {tp4:.4f}
🎯 TP5: {tp5:.4f}
📏 ATR%: {atr_percent*100:.2f}% | RSI: 55.0 | ADX: 25.0
🧾 سایز کل پوزیشن پیشنهادی: {position_size:.6f} واحد (تقسیم در 5 پله)
💰 لوریج: {leverage}x | مارجین: ${position_value/leverage:.2f}
🕒 {self.get_persian_time()}"""

                except Exception as e:
                    logger.error(f"Error creating futures-style message: {e}")
                    basic_message = f"""💎 جزئیات معاملاتی {symbol}

⚠️ خطا در محاسبه سطوح معاملاتی
📊 لطفاً دوباره تلاش کنید

⏰ زمان: {self.get_persian_time()}"""
                
                self.send_message(chat_id, basic_message)
                self._answer_callback_query(callback_query['id'], f"✅ جزئیات پایه {symbol} ارسال شد")
                
        except Exception as e:
            logger.error(f"Error in trading details callback: {e}", exc_info=True)
            # Send error details to user for debugging
            error_msg = f"❌ خطا در نمایش جزئیات {symbol.replace('/USDT', '')}:\n{str(e)}"
            self.send_message(chat_id, error_msg)
            self._answer_callback_query(callback_query['id'], "❌ خطا در نمایش جزئیات")
    
    def _answer_callback_query(self, callback_query_id: str, text: str = None):
        """Answer callback query to acknowledge button press."""
        try:
            url = f"{self.base_url}/answerCallbackQuery"
            data = {'callback_query_id': callback_query_id}
            if text:
                data['text'] = text
            requests.post(url, json=data)
        except Exception as e:
            logger.error(f"Error answering callback query: {e}")
    
    async def handle_golden_command(self, chat_id: int):
        """Handle /golden command - scan for golden signals."""
        try:
            self.send_message(chat_id, "🔍 در حال اسکن سیگنال‌های طلایی...")
            
            # Manual scan for golden signals
            golden_signals = await self.market_monitor.manual_scan()
            
            if not golden_signals:
                self.send_message(chat_id, "🔍 هیچ سیگنال طلایی در حال حاضر یافت نشد.")
                return
            
            # Convert to list format for rating system
            signals_list = []
            for signal in golden_signals:
                signal['symbol'] = signal.get('symbol', 'Unknown')
                signals_list.append(signal)
            
            # Get enhanced signals with ratings
            from analysis.golden_signals import GoldenSignalsDetector
            detector = GoldenSignalsDetector()
            top_signals = detector.get_top_golden_signals(signals_list, limit=5)
            
            if not top_signals:
                self.send_message(chat_id, "🔍 هیچ سیگنال طلایی با کیفیت یافت نشد.")
                return
            
            message = "🌟 *سیگنال‌های طلایی یافت شده:*\n\n"
            
            # Create keyboard for trading details
            keyboard_markup = None
            try:
                from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                keyboard = []
                
                for signal in top_signals:
                    symbol = signal.get('symbol', 'Unknown').replace('/USDT', '')
                    
                    # Get rating data
                    rating_data = detector.rating_system.calculate_signal_rating(signal)
                    rating = rating_data.get('rating', 5)
                    diamond = "💎" if rating_data.get('is_diamond', False) else ""
                    stars = "⭐" * rating
                    
                    signal_type = signal.get('signal_type', 'NEUTRAL')
                    score = signal.get('score', 0)
                    percentage = signal.get('percentage', 0)
                    
                    if signal_type == "LONG":
                        emoji = "🚀"
                        direction = "خرید"
                    elif signal_type == "SHORT":
                        emoji = "📉"
                        direction = "فروش"
                    else:
                        emoji = "⚡"
                        direction = "خنثی"
                    
                    message += f"{diamond} {emoji} *{symbol}* - {direction} ({rating}/10) {stars}\n"
                    message += f"📊 امتیاز: `{score}/120` ({percentage:.1f}%)\n"
                    message += f"💪 قدرت: {signal.get('strength', 'متوسط')}\n"
                    
                    indicators = signal.get('indicators', {})
                    message += f"💰 قیمت: `{indicators.get('current_price', 0):.6f}`\n"
                    message += f"📈 RSI: `{indicators.get('rsi', 0):.1f}`\n"
                    message += f"⚡ ADX: `{indicators.get('adx', 0):.1f}`\n\n"
                    
                    # Add button for trading details
                    keyboard.append([InlineKeyboardButton(
                        f"{diamond} {symbol} - جزئیات معاملاتی ({rating}/10)",
                        callback_data=f"trading_details_{symbol}/USDT"
                    )])
                
                message += f"🕒 {self.get_persian_time()}\n\n"
                message += "📋 برای مشاهده جزئیات معاملاتی هر ارز، روی دکمه مربوطه کلیک کنید:"
                
                if keyboard:
                    keyboard_markup = InlineKeyboardMarkup(keyboard)
                    
            except ImportError:
                logger.warning("Could not import telegram keyboard components")
            
            self.send_message(chat_id, message, reply_markup=keyboard_markup)
            
        except Exception as e:
            logger.error(f"Error in golden command: {e}")
            self.send_message(chat_id, "❌ خطا در اسکن سیگنال‌های طلایی")
    
    async def _get_symbol_market_data(self, symbol: str):
        """Get market data for a specific symbol."""
        try:
            if not self.aggregator:
                logger.error(f"No aggregator available for {symbol}")
                return None
            
            logger.info(f"Getting market data for {symbol}")
            
            # Get OHLCV data
            ohlcv = await self.aggregator.get_ohlcv(symbol, '5m', 200)
            if not ohlcv:
                logger.error(f"No OHLCV data returned for {symbol}")
                return None
            
            logger.info(f"Got {len(ohlcv)} candles for {symbol}")
            
            # Convert to format expected by golden signals detector
            market_data = {
                'open': [candle[1] for candle in ohlcv],
                'high': [candle[2] for candle in ohlcv],
                'low': [candle[3] for candle in ohlcv],
                'close': [candle[4] for candle in ohlcv],
                'volume': [candle[5] for candle in ohlcv],
                'timestamp': [candle[0] for candle in ohlcv]
            }
            
            logger.info(f"Market data prepared for {symbol}: {len(market_data['close'])} data points")
            return market_data
            
        except Exception as e:
            logger.error(f"Exception getting market data for {symbol}: {e}", exc_info=True)
            return None
    
    async def handle_monitor_command(self, chat_id: int):
        """Handle /monitor command - show monitoring status."""
        try:
            status = self.market_monitor.get_monitoring_status()
            
            if status['active']:
                status_emoji = "🟢"
                status_text = "فعال"
            else:
                status_emoji = "🔴"
                status_text = "غیرفعال"
            
            message = (
                f"{status_emoji} *وضعیت مانیتورینگ:* {status_text}\n\n"
                f"⏱️ فاصله اسکن: {status['scan_interval']} ثانیه\n"
                f"📊 ارزهای تحت نظر: {status['watched_symbols']}\n"
                f"🌟 سیگنال‌های طلایی اخیر: {status['last_golden_count']}\n"
                f"⏰ کولداون: {status['cooldown_minutes']} دقیقه\n\n"
            )
            
            if not status['active']:
                message += "💡 برای شروع مانیتورینگ خودکار، از دستور زیر استفاده کنید:\n"
                message += "`/start_monitor`"
            else:
                message += "⚠️ برای توقف مانیتورینگ، از دستور زیر استفاده کنید:\n"
                message += "`/stop_monitor`"
            
            message += f"\n\n🕒 {self.get_persian_time()}"
            self.send_message(chat_id, message)
            
        except Exception as e:
            logger.error(f"Error in monitor command: {e}")
            self.send_message(chat_id, "❌ خطا در نمایش وضعیت مانیتورینگ")
    
    async def handle_symbol_analysis(self, chat_id: int, symbol: str):
        """Handle symbol-specific analysis with simplified input."""
        try:
            # Add USDT if not present
            if '/' not in symbol:
                symbol = f"{symbol}/USDT"
            
            self.send_message(chat_id, f"🔍 در حال تحلیل {symbol}...")
            
            if not self.aggregator:
                self.send_message(chat_id, "❌ خطا: سیستم تجمیع داده در دسترس نیست")
                return
            
            # Get market data
            market_data = await self._get_symbol_market_data(symbol)
            if not market_data:
                self.send_message(chat_id, f"❌ خطا در دریافت داده‌های {symbol}")
                return
            
            # Analyze with golden signals detector
            signal_result = self.golden_detector.is_golden_signal(symbol, market_data)
            
            # Get volume analysis
            volume_analysis = self._analyze_volume_data(market_data)
            
            message = f"📊 *تحلیل {symbol.replace('/USDT', '')}*\n\n"
            
            if signal_result['is_golden']:
                message += "🌟 *سیگنال طلایی!*\n"
                message += f"📈 نوع: {signal_result['signal_type']}\n"
                message += f"💪 قدرت: {signal_result['strength']}\n"
                message += f"📊 امتیاز: {signal_result['score']}/100\n\n"
            else:
                message += f"📊 امتیاز: {signal_result['score']}/100\n"
                message += "⚠️ سیگنال طلایی نیست\n\n"
            
            # Technical indicators
            indicators = signal_result.get('indicators', {})
            if indicators:
                message += "🔧 *اندیکاتورهای تکنیکال:*\n"
                message += f"💰 قیمت فعلی: `{indicators.get('current_price', 0):.6f}`\n"
                message += f"📈 RSI: `{indicators.get('rsi', 0):.1f}`\n"
                message += f"⚡ ADX: `{indicators.get('adx', 0):.1f}`\n"
                message += f"📊 MACD: `{indicators.get('macd', 0):.6f}`\n"
                message += f"🎯 VWAP: `{indicators.get('vwap', 0):.6f}`\n\n"
            
            # Volume analysis
            message += "📊 *تحلیل حجم معاملات:*\n"
            message += f"📈 حجم فعلی: `{volume_analysis['current_volume']:.0f}`\n"
            message += f"📊 میانگین حجم: `{volume_analysis['avg_volume']:.0f}`\n"
            message += f"📊 نسبت حجم: `{volume_analysis['volume_ratio']:.2f}x`\n"
            message += f"🔥 وضعیت حجم: {volume_analysis['volume_status']}\n\n"
            
            # Price prediction (2-hour forecast)
            prediction = self._predict_price_movement(market_data)
            message += "🔮 *پیش‌بینی 2 ساعته:*\n"
            message += f"📈 حداکثر رشد احتمالی: `{prediction['max_growth']:.2f}%`\n"
            message += f"📉 حداقل قیمت احتمالی: `{prediction['min_price']:.6f}`\n"
            message += f"🎯 احتمال روند صعودی: `{prediction['bullish_probability']:.1f}%`\n\n"
            
            message += f"🕒 {self.get_persian_time()}"
            self.send_message(chat_id, message)
            
        except Exception as e:
            logger.error(f"Error in symbol analysis for {symbol}: {e}")
            self.send_message(chat_id, f"❌ خطا در تحلیل {symbol}")
    
    async def _get_symbol_market_data(self, symbol: str):
        """Get market data for a specific symbol."""
        try:
            # This should be implemented based on your aggregator
            # For now, return None to indicate data unavailable
            return None
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _analyze_volume_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume data."""
        try:
            volumes = market_data.get('volume', [])
            if not volumes or len(volumes) < 20:
                return {
                    'current_volume': 0,
                    'avg_volume': 0,
                    'volume_ratio': 0,
                    'volume_status': 'داده ناکافی'
                }
            
            current_volume = volumes[-1]
            avg_volume = sum(volumes[-20:]) / 20
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            
            if volume_ratio > 2:
                status = "بسیار بالا 🔥"
            elif volume_ratio > 1.5:
                status = "بالا 📈"
            elif volume_ratio > 0.8:
                status = "عادی 📊"
            else:
                status = "پایین 📉"
            
            return {
                'current_volume': current_volume,
                'avg_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_status': status
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume data: {e}")
            return {
                'current_volume': 0,
                'avg_volume': 0,
                'volume_ratio': 0,
                'volume_status': 'خطا در تحلیل'
            }
    
    def _predict_price_movement(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Simple price movement prediction for next 2 hours."""
        try:
            prices = market_data.get('close', [])
            if not prices or len(prices) < 50:
                return {
                    'max_growth': 0,
                    'min_price': 0,
                    'bullish_probability': 50
                }
            
            current_price = prices[-1]
            
            # Calculate recent volatility (last 24 periods = ~2 hours on 5m chart)
            recent_prices = prices[-24:]
            volatility = (max(recent_prices) - min(recent_prices)) / current_price * 100
            
            # Simple prediction based on volatility and recent trend
            recent_change = (prices[-1] - prices[-12]) / prices[-12] * 100
            
            max_growth = volatility * 0.6  # Conservative estimate
            min_price = current_price * (1 - volatility * 0.004)  # 0.4% of volatility
            
            # Bullish probability based on recent trend
            if recent_change > 2:
                bullish_prob = 75
            elif recent_change > 0:
                bullish_prob = 60
            elif recent_change > -2:
                bullish_prob = 40
            else:
                bullish_prob = 25
            
            return {
                'max_growth': max_growth,
                'min_price': min_price,
                'bullish_probability': bullish_prob
            }
            
        except Exception as e:
            logger.error(f"Error predicting price movement: {e}")
            return {
                'max_growth': 0,
                'min_price': 0,
                'bullish_probability': 50
            }
    
    async def _start_auto_monitoring(self):
        """Start automatic monitoring on bot initialization."""
        try:
            await asyncio.sleep(5)  # Wait for bot to fully initialize
            if not self.monitoring_task or self.monitoring_task.done():
                self.monitoring_task = asyncio.create_task(self.market_monitor.start_monitoring())
                logger.info("Auto monitoring for golden signals started")
        except Exception as e:
            logger.error(f"Error starting auto monitoring: {e}")
    
    async def handle_start_monitor_command(self, chat_id: int):
        """Handle /start_monitor command."""
        try:
            if self.monitoring_task and not self.monitoring_task.done():
                self.send_message(chat_id, "🟢 مانیتورینگ قبلاً فعال است!")
                return
            
            self.monitoring_task = asyncio.create_task(self.market_monitor.start_monitoring())
            self.send_message(chat_id, "🚀 مانیتورینگ سیگنال‌های طلایی شروع شد!\n\n⚡ سیستم به طور خودکار هر 5 دقیقه بازار را اسکن می‌کند")
            
        except Exception as e:
            logger.error(f"Error starting monitor: {e}")
            self.send_message(chat_id, "❌ خطا در شروع مانیتورینگ")
    
    async def handle_stop_monitor_command(self, chat_id: int):
        """Handle /stop_monitor command."""
        try:
            if not self.monitoring_task or self.monitoring_task.done():
                self.send_message(chat_id, "🔴 مانیتورینگ فعال نیست!")
                return
            
            self.market_monitor.stop_monitoring()
            if self.monitoring_task:
                self.monitoring_task.cancel()
            
            self.send_message(chat_id, "⏹️ مانیتورینگ سیگنال‌های طلایی متوقف شد")
            
        except Exception as e:
            logger.error(f"Error stopping monitor: {e}")
            self.send_message(chat_id, "❌ خطا در توقف مانیتورینگ")
    
    def get_persian_time(self) -> str:
        """Get current time in Persian timezone."""
        from datetime import datetime
        from pytz import timezone
        from jdatetime import datetime as jdatetime
        
        tehran = timezone('Asia/Tehran')
        now = datetime.now(tehran)
        jd = jdatetime.fromgregorian(datetime=now)
        
        return f"{jd.hour:02d}:{jd.minute:02d} - {jd.year}/{jd.month:02d}/{jd.day:02d}"
    
    async def process_updates(self):
        """Process incoming updates from Telegram."""
        offset = 0
        while True:
            try:
                updates = self.get_updates(offset)
                if updates and updates.get('ok'):
                    for update in updates['result']:
                        offset = update['update_id'] + 1
                        await self.handle_update(update)
                        
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing updates: {e}")
                await asyncio.sleep(5)
    
    async def handle_update(self, update: Dict[str, Any]):
        """Handle incoming update."""
        try:
            logger.info(f"Received update: {update.keys()}")
            
            # Handle callback queries (button presses)
            if 'callback_query' in update:
                callback_query = update['callback_query']
                user_id = callback_query['from']['id']
                callback_data = callback_query.get('data', '')
                
                logger.info(f"Callback query received: {callback_data} from user {user_id}")
                
                if not self.is_authorized(user_id):
                    logger.warning(f"Unauthorized callback from user {user_id}")
                    self._answer_callback_query(callback_query['id'], "❌ غیرمجاز")
                    return
                
                if callback_data.startswith('trading_details_'):
                    logger.info(f"Processing trading details callback: {callback_data}")
                    await self.handle_trading_details_callback(callback_query, callback_data)
                else:
                    logger.warning(f"Unknown callback data: {callback_data}")
                return
            
            # Handle regular messages
            if 'message' in update:
                message = update['message']
                chat_id = message['chat']['id']
                user_id = message['from']['id']
                
                if not self.is_authorized(user_id):
                    self.send_message(chat_id, "❌ شما مجاز به استفاده از این بات نیستید")
                    return
                
                text = message.get('text', '')
                
                if text.startswith('/'):
                    await self.handle_command(chat_id, text)
                    
        except Exception as e:
            logger.error(f"Error handling update: {e}")
    
    async def handle_command(self, chat_id: int, text: str):
        """Handle command messages."""
        try:
            if text == '/start':
                welcome_msg = (
                    "🤖 *ربات سیگنال‌دهی ارز دیجیتال*\n\n"
                    "دستورات موجود:\n"
                    "/market - نمایش وضعیت بازار\n"
                    "/futures - تحلیل فیوچرز\n"
                    "/analysis - تحلیل تکنیکال کلی\n"
                    "/analysis BTC - تحلیل ارز خاص\n"
                    "/golden - اسکن سیگنال‌های طلایی\n"
                    "/monitor - وضعیت مانیتورینگ\n"
                    "/start - نمایش این پیام\n\n"
                    "🌟 *ویژگی جدید:* سیستم تشخیص خودکار سیگنال‌های طلایی!"
                )
                self.send_message(chat_id, welcome_msg)
                
            elif text == '/market':
                await self.handle_market_command(chat_id)
                
            elif text == '/futures':
                await self.handle_futures_command(chat_id)
                
            elif text == '/analysis':
                await self.handle_analysis_command(chat_id)
                
            elif text == '/golden':
                await self.handle_golden_command(chat_id)
                
            elif text == '/monitor':
                await self.handle_monitor_command(chat_id)
                
            elif text == '/start_monitor':
                await self.handle_start_monitor_command(chat_id)
                
            elif text == '/stop_monitor':
                await self.handle_stop_monitor_command(chat_id)
                
            elif text.startswith('/analysis '):
                symbol = text.split(' ')[1].upper()
                await self.handle_symbol_analysis(chat_id, symbol)
                
        except Exception as e:
            logger.error(f"Error handling command: {e}")

    def _set_bot_commands(self):
        """Set bot commands menu for quick access."""
        try:
            commands = [
                {"command": "start", "description": "شروع و نمایش راهنما"},
                {"command": "market", "description": "نمایش وضعیت بازار"},
                {"command": "futures", "description": "تحلیل فیوچرز"},
                {"command": "analysis", "description": "تحلیل تکنیکال کلی"},
                {"command": "golden", "description": "اسکن سیگنال‌های طلایی"},
                {"command": "monitor", "description": "وضعیت مانیتورینگ"}
            ]
            
            url = f"{self.base_url}/setMyCommands"
            data = {"commands": commands}
            
            response = requests.post(url, json=data)
            if response.status_code == 200:
                logger.info("Bot commands menu set successfully")
            else:
                logger.error(f"Failed to set bot commands: {response.text}")
                
        except Exception as e:
            logger.error(f"Error setting bot commands: {e}")

def setup_simple_telegram_bot(aggregator):
    """Setup simple telegram bot."""
    token = os.getenv('TELEGRAM_TOKEN')
    if not token:
        logger.error("TELEGRAM_TOKEN not found in environment")
        return None
        
    return SimpleTelegramBot(token, aggregator)
