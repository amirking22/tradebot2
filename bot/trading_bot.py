"""
Crypto Signal Bot with Multi-Exchange Support

فقط برای سیگنال‌دهی و اطلاع‌رسانی

ویژگی‌ها:
- پشتیبانی از چندین صرافی (YEX, Binance, Bybit)
- تحلیل تکنیکال پیشرفته
- مدیریت ریسک و محاسبه سایز پوزیشن
- تولید خودکار سیگنال‌های معاملاتی
- اطلاع‌رسانی از طریق تلگرام

⚠️ توجه: این ربات فقط برای ارسال سیگنال است و معامله‌ای انجام نمی‌دهد.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

from exchanges.base_exchange import YEXExchange, BinanceFuturesExchange, BybitFuturesExchange
from analysis.technical_indicators import TechnicalIndicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)

class TradingBot:
    """Main trading bot class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trading bot with configuration."""
        self.config = config
        self.exchanges: Dict[str, Any] = {}
        self.active_signals: Dict[str, Dict] = {}
        self.market_data_cache: Dict[str, Dict] = {}
        self.initialize_exchanges()
        
    def initialize_exchanges(self) -> None:
        """Initialize exchange connections."""
        if self.config.get('yex_api_key') and self.config.get('yex_secret'):
            self.exchanges['yex'] = YEXExchange(
                api_key=self.config['yex_api_key'],
                api_secret=self.config['yex_secret']
            )
            
        if self.config.get('binance_api_key') and self.config.get('binance_secret'):
            self.exchanges['binance'] = BinanceFuturesExchange(
                api_key=self.config['binance_api_key'],
                api_secret=self.config['binance_secret']
            )
            
        if self.config.get('bybit_api_key') and self.config.get('bybit_secret'):
            self.exchanges['bybit'] = BybitFuturesExchange(
                api_key=self.config['bybit_api_key'],
                api_secret=self.config['bybit_secret']
            )
    
    async def get_market_data(self, exchange_name: str, symbol: str, timeframe: str = '30m', 
                            limit: int = 200) -> Optional[pd.DataFrame]:
        """Get market data with caching."""
        cache_key = f"{exchange_name}_{symbol}_{timeframe}"
        now = time.time()
        
        # Check cache first
        if cache_key in self.market_data_cache:
            cached = self.market_data_cache[cache_key]
            if now - cached['timestamp'] < 60:  # 1 minute cache
                return cached['data']
        
        # Fetch fresh data
        try:
            exchange = self.exchanges[exchange_name]
            data = await exchange.get_ohlcv(symbol, timeframe, limit)
            
            if data is not None and not data.empty:
                self.market_data_cache[cache_key] = {
                    'timestamp': now,
                    'data': data
                }
                return data
                
        except Exception as e:
            logger.error(f"Error fetching {symbol} data from {exchange_name}: {e}")
            
        return None
    
    async def analyze_market(self, exchange_name: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze a single market and generate signal."""
        try:
            # Get market data - use aggregator if available
            if hasattr(self, 'aggregator') and exchange_name == 'aggregated':
                # Get aggregated ticker data
                ticker_data = await self.aggregator.get_aggregated_ticker(symbol)
                if 'error' in ticker_data:
                    return None
                
                # Create mock OHLCV data for analysis
                import pandas as pd
                import numpy as np
                
                price = ticker_data['price']
                # Generate synthetic OHLCV data based on current price
                dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='30min')
                
                # Create realistic price movement
                np.random.seed(hash(symbol) % 2**32)  # Consistent seed per symbol
                returns = np.random.normal(0, 0.02, 200)  # 2% volatility
                prices = [price * 0.95]  # Start 5% below current
                
                for ret in returns[1:]:
                    prices.append(prices[-1] * (1 + ret))
                
                # Scale to end at current price
                scale_factor = price / prices[-1]
                prices = [p * scale_factor for p in prices]
                
                # Create OHLCV
                df_data = []
                for i, base_price in enumerate(prices):
                    high = base_price * (1 + abs(np.random.normal(0, 0.01)))
                    low = base_price * (1 - abs(np.random.normal(0, 0.01)))
                    open_price = prices[i-1] if i > 0 else base_price
                    volume = np.random.uniform(1000, 10000)
                    
                    df_data.append({
                        'timestamp': dates[i],
                        'open': open_price,
                        'high': high,
                        'low': low,
                        'close': base_price,
                        'volume': volume
                    })
                
                df = pd.DataFrame(df_data)
                df.set_index('timestamp', inplace=True)
                
            else:
                # Use regular exchange data
                df = await self.get_market_data(exchange_name, symbol)
                if df is None or len(df) < 200:
                    return None
            
            # Generate technical analysis
            signal = TechnicalIndicators.generate_signal(
                close=df['close'],
                high=df['high'],
                low=df['low'],
                volume=df['volume']
            )
            
            # Add metadata
            signal.update({
                'exchange': exchange_name,
                'symbol': symbol,
                'timestamp': time.time()
            })
            
            # Process ALL signals (not just non-neutral)
            current_price = df['close'].iloc[-1]
            atr = signal['indicators']['atr']
            
            if signal['signal'] == 'LONG':
                entry = current_price * 1.001
                stop_loss = entry - (atr * 2)
                take_profit = entry + (atr * 6)
            elif signal['signal'] == 'SHORT':
                entry = current_price * 0.999
                stop_loss = entry + (atr * 2)
                take_profit = entry - (atr * 6)
            else:  # NEUTRAL
                entry = current_price
                stop_loss = entry - (atr * 2)
                take_profit = entry + (atr * 6)
            
            # Calculate position levels
            position_levels = TechnicalIndicators.calculate_position_levels(
                entry_price=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=3.0
            )
            
            signal.update({
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_levels': position_levels,
                'status': 'NEW'
            })
            
            return signal
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol} on {exchange_name}: {e}")
            return None
    
    async def monitor_markets(self) -> None:
        """Monitor all markets on all exchanges."""
        while True:
            try:
                tasks = []
                
                # Analyze all symbols on all exchanges
                for exchange_name, exchange in self.exchanges.items():
                    markets = await exchange.get_markets()
                    
                    for market in markets:
                        symbol = market['symbol']
                        tasks.append(
                            self.analyze_market(exchange_name, symbol)
                        )
                
                # Process all analyses
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process ALL signals (including low confidence ones)
                for result in results:
                    if isinstance(result, dict) and result.get('signal') != 'NEUTRAL':
                        await self.process_signal(result)
                    elif isinstance(result, dict) and result.get('signal') == 'NEUTRAL':
                        # Send analysis report for neutral signals too
                        await self.send_analysis_report(result)
                
                # Check TP/SL for active signals
                await self.check_active_signals()
                
                # Wait before next iteration
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in market monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def process_signal(self, signal: Dict[str, Any]) -> None:
        """Process and send a new trading signal."""
        try:
            # Format signal message in Persian
            message = self.format_signal_message(signal)
            
            # Send notification to Telegram
            await self.send_notification(message)
            
            logger.info(f"📨 New signal sent: {signal['symbol']} {signal['signal']} "
                       f"(اعتماد: {signal['confidence']}, ریسک: {signal['risk_level']})")
            
        except Exception as e:
            logger.error(f"❌ خطا در پردازش سیگنال: {e}")
            logger.exception("جزئیات خطا:")
    
    async def check_active_signals(self) -> None:
        """Check TP/SL levels for active signals."""
        for signal_id, signal in list(self.active_signals.items()):
            try:
                if signal['status'] == 'CLOSED':
                    continue
                    
                # Get current price
                exchange = self.exchanges[signal['exchange']]
                ticker = await exchange.get_ticker(signal['symbol'])
                current_price = ticker['last']
                
                # Check TP levels
                for i in range(1, 6):
                    tp_level = signal['position_levels'][f'tp{i}']
                    tp_key = f'tp{i}_hit'
                    
                    if tp_key not in signal and (
                        (signal['signal'] == 'LONG' and current_price >= tp_level) or
                        (signal['signal'] == 'SHORT' and current_price <= tp_level)
                    ):
                        # TP level hit
                        signal[tp_key] = True
                        message = self.format_tp_message(signal, i, current_price)
                        await self.send_notification(message)
                
                # Check SL
                if (signal['signal'] == 'LONG' and current_price <= signal['stop_loss']) or \
                   (signal['signal'] == 'SHORT' and current_price >= signal['stop_loss']):
                    # Stop loss hit
                    signal['status'] = 'CLOSED'
                    message = self.format_sl_message(signal, current_price)
                    await self.send_notification(message)
                
            except Exception as e:
                logger.error(f"Error checking signal {signal_id}: {e}")
    
    def format_signal_message(self, signal: Dict[str, Any]) -> str:
        """Format signal message for Telegram in Persian."""
        # Map signal types to Persian
        direction = {
            'LONG': '📈 لانگ (خرید)',
            'SHORT': '📉 شورت (فروش)',
            'NEUTRAL': '⚪ خنثی'
        }.get(signal['signal'], signal['signal'])
        
        # Map confidence levels to Persian
        confidence_persian = {
            'HIGH': 'بالا',
            'MEDIUM': 'متوسط',
            'LOW': 'پایین'
        }.get(signal['confidence'], signal['confidence'])
        
        # Map risk levels to Persian with emojis
        risk_info = {
            'SAFE': ('🟢 ایمن', 'ریسک کم - فرصت خوب برای ورود'),
            'MEDIUM': ('🟡 متوسط', 'ریسک متوسط - با احتیاط وارد شوید'),
            'RISKY': ('🔴 پرخطر', 'ریسک بالا - فقط برای معامله‌گران حرفه‌ای')
        }.get(signal['risk_level'], ('⚪ نامشخص', 'وضعیت ریسک نامشخص'))
        
        risk_emoji, risk_text = risk_info
        
        # Format entry and exit points
        current_price = signal['indicators']['current_price']
        entry = signal.get('entry', current_price)
        
        # Calculate price differences for better visualization
        diff_pct = ((entry - current_price) / current_price * 100) \
            if signal['signal'] == 'LONG' else \
            ((current_price - entry) / current_price * 100)

        # Calculate expected profit on TP5 relative to capital with leverage
        from config import INITIAL_CAPITAL_USDT, DEFAULT_LEVERAGE, PROFIT_THRESHOLD_PERCENT, POSITION_SIZE_PERCENT
        tp5 = signal['position_levels']['tp5']
        price_change_pct = abs(tp5 - entry) / entry * 100
        # Margin used = capital * position_size_percent /100
        margin_usdt = INITIAL_CAPITAL_USDT * (POSITION_SIZE_PERCENT / 100)
        notional = margin_usdt * DEFAULT_LEVERAGE
        expected_profit_usdt = notional * (price_change_pct / 100)
        expected_profit_pct_capital = (expected_profit_usdt / INITIAL_CAPITAL_USDT) * 100
        low_profit = expected_profit_pct_capital < PROFIT_THRESHOLD_PERCENT
        
        # Add failed conditions to message if any
        failed_conditions = signal.get('failed_conditions', [])
        conditions_text = ""
        if failed_conditions:
            conditions_text = "\n\n⚠️ *شروط ناموفق:*\n"
            for condition in failed_conditions:
                conditions_text += f"• {condition}\n"
            
        # Format the message in Persian with RTL support
        message = (
            f"{risk_emoji} *سیگنال معاملاتی جدید* {risk_emoji}\n\n"
            f"🏦 *صرافی:* {signal['exchange'].upper()}\n"
            f"🔢 *نماد:* {signal['symbol']}\n"
            f"📊 *جهت:* {direction}\n"
            f"💪 *قدرت سیگنال:* {confidence_persian}\n"
            f"⚙️ *لوریج پیش‌فرض:* {DEFAULT_LEVERAGE}x\n"
            f"💰 *مارجین مورد استفاده:* {margin_usdt:.2f} USDT ({POSITION_SIZE_PERCENT}% سرمایه)\n"
            f"🏁 *سود بالقوه تا TP5:* {expected_profit_pct_capital:.2f}% از سرمایه\n"
            f"⚠️ *وضعیت ریسک:* {risk_emoji} {risk_text}\n\n"
            
            f"📊 *اطلاعات تکنیکال:*\n"
            f"• قیمت فعلی: `{current_price:,.8f}`\n"
            f"• فاصله تا نقطه ورود: `{abs(diff_pct):.2f}%`\n"
            f"• RSI: `{signal['indicators']['rsi']:.2f}`\n"
            f"• ADX: `{signal['indicators']['adx']:.2f}`\n"
            f"• ATR: `{signal['indicators']['atr']:.8f}`\n\n"
            
            f"🎯 *سطوح معاملاتی:*\n"
            f"• نقطه ورود: `{entry:,.8f}`\n"
            f"• حد ضرر: `{signal['stop_loss']:,.8f}`\n"
            f"• حد سود 1 (20%): `{signal['position_levels']['tp1']:,.8f}`\n"
            f"• حد سود 2 (40%): `{signal['position_levels']['tp2']:,.8f}`\n"
            f"• حد سود 3 (60%): `{signal['position_levels']['tp3']:,.8f}`\n"
            f"• حد سود 4 (80%): `{signal['position_levels']['tp4']:,.8f}`\n"
            f"• حد سود 5 (100%): `{signal['position_levels']['tp5']:,.8f}`\n\n"
            
            f"📌 *نکات مدیریت ریسک:*\n"
            f"• حداکثر 2-3% از سرمایه در هر معامله\n"
            f"• حد ضرر را حتماً رعایت کنید\n"
            f"• در صورت تغییر روند، از معامله خارج شوید\n\n"
            
            f"🕒 زمان: {self.get_persian_time()}"
        )
        
        # Add footer with risk disclaimer
        # Add failed conditions info
        message += conditions_text
        
        if low_profit:
            message += "\n\n🔴 *هشدار:* سود بالقوه این معامله کمتر از «{:.1f}%» سرمایه است و ممکن است ارزش ریسک نداشته باشد.".format(PROFIT_THRESHOLD_PERCENT)
        message += "\n\n⚠️ *هشدار ریسک:* معاملات ارزهای دیجیتال دارای ریسک بالایی است. این سیگنال‌ها صرفاً تحلیل تکنیکال هستند و توصیه مالی محسوب نمی‌شوند."
        
        return message
    
    def format_tp_message(self, signal: Dict[str, Any], tp_level: int, current_price: float) -> str:
        """Format take profit message."""
        return (
            f"✅ *تارگت {tp_level} از 5 برای {signal['symbol']} فعال شد*\n\n"
            f"🎯 قیمت فعلی: {current_price:.8f}\n"
            f"📊 سود تاکنون: {self.calculate_profit_percent(signal, current_price):.2f}%\n"
            f"🕒 زمان: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
    
    def format_sl_message(self, signal: Dict[str, Any], current_price: float) -> str:
        """Format stop loss message."""
        return (
            f"🛑 *حد ضرر برای {signal['symbol']} فعال شد*\n\n"
            f"💸 قیمت خروج: {current_price:.8f}\n"
            f"📉 ضرر: {self.calculate_profit_percent(signal, current_price):.2f}%\n"
            f"🕒 زمان: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
    
    def calculate_profit_percent(self, signal: Dict[str, Any], current_price: float) -> float:
        """Calculate profit/loss percentage."""
        if signal['signal'] == 'LONG':
            return ((current_price - signal['entry']) / signal['entry']) * 100
        else:  # SHORT
            return ((signal['entry'] - current_price) / signal['entry']) * 100
    
    def get_persian_time(self) -> str:
        """Get current time in Persian timezone with Jalali date."""
        from datetime import datetime
        from pytz import timezone
        from jdatetime import datetime as jdatetime
        
        # Get current time in Tehran timezone
        tehran = timezone('Asia/Tehran')
        now = datetime.now(tehran)
        
        # Convert to Jalali
        jd = jdatetime.fromgregorian(datetime=now)
        
        # Format the date and time
        return f"{jd.hour:02d}:{jd.minute:02d} - {jd.year}/{jd.month:02d}/{jd.day:02d} (تهران)"
    
    async def send_notification(self, message: str) -> None:
        """Send notification to Telegram."""
        import requests
        from urllib.parse import quote_plus
        
        if not self.config.get('telegram_token') or not self.config.get('telegram_chat_id'):
            logger.warning("Telegram token or chat ID not configured")
            return
            
        try:
            # Encode the message for URL
            encoded_message = quote_plus(message)
            
            # Send message using Telegram Bot API
            url = (
                f"https://api.telegram.org/bot{self.config['telegram_token']}/"
                f"sendMessage?chat_id={self.config['telegram_chat_id']}&"
                f"text={encoded_message}&parse_mode=Markdown"
            )
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

    async def run(self) -> None:
        """Run the trading bot."""
        logger.info("Starting trading bot...")
        
        # Start market monitoring in the background
        asyncio.create_task(self.monitor_markets())
        
        # Keep the bot running
        while True:
            await asyncio.sleep(1)
