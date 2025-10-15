"""
Market Monitor System
سیستم مانیتورینگ مداوم بازار برای تشخیص فرصت‌های طلایی
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
from .golden_signals import GoldenSignalsDetector

logger = logging.getLogger(__name__)

class MarketMonitor:
    """کلاس مانیتورینگ مداوم بازار"""
    
    def __init__(self, exchange_aggregator=None, golden_detector=None, telegram_bot=None, aggregator=None):
        self.exchange_aggregator = exchange_aggregator or aggregator
        self.aggregator = self.exchange_aggregator  # برای سازگاری
        self.golden_detector = golden_detector or GoldenSignalsDetector(self.exchange_aggregator)
        self.telegram_bot = telegram_bot
        self.last_notifications = {}  # برای جلوگیری از ارسال مکرر
        self.notification_cooldown = 300  # 5 دقیقه کولداون
        self.stored_signals = {}  # ذخیره جزئیات سیگنال‌ها برای دکمه‌های جزئیات برای هر سیگنال
        self.monitoring_active = False
        self.scan_interval = 300  # 5 دقیقه
        self.last_golden_signals = {}  # ذخیره آخرین سیگنال‌های طلایی
        
        # لیست ارزهای مورد نظر برای اسکن
        self.watch_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'AVAX/USDT',
            'LINK/USDT', 'UNI/USDT', 'LTC/USDT', 'BCH/USDT', 'ATOM/USDT',
            'FIL/USDT', 'TRX/USDT', 'ETC/USDT', 'XLM/USDT', 'VET/USDT'
        ]
    
    async def start_monitoring(self):
        """شروع مانیتورینگ مداوم"""
        if self.monitoring_active:
            logger.info("Monitoring is already active")
            return
        
        self.monitoring_active = True
        logger.info("Starting continuous market monitoring for golden signals")
        
        while self.monitoring_active:
            try:
                await self._scan_market_for_golden_signals()
                await asyncio.sleep(self.scan_interval)
            except Exception as e:
                logger.error(f"Error in market monitoring: {e}")
                await asyncio.sleep(60)  # استراحت کوتاه در صورت خطا
    
    def stop_monitoring(self):
        """توقف مانیتورینگ"""
        self.monitoring_active = False
        logger.info("Market monitoring stopped")
    
    async def _scan_market_for_golden_signals(self):
        """اسکن بازار برای یافتن سیگنال‌های طلایی"""
        try:
            if not self.aggregator:
                logger.warning("Aggregator در دسترس نیست")
                return
            
            logger.info(f"Starting scan of {len(self.watch_symbols)} symbols for golden signals")
            
            golden_signals = []
            
            for symbol in self.watch_symbols:
                try:
                    # دریافت داده‌های بازار
                    market_data = await self._get_market_data(symbol)
                    if not market_data:
                        continue
                    
                    # بررسی سیگنال طلایی
                    signal_result = self.golden_detector.is_golden_signal(symbol, market_data)
                    
                    if signal_result['is_golden']:
                        # بررسی کولداون
                        if self._should_notify(symbol, signal_result):
                            golden_signals.append(signal_result)
                            self.last_golden_signals[symbol] = {
                                'timestamp': datetime.now(),
                                'score': signal_result['score'],
                                'signal_type': signal_result['signal_type']
                            }
                    
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
                    continue
            
            # ارسال نوتیفیکیشن سیگنال‌های طلایی
            if golden_signals:
                await self._send_golden_signal_notifications(golden_signals)
            
            logger.info(f"Scan completed. {len(golden_signals)} golden signals found")
            
        except Exception as e:
            logger.error(f"Error in overall market scan: {e}")
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """دریافت داده‌های بازار برای یک ارز"""
        try:
            # دریافت داده‌های کندل (200 کندل برای محاسبه EMA200)
            klines = await self.aggregator.get_klines(symbol, '5m', 200)
            if not klines or len(klines) < 200:
                return None
            
            # تبدیل به فرمت مورد نیاز
            market_data = {
                'close': [float(k[4]) for k in klines],
                'high': [float(k[2]) for k in klines],
                'low': [float(k[3]) for k in klines],
                'volume': [float(k[5]) for k in klines],
                'timestamp': [int(k[0]) for k in klines]
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"خطا در دریافت داده‌های {symbol}: {e}")
            return None
    
    def _should_notify(self, symbol: str, signal_result: Dict[str, Any]) -> bool:
        """بررسی اینکه آیا باید نوتیفیکیشن ارسال شود یا نه"""
        if symbol not in self.last_golden_signals:
            return True
        
        last_signal = self.last_golden_signals[symbol]
        time_diff = datetime.now() - last_signal['timestamp']
        
        # اگر کولداون گذشته باشد
        if time_diff.total_seconds() > self.notification_cooldown:
            return True
        
        # اگر امتیاز به طور قابل توجهی بهتر شده باشد
        if signal_result['score'] > last_signal['score'] + 10:
            return True
        
        # اگر نوع سیگنال تغییر کرده باشد
        if signal_result['signal_type'] != last_signal['signal_type']:
            return True
        
        return False
    
    async def _send_golden_signal_notifications(self, golden_signals: List[Dict[str, Any]]):
        """ارسال نوتیفیکیشن سیگنال‌های طلایی"""
        try:
            if not self.telegram_bot:
                logger.warning("Telegram bot در دسترس نیست")
                return
            
            # مرتب‌سازی بر اساس امتیاز
            golden_signals.sort(key=lambda x: x['score'], reverse=True)
            
            # ساخت پیام با دکمه‌های جزئیات
            message = "🌟 *سیگنال‌های طلایی شناسایی شد!* 🌟\n\n"
            
            # ساخت دکمه‌ها برای جزئیات
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup
            keyboard = []
            
            for i, signal in enumerate(golden_signals[:5], 1):  # حداکثر 5 سیگنال
                symbol = signal['symbol'].replace('/USDT', '')
                signal_type = signal['signal_type']
                score = signal['score']
                percentage = signal['percentage']
                
                # تعیین ایموجی بر اساس نوع سیگنال
                if signal_type == "LONG":
                    emoji = "🚀"
                    direction = "خرید"
                elif signal_type == "SHORT":
                    emoji = "📉"
                    direction = "فروش"
                else:
                    emoji = "⚡"
                    direction = "خنثی"
                
                # محاسبه امتیاز 1-10 و علامت الماس
                rating_data = self.golden_detector.rating_system.calculate_signal_rating(signal)
                rating = rating_data.get('rating', 5)
                diamond = "💎" if rating_data.get('is_diamond', False) else ""
                stars = "⭐" * rating
                
                # ذخیره جزئیات سیگنال برای استفاده در دکمه‌های جزئیات
                signal['rating_data'] = rating_data
                
                message += f"{diamond} {emoji} *{symbol}* - {direction} ({rating}/10) {stars}\n"
                message += f"📊 امتیاز: `{score}/120` ({percentage:.1f}%)\n"
                message += f"💪 قدرت: {signal['strength']}\n"
                
                # نمایش اندیکاتورهای کلیدی
                indicators = signal['indicators']
                message += f"💰 قیمت: `{indicators['current_price']:.6f}`\n"
                message += f"📈 RSI: `{indicators['rsi']:.1f}`\n"
                message += f"⚡ ADX: `{indicators['adx']:.1f}`\n\n"
                
                # اضافه کردن دکمه برای جزئیات معاملاتی
                keyboard.append([InlineKeyboardButton(
                    f"{diamond} {symbol} - جزئیات معاملاتی ({rating}/10)",
                    callback_data=f"trading_details_{symbol}"
                )])
                
                # ذخیره سیگنال کامل برای دسترسی بعدی
                self._store_signal_details(f"{symbol}/USDT", signal)
            
            message += f"🕒 زمان: {datetime.now().strftime('%H:%M:%S')}\n"
            message += "⚠️ *توجه: این تحلیل خودکار است و نباید تنها مبنای تصمیم‌گیری باشد*\n\n"
            message += "📋 برای مشاهده جزئیات معاملاتی هر ارز، روی دکمه مربوطه کلیک کنید:"
            
            # ارسال به کاربران مجاز با دکمه‌ها
            reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
            await self._send_to_authorized_users(message, reply_markup)
            
        except Exception as e:
            logger.error(f"خطا در ارسال نوتیفیکیشن سیگنال‌های طلایی: {e}")
    
    def _store_signal_details(self, symbol: str, signal_data: Dict[str, Any]):
        """ذخیره جزئیات سیگنال برای استفاده در دکمه‌های جزئیات"""
        try:
            # ذخیره سیگنال با timestamp برای مدیریت انقضا
            self.stored_signals[symbol] = {
                'signal_data': signal_data,
                'timestamp': datetime.now(),
                'expiry': datetime.now() + timedelta(hours=24)  # انقضا بعد از 24 ساعت
            }
            
            # پاک کردن سیگنال‌های منقضی شده
            current_time = datetime.now()
            expired_symbols = [sym for sym, data in self.stored_signals.items() 
                             if data['expiry'] < current_time]
            
            for sym in expired_symbols:
                del self.stored_signals[sym]
                
            logger.debug(f"Signal details stored for {symbol}")
            
        except Exception as e:
            logger.error(f"Error storing signal details for {symbol}: {e}")
    
    def get_stored_signal(self, symbol: str) -> Dict[str, Any]:
        """دریافت جزئیات ذخیره شده سیگنال"""
        try:
            if symbol in self.stored_signals:
                stored_data = self.stored_signals[symbol]
                if stored_data['expiry'] > datetime.now():
                    return stored_data['signal_data']
                else:
                    # حذف سیگنال منقضی شده
                    del self.stored_signals[symbol]
                    
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving stored signal for {symbol}: {e}")
            return None
    
    async def _send_to_authorized_users(self, message: str, reply_markup=None):
        """ارسال پیام به کاربران مجاز با دکمه‌ها"""
        try:
            import os
            authorized_users = os.getenv('AUTHORIZED_USER_IDS', '').split(',')
            
            for user_id in authorized_users:
                if user_id.strip():
                    try:
                        self.telegram_bot.send_message(
                            chat_id=int(user_id.strip()), 
                            text=message, 
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                        await asyncio.sleep(0.5)  # تاخیر کوتاه بین ارسال‌ها
                    except Exception as e:
                        logger.error(f"خطا در ارسال پیام به {user_id}: {e}")
            
        except Exception as e:
            logger.error(f"خطا در ارسال به کاربران مجاز: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """دریافت وضعیت مانیتورینگ"""
        return {
            'active': self.monitoring_active,
            'scan_interval': self.scan_interval,
            'watched_symbols': len(self.watch_symbols),
            'last_golden_count': len(self.last_golden_signals),
            'cooldown_minutes': self.notification_cooldown // 60
        }
    
    def add_symbol(self, symbol: str):
        """اضافه کردن ارز جدید به لیست نظارت"""
        if symbol not in self.watch_symbols:
            self.watch_symbols.append(symbol)
            logger.info(f"ارز {symbol} به لیست نظارت اضافه شد")
    
    def remove_symbol(self, symbol: str):
        """حذف ارز از لیست نظارت"""
        if symbol in self.watch_symbols:
            self.watch_symbols.remove(symbol)
            logger.info(f"ارز {symbol} از لیست نظارت حذف شد")
    
    def set_scan_interval(self, seconds: int):
        """تنظیم فاصله زمانی اسکن"""
        if seconds >= 60:  # حداقل 1 دقیقه
            self.scan_interval = seconds
            logger.info(f"فاصله اسکن به {seconds} ثانیه تنظیم شد")
    
    async def manual_scan(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """اسکن دستی برای سیگنال‌های طلایی"""
        try:
            symbols_to_scan = [symbol] if symbol else self.watch_symbols
            golden_signals = []
            
            for sym in symbols_to_scan:
                market_data = await self._get_market_data(sym)
                if market_data:
                    signal_result = self.golden_detector.is_golden_signal(sym, market_data)
                    if signal_result['is_golden']:
                        golden_signals.append(signal_result)
            
            return golden_signals
            
        except Exception as e:
            logger.error(f"خطا در اسکن دستی: {e}")
            return []
