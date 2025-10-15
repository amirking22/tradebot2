"""
Telegram bot handler with market command and access control.
"""
import asyncio
import logging
from typing import Dict, Any, List
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CommandHandler, CallbackContext
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Authorized user ID
AUTHORIZED_USER_ID = 37292924  # @Manti77

logger = logging.getLogger(__name__)

class TelegramHandler:
    def __init__(self, aggregator):
        self.aggregator = aggregator
        self.token = os.getenv('TELEGRAM_TOKEN')
        
    def is_authorized(self, user_id: int) -> bool:
        """Check if user is authorized to use the bot."""
        return user_id == AUTHORIZED_USER_ID
    
    def start_command(self, update: Update, context: CallbackContext):
        """Handle /start command."""
        if not self.is_authorized(update.effective_user.id):
            update.message.reply_text("⛔ شما مجاز به استفاده از این ربات نیستید.")
            return
            
        keyboard = [
            [InlineKeyboardButton("📊 وضعیت بازار", callback_data="market_status")],
            [InlineKeyboardButton("🪙 فیوچرز", callback_data="futures_list")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        update.message.reply_text(
            "🤖 *ربات سیگنال‌دهی ارز دیجیتال*\n\n"
            "برای مشاهده وضعیت بازار یا لیست فیوچرز، دکمه مربوطه را انتخاب کنید:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    def market_command(self, update: Update, context: CallbackContext):
        """Handle /market command - show all crypto status with 1h changes."""
        if not self.is_authorized(update.effective_user.id):
            update.message.reply_text("⛔ شما مجاز به استفاده از این ربات نیستید.")
            return
            
        try:
            # Get top markets from aggregator
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            top_symbols = loop.run_until_complete(self.aggregator.get_top_markets(20))
            
            message = "📊 *وضعیت بازار ارزهای دیجیتال*\n\n"
            
            for symbol in top_symbols:
                try:
                    ticker_data = loop.run_until_complete(self.aggregator.get_aggregated_ticker(symbol))
                    
                    if 'error' in ticker_data:
                        continue
                        
                    price = ticker_data['price']
                    change_24h = ticker_data['change_24h']
                    reliability = ticker_data['reliability']
                    source_count = ticker_data['total_sources']
                    
                    # Format with emoji based on profit/loss
                    if change_24h > 0:
                        emoji = "🟢"
                        sign = "+"
                    elif change_24h < 0:
                        emoji = "🔴"
                        sign = ""
                    else:
                        emoji = "⚪"
                        sign = ""
                    
                    # Reliability indicator
                    rel_emoji = "🟢" if reliability == "high" else "🟡" if reliability == "medium" else "🔴"
                    
                    message += f"{emoji} *{symbol}*: `{price:.6f}` ({sign}{change_24h:.2f}%) {rel_emoji}({source_count})\n"
                    
                except Exception as e:
                    logger.error(f"Error getting ticker for {symbol}: {e}")
                    continue
            
            message += f"\n{rel_emoji} اعتبار: 🟢بالا 🟡متوسط 🔴پایین (تعداد منابع)\n"
            message += f"🕒 آخرین بروزرسانی: {self.get_persian_time()}"
            
            update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in market command: {e}")
            update.message.reply_text("❌ خطا در دریافت اطلاعات بازار")
    
    def get_persian_time(self) -> str:
        """Get current time in Persian timezone."""
        from datetime import datetime
        from pytz import timezone
        from jdatetime import datetime as jdatetime
        
        tehran = timezone('Asia/Tehran')
        now = datetime.now(tehran)
        jd = jdatetime.fromgregorian(datetime=now)
        
        return f"{jd.hour:02d}:{jd.minute:02d} - {jd.year}/{jd.month:02d}/{jd.day:02d}"

def setup_telegram_bot(aggregator):
    """Setup Telegram bot with handlers."""
    handler = TelegramHandler(aggregator)
    
    updater = Updater(token=handler.token, use_context=True)
    dispatcher = updater.dispatcher
    
    # Add command handlers
    dispatcher.add_handler(CommandHandler("start", handler.start_command))
    dispatcher.add_handler(CommandHandler("market", handler.market_command))
    
    return updater, handler
