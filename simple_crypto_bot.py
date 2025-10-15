"""A simple crypto bot using python-telegram-bot v20+"""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)
from telegram.constants import ParseMode

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    keyboard = [
        [InlineKeyboardButton("📊 وضعیت بازار", callback_data='market')],
        [InlineKeyboardButton("🔍 تحلیل نماد", callback_data='analyze')],
        [InlineKeyboardButton("ℹ️ راهنما", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        '🤖 *به ربات تحلیل ارز دیجیتال خوش آمدید!*\n\n'
        'من می‌توانم به شما در تحلیل بازار ارزهای دیجیتال کمک کنم.\n'
        'از منوی زیر گزینه مورد نظر را انتخاب کنید:',
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button presses."""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'market':
        await query.edit_message_text(
            "📊 *وضعیت بازار*\n\n"
            "وضعیت فعلی بازار به شرح زیر است:\n\n"
            "• بیت‌کوین (BTC): ۵۰,۰۰۰ دلار (۲.۵%+)\n"
            "• اتریوم (ETH): ۳,۲۰۰ دلار (۱.۸%+)\n"
            "• بایننس کوین (BNB): ۴۰۰ دلار (۰.۹%+)",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data='market')],
                [InlineKeyboardButton("🔙 بازگشت", callback_data='back')]
            ])
        )
    elif query.data == 'analyze':
        await query.edit_message_text(
            "🔍 *تحلیل نماد*\n\n"
            "لطفاً نماد معاملاتی مورد نظر را وارد کنید (مثلاً BTCUSDT یا ETHUSDT):",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 بازگشت", callback_data='back')]
            ])
        )
    elif query.data == 'help':
        await query.edit_message_text(
            "ℹ️ *راهنما*\n\n"
            "*دستورات موجود:*\n"
            "/start - نمایش منوی اصلی\n"
            "/market - نمایش وضعیت بازار\n"
            "/analyze - تحلیل یک نماد معاملاتی\n"
            "/help - نمایش این راهنما",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 بازگشت", callback_data='back')]
            ])
        )
    elif query.data == 'back':
        keyboard = [
            [InlineKeyboardButton("📊 وضعیت بازار", callback_data='market')],
            [InlineKeyboardButton("🔍 تحلیل نماد", callback_data='analyze')],
            [InlineKeyboardButton("ℹ️ راهنما", callback_data='help')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            '📋 *منوی اصلی*\n\nگزینه مورد نظر را انتخاب کنید:',
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )

async def analyze_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze a specific symbol."""
    symbol = update.message.text.upper()
    
    # Simple analysis response
    analysis = f"📈 *تحلیل {symbol}*\n\n"
    analysis += "• قیمت فعلی: ۵۰,۰۰۰ دلار\n"
    analysis += "• تغییر ۲۴ ساعت: ۲.۵%+\n"
    analysis += "• بیشترین قیمت ۲۴ ساعت: ۵۱,۲۰۰ دلار\n"
    analysis += "• کمترین قیمت ۲۴ ساعت: ۴۹,۵۰۰ دلار\n"
    analysis += "• حجم معاملات ۲۴ ساعت: ۲۵,۰۰۰ بیت‌کوین"
    
    await update.message.reply_text(
        analysis,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 بازگشت به منو", callback_data='back')]
        ])
    )

def main() -> None:
    """Start the bot."""
    # Load token from environment variable
    from config import TELEGRAM_TOKEN
    
    if not TELEGRAM_TOKEN:
        logger.error("No TELEGRAM_TOKEN found. Please set it in .env file.")
        return
    
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(button_handler))
    
    # Add message handler for symbol analysis
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_symbol))

    # Start the Bot
    logger.info("Starting bot...")
    application.run_polling()

if __name__ == "__main__":
    main()
