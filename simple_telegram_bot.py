"""A simple Telegram bot using the telegram package"""
import logging
import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Updater, CommandHandler, CallbackQueryHandler,
    MessageHandler, Filters, CallbackContext
)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    keyboard = [
        [
            InlineKeyboardButton("📊 وضعیت بازار", callback_data='market'),
            InlineKeyboardButton("🔍 تحلیل نماد", callback_data='analyze')
        ],
        [InlineKeyboardButton("ℹ️ راهنما", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    update.message.reply_text(
        '🤖 *به ربات تحلیل ارز دیجیتال خوش آمدید!*\n\n'
        'من می‌توانم به شما در تحلیل بازار ارزهای دیجیتال کمک کنم.\n'
        'از منوی زیر گزینه مورد نظر را انتخاب کنید:',
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

def button(update: Update, context: CallbackContext) -> None:
    """Handle button presses."""
    query = update.callback_query
    query.answer()
    
    if query.data == 'market':
        query.edit_message_text(
            "📊 *وضعیت بازار*\n\n"
            "وضعیت فعلی بازار به شرح زیر است:\n\n"
            "• بیت‌کوین (BTC): ۵۰,۰۰۰ دلار (۲.۵%+)\n"
            "• اتریوم (ETH): ۳,۲۰۰ دلار (۱.۸%+)\n"
            "• بایننس کوین (BNB): ۴۰۰ دلار (۰.۹%+)",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 به‌روزرسانی", callback_data='market')],
                [InlineKeyboardButton("🔙 بازگشت", callback_data='back')]
            ])
        )
        
    elif query.data == 'analyze':
        query.edit_message_text(
            "🔍 *تحلیل نماد*\n\n"
            "لطفاً نماد معاملاتی مورد نظر را وارد کنید (مثلاً BTCUSDT یا ETHUSDT):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 بازگشت", callback_data='back')]
            ])
        )
        
    elif query.data == 'help':
        query.edit_message_text(
            "ℹ️ *راهنما*\n\n"
            "*دستورات موجود:*\n"
            "/start - نمایش منوی اصلی\n"
            "/market - نمایش وضعیت بازار\n"
            "/analyze - تحلیل یک نماد معاملاتی\n"
            "/help - نمایش این راهنما",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 بازگشت", callback_data='back')]
            ])
        )
        
    elif query.data == 'back':
        keyboard = [
            [
                InlineKeyboardButton("📊 وضعیت بازار", callback_data='market'),
                InlineKeyboardButton("🔍 تحلیل نماد", callback_data='analyze')
            ],
            [InlineKeyboardButton("ℹ️ راهنما", callback_data='help')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(
            '📋 *منوی اصلی*\n\nگزینه مورد نظر را انتخاب کنید:',
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

def analyze_symbol(update: Update, context: CallbackContext) -> None:
    """Analyze a specific symbol."""
    symbol = update.message.text.upper()
    
    # Simple analysis response
    analysis = f"📈 *تحلیل {symbol}*\n\n"
    analysis += "• قیمت فعلی: ۵۰,۰۰۰ دلار\n"
    analysis += "• تغییر ۲۴ ساعت: ۲.۵%+\n"
    analysis += "• بیشترین قیمت ۲۴ ساعت: ۵۱,۲۰۰ دلار\n"
    analysis += "• کمترین قیمت ۲۴ ساعت: ۴۹,۵۰۰ دلار\n"
    analysis += "• حجم معاملات ۲۴ ساعت: ۲۵,۰۰۰ بیت‌کوین"
    
    update.message.reply_text(
        analysis,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 بازگشت به منو", callback_data='back')]
        ])
    )

def error(update: Update, context: CallbackContext) -> None:
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)

def main() -> None:
    """Start the bot."""
    # Load token from environment variable
    from config import TELEGRAM_TOKEN
    
    if not TELEGRAM_TOKEN:
        logger.error("No TELEGRAM_TOKEN found. Please set it in .env file.")
        return
    
    # Create the Updater and pass it your bot's token.
    updater = Updater(TELEGRAM_TOKEN, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", lambda u, c: button(u, c, query_data='help')))
    dp.add_handler(CommandHandler("market", lambda u, c: button(u, c, query_data='market')))
    dp.add_handler(CommandHandler("analyze", lambda u, c: button(u, c, query_data='analyze')))
    
    # on non-command messages
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, analyze_symbol))
    
    # Add callback query handler
    dp.add_handler(CallbackQueryHandler(button))
    
    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    logger.info("Starting bot...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
