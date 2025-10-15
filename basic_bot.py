"""A basic Telegram bot for crypto signals using python-telegram-bot v13.7"""
import logging
import sys

# Add a workaround for missing imghdr in Python 3.10+
if sys.version_info >= (3, 10):
    import shutil
    from importlib import import_module
    
    class ImghdrCompat:
        def __init__(self):
            self._original_imghdr = None
            
        def __enter__(self):
            try:
                import imghdr
                self._original_imghdr = imghdr
            except ImportError:
                # Create a minimal imghdr module
                import types
                imghdr = types.ModuleType('imghdr')
                imghdr.what = lambda file, h=None: 'png'  # Simple fallback
                sys.modules['imghdr'] = imghdr
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self._original_imghdr:
                sys.modules['imghdr'] = self._original_imghdr
    
    # Apply the workaround
    _imghdr_compat = ImghdrCompat()
    _imghdr_compat.__enter__()

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

# Define menu states
MENU, GET_SYMBOL = range(2)

def start(update: Update, context: CallbackContext) -> int:
    """Send welcome message and show main menu."""
    keyboard = [
        [InlineKeyboardButton("📊 Market Overview", callback_data='market')],
        [InlineKeyboardButton("🔍 Analyze Symbol", callback_data='analyze')],
        [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    update.message.reply_text(
        '🤖 *Welcome to Crypto Bot!*\n\n'
        'I can help you with crypto market analysis. '
        'Use the buttons below to get started.',
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )
    return MENU

def button(update: Update, context: CallbackContext) -> int:
    """Handle button presses."""
    query = update.callback_query
    query.answer()
    
    if query.data == 'market':
        query.edit_message_text(
            "📊 *Market Overview*\n\n"
            "Here's the current market overview...\n\n"
            "• BTC/USD: $50,000 (+2.5%)\n"
            "• ETH/USD: $3,200 (+1.8%)\n"
            "• BNB/USD: $400 (+0.9%)",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Refresh", callback_data='market')],
                [InlineKeyboardButton("🔙 Back", callback_data='back')]
            ])
        )
        return MENU
        
    elif query.data == 'analyze':
        query.edit_message_text(
            "🔍 *Analyze Symbol*\n\n"
            "Please enter a trading pair (e.g., BTCUSDT or ETHUSDT):",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data='back')]
            ])
        )
        return GET_SYMBOL
        
    elif query.data == 'help':
        query.edit_message_text(
            "ℹ️ *Help*\n\n"
            "*Available commands:*\n"
            "/start - Show main menu\n"
            "/market - Show market overview\n"
            "/analyze [symbol] - Analyze a trading pair\n"
            "/help - Show this help message",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data='back')]
            ])
        )
        return MENU
        
    elif query.data == 'back':
        keyboard = [
            [InlineKeyboardButton("📊 Market Overview", callback_data='market')],
            [InlineKeyboardButton("🔍 Analyze Symbol", callback_data='analyze')],
            [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        query.edit_message_text(
            '📋 *Main Menu*\n\nSelect an option:',
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        return MENU

def analyze_symbol(update: Update, context: CallbackContext) -> int:
    """Analyze a specific symbol."""
    symbol = update.message.text.upper()
    
    # Simple analysis response
    analysis = f"📈 *{symbol} Analysis*\n\n"
    analysis += "• Current Price: $50,000\n"
    analysis += "• 24h Change: +2.5%\n"
    analysis += "• 24h High: $51,200\n"
    analysis += "• 24h Low: $49,500\n"
    analysis += "• 24h Volume: 25,000 BTC"
    
    update.message.reply_text(
        analysis,
        parse_mode='Markdown',
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Back to Menu", callback_data='back')]
        ])
    )
    return MENU

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

    # Add conversation handler with the states
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MENU: [
                CallbackQueryHandler(button),
                CommandHandler('market', lambda u, c: button(u, c, query_data='market')),
                CommandHandler('analyze', lambda u, c: button(u, c, query_data='analyze')),
                CommandHandler('help', lambda u, c: button(u, c, query_data='help')),
            ],
            GET_SYMBOL: [
                MessageHandler(Filters.text & ~Filters.command, analyze_symbol)
            ]
        },
        fallbacks=[CommandHandler('start', start)]
    )

    dp.add_handler(conv_handler)
    dp.add_error_handler(error)

    # Start the Bot
    logger.info("Starting bot...")
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
