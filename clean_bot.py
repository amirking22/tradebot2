"""A clean and functional Telegram bot for crypto signals."""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)
from telegram.constants import ParseMode

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
MENU, GET_SYMBOL = range(2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send welcome message and show main menu."""
    keyboard = [
        [InlineKeyboardButton("📊 Market Overview", callback_data='market_overview')],
        [InlineKeyboardButton("🔍 Analyze Symbol", callback_data='analyze_symbol')],
        [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"🤖 *Welcome to Crypto Bot!*\n\n"
        "I can help you with crypto market analysis. Use the buttons below to get started.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    return MENU

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle button presses."""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'market_overview':
        await query.edit_message_text(
            "📊 *Market Overview*\n\n"
            "Here's the current market overview...\n\n"
            "• BTC/USD: $50,000 (+2.5%)\n"
            "• ETH/USD: $3,200 (+1.8%)\n"
            "• BNB/USD: $400 (+0.9%)\n\n"
            "*Last updated: Just now*",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Refresh", callback_data='market_overview')],
                [InlineKeyboardButton("🔙 Back", callback_data='back_to_menu')]
            ])
        )
        return MENU
        
    elif query.data == 'analyze_symbol':
        await query.edit_message_text(
            "🔍 *Analyze Symbol*\n\n"
            "Please enter a trading pair (e.g., BTCUSDT or ETHUSDT):",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data='back_to_menu')]
            ])
        )
        return GET_SYMBOL
        
    elif query.data == 'help':
        await query.edit_message_text(
            "ℹ️ *Help*\n\n"
            "*Available commands:*\n"
            "/start - Show main menu\n"
            "/market - Show market overview\n"
            "/analyze [symbol] - Analyze a trading pair\n"
            "/help - Show this help message\n\n"
            "*How to use:*\n"
            "1. Use the menu buttons to navigate\n"
            "2. Enter a symbol when prompted for analysis\n"
            "3. Check back regularly for updates",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data='back_to_menu')]
            ])
        )
        return MENU
        
    elif query.data == 'back_to_menu':
        keyboard = [
            [InlineKeyboardButton("📊 Market Overview", callback_data='market_overview')],
            [InlineKeyboardButton("🔍 Analyze Symbol", callback_data='analyze_symbol')],
            [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "📋 *Main Menu*\n\n"
            "Select an option from the menu below:",
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
        return MENU

async def analyze_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle symbol analysis."""
    symbol = update.message.text.upper()
    
    # Here you would add your analysis logic
    analysis = f"📈 *{symbol} Analysis*\n\n"
    analysis += "• Current Price: $50,000\n"
    analysis += "• 24h Change: +2.5%\n"
    analysis += "• 24h High: $51,200\n"
    analysis += "• 24h Low: $49,500\n"
    analysis += "• 24h Volume: 25,000 BTC\n\n"
    analysis += "*Technical Indicators:*\n"
    analysis += "• RSI (14): 62 (Neutral)\n"
    analysis += "• MACD: Bullish\n"
    analysis += "• Support: $49,000\n"
    analysis += "• Resistance: $52,000\n\n"
    analysis += "*Last updated: Just now*"
    
    await update.message.reply_text(
        analysis,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📊 View Chart", callback_data=f'chart_{symbol}')],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_menu')]
        ])
    )
    
    return MENU

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors and handle them gracefully."""
    logger.error("Exception while handling an update:", exc_info=context.error)
    
    if update and hasattr(update, 'message') and update.message:
        await update.message.reply_text(
            '⚠️ An error occurred. Please try again later.',
            parse_mode=ParseMode.MARKDOWN
        )

def main() -> None:
    """Start the bot."""
    # Load token from environment variable
    from config import TELEGRAM_TOKEN
    import pytz
    
    if not TELEGRAM_TOKEN:
        logger.error("No TELEGRAM_TOKEN found. Please set it in .env file.")
        return
    
    # Create the Application with timezone configuration
    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .job_queue(
            job_queue={
                'timezone': pytz.utc  # Set timezone to UTC
            }
        )
        .build()
    )

    # Add conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MENU: [
                CallbackQueryHandler(button_handler),
                CommandHandler('market', lambda u, c: button_handler(u, c, query_data='market_overview')),
                CommandHandler('analyze', lambda u, c: button_handler(u, c, query_data='analyze_symbol')),
                CommandHandler('help', lambda u, c: button_handler(u, c, query_data='help')),
            ],
            GET_SYMBOL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_symbol)
            ]
        },
        fallbacks=[CommandHandler('start', start)]
    )

    application.add_handler(conv_handler)
    application.add_error_handler(error_handler)

    # Start the Bot
    logger.info("Starting bot...")
    application.run_polling()

if __name__ == "__main__":
    main()
