"""A minimal Telegram bot for crypto signals."""
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    ContextTypes, MessageHandler, filters
)
from telegram.constants import ParseMode

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    keyboard = [
        [InlineKeyboardButton("📊 Market Overview", callback_data='market')],
        [InlineKeyboardButton("🔍 Analyze Symbol", callback_data='analyze')],
        [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        '🤖 *Welcome to Crypto Bot!*\n\n'
        'I can help you with crypto market analysis. Use the buttons below to get started.',
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button presses."""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'market':
        await query.edit_message_text(
            "📊 *Market Overview*\n\n"
            "Here's the current market overview...\n\n"
            "• BTC/USD: $50,000 (+2.5%)\n"
            "• ETH/USD: $3,200 (+1.8%)\n"
            "• BNB/USD: $400 (+0.9%)",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔄 Refresh", callback_data='market')],
                [InlineKeyboardButton("🔙 Back", callback_data='back')]
            ])
        )
    elif query.data == 'analyze':
        await query.edit_message_text(
            "🔍 *Analyze Symbol*\n\n"
            "Please enter a trading pair (e.g., BTCUSDT or ETHUSDT):",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data='back')]
            ])
        )
    elif query.data == 'help':
        await query.edit_message_text(
            "ℹ️ *Help*\n\n"
            "*Available commands:*\n"
            "/start - Show main menu\n"
            "/market - Show market overview\n"
            "/analyze [symbol] - Analyze a trading pair\n"
            "/help - Show this help message",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data='back')]
            ])
        )
    elif query.data == 'back':
        keyboard = [
            [InlineKeyboardButton("📊 Market Overview", callback_data='market')],
            [InlineKeyboardButton("🔍 Analyze Symbol", callback_data='analyze')],
            [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            '📋 *Main Menu*\n\nSelect an option:',
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze a specific symbol."""
    symbol = update.message.text.upper()
    
    # Simple analysis response
    analysis = f"📈 *{symbol} Analysis*\n\n"
    analysis += "• Current Price: $50,000\n"
    analysis += "• 24h Change: +2.5%\n"
    analysis += "• 24h High: $51,200\n"
    analysis += "• 24h Low: $49,500\n"
    analysis += "• 24h Volume: 25,000 BTC"
    
    await update.message.reply_text(
        analysis,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("🔙 Back to Menu", callback_data='back')]
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
    application.add_handler(CommandHandler("market", lambda u, c: button(u, c, query_data='market')))
    application.add_handler(CommandHandler("analyze", lambda u, c: button(u, c, query_data='analyze')))
    application.add_handler(CommandHandler("help", lambda u, c: button(u, c, query_data='help')))
    
    # Add callback query handler
    application.add_handler(CallbackQueryHandler(button))
    
    # Add message handler for symbol analysis
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze))

    # Start the Bot
    logger.info("Starting bot...")
    application.run_polling()

if __name__ == "__main__":
    main()
