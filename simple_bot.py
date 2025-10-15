"""A simple Telegram bot for crypto signals."""
import logging
import os
from datetime import datetime

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, 
    ContextTypes, MessageHandler, filters
)
from telegram.constants import ParseMode

from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
MENU, GET_SYMBOL = range(2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Send welcome message and show main menu."""
    user = update.effective_user
    keyboard = [
        [InlineKeyboardButton("📊 Market Overview", callback_data='market_overview')],
        [InlineKeyboardButton("🔍 Analyze Symbol", callback_data='analyze_symbol')],
        [InlineKeyboardButton("ℹ️ Help", callback_data='help')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"🤖 *Welcome {user.first_name}!*\n\n"
        "I'm your crypto trading assistant. Use the buttons below to get started.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    return MENU

async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle button presses."""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'market_overview':
        await query.edit_message_text(
            "📊 *Market Overview*\n\n"
            "Here's the current market overview...",
            parse_mode=ParseMode.MARKDOWN
        )
    elif query.data == 'analyze_symbol':
        await query.edit_message_text(
            "🔍 *Analyze Symbol*\n\n"
            "Please enter a trading pair (e.g., BTCUSDT):",
            parse_mode=ParseMode.MARKDOWN
        )
        return GET_SYMBOL
    elif query.data == 'help':
        await query.edit_message_text(
            "ℹ️ *Help*\n\n"
            "*Available commands:*\n"
            "/start - Show main menu\n"
            "/market - Show market overview\n"
            "/analyze [symbol] - Analyze a trading pair\n"
            "/help - Show this help message",
            parse_mode=ParseMode.MARKDOWN
        )
    
    return MENU

async def analyze_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle symbol analysis."""
    symbol = update.message.text.upper()
    
    # Here you would add your analysis logic
    analysis = f"📈 *{symbol} Analysis*\n\n"
    analysis += "• Price: $50,000\n"
    analysis += "• 24h Change: +2.5%\n"
    analysis += "• Volume: $1.2B\n"
    
    keyboard = [[InlineKeyboardButton("🔙 Back to Menu", callback_data='back_to_menu')]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        analysis,
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )
    
    return MENU

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add conversation handler
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            MENU: [
                CallbackQueryHandler(button),
                CommandHandler('market', lambda u, c: button(u, c, query_data='market_overview')),
                CommandHandler('analyze', lambda u, c: button(u, c, query_data='analyze_symbol')),
                CommandHandler('help', lambda u, c: button(u, c, query_data='help')),
            ],
            GET_SYMBOL: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_symbol)
            ]
        },
        fallbacks=[CommandHandler('start', start)]
    )

    application.add_handler(conv_handler)

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
