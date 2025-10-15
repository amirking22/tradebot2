"""Telegram bot handlers for the crypto signal bot."""
import logging
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup,
    ReplyKeyboardMarkup, ReplyKeyboardRemove, ParseMode
)
from telegram.ext import (
    CallbackContext, CommandHandler, MessageHandler, Filters,
    CallbackQueryHandler, ConversationHandler
)

from exchanges.yex_market import yex_market
from config import YEX_FUTURES_SYMBOLS
from visualization import BacktestVisualizer

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
SELECTING_ACTION, SELECTING_SYMBOL, SELECTING_INTERVAL = range(3)

# Constants
INTERVALS = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
LOOKBACK_PERIODS = {
    '1h': '24h',
    '4h': '7d',
    '1d': '30d'
}

# In-memory storage for user data
user_data = {}


def get_main_menu_keyboard() -> List[List[InlineKeyboardButton]]:
    """Create the main menu keyboard."""
    return [
        [
            InlineKeyboardButton("📊 Market Overview", callback_data='market_overview'),
            InlineKeyboardButton("📈 Analyze Symbol", callback_data='analyze_symbol')
        ],
        [
            InlineKeyboardButton("📊 Top Gainers", callback_data='top_gainers'),
            InlineKeyboardButton("📉 Top Losers", callback_data='top_losers')
        ],
        [
            InlineKeyboardButton("⚙️ Settings", callback_data='settings'),
            InlineKeyboardButton("ℹ️ Help", callback_data='help')
        ]
    ]


def get_interval_keyboard() -> List[List[InlineKeyboardButton]]:
    """Create the interval selection keyboard."""
    return [
        [
            InlineKeyboardButton("1 Minute", callback_data='interval_1m'),
            InlineKeyboardButton("5 Minutes", callback_data='interval_5m'),
            InlineKeyboardButton("15 Minutes", callback_data='interval_15m')
        ],
        [
            InlineKeyboardButton("30 Minutes", callback_data='interval_30m'),
            InlineKeyboardButton("1 Hour", callback_data='interval_1h'),
            InlineKeyboardButton("4 Hours", callback_data='interval_4h')
        ],
        [
            InlineKeyboardButton("1 Day", callback_data='interval_1d'),
            InlineKeyboardButton("🔙 Back", callback_data='back_to_main')
        ]
    ]


def get_symbols_keyboard(symbols: List[str], page: int = 0, per_page: int = 8) -> Tuple[List[List[InlineKeyboardButton]], int]:
    """Create a paginated keyboard for symbol selection."""
    total_pages = (len(symbols) + per_page - 1) // per_page
    start_idx = page * per_page
    end_idx = start_idx + per_page
    
    # Get symbols for current page
    page_symbols = symbols[start_idx:end_idx]
    
    # Create buttons (2 per row)
    keyboard = []
    for i in range(0, len(page_symbols), 2):
        row = []
        if i < len(page_symbols):
            row.append(InlineKeyboardButton(
                page_symbols[i], 
                callback_data=f'symbol_{page_symbols[i]}'
            ))
        if i + 1 < len(page_symbols):
            row.append(InlineKeyboardButton(
                page_symbols[i + 1], 
                callback_data=f'symbol_{page_symbols[i + 1]}'
            ))
        keyboard.append(row)
    
    # Add navigation buttons
    nav_buttons = []
    if page > 0:
        nav_buttons.append(InlineKeyboardButton("⬅️ Prev", callback_data=f'page_{page-1}'))
    
    nav_buttons.append(InlineKeyboardButton(
        f"{page+1}/{total_pages}", 
        callback_data='noop'
    ))
    
    if page < total_pages - 1:
        nav_buttons.append(InlineKeyboardButton("Next ➡️", callback_data=f'page_{page+1}'))
    
    if nav_buttons:
        keyboard.append(nav_buttons)
    
    # Add back button
    keyboard.append([InlineKeyboardButton("🔙 Back", callback_data='back_to_main')])
    
    return keyboard, total_pages


async def start(update: Update, context: CallbackContext) -> int:
    """Send a welcome message and show the main menu."""
    user = update.effective_user
    welcome_message = (
        f"👋 Welcome *{user.first_name}* to *Crypto Signal Bot*!\n\n"
        "I can help you analyze cryptocurrency markets and provide trading signals.\n"
        "Use the buttons below to get started:"
    )
    
    # Initialize user data
    user_id = update.effective_user.id
    if user_id not in user_data:
        user_data[user_id] = {
            'symbol': None,
            'interval': '1h',
            'page': 0
        }
    
    # Send welcome message with main menu
    await update.message.reply_text(
        welcome_message,
        reply_markup=InlineKeyboardMarkup(get_main_menu_keyboard()),
        parse_mode=ParseMode.MARKDOWN
    )
    
    return SELECTING_ACTION


async def handle_callback(update: Update, context: CallbackContext) -> int:
    """Handle callback queries from inline keyboards."""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    data = query.data
    
    # Initialize user data if not exists
    if user_id not in user_data:
        user_data[user_id] = {
            'symbol': None,
            'interval': '1h',
            'page': 0
        }
    
    # Handle back button
    if data == 'back_to_main':
        await query.edit_message_text(
            "📊 *Main Menu*\n\nSelect an option:",
            reply_markup=InlineKeyboardMarkup(get_main_menu_keyboard()),
            parse_mode=ParseMode.MARKDOWN
        )
        return SELECTING_ACTION
    
    # Handle main menu actions
    if data in ['market_overview', 'top_gainers', 'top_losers']:
        return await handle_market_actions(update, context, data)
    
    # Handle symbol analysis
    elif data == 'analyze_symbol':
        return await handle_analyze_symbol(update, context)
    
    # Handle interval selection
    elif data.startswith('interval_'):
        interval = data.split('_')[1]
        user_data[user_id]['interval'] = interval
        
        # If we already have a symbol, show analysis
        if user_data[user_id].get('symbol'):
            return await show_symbol_analysis(update, context)
        
        # Otherwise, show symbol selection
        return await handle_analyze_symbol(update, context)
    
    # Handle symbol selection
    elif data.startswith('symbol_'):
        symbol = data.split('_', 1)[1]
        user_data[user_id]['symbol'] = symbol
        return await show_symbol_analysis(update, context)
    
    # Handle pagination
    elif data.startswith('page_'):
        page = int(data.split('_')[1])
        user_data[user_id]['page'] = page
        return await handle_analyze_symbol(update, context)
    
    # Handle settings and help
    elif data == 'settings':
        await query.edit_message_text(
            "⚙️ *Settings*\n\n"
            "Here you can configure your preferences.\n\n"
            "*Default Interval:* {}\n"
            "*Default Lookback:* {}".format(
                user_data[user_id].get('interval', '1h'),
                LOOKBACK_PERIODS.get(user_data[user_id].get('interval', '1h'), '24h')
            ),
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data='back_to_main')]
            ]),
            parse_mode=ParseMode.MARKDOWN
        )
        return SELECTING_ACTION
    
    elif data == 'help':
        help_text = (
            "ℹ️ *Help*\n\n"
            "*Available Commands:*\n"
            "• /start - Start the bot and show main menu\n"
            "• /market - Show market overview\n"
            "• /analyze [symbol] [interval] - Analyze a specific symbol\n"
            "• /status - Show bot status\n"
            "• /active - List active signals\n"
            "• /close [signal_id] - Close a signal\n"
            "• /help - Show this help message"
        )
        await query.edit_message_text(
            help_text,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🔙 Back", callback_data='back_to_main')]
            ])
        )
