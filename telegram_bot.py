"""Telegram bot for crypto trading signals and market analysis."""
from __future__ import annotations
import asyncio
import logging
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

from telegram import (
    Update, InlineKeyboardButton, 
    InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
)
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ContextTypes,
    MessageHandler, filters, ConversationHandler
)
from telegram.constants import ParseMode

from config import (
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, YEX_API_KEY, YEX_SECRET_KEY,
    YEX_FUTURES_SYMBOLS, ensure_config
)
from exchanges.yex_market import YEXFuturesMarket
from signal_monitor import SignalMonitor, create_signal, SignalDirection
from cache import signal_tracker, SignalStatus

# Import bot handlers
from bot_handlers import (
    start, handle_callback, handle_market_actions, handle_analyze_symbol,
    show_symbol_analysis, cancel, error_handler, SELECTING_ACTION,
    SELECTING_SYMBOL, SELECTING_INTERVAL
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class CryptoSignalBot:
    """Advanced crypto signal bot with market analysis capabilities."""
    
    def __init__(self, token: str, chat_id: int):
        """Initialize the bot with token and chat ID."""
        self.token = token
        self.chat_id = chat_id
        self.application = Application.builder().token(token).build()
        self.market = YEXFuturesMarket(api_key=YEX_API_KEY, api_secret=YEX_SECRET_KEY)
        self.signal_monitor = SignalMonitor()
        
        # Register handlers
        self._register_handlers()
        
        # Initialize data
        self.symbols = YEX_FUTURES_SYMBOLS
        self.user_data = {}
        
        logger.info("CryptoSignalBot initialized")

    def _register_handlers(self):
        """Register all command and callback handlers."""
        # Define conversation states
        SELECTING_ACTION, SELECTING_SYMBOL, SELECTING_INTERVAL = range(3)
        
        # Store states as instance variables
        self.SELECTING_ACTION = SELECTING_ACTION
        self.SELECTING_SYMBOL = SELECTING_SYMBOL
        self.SELECTING_INTERVAL = SELECTING_INTERVAL
        
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._start))
        self.application.add_handler(CommandHandler("help", self._help))
        self.application.add_handler(CommandHandler("market", self._market_overview))
        self.application.add_handler(CommandHandler("analyze", self._analyze_symbol))
        
        # Main callback query handler
        self.application.add_handler(CallbackQueryHandler(self._handle_callback, pattern='^[^/]'))
        
        # Conversation handler for navigation
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', self._start)],
            states={
                SELECTING_ACTION: [
                    CallbackQueryHandler(self._handle_callback, pattern='^market_overview$'),
                    CallbackQueryHandler(self._handle_callback, pattern='^top_gainers$'),
                    CallbackQueryHandler(self._handle_callback, pattern='^top_losers$'),
                    CallbackQueryHandler(self._handle_callback, pattern='^analyze_symbol$'),
                    CallbackQueryHandler(self._handle_callback, pattern='^settings$'),
                    CallbackQueryHandler(self._handle_callback, pattern='^help$'),
                ],
                SELECTING_SYMBOL: [
                    CallbackQueryHandler(self._handle_callback, pattern='^symbol_'),
                    CallbackQueryHandler(self._handle_callback, pattern='^page_'),
                    CallbackQueryHandler(self._handle_callback, pattern='^back_to_main$'),
                ],
                SELECTING_INTERVAL: [
                    CallbackQueryHandler(self._handle_callback, pattern='^interval_'),
                    CallbackQueryHandler(self._handle_callback, pattern='^back_to_main$'),
                ],
            },
            fallbacks=[CommandHandler('cancel', self._cancel)],
            allow_reentry=True
        )
        self.application.add_handler(conv_handler)
        
        # Error handler
        self.application.add_error_handler(self._error_handler)

    async def start(self) -> None:
        """Start the bot, initialize market data, and start monitoring."""
        try:
            # Initialize market data
            logger.info("Initializing YEX market data...")
            if await self.market.initialize():
                logger.info(f"Successfully initialized with {len(self.market.symbols)} symbols")
            else:
                logger.warning("Failed to initialize market data")
            
            # Start the bot
            self.application.run_polling()
            logger.info("Bot started")
            
            # Start signal monitoring in the background
            asyncio.create_task(self._start_signal_monitoring())
            
            # Keep the bot running
            self.application.run_polling()
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            raise
            
    async def _start_signal_monitoring(self):
        """Start the signal monitoring system."""
        try:
            await self.signal_monitor.start_monitoring(self.chat_id)
        except Exception as e:
            logger.error(f"Error in signal monitoring: {e}", exc_info=True)

    async def _update_market_data(self, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Periodically update market data."""
        try:
            logger.info("Updating market data...")
            await self.market.initialize()  # Refresh symbols and data
            logger.info("Market data updated successfully")
        except Exception as e:
            logger.error(f"Error updating market data: {e}")

    async def _market_overview(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show market overview with top gainers and losers."""
        try:
            overview = await self.market.get_market_overview()
            
            # Format message
            message = "📊 *Market Overview*\n\n"
            
            # Add top gainers
            message += "📈 *Top Gainers (24h):*\n"
            for i, symbol in enumerate(overview['top_gainers'][:5], 1):
                change = symbol.get('priceChangePercent', 0)
                message += f"{i}. {symbol['symbol']}: *{change:+.2f}%*\n"
            
            # Add top losers
            message += "\n📉 *Top Losers (24h):*\n"
            for i, symbol in enumerate(overview['top_losers'][:5], 1):
                change = symbol.get('priceChangePercent', 0)
                message += f"{i}. {symbol['symbol']}: *{change:+.2f}%*\n"
            
            # Add total volume
            message += f"\n💱 *Total Volume (24h):* ${overview['total_volume']:,.2f}"
            
            # Send message
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("🔄 Refresh", callback_data='market_overview')],
                    [InlineKeyboardButton("📈 Analyze Symbol", callback_data='analyze_symbol')]
                ])
            )
            
        except Exception as e:
            logger.error(f"Error in market overview: {e}")
            await update.message.reply_text(
                "❌ Error fetching market data. Please try again later.",
                parse_mode=ParseMode.MARKDOWN
            )

    async def _analyze_symbol(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Analyze a specific symbol."""
        try:
            if not context.args:
                await update.message.reply_text(
                    "Please specify a symbol. Example: `/analyze BTCUSDT`",
                    parse_mode=ParseMode.MARKDOWN
                )
                return
            
            symbol = context.args[0].upper()
            interval = context.args[1] if len(context.args) > 1 else '1h'
            
            # Validate interval
            if interval not in ['1h', '4h', '1d']:
                await update.message.reply_text(
                    f"Invalid interval. Available intervals: 1h, 4h, 1d",
                    parse_mode=ParseMode.MARKDOWN
                )
                return
            
            # Get analysis
            analysis = await self.market.analyze_symbol(symbol, interval)
            
            if not analysis.get('success', False):
                await update.message.reply_text(
                    f"❌ Could not analyze {symbol}. Please check the symbol and try again.",
                    parse_mode=ParseMode.MARKDOWN
                )
                return
            
            # Format analysis message
            message = (
                f"📊 *{symbol} Analysis ({interval})*\n\n"
                f"💰 *Price:* ${analysis['current_price']:,.2f} "
                f"({analysis['price_change_24h']:+.2f}% 24h)\n"
                f"📈 *Trend:* {analysis['trend'].capitalize()}\n"
                f"📊 *RSI (14):* {analysis['rsi']:.2f} "
                f"{'🔴' if analysis['rsi'] > 70 else '🟢' if analysis['rsi'] < 30 else '🟡'}\n"
                f"📉 *Volume (24h):* ${analysis['volume_24h']:,.2f}\n\n"
                f"📈 *Support Levels:*\n" + "\n".join([
                    f"  • ${level:,.2f}" for level in analysis['support_levels']
                ]) + "\n\n"
                f"📉 *Resistance Levels:*\n" + "\n".join([
                    f"  • ${level:,.2f}" for level in analysis['resistance_levels']
                ])
            )
            
            # Generate chart
            chart_path = await self.market.plot_analysis(symbol, interval)
            
            # Send message with chart
            if chart_path and os.path.exists(chart_path):
                with open(chart_path, 'rb') as photo:
                    await update.message.reply_photo(
                        photo=photo,
                        caption=message,
                        parse_mode=ParseMode.MARKDOWN,
                        reply_markup=InlineKeyboardMarkup([
                            [
                                InlineKeyboardButton("1h", callback_data=f'analyze_{symbol}_1h'),
                                InlineKeyboardButton("4h", callback_data=f'analyze_{symbol}_4h'),
                                InlineKeyboardButton("1d", callback_data=f'analyze_{symbol}_1d')
                            ],
                            [
                                InlineKeyboardButton("🔄 Refresh", callback_data=f'analyze_{symbol}_{interval}'),
                                InlineKeyboardButton("📊 More Analysis", callback_data=f'more_{symbol}')
                            ]
                        ])
                    )
                # Clean up the chart file
                try:
                    os.remove(chart_path)
                except:
                    pass
            else:
                await update.message.reply_text(
                    message,
                    parse_mode=ParseMode.MARKDOWN,
                    reply_markup=InlineKeyboardMarkup([
                        [
                            InlineKeyboardButton("1h", callback_data=f'analyze_{symbol}_1h'),
                            InlineKeyboardButton("4h", callback_data=f'analyze_{symbol}_4h'),
                            InlineKeyboardButton("1d", callback_data=f'analyze_{symbol}_1d')
                        ],
                        [
                            InlineKeyboardButton("🔄 Refresh", callback_data=f'analyze_{symbol}_{interval}'),
                            InlineKeyboardButton("📊 More Analysis", callback_data=f'more_{symbol}')
                        ]
                    ])
                )
                
        except Exception as e:
            logger.error(f"Error in symbol analysis: {e}")
            await update.message.reply_text(
                "❌ Error analyzing symbol. Please try again later.",
                parse_mode=ParseMode.MARKDOWN
            )

    async def send_signal(
        self, 
        signal: Dict[str, Any],
        position: PositionSize,
        exchange_name: str,
        timeframe: str
    ) -> None:
        """
        Send a trading signal to the configured chat.
        
        Args:
            signal: Trade signal dictionary
            position: PositionSize object with calculated values
            exchange_name: Name of the exchange
            timeframe: Timeframe of the signal (e.g., '1h', '4h')
        """
        try:
            # Format the message
            message = self._format_signal_message(signal, position, exchange_name, timeframe)
            
            # Send the message
            await self.application.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN_V2,
                disable_web_page_preview=True
            )
            
            logger.info(f"Signal sent for {signal['symbol']}")
        except Exception as e:
            logger.error(f"Error sending signal: {e}")

    def _format_signal_message(
        self,
        signal: Dict[str, Any],
        position: PositionSize,
        exchange_name: str,
        timeframe: str
    ) -> str:
        """Format the signal message with Markdown."""
        # Escape special characters for MarkdownV2
        def escape(text):
            return escape_markdown(str(text), version=2)
        
        symbol = escape(signal['symbol'])
        side = escape(signal['side'].upper())
        entry = escape(f"{signal['entry']:.8f}".rstrip('0').rstrip('.'))
        stop_loss = escape(f"{position.stop_loss:.8f}".rstrip('0').rstrip('.'))
        
        # Format take profit levels
        take_profits = []
        for i, tp in enumerate(position.take_profit, 1):
            tp_str = escape(f"{tp:.8f}".rstrip('0').rstrip('.'))
            take_profits.append(f"🎯 TP{i}: `{tp_str}`")
        
        # Format risk/reward info
        risk_reward = (
            f"📏 *ATR%:* `{position.atr_pct:.2f}%` | "
            f"📊 *RSI:* `{position.rsi:.1f}` | "
            f"📈 *ADX:* `{position.adx:.1f}`"
        )
        
        # Format position size info
        position_size = (
            f"🧮 *Position Size:* `{position.position_size:.2f} USDT`\n"
            f"📊 *Leverage:* `{position.leverage}x` | "
            f"🔒 *Risk:* `{position.risk_per_trade}%`"
        )
        
        # Compose the full message
        message = (
            f"🚀 *{exchange_name.upper()} Signal* \n\n"
            f"📊 *{symbol}* | `{timeframe}` | *{side}*\n"
            f"🔹 *Entry:* `{entry}`\n"
            f"🛑 *Stop Loss:* `{stop_loss}`\n\n"
            f"{' '.join(take_profits)}\n\n"
            f"{risk_reward}\n"
            f"{position_size}\n\n"
            f"🕒 *Time:* `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC`"
        )
        
        return message

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Send welcome message and show main menu."""
        user = update.effective_user
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"🤖 *Welcome {user.first_name}!*\n\n"
                 "I'm your crypto trading assistant. Use the buttons below to get started.",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_main_menu_keyboard()
        )

    async def _help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        help_text = (
            "*🤖 Crypto Signal Bot Help*\n\n"
            "*Available commands:*\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/status - Show bot status and configuration"
        )
        
        await update.message.reply_text(
            help_text,
            parse_mode=ParseMode.MARKDOWN_V2
        )
    
    async def _status(self, update: Update, context: CallbackContext) -> None:
        """Handle /status command."""
        status_text = (
            "*🤖 Bot Status*\n\n"
            "*Status:* ✅ Running\n"
            f"*Chat ID:* `{self.chat_id}`\n"
            f"*Last Update:* `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC`"
        )
        
        await update.message.reply_text(
            status_text,
            parse_mode=ParseMode.MARKDOWN_V2
        )


async def main() -> None:
    """Run the crypto signal bot."""
    bot = None
    try:
        # Ensure required configs are set
        await ensure_config()
        
        # Initialize and start the bot
        bot = CryptoSignalBot(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        await bot.start()
        bot.run()
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        # Cleanup
        if bot is not None:
            await bot.market.close()


if __name__ == "__main__":
    # Run the main function
    import asyncio
    asyncio.run(main())
