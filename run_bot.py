"""
Main script to run the trading bot.
"""
import asyncio
import logging
import os
from dotenv import load_dotenv
from bot.trading_bot import TradingBot
from simple_telegram import setup_simple_telegram_bot
from exchanges.multi_exchange import MultiExchangeAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('trading_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    # Exchange API Keys
    'yex_api_key': os.getenv('YEX_API_KEY'),
    'yex_secret': os.getenv('YEX_SECRET'),
    'binance_api_key': os.getenv('BINANCE_API_KEY'),
    'binance_secret': os.getenv('BINANCE_SECRET'),
    'bybit_api_key': os.getenv('BYBIT_API_KEY'),
    'bybit_secret': os.getenv('BYBIT_SECRET'),
    
    # Trading settings
    'risk_reward_ratio': float(os.getenv('RISK_REWARD_RATIO', '3.0')),
    'position_size_percent': float(os.getenv('POSITION_SIZE_PERCENT', '10.0')),
    'max_open_positions': int(os.getenv('MAX_OPEN_POSITIONS', '5')),
    
    # Technical indicators
    'rsi_period': int(os.getenv('RSI_PERIOD', '14')),
    'adx_period': int(os.getenv('ADX_PERIOD', '14')),
    'atr_period': int(os.getenv('ATR_PERIOD', '14')),
    'min_adx': float(os.getenv('MIN_ADX', '25.0')),
    'rsi_overbought': float(os.getenv('RSI_OVERBOUGHT', '70.0')),
    'rsi_oversold': float(os.getenv('RSI_OVERSOLD', '30.0')),
    
    # Telegram settings
    'telegram_token': os.getenv('TELEGRAM_TOKEN'),
    'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID')
}

async def main():
    """Main function to run the trading bot."""
    # Check required environment variables
    if not CONFIG['telegram_token'] or not CONFIG['telegram_chat_id']:
        logger.error("Telegram token and chat ID are required in .env file")
        return
    
    # Check if at least one exchange is configured
    has_exchange = any([
        CONFIG['yex_api_key'] and CONFIG['yex_secret'],
        CONFIG['binance_api_key'] and CONFIG['binance_secret'],
        CONFIG['bybit_api_key'] and CONFIG['bybit_secret']
    ])
    
    if not has_exchange:
        logger.error("At least one exchange API key and secret are required in .env file")
        return
    
    # Initialize multi-exchange aggregator
    async with MultiExchangeAggregator() as aggregator:
        # Initialize trading bot
        bot = TradingBot(CONFIG)
        bot.aggregator = aggregator
        
        # Setup simple Telegram bot
        telegram_bot = setup_simple_telegram_bot(aggregator)
        
        if telegram_bot:
            # Start both bots concurrently
            await asyncio.gather(
                bot.run(),
                telegram_bot.process_updates()
            )
        else:
            # Run only trading bot if telegram setup failed
            await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Trading bot stopped")
