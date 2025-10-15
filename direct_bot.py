"""A simple Telegram bot using direct API calls"""
import logging
import json
import time
from typing import Dict, Any, Optional
import requests

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SimpleCryptoBot:
    def __init__(self, token: str):
        self.token = token
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.offset = 0
        
    def get_updates(self) -> Dict[str, Any]:
        """Get new updates from Telegram"""
        url = f"{self.base_url}/getUpdates"
        params = {"offset": self.offset + 1, "timeout": 30}
        try:
            response = requests.get(url, params=params, timeout=35)
            return response.json()
        except Exception as e:
            logger.error(f"Error getting updates: {e}")
            return {"ok": False, "result": []}
    
    def send_message(self, chat_id: int, text: str, reply_markup: Optional[Dict] = None) -> bool:
        """Send a message to a chat"""
        url = f"{self.base_url}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
            
        try:
            response = requests.post(url, json=data)
            return response.json().get("ok", False)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def edit_message(self, chat_id: int, message_id: int, text: str, reply_markup: Optional[Dict] = None) -> bool:
        """Edit an existing message"""
        url = f"{self.base_url}/editMessageText"
        data = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        if reply_markup:
            data["reply_markup"] = json.dumps(reply_markup)
            
        try:
            response = requests.post(url, json=data)
            return response.json().get("ok", False)
        except Exception as e:
            logger.error(f"Error editing message: {e}")
            return False
    
    def create_keyboard(self, buttons: list) -> Dict:
        """Create an inline keyboard"""
        return {
            "inline_keyboard": buttons
        }
    
    def create_button(self, text: str, callback_data: str) -> Dict:
        """Create an inline button"""
        return [{"text": text, "callback_data": callback_data}]
    
    def create_row_buttons(self, *buttons: Dict) -> list:
        """Create a row of buttons"""
        return [button for button in buttons]
    
    def get_main_menu(self) -> Dict:
        """Create the main menu keyboard"""
        buttons = [
            self.create_button("📊 وضعیت بازار", "market"),
            self.create_button("🔍 تحلیل نماد", "analyze")
        ]
        help_button = [self.create_button("ℹ️ راهنما", "help")]
        return self.create_keyboard(buttons + help_button)
    
    def get_back_button(self) -> Dict:
        """Create a back button"""
        return self.create_keyboard([self.create_button("🔙 بازگشت", "back")])
    
    def handle_message(self, chat_id: int, text: str) -> None:
        """Handle incoming text messages"""
        if text.startswith('/'):
            command = text.split('@')[0].lower()
            if command == "/start":
                self.show_main_menu(chat_id)
            elif command == "/market":
                self.show_market(chat_id)
            elif command == "/analyze":
                self.ask_for_symbol(chat_id)
            elif command == "/help":
                self.show_help(chat_id)
        else:
            # Assume it's a symbol for analysis
            self.analyze_symbol(chat_id, text.upper())
    
    def handle_callback(self, chat_id: int, message_id: int, data: str) -> None:
        """Handle callback queries"""
        if data == "market":
            self.show_market(chat_id, message_id)
        elif data == "analyze":
            self.ask_for_symbol(chat_id, message_id)
        elif data == "help":
            self.show_help(chat_id, message_id)
        elif data == "back":
            self.show_main_menu(chat_id, message_id)
    
    def show_main_menu(self, chat_id: int, message_id: int = None) -> None:
        """Show the main menu"""
        text = ("🤖 *به ربات تحلیل ارز دیجیتال خوش آمدید!*\n\n"
               "من می‌توانم به شما در تحلیل بازار ارزهای دیجیتال کمک کنم.\n"
               "از منوی زیر گزینه مورد نظر را انتخاب کنید:")
        
        if message_id:
            self.edit_message(chat_id, message_id, text, self.get_main_menu())
        else:
            self.send_message(chat_id, text, self.get_main_menu())
    
    def show_market(self, chat_id: int, message_id: int = None) -> None:
        """Show market overview"""
        text = ("📊 *وضعیت بازار*\n\n"
               "وضعیت فعلی بازار به شرح زیر است:\n\n"
               "• بیت‌کوین (BTC): ۵۰,۰۰۰ دلار (۲.۵%+)\n"
               "• اتریوم (ETH): ۳,۲۰۰ دلار (۱.۸%+)\n"
               "• بایننس کوین (BNB): ۴۰۰ دلار (۰.۹%+)")
        
        keyboard = self.create_keyboard([
            self.create_button("🔄 به‌روزرسانی", "market")[0],
            self.create_button("🔙 بازگشت", "back")[0]
        ])
        
        if message_id:
            self.edit_message(chat_id, message_id, text, keyboard)
        else:
            self.send_message(chat_id, text, keyboard)
    
    def ask_for_symbol(self, chat_id: int, message_id: int = None) -> None:
        """Ask user to enter a symbol"""
        text = "🔍 *تحلیل نماد*\n\nلطفاً نماد معاملاتی مورد نظر را وارد کنید (مثلاً BTCUSDT یا ETHUSDT):"
        
        if message_id:
            self.edit_message(chat_id, message_id, text, self.get_back_button())
        else:
            self.send_message(chat_id, text, self.get_back_button())
    
    def analyze_symbol(self, chat_id: int, symbol: str, message_id: int = None) -> None:
        """Analyze a specific symbol"""
        # In a real app, you would fetch real market data here
        analysis = (f"📈 *تحلیل {symbol}*\n\n"
                   "• قیمت فعلی: ۵۰,۰۰۰ دلار\n"
                   "• تغییر ۲۴ ساعت: ۲.۵%+\n"
                   "• بیشترین قیمت ۲۴ ساعت: ۵۱,۲۰۰ دلار\n"
                   "• کمترین قیمت ۲۴ ساعت: ۴۹,۵۰۰ دلار\n"
                   "• حجم معاملات ۲۴ ساعت: ۲۵,۰۰۰ بیت‌کوین")
        
        keyboard = self.create_keyboard([self.create_button("🔙 بازگشت به منو", "back")])
        
        if message_id:
            self.edit_message(chat_id, message_id, analysis, keyboard)
        else:
            self.send_message(chat_id, analysis, keyboard)
    
    def show_help(self, chat_id: int, message_id: int = None) -> None:
        """Show help message"""
        text = ("ℹ️ *راهنما*\n\n"
               "*دستورات موجود:*\n"
               "/start - نمایش منوی اصلی\n"
               "/market - نمایش وضعیت بازار\n"
               "/analyze - تحلیل یک نماد معاملاتی\n"
               "/help - نمایش این راهنما")
        
        if message_id:
            self.edit_message(chat_id, message_id, text, self.get_back_button())
        else:
            self.send_message(chat_id, text, self.get_back_button())
    
    def run(self) -> None:
        """Run the bot"""
        logger.info("Starting bot...")
        
        while True:
            try:
                updates = self.get_updates()
                
                if not updates.get("ok"):
                    logger.error("Failed to get updates")
                    time.sleep(5)
                    continue
                
                for update in updates.get("result", []):
                    self.offset = max(self.offset, update["update_id"])
                    
                    if "message" in update and "text" in update["message"]:
                        chat_id = update["message"]["chat"]["id"]
                        text = update["message"]["text"]
                        self.handle_message(chat_id, text)
                        
                    elif "callback_query" in update:
                        callback = update["callback_query"]
                        chat_id = callback["message"]["chat"]["id"]
                        message_id = callback["message"]["message_id"]
                        data = callback["data"]
                        self.handle_callback(chat_id, message_id, data)
                
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(5)

def main():
    """Main function"""
    from config import TELEGRAM_TOKEN
    
    if not TELEGRAM_TOKEN:
        logger.error("No TELEGRAM_TOKEN found. Please set it in .env file.")
        return
    
    bot = SimpleCryptoBot(TELEGRAM_TOKEN)
    bot.run()

if __name__ == "__main__":
    main()
