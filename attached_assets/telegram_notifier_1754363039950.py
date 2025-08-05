"""
Enhanced Telegram Notifier for Trading Bot
Provides real-time notifications for trade alerts and bot status
"""

import requests
import datetime
from config import config

class TelegramNotifier:
    def __init__(self):
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.enabled = (self.bot_token != "your_bot_token_here" and 
                       self.chat_id != "your_chat_id_here")
        
        if self.enabled:
            print("✅ Telegram notifier initialized")
        else:
            print("⚠️ Telegram notifier disabled - Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram"""
        if not self.enabled:
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }
            
            response = requests.post(url, data=data, timeout=5)
            return response.status_code == 200
            
        except Exception as e:
            print(f"❌ Telegram send error: {e}")
            return False
    
    def send_trade_alert(self, signal: str, symbol: str, price: float, tp: float, sl: float) -> bool:
        """Send formatted trade alert"""
        message = f"""
🎯 <b>TRADE ALERT</b>
📊 Signal: <b>{signal}</b>
💰 Symbol: <b>{symbol}</b>
💵 Price: <b>{price:.5f}</b>
🎯 TP: <b>{tp:.5f}</b>
🛡️ SL: <b>{sl:.5f}</b>
⏰ Time: {datetime.datetime.now().strftime('%H:%M:%S')}
"""
        return self.send_message(message)
    
    def send_bot_status(self, status: str, details: str = "") -> bool:
        """Send bot status update"""
        message = f"""
🤖 <b>BOT STATUS</b>
📊 Status: <b>{status}</b>
📝 Details: {details}
⏰ Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return self.send_message(message)

# Global instance
telegram_notifier = TelegramNotifier()