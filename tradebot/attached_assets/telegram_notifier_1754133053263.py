"""
Telegram Notification Module for Trading Bot
"""

import requests
import threading
import time
from datetime import datetime

class TelegramNotifier:
    def __init__(self, config):
        self.config = config
        self.bot_token = config.TELEGRAM_BOT_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.message_queue = []
        self.last_message_time = {}
        self.rate_limit_delay = 1  # Minimum seconds between messages
        
        # Start message sender thread
        self.sender_thread = threading.Thread(target=self._message_sender_loop, daemon=True)
        self.sender_thread.start()
    
    def send_message(self, text, urgent=False):
        """Add message to queue for sending"""
        try:
            message = {
                'text': text,
                'timestamp': datetime.now(),
                'urgent': urgent
            }
            
            # Check for rate limiting (avoid spam)
            current_time = time.time()
            message_hash = hash(text[:50])  # Hash first 50 chars to detect similar messages
            
            if not urgent and message_hash in self.last_message_time:
                if current_time - self.last_message_time[message_hash] < 300:  # 5 minutes
                    return False  # Skip duplicate message
            
            self.message_queue.append(message)
            self.last_message_time[message_hash] = current_time
            
            return True
            
        except Exception as e:
            print(f"Error queuing Telegram message: {e}")
            return False
    
    def _message_sender_loop(self):
        """Background thread to send queued messages"""
        while True:
            try:
                if self.message_queue:
                    message = self.message_queue.pop(0)
                    self._send_message_direct(message['text'])
                    time.sleep(self.rate_limit_delay)
                else:
                    time.sleep(1)
                    
            except Exception as e:
                print(f"Error in message sender loop: {e}")
                time.sleep(5)
    
    def _send_message_direct(self, text):
        """Send message directly to Telegram"""
        try:
            if not self.bot_token or self.bot_token == "your_bot_token_here":
                return False
            
            if not self.chat_id or self.chat_id == "your_chat_id_here":
                return False
            
            url = f"{self.base_url}/sendMessage"
            
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                return True
            else:
                print(f"Telegram API error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print("Telegram message timeout")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Telegram request error: {e}")
            return False
        except Exception as e:
            print(f"Error sending Telegram message: {e}")
            return False
    
    def send_trade_notification(self, signal, symbol, price, tp, sl, lot_size, confidence):
        """Send formatted trade notification"""
        try:
            emoji = "📈" if signal == "BUY" else "📉"
            
            message = (
                f"{emoji} <b>Trade Executed</b>\n\n"
                f"🔸 <b>Symbol:</b> {symbol}\n"
                f"🔸 <b>Action:</b> {signal}\n"
                f"🔸 <b>Price:</b> {price:.5f}\n"
                f"🔸 <b>Volume:</b> {lot_size}\n"
                f"🔸 <b>Take Profit:</b> {tp:.5f}\n"
                f"🔸 <b>Stop Loss:</b> {sl:.5f}\n"
                f"🔸 <b>Confidence:</b> {confidence:.1f}%\n"
                f"🔸 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return self.send_message(message, urgent=True)
            
        except Exception as e:
            print(f"Error sending trade notification: {e}")
            return False
    
    def send_position_closed(self, symbol, signal, entry_price, exit_price, lot_size, profit):
        """Send position closed notification"""
        try:
            emoji = "💰" if profit > 0 else "💸"
            profit_text = "PROFIT" if profit > 0 else "LOSS"
            
            message = (
                f"{emoji} <b>Position Closed</b>\n\n"
                f"🔸 <b>Symbol:</b> {symbol}\n"
                f"🔸 <b>Action:</b> {signal}\n"
                f"🔸 <b>Entry:</b> {entry_price:.5f}\n"
                f"🔸 <b>Exit:</b> {exit_price:.5f}\n"
                f"🔸 <b>Volume:</b> {lot_size}\n"
                f"🔸 <b>Result:</b> {profit_text} ${profit:.2f}\n"
                f"🔸 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return self.send_message(message, urgent=True)
            
        except Exception as e:
            print(f"Error sending position closed notification: {e}")
            return False
    
    def send_daily_summary(self, total_trades, profit_loss, win_rate, balance):
        """Send daily trading summary"""
        try:
            emoji = "🎉" if profit_loss > 0 else "😔" if profit_loss < 0 else "😐"
            
            message = (
                f"{emoji} <b>Daily Summary</b>\n\n"
                f"📊 <b>Total Trades:</b> {total_trades}\n"
                f"💰 <b>P&L:</b> ${profit_loss:.2f}\n"
                f"🎯 <b>Win Rate:</b> {win_rate:.1f}%\n"
                f"💳 <b>Balance:</b> ${balance:.2f}\n"
                f"📅 <b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}"
            )
            
            return self.send_message(message)
            
        except Exception as e:
            print(f"Error sending daily summary: {e}")
            return False
    
    def send_alert(self, alert_type, message_text):
        """Send alert notification"""
        try:
            alert_emojis = {
                'ERROR': '🚨',
                'WARNING': '⚠️',
                'INFO': 'ℹ️',
                'SUCCESS': '✅',
                'CRITICAL': '🔴'
            }
            
            emoji = alert_emojis.get(alert_type, 'ℹ️')
            
            message = (
                f"{emoji} <b>{alert_type}</b>\n\n"
                f"{message_text}\n"
                f"🕐 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
            )
            
            urgent = alert_type in ['ERROR', 'CRITICAL']
            return self.send_message(message, urgent=urgent)
            
        except Exception as e:
            print(f"Error sending alert: {e}")
            return False
    
    def send_bot_status(self, status, additional_info=""):
        """Send bot status update"""
        try:
            status_emojis = {
                'STARTED': '🟢',
                'STOPPED': '🔴',
                'PAUSED': '🟡',
                'ERROR': '🚨'
            }
            
            emoji = status_emojis.get(status, '🤖')
            
            message = (
                f"{emoji} <b>Bot Status: {status}</b>\n\n"
                f"🕐 <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            if additional_info:
                message += f"\n\n📝 <b>Info:</b> {additional_info}"
            
            return self.send_message(message, urgent=(status == 'ERROR'))
            
        except Exception as e:
            print(f"Error sending bot status: {e}")
            return False
    
    def send_market_update(self, symbol, price, change_percent, trend):
        """Send market update notification"""
        try:
            trend_emoji = "📈" if trend == "UPTREND" else "📉" if trend == "DOWNTREND" else "➡️"
            change_emoji = "🟢" if change_percent > 0 else "🔴" if change_percent < 0 else "⚪"
            
            message = (
                f"{trend_emoji} <b>Market Update</b>\n\n"
                f"🔸 <b>Symbol:</b> {symbol}\n"
                f"🔸 <b>Price:</b> {price:.5f}\n"
                f"🔸 <b>Change:</b> {change_emoji} {change_percent:+.2f}%\n"
                f"🔸 <b>Trend:</b> {trend}\n"
                f"🔸 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}"
            )
            
            return self.send_message(message)
            
        except Exception as e:
            print(f"Error sending market update: {e}")
            return False
    
    def test_connection(self):
        """Test Telegram bot connection"""
        try:
            test_message = f"🤖 Trading Bot Test Message\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            return self._send_message_direct(test_message)
            
        except Exception as e:
            print(f"Error testing Telegram connection: {e}")
            return False
    
    def get_bot_info(self):
        """Get bot information from Telegram"""
        try:
            if not self.bot_token or self.bot_token == "your_bot_token_here":
                return None
            
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            print(f"Error getting bot info: {e}")
            return None
    
    def clear_message_queue(self):
        """Clear pending messages in queue"""
        try:
            cleared_count = len(self.message_queue)
            self.message_queue.clear()
            return cleared_count
        except Exception as e:
            print(f"Error clearing message queue: {e}")
            return 0
