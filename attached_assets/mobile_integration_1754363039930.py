"""
Mobile Integration & Push Notifications
Remote monitoring and control capabilities
"""

import requests
import json
from typing import Dict, List, Optional
import datetime

class MobileIntegration:
    def __init__(self):
        self.push_endpoints = []
        self.notification_history = []
        self.remote_commands = {}
        
    def add_push_endpoint(self, service: str, token: str, endpoint: str):
        """Add push notification endpoint"""
        self.push_endpoints.append({
            'service': service,
            'token': token,
            'endpoint': endpoint,
            'active': True
        })
    
    def send_push_notification(self, title: str, message: str, priority: str = 'normal'):
        """Send push notification to all registered devices"""
        notification = {
            'title': title,
            'message': message,
            'priority': priority,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Store in history
        self.notification_history.append(notification)
        
        # Keep only last 100 notifications
        if len(self.notification_history) > 100:
            self.notification_history = self.notification_history[-100:]
        
        # Send to all endpoints
        for endpoint in self.push_endpoints:
            if endpoint['active']:
                self.send_to_endpoint(endpoint, notification)
    
    def send_to_endpoint(self, endpoint: Dict, notification: Dict):
        """Send notification to specific endpoint"""
        try:
            if endpoint['service'] == 'telegram':
                self.send_telegram_notification(endpoint, notification)
            elif endpoint['service'] == 'pushover':
                self.send_pushover_notification(endpoint, notification)
            elif endpoint['service'] == 'webhook':
                self.send_webhook_notification(endpoint, notification)
        except Exception as e:
            print(f"Failed to send notification to {endpoint['service']}: {e}")
    
    def send_telegram_notification(self, endpoint: Dict, notification: Dict):
        """Send notification via Telegram"""
        url = f"https://api.telegram.org/bot{endpoint['token']}/sendMessage"
        
        data = {
            'chat_id': endpoint.get('chat_id'),
            'text': f"🚀 *{notification['title']}*\n\n{notification['message']}",
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, json=data, timeout=10)
        return response.status_code == 200
    
    def send_pushover_notification(self, endpoint: Dict, notification: Dict):
        """Send notification via Pushover"""
        url = "https://api.pushover.net/1/messages.json"
        
        data = {
            'token': endpoint['token'],
            'user': endpoint.get('user_key'),
            'title': notification['title'],
            'message': notification['message'],
            'priority': 1 if notification['priority'] == 'high' else 0
        }
        
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
    
    def send_webhook_notification(self, endpoint: Dict, notification: Dict):
        """Send notification via webhook"""
        response = requests.post(
            endpoint['endpoint'], 
            json=notification, 
            timeout=10,
            headers={'Authorization': f"Bearer {endpoint['token']}"}
        )
        return response.status_code == 200
    
    def send_trade_notification(self, trade_data: Dict):
        """Send trade-specific notification"""
        action = trade_data.get('action', 'UNKNOWN')
        symbol = trade_data.get('symbol', 'UNKNOWN')
        price = trade_data.get('price', 0)
        lot = trade_data.get('lot', 0)
        
        title = f"🔄 Trade Executed - {action}"
        message = f"""
Symbol: {symbol}
Action: {action}
Price: {price}
Lot Size: {lot}
Time: {datetime.datetime.now().strftime('%H:%M:%S')}
        """
        
        self.send_push_notification(title, message.strip(), 'normal')
    
    def send_alert_notification(self, alert_type: str, message: str):
        """Send alert notification"""
        icons = {
            'error': '❌',
            'warning': '⚠️',
            'success': '✅',
            'info': 'ℹ️'
        }
        
        icon = icons.get(alert_type, '🔔')
        title = f"{icon} Trading Alert"
        
        self.send_push_notification(title, message, 'high' if alert_type == 'error' else 'normal')
    
    def send_daily_summary(self, summary_data: Dict):
        """Send daily trading summary"""
        title = "📊 Daily Trading Summary"
        
        message = f"""
Trades Executed: {summary_data.get('total_trades', 0)}
Success Rate: {summary_data.get('success_rate', 0):.1f}%
P/L: ${summary_data.get('total_pnl', 0):.2f}
Balance: ${summary_data.get('current_balance', 0):.2f}
        """
        
        self.send_push_notification(title, message.strip(), 'normal')
    
    def register_remote_command(self, command: str, handler):
        """Register remote command handler"""
        self.remote_commands[command] = handler
    
    def process_remote_command(self, command_data: Dict) -> Dict:
        """Process remote command from mobile app"""
        command = command_data.get('command')
        
        if command not in self.remote_commands:
            return {'success': False, 'error': 'Unknown command'}
        
        try:
            result = self.remote_commands[command](command_data.get('params', {}))
            return {'success': True, 'result': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_status_for_mobile(self) -> Dict:
        """Get current bot status for mobile app"""
        return {
            'timestamp': datetime.datetime.now().isoformat(),
            'bot_status': 'running',  # This would be actual status
            'account_balance': 0,  # This would be actual balance
            'open_positions': 0,  # This would be actual positions
            'daily_pnl': 0,  # This would be actual P/L
            'recent_notifications': self.notification_history[-10:]
        }
    
    def create_mobile_dashboard_data(self) -> Dict:
        """Create data structure for mobile dashboard"""
        return {
            'overview': {
                'bot_status': 'Active',
                'trading_mode': 'Normal',
                'account_balance': 10000.0,
                'daily_pnl': 150.50,
                'open_positions': 2
            },
            'recent_trades': [
                {
                    'time': '14:30:15',
                    'symbol': 'XAUUSDm',
                    'action': 'BUY',
                    'price': 2650.50,
                    'pnl': 45.20
                }
            ],
            'performance': {
                'success_rate': 78.5,
                'total_trades': 156,
                'profit_factor': 1.45
            },
            'alerts': [
                {
                    'type': 'info',
                    'message': 'High impact news in 30 minutes',
                    'time': '14:15:00'
                }
            ]
        }

class RemoteControl:
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot
        self.allowed_commands = [
            'get_status',
            'stop_trading',
            'start_trading',
            'close_positions',
            'get_account_info',
            'emergency_stop'
        ]
    
    def execute_command(self, command: str, params: Dict = None) -> Dict:
        """Execute remote command safely"""
        if command not in self.allowed_commands:
            return {'error': 'Command not allowed'}
        
        try:
            if command == 'get_status':
                return self.get_bot_status()
            elif command == 'stop_trading':
                return self.stop_trading()
            elif command == 'start_trading':
                return self.start_trading()
            elif command == 'close_positions':
                return self.close_all_positions()
            elif command == 'get_account_info':
                return self.get_account_info()
            elif command == 'emergency_stop':
                return self.emergency_stop()
        except Exception as e:
            return {'error': str(e)}
    
    def get_bot_status(self) -> Dict:
        """Get comprehensive bot status"""
        return {
            'running': getattr(self.trading_bot, 'running', False),
            'connected': getattr(self.trading_bot, 'mt5_connected', False),
            'mode': getattr(self.trading_bot, 'current_mode', 'Unknown'),
            'orders_today': getattr(self.trading_bot, 'order_counter', 0)
        }
    
    def stop_trading(self) -> Dict:
        """Stop trading remotely"""
        if hasattr(self.trading_bot, 'stop_bot'):
            self.trading_bot.stop_bot()
            return {'success': True, 'message': 'Trading stopped'}
        return {'error': 'Stop method not available'}
    
    def start_trading(self) -> Dict:
        """Start trading remotely"""
        if hasattr(self.trading_bot, 'start_bot'):
            self.trading_bot.start_bot()
            return {'success': True, 'message': 'Trading started'}
        return {'error': 'Start method not available'}
    
    def close_all_positions(self) -> Dict:
        """Close all positions remotely"""
        if hasattr(self.trading_bot, 'close_all_positions'):
            result = self.trading_bot.close_all_positions()
            return {'success': True, 'message': 'All positions closed', 'result': result}
        return {'error': 'Close positions method not available'}
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        # This would get actual account info from MT5
        return {
            'balance': 10000.0,
            'equity': 10150.50,
            'margin': 250.0,
            'free_margin': 9900.50
        }
    
    def emergency_stop(self) -> Dict:
        """Emergency stop all trading activities"""
        try:
            if hasattr(self.trading_bot, 'emergency_stop'):
                self.trading_bot.emergency_stop()
            else:
                # Fallback emergency stop
                if hasattr(self.trading_bot, 'running'):
                    self.trading_bot.running = False
                if hasattr(self.trading_bot, 'close_all_positions'):
                    self.trading_bot.close_all_positions()
            
            return {'success': True, 'message': 'Emergency stop executed'}
        except Exception as e:
            return {'error': f'Emergency stop failed: {e}'}