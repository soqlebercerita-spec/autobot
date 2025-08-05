
"""
Main Configuration for AuraTrade Trading Bot
Balance-based TP/SL system with HFT capabilities
"""

import os
from typing import Dict, List, Any

class TradingConfig:
    """Main trading configuration class"""
    
    def __init__(self):
        # Core Trading Parameters (Balance-based TP/SL)
        self.TP_PERCENT = float(os.getenv("TP_PERCENT", "1.5"))  # 1.5% of balance
        self.SL_PERCENT = float(os.getenv("SL_PERCENT", "0.5"))  # 0.5% of balance
        self.MIN_BALANCE = float(os.getenv("MIN_BALANCE", "1000"))  # Minimum $1000
        
        # HFT Parameters
        self.HFT_ENABLED = os.getenv("HFT_ENABLED", "true").lower() == "true"
        self.MAX_LATENCY_MS = float(os.getenv("MAX_LATENCY_MS", "1.0"))  # <1ms target
        self.HFT_MIN_PROFIT_PIPS = float(os.getenv("HFT_MIN_PROFIT_PIPS", "0.5"))
        
        # Multi-Symbol Trading
        self.SYMBOLS = [
            "XAUUSD",  # Gold
            "EURUSD",  # Euro
            "GBPUSD",  # Pound
            "USDJPY",  # Yen
            "BTCUSD",  # Bitcoin (if available)
        ]
        
        # Risk Management
        self.MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "2.0"))  # 2%
        self.MAX_DAILY_DRAWDOWN = float(os.getenv("MAX_DAILY_DRAWDOWN", "5.0"))  # 5%
        self.MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "10"))
        
        # Technical Analysis (Custom indicators only)
        self.RSI_PERIOD = 14
        self.RSI_OVERSOLD = 30
        self.RSI_OVERBOUGHT = 70
        self.MA_FAST = 10
        self.MA_SLOW = 20
        self.MACD_FAST = 12
        self.MACD_SLOW = 26
        self.MACD_SIGNAL = 9
        
        # Trading Hours (24/5 for FX, varies for others)
        self.TRADING_START_HOUR = 0
        self.TRADING_END_HOUR = 24
        self.FRIDAY_CLOSE_HOUR = 22  # Close early on Friday
        
        # Telegram Notifications
        self.TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
        
        # MT5 Settings
        self.MT5_LOGIN = os.getenv("MT5_LOGIN", "")
        self.MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
        self.MT5_SERVER = os.getenv("MT5_SERVER", "")
        self.MT5_MAGIC_NUMBER = int(os.getenv("MT5_MAGIC_NUMBER", "234000"))
        
        # Performance Targets
        self.TARGET_WIN_RATE = 0.90  # 90%+
        self.TARGET_DAILY_ROI = 0.02  # 2% daily
        self.TARGET_MONTHLY_ROI = 0.25  # 25% monthly
        
    def get_symbol_config(self, symbol: str) -> Dict[str, Any]:
        """Get symbol-specific configuration"""
        configs = {
            "XAUUSD": {
                "lot_size": 0.01,
                "pip_value": 0.01,
                "spread_buffer": 2.0,
                "volatility_multiplier": 1.5
            },
            "EURUSD": {
                "lot_size": 0.01, 
                "pip_value": 0.0001,
                "spread_buffer": 1.0,
                "volatility_multiplier": 1.0
            },
            "GBPUSD": {
                "lot_size": 0.01,
                "pip_value": 0.0001, 
                "spread_buffer": 1.5,
                "volatility_multiplier": 1.2
            },
            "USDJPY": {
                "lot_size": 0.01,
                "pip_value": 0.01,
                "spread_buffer": 1.0,
                "volatility_multiplier": 1.1
            },
            "BTCUSD": {
                "lot_size": 0.001,
                "pip_value": 1.0,
                "spread_buffer": 5.0,
                "volatility_multiplier": 2.0
            }
        }
        return configs.get(symbol, configs["EURUSD"])
    
    def calculate_lot_size(self, balance: float, symbol: str, risk_percent: float = None) -> float:
        """Calculate lot size based on balance and risk percentage"""
        if risk_percent is None:
            risk_percent = self.MAX_RISK_PER_TRADE
        
        risk_amount = balance * (risk_percent / 100)
        symbol_config = self.get_symbol_config(symbol)
        
        # Simple lot calculation (can be enhanced)
        lot_size = min(risk_amount / 1000, symbol_config["lot_size"] * 10)
        return max(symbol_config["lot_size"], lot_size)

# Global config instance
Config = TradingConfig()
