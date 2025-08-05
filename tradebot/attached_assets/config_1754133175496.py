"""
Configuration settings for the Trading Bot
"""

import os

class TradingConfig:
    def __init__(self):
        # Telegram Configuration
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_bot_token_here")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your_chat_id_here")
        
        # Trading Parameters
        self.TP_PERSEN_DEFAULT = 0.01  # 1%
        self.SL_PERSEN_DEFAULT = 0.05  # 5%
        self.SCALPING_TP_PERSEN = 0.005  # 0.5%
        self.SCALPING_SL_PERSEN = 0.01  # 1%
        self.SCALPING_OVERRIDE_ENABLED = True
        
        # Risk Management
        self.MAX_ORDER_PER_SESSION = 10
        self.SALDO_MINIMAL = 500000  # Minimum balance
        self.TARGET_PROFIT_PERSEN = 10  # 10% profit target
        self.LONJAKAN_THRESHOLD = 10  # Price spike threshold
        self.SKOR_MINIMAL = 4  # Minimum signal strength
        self.MAX_RISK_PER_TRADE = 2.0  # Maximum 2% risk per trade
        self.MAX_DRAWDOWN = 20.0  # Maximum 20% drawdown
        
        # Trading Hours
        self.TRADING_START_HOUR = 7
        self.TRADING_END_HOUR = 21
        self.RESET_ORDER_HOUR = 0
        
        # Technical Settings
        self.TRAILING_STOP_PIPS = 50
        self.DEFAULT_SYMBOL = "XAUUSDm"
        self.DEFAULT_LOT = 0.01
        self.DEFAULT_INTERVAL = 10  # seconds
        
        # MetaTrader5 Settings
        self.MT5_MAGIC_NUMBER = 234000
        self.MT5_DEVIATION = 20
        self.MT5_TIMEOUT = 10000  # milliseconds
        
        # Indicator Settings
        self.MA_PERIODS = [10, 20, 50]
        self.EMA_PERIODS = [9, 21, 50]
        self.WMA_PERIODS = [5, 10]
        self.RSI_PERIOD = 14
        self.BB_PERIOD = 20
        self.BB_DEVIATION = 2
        
        # File Paths
        self.LOG_FILE = "trading_log.txt"
        self.TRADE_LOG_FILE = "trade_log.csv"
        self.SETTINGS_FILE = "bot_settings.json"
        self.ERROR_LOG_FILE = "error_log.txt"
