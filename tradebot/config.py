
"""
Enhanced Configuration for Trading Bot
Complete configuration with all required parameters
"""

import os

class TradingConfig:
    def __init__(self):
        # Basic Trading Parameters
        self.DEFAULT_SYMBOL = "EURUSD"
        self.DEFAULT_LOT = 0.01
        self.DEFAULT_INTERVAL = 5
        
        # Balance-based TP/SL system (percentage of balance)
        self.TP_PERSEN_BALANCE = 0.01      # 1% of balance
        self.SL_PERSEN_BALANCE = 0.005     # 0.5% of balance
        
        # HFT Settings  
        self.HFT_TP_PERSEN_BALANCE = 0.003  # 0.3% for HFT
        self.HFT_SL_PERSEN_BALANCE = 0.002  # 0.2% for HFT
        self.HFT_INTERVAL = 1               # 1 second
        
        # Scalping Settings
        self.SCALPING_TP_PERSEN_BALANCE = 0.005  # 0.5% for scalping
        self.SCALPING_SL_PERSEN_BALANCE = 0.003  # 0.3% for scalping
        self.SCALPING_OVERRIDE_ENABLED = True
        
        # Risk Management
        self.MAX_ORDER_PER_SESSION = 50
        self.MAX_ORDER_PER_SESSION_HFT = 200
        self.MAX_DRAWDOWN = 10  # 10%
        self.SALDO_MINIMAL = 100
        self.MAX_RISK_PER_TRADE = 2  # 2% per trade
        
        # Signal Thresholds
        self.SIGNAL_CONFIDENCE_THRESHOLD = 0.65
        self.SIGNAL_CONFIDENCE_THRESHOLD_HFT = 0.55
        self.SIGNAL_STRENGTH_MULTIPLIER = 1.5
        self.LONJAKAN_THRESHOLD = 8  # More sensitive threshold
        self.SKOR_MINIMAL = 3
        self.TREND_STRENGTH_MIN = 0.3
        
        # Trading Hours
        self.TRADING_START_HOUR = 0
        self.TRADING_END_HOUR = 23
        self.ENABLE_24_7_TRADING = True
        self.RESET_ORDER_HOUR = 0
        
        # Technical Indicators
        self.MA_PERIODS = [10, 20, 50]
        self.EMA_PERIODS = [9, 21]
        self.WMA_PERIODS = [5, 10]
        self.RSI_PERIOD = 14
        self.RSI_OVERSOLD = 30
        self.RSI_OVERBOUGHT = 70
        self.BB_PERIOD = 20
        self.BB_DEVIATION = 2
        
        # Data and Fetch Settings
        self.PRICE_FETCH_RETRY = 3
        self.DATA_BUFFER_SIZE = 100
        self.MIN_DATA_POINTS = 20
        
        # MetaTrader5 Settings
        self.MT5_MAGIC_NUMBER = 234000
        self.MT5_DEVIATION = 20
        self.MT5_TIMEOUT = 10000  # milliseconds
        
        # Files
        self.LOG_FILE = "trading_log.txt"
        self.TRADE_LOG_FILE = "trade_log.csv"
        
        # Telegram
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_bot_token_here")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your_chat_id_here")

# Global config instance
config = TradingConfig()
