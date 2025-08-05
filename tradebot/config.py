"""
Enhanced Configuration settings for the Trading Bot
Optimized for better opportunity capture
"""

import os

class TradingConfig:
    def __init__(self):
        # Telegram Configuration
        self.TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "your_bot_token_here")
        self.TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "your_chat_id_here")
        
        # UNIFIED BALANCE-BASED TP/SL SYSTEM - SEMUA PAKAI MODAL CALCULATION
        # Normal Trading Mode
        self.TP_PERSEN_BALANCE = 0.01   # 1% dari modal untuk TP normal
        self.SL_PERSEN_BALANCE = 0.03   # 3% dari modal untuk SL normal (lebih aman)
        
        # Scalping Trading Mode  
        self.SCALPING_TP_PERSEN_BALANCE = 0.005  # 0.5% dari modal untuk scalping TP
        self.SCALPING_SL_PERSEN_BALANCE = 0.02   # 2% dari modal untuk scalping SL
        
        # HFT Trading Mode
        self.HFT_TP_PERSEN_BALANCE = 0.003   # 0.3% dari modal untuk HFT TP (sangat ketat)
        self.HFT_SL_PERSEN_BALANCE = 0.015   # 1.5% dari modal untuk HFT SL (tighter control)
        
        # Trading Mode Settings
        self.SCALPING_OVERRIDE_ENABLED = True
        self.BALANCE_BASED_ALWAYS = True  # Force balance-based calculation
        
        # Legacy compatibility (for backward compatibility)
        self.TP_PERSEN_DEFAULT = self.TP_PERSEN_BALANCE
        self.SL_PERSEN_DEFAULT = self.SL_PERSEN_BALANCE
        self.SCALPING_TP_PERSEN = self.SCALPING_TP_PERSEN_BALANCE
        self.SCALPING_SL_PERSEN = self.SCALPING_SL_PERSEN_BALANCE
        
        # Enhanced Risk Management - HFT Scalping Support
        self.MAX_ORDER_PER_SESSION = 50     # Significantly increased for HFT scalping
        self.MAX_ORDER_PER_SESSION_HFT = 100  # Special limit for HFT mode
        self.SALDO_MINIMAL = 100  # Reduced minimum balance for more accessibility
        self.TARGET_PROFIT_PERSEN = 12  # Increased to 12% for better performance
        self.LONJAKAN_THRESHOLD = 2     # Further reduced to 2% for more opportunities
        self.SKOR_MINIMAL = 2           # Reduced to 2 for higher winrate
        self.MAX_RISK_PER_TRADE = 1.0   # Reduced to 1.0% for safer trading
        self.MAX_DRAWDOWN = 10.0        # Reduced to 10% for tighter control
        
        # Trading Hours - 24/7 UNLIMITED
        self.TRADING_START_HOUR = 0     # 24/7 operation
        self.TRADING_END_HOUR = 23      # 24/7 operation  
        self.RESET_ORDER_HOUR = 0       # Daily reset only
        self.ENABLE_24_7_TRADING = True # 24/7 mode enabled
        
        # Enhanced Technical Settings
        self.TRAILING_STOP_PIPS = 30    # Reduced for quicker exits
        self.DEFAULT_SYMBOL = "XAUUSDm"
        self.DEFAULT_LOT = 0.01
        self.DEFAULT_INTERVAL = 8       # Faster scanning - reduced from 10
        self.HFT_INTERVAL = 1            # HFT ultra-fast scanning (1 second)
        
        # MetaTrader5 Settings
        self.MT5_MAGIC_NUMBER = 234000
        self.MT5_DEVIATION = 30         # Increased slippage tolerance
        self.MT5_TIMEOUT = 15000        # Increased timeout
        
        # Optimized Indicator Settings for Better Signals
        self.MA_PERIODS = [8, 15, 30]   # Shorter periods for faster signals
        self.EMA_PERIODS = [7, 17, 40]  # Optimized EMA periods
        self.WMA_PERIODS = [4, 8]       # Shorter WMA periods
        self.RSI_PERIOD = 12            # Shorter RSI for more sensitivity
        self.RSI_OVERSOLD = 35          # More realistic oversold level
        self.RSI_OVERBOUGHT = 65        # More realistic overbought level
        self.BB_PERIOD = 16             # Shorter Bollinger Band period
        self.BB_DEVIATION = 1.8         # Tighter bands for more signals
        
        # Enhanced Market Data Settings
        self.PRICE_FETCH_RETRY = 3      # Retry failed requests
        self.PRICE_FETCH_TIMEOUT = 5    # Seconds timeout
        self.DATA_BUFFER_SIZE = 100     # Larger data buffer
        self.MIN_DATA_POINTS = 30       # Reduced minimum data requirement
        
        # Signal Enhancement Settings - Higher Winrate
        self.SIGNAL_CONFIDENCE_THRESHOLD = 0.25  # Even lower for more signals
        self.SIGNAL_CONFIDENCE_THRESHOLD_HFT = 0.15  # Ultra-low for HFT mode
        self.TREND_STRENGTH_MIN = 0.15          # Lower minimum trend strength
        self.VOLUME_THRESHOLD = 0.25            # Lower volume threshold
        
        # Winrate Enhancement Settings
        self.WINRATE_BOOST_ENABLED = True      # Enable winrate boost features
        self.MULTI_CONFIRMATION_REQUIRED = 2   # Require 2+ indicators agreement
        self.TREND_CONFIRMATION_PERIOD = 5     # Look back 5 periods for trend
        self.SIGNAL_STRENGTH_MULTIPLIER = 1.5  # Boost strong signals
        
        # File Paths
        self.LOG_FILE = "trading_log.txt"
        self.TRADE_LOG_FILE = "trade_log.csv"
        self.SETTINGS_FILE = "bot_settings.json"
        self.ERROR_LOG_FILE = "error_log.txt"
        self.PERFORMANCE_LOG_FILE = "performance_log.json"

# Global config instance
config = TradingConfig()
