# Enhanced Trading Bot Configuration
import os
from datetime import datetime

# Trading Parameters
DEFAULT_SYMBOL = "XAUUSDm"
DEFAULT_LOT = 0.01
DEFAULT_INTERVAL = 8
DATA_BUFFER_SIZE = 100
MIN_DATA_POINTS = 20

# Risk Management - Balance-based TP/SL
TP_PERSEN_BALANCE = 0.008  # 0.8% dari modal
SL_PERSEN_BALANCE = 0.04   # 4% dari modal

# Scalping Mode
SCALPING_TP_PERSEN_BALANCE = 0.005  # 0.5% dari modal
SCALPING_SL_PERSEN_BALANCE = 0.02   # 2% dari modal
SCALPING_OVERRIDE_ENABLED = False

# HFT Mode
HFT_TP_PERSEN_BALANCE = 0.003   # 0.3% dari modal
HFT_SL_PERSEN_BALANCE = 0.015   # 1.5% dari modal

# Price-based fallback TP/SL
TP_PERSEN_DEFAULT = 0.008
SL_PERSEN_DEFAULT = 0.04
SCALPING_TP_PERSEN = 0.003
SCALPING_SL_PERSEN = 0.015

# Signal Generation
LONJAKAN_THRESHOLD = 2.0  # Reduced for more opportunities
SIGNAL_CONFIDENCE_THRESHOLD = 0.4  # Reduced for more signals
SIGNAL_CONFIDENCE_THRESHOLD_HFT = 0.2  # Ultra-low for HFT
SIGNAL_STRENGTH_MULTIPLIER = 1.5

# Winrate Enhancement
WINRATE_BOOST_ENABLED = True
TREND_CONFIRMATION_PERIOD = 5
MULTI_CONFIRMATION_REQUIRED = 2

# Trading Limits
MAX_ORDER_PER_SESSION = 50
MAX_ORDER_PER_SESSION_HFT = 100
SALDO_MINIMAL = 1000

# Trading Hours
TRADING_START_HOUR = 0
TRADING_END_HOUR = 23
ENABLE_24_7_TRADING = True

# Price Fetching
PRICE_FETCH_RETRY = 5

# MT5 Settings
MT5_DEVIATION = 20
MT5_MAGIC_NUMBER = 234000

# Logging
LOG_FILE = "tradebot/trading_log.txt"
TRADE_LOG_FILE = "tradebot/trade_log.csv"

# Telegram
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"

# Create a config object for backward compatibility
class Config:
    def __init__(self):
        for key, value in globals().items():
            if not key.startswith('_') and key.isupper():
                setattr(self, key, value)

config = Config()