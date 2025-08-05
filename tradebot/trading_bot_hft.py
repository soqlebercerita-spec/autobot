#!/usr/bin/env python3
"""
HFT Trading Bot Launcher - High Frequency Trading Mode
Ultra-fast trading with 24/7 operation
"""

import sys
import threading
import time
from datetime import datetime

def launch_hft_mode():
    """Launch bot in HFT mode"""
    try:
        print("⚡ LAUNCHING HFT MODE")
        print("=" * 50)
        print("🚀 High-Frequency Trading Bot")
        print("⏰ 24/7 Operation: ENABLED")
        print("🔥 Ultra-Fast Execution: ENABLED") 
        print("📊 Max Speed: 10 trades/second")
        print("⚡ Scan Interval: 1 second")
        print("=" * 50)
        
        # Import HFT config
        from hft_config import hft_config
        hft_config.enable_hft_mode()
        
        # Import main bot
        from trading_bot_integrated import TradingBot
        
        # Create bot instance with HFT modifications
        bot = TradingBot()
        
        # Override default settings with HFT settings
        bot.interval_var.set("1")  # 1 second scanning
        bot.lot_var.set("0.01")    # Small lots for HFT
        bot.tp_var.set("0.1")      # 0.1% TP (very tight)
        bot.sl_var.set("0.3")      # 0.3% SL (tight)
        
        # Enable scalping mode (closest to HFT)
        bot.scalping_mode_var.set(True)
        
        # Log HFT activation
        bot.log("⚡ HFT MODE ACTIVATED!")
        bot.log("   • Ultra-fast execution enabled")
        bot.log("   • 24/7 operation active")
        bot.log("   • High-frequency scanning active") 
        bot.log("   • Tight profit targets set")
        bot.log("   • Max 10 trades/second capability")
        
        # Start GUI
        bot.root.mainloop()
        
    except Exception as e:
        print(f"❌ HFT launch error: {e}")
        return False

if __name__ == "__main__":
    launch_hft_mode()