#!/usr/bin/env python3
"""
Quick launcher for Enhanced Trading Bot
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main bot
if __name__ == "__main__":
    try:
        from trading_bot_integrated import TradingBot
        
        print("🚀 Starting Enhanced Trading Bot...")
        print("✅ Fixed Price Retrieval")
        print("⚡ Optimized Signal Generation") 
        print("🎯 Enhanced Opportunity Capture")
        print("🛡️ Safe Virtual Trading")
        print()
        
        # Create and run bot
        bot = TradingBot()
        bot.root.mainloop()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check the error logs.")