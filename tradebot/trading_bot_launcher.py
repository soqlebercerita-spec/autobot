#!/usr/bin/env python3
"""
Enhanced Trading Bot Launcher - Safe Startup
Detects issues and provides safe fallback options
"""

import sys
import os
import traceback

def safe_import_test():
    """Test all imports safely"""
    print("🔍 Testing imports...")
    
    try:
        from trading_bot_integrated import TradingBot
        print("✅ Main trading bot import successful")
        return True
    except Exception as e:
        print(f"❌ Trading bot import failed: {e}")
        return False

def safe_dependency_test():
    """Test dependencies safely"""
    print("🔍 Testing dependencies...")
    
    missing_deps = []
    
    try:
        import numpy
        print("✅ numpy available")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import requests
        print("✅ requests available")
    except ImportError:
        missing_deps.append("requests")
    
    try:
        import tkinter
        print("✅ tkinter available")
    except ImportError:
        missing_deps.append("tkinter")
    
    return missing_deps

def launch_bot():
    """Launch bot with error handling"""
    try:
        print("🚀 Enhanced Trading Bot Launcher")
        print("=" * 50)
        
        # Test dependencies
        missing = safe_dependency_test()
        if missing:
            print(f"❌ Missing dependencies: {', '.join(missing)}")
            return False
        
        # Test imports
        if not safe_import_test():
            print("❌ Import test failed")
            return False
        
        print("🎯 All tests passed! Starting bot...")
        print("=" * 50)
        
        # Import and run
        from trading_bot_integrated import TradingBot
        bot = TradingBot()
        bot.root.mainloop()
        
        return True
        
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
        return True
    except Exception as e:
        print(f"❌ Launch error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Check if all dependencies are installed")
        print("2. Verify environment configuration")
        print("3. Try running: python trading_bot_integrated.py")
        print(f"\nFull error:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = launch_bot()
    if not success:
        sys.exit(1)