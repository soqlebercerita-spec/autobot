
#!/usr/bin/env python3
"""
AuraTrade System Status Checker
Quick diagnostic tool
"""

import sys
import os
import importlib.util

def check_all_systems():
    """Comprehensive system check"""
    print("🔍 AURA TRADE - SYSTEM STATUS CHECK")
    print("=" * 50)
    
    # Check Python version
    print(f"🐍 Python Version: {sys.version.split()[0]}")
    if sys.version_info < (3, 11):
        print("⚠️  Warning: Python 3.11+ recommended")
    else:
        print("✅ Python version OK")
    
    # Check core dependencies  
    core_deps = ['numpy', 'pandas', 'requests', 'matplotlib']
    print(f"\n📦 Checking Core Dependencies:")
    for dep in core_deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - MISSING")
    
    # Check trading bot files
    print(f"\n📁 Checking Trading Bot Files:")
    critical_files = [
        'tradebot/trading_bot_integrated.py',
        'tradebot/config.py', 
        'tradebot/market_data_api.py',
        'config/config.py'
    ]
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
    
    # Check configuration
    print(f"\n⚙️  Checking Configuration:")
    try:
        from config.config import Config
        print("✅ Configuration loaded")
        print(f"   TP: {Config.TP_PERCENT}% | SL: {Config.SL_PERCENT}%")
        print(f"   Symbols: {len(Config.SYMBOLS)} configured")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
    
    # Check MT5 availability
    print(f"\n🏦 Checking MT5:")
    try:
        import MetaTrader5
        print("✅ MetaTrader5 available")
    except ImportError:
        print("⚠️  MetaTrader5 not available (simulation mode only)")
    
    print(f"\n{'='*50}")
    print("💡 To start trading: python bot.py")
    print("📖 For help: check tradebot/README.md")

if __name__ == "__main__":
    check_all_systems()
