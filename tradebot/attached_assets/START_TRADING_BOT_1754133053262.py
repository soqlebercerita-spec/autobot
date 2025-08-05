#!/usr/bin/env python3
"""
🚀 TRADING BOT LAUNCHER - Windows MT5
Launcher untuk memulai trading bot dengan MT5
"""

import os
import sys
import subprocess
import platform

def check_requirements():
    """Cek requirements dan dependencies"""
    print("🔍 Checking system requirements...")
    
    # Check OS
    if platform.system() != "Windows":
        print("❌ This bot requires Windows OS for MetaTrader5 integration")
        return False
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        return False
    
    print(f"✅ Python {sys.version}")
    
    # Check required packages
    required_packages = ["MetaTrader5", "numpy", "requests"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} not found")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All packages installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_mt5():
    """Cek MT5 tersedia"""
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            terminal_info = mt5.terminal_info()
            if terminal_info:
                print(f"✅ MetaTrader5 connected: {terminal_info.name}")
                mt5.shutdown()
                return True
        print("⚠️  MetaTrader5 found but not connected")
        print("   Please ensure MT5 is running and logged in")
        return True
    except ImportError:
        print("⚠️  MetaTrader5 package not installed")
        return False
    except Exception as e:
        print(f"❌ MetaTrader5 error: {e}")
        return False

def main():
    """Main launcher function"""
    print("=" * 60)
    print("🤖 TRADING BOT LAUNCHER")
    print("   Auto Trading & Scalping untuk MetaTrader5")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        input("Press Enter to exit...")
        return
    
    # Check MT5
    if not check_mt5():
        print("\n⚠️  MT5 not available - will run in simulation mode")
    
    print("\n🚀 Starting Trading Bot...")
    
    # Try to run Windows version first
    if os.path.exists("trading_bot_windows.py"):
        print("📊 Loading Windows MT5 version...")
        try:
            subprocess.run([sys.executable, "trading_bot_windows.py"])
        except KeyboardInterrupt:
            print("\n👋 Bot stopped by user")
        except Exception as e:
            print(f"❌ Error running Windows version: {e}")
            print("🔄 Trying simulation version...")
            try:
                subprocess.run([sys.executable, "trading_bot_integrated.py"])
            except Exception as e2:
                print(f"❌ Error running simulation: {e2}")
    else:
        print("📊 Loading simulation version...")
        try:
            subprocess.run([sys.executable, "trading_bot_integrated.py"])
        except KeyboardInterrupt:
            print("\n👋 Bot stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Trading Bot session ended")
    input("Press Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Launcher error: {e}")
        input("Press Enter to exit...")