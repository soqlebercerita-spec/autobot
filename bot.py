
#!/usr/bin/env python3
"""
Enhanced AuraTrade Bot Launcher - Fixed Version
Institutional-level trading bot with advanced features
"""

import sys
import os
import subprocess
import importlib.util
import platform
from pathlib import Path

# Enhanced package installation
def install_package(package_name, import_name=None):
    """Install package with better error handling"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} already available")
        return True
    except ImportError:
        print(f"🔄 Installing {package_name}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--user", "--upgrade", package_name
            ])
            # Try importing again
            importlib.import_module(import_name)
            print(f"✅ Successfully installed {package_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False

def main():
    """Enhanced main launcher"""
    print("🐍 Python:", sys.version.split()[0])
    print("📁 Working Directory:", os.getcwd())
    print("🖥️  Platform:", platform.system())
    
    # Check platform and MT5 availability
    is_windows = platform.system() == "Windows"
    if not is_windows:
        print("⚠️  Non-Windows platform - MT5 live trading not available")
        print("🔄 Will use simulation mode")
    else:
        print("✅ Windows platform - MT5 live trading available")
    
    print("=" * 60)
    print("🤖 AURA TRADE - INSTITUTIONAL TRADING BOT")
    print("   Advanced HFT • Multi-Symbol • AI-Powered")
    print("=" * 60)
    
    # Install required packages
    required_packages = [
        "requests", "numpy", "pandas", "matplotlib", 
        "pillow", "psutil", "schedule", "colorama"
    ]
    
    print(f"🚀 Installing required packages: {', '.join(required_packages)}")
    
    success_count = 0
    for package in required_packages:
        if install_package(package):
            success_count += 1
    
    if success_count != len(required_packages):
        print(f"⚠️  Only {success_count}/{len(required_packages)} packages installed successfully")
    
    # Force reload Python path
    importlib.invalidate_caches()
    
    # Try to find and launch the best available bot
    bot_options = [
        ("tradebot/trading_bot_integrated.py", "Integrated Trading Bot (Best)"),
        ("tradebot/trading_bot_windows.py", "Windows MT5 Trading Bot"),
        ("tradebot/trading_bot_real.py", "Real Money Trading Bot"),
        ("core/trading_engine.py", "Core Trading Engine"),
    ]
    
    for bot_path, bot_name in bot_options:
        if os.path.exists(bot_path):
            print(f"✅ Found: {bot_name} at {bot_path}")
            try:
                # Change to bot directory
                bot_dir = os.path.dirname(bot_path)
                if bot_dir:
                    sys.path.insert(0, bot_dir)
                
                # Import and run
                spec = importlib.util.spec_from_file_location("trading_bot", bot_path)
                if spec and spec.loader:
                    print(f"\n🚀 Launching: {bot_name}")
                    trading_bot = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(trading_bot)
                    
                    # Try to run main function
                    if hasattr(trading_bot, 'main'):
                        trading_bot.main()
                    elif hasattr(trading_bot, 'TradingBot'):
                        # Create and run bot instance
                        bot = trading_bot.TradingBot()
                        if hasattr(bot, 'root'):
                            bot.root.mainloop()
                    return
                    
            except Exception as e:
                print(f"❌ Failed to launch {bot_name}: {e}")
                continue
    
    print("❌ No working trading bot found")
    print("\n🔧 TROUBLESHOOTING:")
    print("1. Make sure all files are in place")
    print("2. Check internet connection for package installation") 
    print("3. Try running from tradebot/ folder directly")
    print("4. Check the tradebot/README.md for specific instructions")
    
    print("\n" + "=" * 60)
    print("📞 Support: Check README files in tradebot/ folder")
    print("🔄 To restart: python bot.py")
    print("=" * 60)
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
