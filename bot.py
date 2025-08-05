#!/usr/bin/env python3
"""
AuraTrade - Institutional-Level Python Trading Bot
Main Entry Point with Auto-Detection and Safe Launch
"""

import sys
import os
import traceback
import platform
import importlib.util

def check_and_install_requirements():
    """Check and install required packages if missing"""
    required_packages = [
        'requests', 'numpy', 'pandas', 'matplotlib', 
        'pillow', 'psutil', 'schedule', 'colorama'
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == 'pillow':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"🚀 Installing missing packages: {', '.join(missing_packages)}")
        import subprocess
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")

        # Test imports again
        print("🔄 Testing imports after installation...")
        for package in missing_packages:
            try:
                if package == 'pillow':
                    import PIL
                else:
                    __import__(package)
                print(f"✅ {package} now available")
            except ImportError:
                print(f"❌ {package} still not available")

def find_best_trading_bot():
    """Find the best available trading bot to run"""
    # Priority order: most advanced first
    bot_candidates = [
        {
            'path': 'tradebot/trading_bot_integrated.py',
            'name': 'Integrated Trading Bot (Best)',
            'class_name': 'TradingBot'
        },
        {
            'path': 'tradebot/trading_bot_windows.py', 
            'name': 'Windows MT5 Trading Bot',
            'class_name': 'TradingBotWindows'
        },
        {
            'path': 'tradebot/main.py',
            'name': 'Alternative Main',
            'class_name': None
        }
    ]

    for bot in bot_candidates:
        if os.path.exists(bot['path']):
            print(f"✅ Found: {bot['name']} at {bot['path']}")
            return bot

    print("❌ No trading bot found!")
    return None

def safe_import_module(file_path):
    """Safely import a module from file path"""
    try:
        # Add the tradebot directory to sys.path to fix imports
        tradebot_dir = os.path.join(os.getcwd(), 'tradebot')
        if tradebot_dir not in sys.path:
            sys.path.insert(0, tradebot_dir)

        spec = importlib.util.spec_from_file_location("trading_module", file_path)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"❌ Failed to import {file_path}: {e}")
        print(f"   Current sys.path: {sys.path[:3]}...")  # Show first 3 entries
        return None

def launch_trading_bot():
    """Launch the trading bot with proper error handling"""
    print("=" * 60)
    print("🤖 AURA TRADE - INSTITUTIONAL TRADING BOT")
    print("   Advanced HFT • Multi-Symbol • AI-Powered")
    print("=" * 60)

    # Check requirements first
    check_and_install_requirements()

    # Find best bot
    bot_info = find_best_trading_bot()
    if not bot_info:
        print("\n❌ No trading bot files found!")
        input("Press Enter to exit...")
        return False

    print(f"\n🚀 Launching: {bot_info['name']}")

    try:
        # Try to import and run the bot
        if bot_info['path'].endswith('.py'):
            # Import as module and run
            module = safe_import_module(bot_info['path'])
            if module:
                # Try different ways to start the bot
                if hasattr(module, 'main'):
                    print("📊 Starting bot via main() function...")
                    module.main()
                elif hasattr(module, bot_info['class_name']):
                    print(f"📊 Starting bot via {bot_info['class_name']} class...")
                    bot_class = getattr(module, bot_info['class_name'])
                    bot = bot_class()
                    if hasattr(bot, 'run'):
                        bot.run()
                    elif hasattr(bot, 'start'):
                        bot.start()
                    else:
                        print("✅ Bot class instantiated successfully")
                else:
                    print("📊 Executing module directly...")
                    # Module was imported, if it has startup code it should run
            else:
                print("❌ Failed to import bot module")
                return False

        return True

    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user (Ctrl+C)")
        return True
    except Exception as e:
        print(f"\n❌ Error launching bot: {e}")
        print(f"📝 Traceback:\n{traceback.format_exc()}")
        return False

def show_system_info():
    """Show system information"""
    print(f"🐍 Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"📁 Working Directory: {os.getcwd()}")

    # Check platform and MetaTrader5 availability
    import platform
    system = platform.system()
    print(f"🖥️  Platform: {system}")

    if system == "Windows":
        try:
            import MetaTrader5 as mt5
            print("✅ Windows MT5 ready for live trading")

            # Test MT5 initialization
            if mt5.initialize():
                account = mt5.account_info()
                if account:
                    print(f"🔗 MT5 Connected - Account: {account.login}")
                    print(f"💰 Balance: ${account.balance:,.2f}")
                else:
                    print("⚠️  MT5 terminal found but not logged in")
                mt5.shutdown()
            else:
                print("⚠️  MT5 terminal not running or not accessible")

        except ImportError:
            print("❌ MetaTrader5 library not installed")
            print("💡 Install with: pip install MetaTrader5")
    else:
        print("⚠️  Non-Windows platform - MT5 live trading not available")
        print("🔄 Will use simulation mode")

def main():
    """Main entry point"""
    try:
        show_system_info()
        success = launch_trading_bot()

        if not success:
            print("\n🔧 TROUBLESHOOTING:")
            print("1. Make sure all files are in place")
            print("2. Check internet connection for package installation")
            print("3. Try running from tradebot/ folder directly")
            print("4. Check the tradebot/README.md for specific instructions")

        print(f"\n{'='*60}")
        print("📞 Support: Check README files in tradebot/ folder")
        print("🔄 To restart: python bot.py")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n💥 Critical error in main(): {e}")
        print(f"📝 Traceback:\n{traceback.format_exc()}")

    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()