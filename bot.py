
"""
AuraTrade Bot - Enhanced Entry Point
Institutional-Level AI Auto Trading Bot with HFT Platform
Uses the working tradebot system
"""

import sys
import os
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

# Add tradebot folder to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tradebot'))

try:
    from config import config
    from trading_bot_integrated import TradingBot
    print("✅ Successfully imported working trading bot system")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("🔄 Trying alternative import...")
    try:
        from tradebot.config import config
        from tradebot.trading_bot_integrated import TradingBot
        print("✅ Alternative import successful")
    except ImportError as e2:
        print(f"❌ Alternative import failed: {e2}")
        sys.exit(1)

class AuraTradeBotLauncher:
    def __init__(self):
        """Initialize the AuraTrade Bot Launcher"""
        self.bot = None
        self.running = False
        print("🚀 AuraTrade Bot Launcher initialized")
        print(f"📊 Using Balance-Based TP/SL System")
        print(f"   • Normal TP: {config.TP_PERSEN_BALANCE*100}% of balance")
        print(f"   • Normal SL: {config.SL_PERSEN_BALANCE*100}% of balance")
        print(f"   • Scalping TP: {config.SCALPING_TP_PERSEN_BALANCE*100}% of balance")
        print(f"   • HFT TP: {config.HFT_TP_PERSEN_BALANCE*100}% of balance")
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        required_modules = [
            'numpy', 'pandas', 'requests', 'tkinter'
        ]
        
        missing = []
        for module in required_modules:
            try:
                __import__(module)
                print(f"✅ {module} available")
            except ImportError:
                missing.append(module)
                print(f"❌ {module} missing")
        
        return missing
    
    def show_startup_info(self):
        """Show startup information"""
        print("\n" + "="*60)
        print("🎯 AURATRADE - INSTITUTIONAL TRADING BOT")
        print("="*60)
        print("📈 Features:")
        print("   • HFT Engine (<1ms targeting)")
        print("   • Balance-Based TP/SL System")
        print("   • Multi-Symbol Trading (GOLD, FOREX, CRYPTO)")
        print("   • Custom Indicators (No TA-Lib)")
        print("   • Real-time GUI Interface")
        print("   • Telegram Notifications")
        print("   • Advanced Risk Management")
        print("   • Machine Learning Integration")
        print("\n🔧 Trading Modes Available:")
        print("   • Normal Trading (Balanced)")
        print("   • Scalping Mode (Quick profits)")
        print("   • HFT Mode (Ultra-fast execution)")
        print("="*60)
    
    def launch_gui_mode(self):
        """Launch bot with GUI interface"""
        try:
            print("🖥️ Starting GUI Mode...")
            self.bot = TradingBot()
            self.bot.root.protocol("WM_DELETE_WINDOW", self.on_close)
            self.bot.root.mainloop()
            return True
        except Exception as e:
            print(f"❌ GUI launch error: {e}")
            return False
    
    def launch_console_mode(self):
        """Launch bot in console mode (no GUI)"""
        try:
            print("💻 Starting Console Mode...")
            # Import the trading strategies directly
            from trading_strategies import TradingStrategies
            from market_data_api import MarketDataAPI
            from risk_manager import RiskManager
            
            # Initialize components
            market_data = MarketDataAPI()
            risk_manager = RiskManager()
            trading_strategies = TradingStrategies(market_data, risk_manager)
            
            print("✅ Console mode initialized")
            print("🔄 Starting trading loop... (Press Ctrl+C to stop)")
            
            self.running = True
            while self.running:
                try:
                    # Run a trading cycle
                    trading_strategies.run_strategies()
                    time.sleep(config.DEFAULT_INTERVAL)
                except KeyboardInterrupt:
                    print("\n👋 Stopping bot...")
                    self.running = False
                    break
                except Exception as e:
                    print(f"⚠️ Trading cycle error: {e}")
                    time.sleep(5)
            
            return True
        except Exception as e:
            print(f"❌ Console launch error: {e}")
            return False
    
    def on_close(self):
        """Handle window close event"""
        if messagebox.askokcancel("Quit", "Do you want to quit the trading bot?"):
            self.running = False
            if self.bot:
                try:
                    # Try to stop bot gracefully
                    if hasattr(self.bot, 'stop_trading'):
                        self.bot.stop_trading()
                except:
                    pass
            self.bot.root.destroy()
    
    def start(self, mode="auto"):
        """Start the trading bot"""
        try:
            self.show_startup_info()
            
            # Check dependencies
            missing = self.check_dependencies()
            if missing:
                print(f"\n⚠️ Missing dependencies: {', '.join(missing)}")
                print("📦 Installing missing packages...")
                
                import subprocess
                for package in missing:
                    try:
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                        print(f"✅ {package} installed")
                    except:
                        print(f"❌ Failed to install {package}")
            
            # Choose launch mode
            if mode == "auto":
                # Try GUI first, fallback to console
                print("\n🚀 Auto-detecting best launch mode...")
                if self.launch_gui_mode():
                    return
                else:
                    print("🔄 GUI failed, trying console mode...")
                    self.launch_console_mode()
            elif mode == "gui":
                self.launch_gui_mode()
            elif mode == "console":
                self.launch_console_mode()
            else:
                print("❌ Invalid mode. Use 'auto', 'gui', or 'console'")
                
        except Exception as e:
            print(f"❌ Critical startup error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point"""
    launcher = AuraTradeBotLauncher()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode in ['gui', 'console', 'auto']:
            launcher.start(mode)
        else:
            print("Usage: python bot.py [gui|console|auto]")
            print("Default: auto (tries GUI first, then console)")
    else:
        launcher.start("auto")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
