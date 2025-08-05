"""
AuraTrade Bot - Main Entry Point
Institutional-Level AI Auto Trading Bot with HFT Platform
"""

import sys
import os
import threading
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from core.trading_engine import TradingEngine
from core.mt5_connector import MT5Connector
from gui.main_window import MainWindow
from utils.logger import Logger
from utils.notifier import Notifier

class AuraTradingBot:
    def __init__(self):
        """Initialize the AuraTrade Bot"""
        self.config = Config()
        self.logger = Logger()
        self.notifier = Notifier()
        self.mt5_connector = None
        self.trading_engine = None
        self.gui = None
        self.running = False
        
        self.logger.info("AuraTrade Bot initialized")
    
    def initialize(self):
        """Initialize all components"""
        try:
            # Initialize MT5 connector
            self.mt5_connector = MT5Connector()
            if not self.mt5_connector.initialize():
                self.logger.error("Failed to initialize MT5 connector")
                return False
            
            # Initialize trading engine
            self.trading_engine = TradingEngine(self.mt5_connector)
            
            # Initialize GUI
            self.gui = MainWindow(self.trading_engine, self.mt5_connector)
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False
    
    def start_trading(self):
        """Start the trading bot"""
        if not self.initialize():
            return
        
        self.running = True
        
        # Start trading engine in separate thread
        trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        trading_thread.start()
        
        # Start GUI (main thread)
        self.gui.run()
    
    def _trading_loop(self):
        """Main trading loop"""
        self.logger.info("Trading loop started")
        
        while self.running:
            try:
                # Run trading engine cycle
                self.trading_engine.run_cycle()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.001)  # 1ms for HFT simulation
                
            except Exception as e:
                self.logger.error(f"Trading loop error: {str(e)}")
                time.sleep(1)  # Longer delay on error
    
    def stop(self):
        """Stop the trading bot"""
        self.running = False
        if self.trading_engine:
            self.trading_engine.stop()
        if self.mt5_connector:
            self.mt5_connector.shutdown()
        
        self.logger.info("AuraTrade Bot stopped")

if __name__ == "__main__":
    bot = AuraTradingBot()
    try:
        bot.start_trading()
    except KeyboardInterrupt:
        print("\nShutting down...")
        bot.stop()
    except Exception as e:
        print(f"Critical error: {e}")
        bot.stop()
