#!/usr/bin/env python3
"""
Multi-Mode Trading Bot - 3 Separate Forms
Normal Trading, Scalping Mode, and HFT Mode
"""

import time
import datetime
import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.scrolledtext import ScrolledText
import tkinter.messagebox as messagebox
import requests
import csv
import os
import json

# Handle numpy import gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class MockNumpy:
        @staticmethod
        def array(data): return data
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data):
            if not data: return 0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return variance ** 0.5
    np = MockNumpy()

from enhanced_indicators import EnhancedIndicators
from config import config
from mt5_wrapper import mt5, MT5_AVAILABLE

class TradingModeSelector:
    """Main window to select trading mode"""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 Multi-Mode Trading Bot - Select Trading Mode")
        self.root.geometry("600x400")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)
        
        self.create_mode_selector()
        
    def create_mode_selector(self):
        """Create the mode selection interface"""
        # Title
        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.pack(pady=20)
        
        title_label = tk.Label(title_frame, text="🚀 Multi-Mode Trading Bot", 
                              font=("Arial", 20, "bold"), bg="#f0f0f0")
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Select Your Trading Mode", 
                                 font=("Arial", 12), bg="#f0f0f0", fg="gray")
        subtitle_label.pack(pady=5)
        
        # Mode buttons frame
        modes_frame = tk.Frame(self.root, bg="#f0f0f0")
        modes_frame.pack(pady=30, padx=50, fill="both", expand=True)
        
        # Normal Trading Mode
        normal_frame = tk.LabelFrame(modes_frame, text="📊 Normal Trading Mode", 
                                   font=("Arial", 12, "bold"), bg="#e8f4f8", 
                                   fg="#2c3e50", padx=10, pady=10)
        normal_frame.pack(fill="x", pady=10)
        
        tk.Label(normal_frame, text="• Conservative approach with balanced risk", 
                bg="#e8f4f8", font=("Arial", 10)).pack(anchor="w")
        tk.Label(normal_frame, text="• TP: 1% of balance | SL: 3% of balance", 
                bg="#e8f4f8", font=("Arial", 10)).pack(anchor="w")
        tk.Label(normal_frame, text="• Suitable for steady long-term growth", 
                bg="#e8f4f8", font=("Arial", 10)).pack(anchor="w")
        
        tk.Button(normal_frame, text="🔵 Start Normal Mode", 
                 command=self.start_normal_mode, font=("Arial", 12, "bold"),
                 bg="#3498db", fg="white", pady=5).pack(fill="x", pady=5)
        
        # Scalping Mode
        scalping_frame = tk.LabelFrame(modes_frame, text="⚡ Scalping Mode", 
                                     font=("Arial", 12, "bold"), bg="#f0f8e8", 
                                     fg="#e67e22", padx=10, pady=10)
        scalping_frame.pack(fill="x", pady=10)
        
        tk.Label(scalping_frame, text="• Quick trades with small profits", 
                bg="#f0f8e8", font=("Arial", 10)).pack(anchor="w")
        tk.Label(scalping_frame, text="• TP: 0.5% of balance | SL: 2% of balance", 
                bg="#f0f8e8", font=("Arial", 10)).pack(anchor="w")
        tk.Label(scalping_frame, text="• Fast-paced trading with tight controls", 
                bg="#f0f8e8", font=("Arial", 10)).pack(anchor="w")
        
        tk.Button(scalping_frame, text="🟡 Start Scalping Mode", 
                 command=self.start_scalping_mode, font=("Arial", 12, "bold"),
                 bg="#f39c12", fg="white", pady=5).pack(fill="x", pady=5)
        
        # HFT Mode
        hft_frame = tk.LabelFrame(modes_frame, text="🚀 HFT (High Frequency Trading)", 
                                font=("Arial", 12, "bold"), bg="#f8e8e8", 
                                fg="#e74c3c", padx=10, pady=10)
        hft_frame.pack(fill="x", pady=10)
        
        tk.Label(hft_frame, text="• Ultra-fast automated trading", 
                bg="#f8e8e8", font=("Arial", 10)).pack(anchor="w")
        tk.Label(hft_frame, text="• TP: 0.3% of balance | SL: 1.5% of balance", 
                bg="#f8e8e8", font=("Arial", 10)).pack(anchor="w")  
        tk.Label(hft_frame, text="• Maximum 100 trades per session", 
                bg="#f8e8e8", font=("Arial", 10)).pack(anchor="w")
        
        tk.Button(hft_frame, text="🔴 Start HFT Mode", 
                 command=self.start_hft_mode, font=("Arial", 12, "bold"),
                 bg="#e74c3c", fg="white", pady=5).pack(fill="x", pady=5)
        
        # Status frame
        status_frame = tk.Frame(self.root, bg="#f0f0f0")
        status_frame.pack(pady=10)
        
        mt5_status = "✅ Available" if MT5_AVAILABLE else "❌ Not Available (Simulation Mode)"
        tk.Label(status_frame, text=f"MetaTrader5 Status: {mt5_status}", 
                font=("Arial", 10), bg="#f0f0f0").pack()
        
    def start_normal_mode(self):
        """Start Normal Trading Mode"""
        self.root.destroy()
        bot = NormalTradingBot()
        bot.run()
        
    def start_scalping_mode(self):
        """Start Scalping Mode"""
        self.root.destroy()
        bot = ScalpingTradingBot()
        bot.run()
        
    def start_hft_mode(self):
        """Start HFT Mode"""
        self.root.destroy()
        bot = HFTTradingBot()
        bot.run()
        
    def run(self):
        """Run the mode selector"""
        self.root.mainloop()

class BaseTradingBot:
    """Base class for all trading modes"""
    def __init__(self, mode_name="Base"):
        self.mode_name = mode_name
        self.indicators = EnhancedIndicators()
        self.running = False
        self.modal_awal = None
        self.last_price = None
        self.order_counter = 0
        self.total_opportunities_captured = 0
        self.total_opportunities_missed = 0
        self.bot_thread = None
        
    def setup_base_gui(self):
        """Setup base GUI elements"""
        self.root = tk.Tk()
        self.root.title(f"🚀 {self.mode_name} Trading Bot")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Base variables
        self.symbol_var = tk.StringVar(value=config.DEFAULT_SYMBOL)
        self.lot_var = tk.StringVar(value=str(config.DEFAULT_LOT))
        self.interval_var = tk.StringVar(value=str(config.DEFAULT_INTERVAL))
        self.account_info_var = tk.StringVar(value="Account: Not Connected")
        self.profit_var = tk.StringVar(value="Real-time P/L: -")
        self.balance_var = tk.StringVar(value="$0.00")
        self.opportunities_var = tk.StringVar(value="Opportunities: Captured: 0 | Missed: 0")
        
    def calculate_proper_tp_sl(self, signal, current_price, tp_balance_pct, sl_balance_pct):
        """Calculate TP/SL based on ACCOUNT BALANCE percentage for REAL TRADING"""
        try:
            # Get REAL account balance from MT5
            if not MT5_AVAILABLE:
                self.log("❌ MT5 not available - cannot calculate balance-based TP/SL")
                # Emergency fallback
                if signal == "BUY":
                    return current_price * 1.01, current_price * 0.99
                else:
                    return current_price * 0.99, current_price * 1.01
            
            account_info = mt5.account_info()
            if account_info is None:
                self.log("❌ Cannot get MT5 account info")
                # Emergency fallback
                if signal == "BUY":
                    return current_price * 1.01, current_price * 0.99
                else:
                    return current_price * 0.99, current_price * 1.01
            
            balance = account_info.balance
            symbol = self.symbol_var.get()
            lot_size = float(self.lot_var.get())
            
            # Calculate target profit/loss in MONEY
            tp_money_target = balance * tp_balance_pct
            sl_money_limit = balance * sl_balance_pct
            
            # Get symbol information for conversion
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.log(f"❌ Cannot get symbol info for {symbol}")
                # Emergency fallback
                if signal == "BUY":
                    return current_price * 1.01, current_price * 0.99
                else:
                    return current_price * 0.99, current_price * 1.01
            
            # Calculate contract specifications
            contract_size = symbol_info.trade_contract_size
            point = symbol_info.point
            pip_size = 10 * point  # Standard pip size
            
            # Calculate pip value for our lot size
            pip_value = contract_size * pip_size * lot_size
            
            # Convert money targets to pips
            tp_pips = tp_money_target / pip_value if pip_value > 0 else 50
            sl_pips = sl_money_limit / pip_value if pip_value > 0 else 100
            
            # Convert pips to price levels
            if signal == "BUY":
                tp_price = current_price + (tp_pips * pip_size)
                sl_price = current_price - (sl_pips * pip_size)
            else:  # SELL
                tp_price = current_price - (tp_pips * pip_size)
                sl_price = current_price + (sl_pips * pip_size)
            
            # Validate results
            if tp_price <= 0 or sl_price <= 0:
                raise ValueError("Invalid TP/SL calculation")
            
            self.log(f"💰 BALANCE-BASED TP/SL Calculation:")
            self.log(f"   Account Balance: ${balance:,.2f}")
            self.log(f"   TP Money Target: ${tp_money_target:,.2f} ({tp_balance_pct*100:.1f}% of balance)")
            self.log(f"   SL Money Limit: ${sl_money_limit:,.2f} ({sl_balance_pct*100:.1f}% of balance)")
            self.log(f"   TP Price: {tp_price:.5f}")
            self.log(f"   SL Price: {sl_price:.5f}")
            
            return tp_price, sl_price
            
        except Exception as e:
            self.log(f"❌ Balance-based TP/SL calculation error: {e}")
            # Safe fallback
            if signal == "BUY":
                return current_price * 1.01, current_price * 0.99
            else:
                return current_price * 0.99, current_price * 1.01
    
    def log(self, message):
        """Base logging method"""
        if hasattr(self, 'log_text'):
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"{timestamp} - {message}\n")
            self.log_text.see(tk.END)
            self.root.update_idletasks()
    
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            self.stop_bot()
        self.root.destroy()
    
    def run(self):
        """Run the trading bot"""
        self.root.mainloop()

class NormalTradingBot(BaseTradingBot):
    """Normal Trading Mode Bot"""
    def __init__(self):
        super().__init__("Normal")
        self.setup_base_gui()
        self.create_normal_gui()
        
    def create_normal_gui(self):
        """Create Normal Trading GUI"""
        # Main title
        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.pack(pady=10, fill="x")
        
        tk.Label(title_frame, text="📊 Normal Trading Mode", 
                font=("Arial", 18, "bold"), bg="#f0f0f0").pack()
        tk.Label(title_frame, text="Conservative approach with balanced risk management", 
                font=("Arial", 10), bg="#f0f0f0", fg="gray").pack()
        
        # Configuration frame
        config_frame = tk.LabelFrame(self.root, text="Trading Configuration", 
                                   font=("Arial", 12, "bold"))
        config_frame.pack(pady=10, padx=20, fill="x")
        
        # Symbol and lot
        row1_frame = tk.Frame(config_frame)
        row1_frame.pack(fill="x", pady=5)
        
        tk.Label(row1_frame, text="Symbol:", width=10).pack(side="left")
        tk.Entry(row1_frame, textvariable=self.symbol_var, width=15).pack(side="left", padx=5)
        
        tk.Label(row1_frame, text="Lot Size:", width=10).pack(side="left", padx=(20,0))
        tk.Entry(row1_frame, textvariable=self.lot_var, width=10).pack(side="left", padx=5)
        
        tk.Label(row1_frame, text="Interval (s):", width=12).pack(side="left", padx=(20,0))
        tk.Entry(row1_frame, textvariable=self.interval_var, width=10).pack(side="left", padx=5)
        
        # TP/SL settings
        tp_sl_frame = tk.Frame(config_frame)
        tp_sl_frame.pack(fill="x", pady=5)
        
        tk.Label(tp_sl_frame, text="Fixed TP: 1% | Fixed SL: 3%", 
                font=("Arial", 10, "bold"), fg="blue").pack(side="left")
        
        # Control buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=15)
        
        tk.Button(control_frame, text="🔵 Start Normal Trading", 
                 command=self.start_bot, font=("Arial", 12, "bold"),
                 bg="#3498db", fg="white", padx=20, pady=5).pack(side="left", padx=10)
        
        tk.Button(control_frame, text="⏹️ Stop Bot", 
                 command=self.stop_bot, font=("Arial", 12, "bold"),
                 bg="#e74c3c", fg="white", padx=20, pady=5).pack(side="left", padx=10)
        
        # Account info
        info_frame = tk.LabelFrame(self.root, text="Account Information")
        info_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(info_frame, textvariable=self.account_info_var).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.profit_var).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.opportunities_var).pack(anchor="w")
        
        # Log area
        log_frame = tk.LabelFrame(self.root, text="Trading Log")
        log_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.log_text = ScrolledText(log_frame, height=15, width=80)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def start_bot(self):
        """Start normal trading bot"""
        if self.running:
            messagebox.showwarning("Already Running", "Normal trading bot is already running!")
            return
            
        self.running = True
        self.log("🔵 Starting Normal Trading Mode...")
        self.log("📊 Configuration: TP=1%, SL=3%, Conservative approach")
        
        self.bot_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.bot_thread.start()
        
    def stop_bot(self):
        """Stop trading bot"""
        self.running = False
        self.log("⏹️ Normal trading bot stopped")
        
    def trading_loop(self):
        """Main trading loop for normal mode"""
        while self.running:
            try:
                signal, tp, sl = self.check_trading_signal()
                if signal:
                    self.log(f"🎯 Normal Signal: {signal} | TP: {tp:.5f} | SL: {sl:.5f}")
                    # Execute trade logic here
                    
                interval = int(self.interval_var.get())
                time.sleep(interval)
                
            except Exception as e:
                self.log(f"❌ Trading loop error: {e}")
                time.sleep(5)
                
    def check_trading_signal(self):
        """Check for trading signals in normal mode"""
        try:
            # Get REAL price from MT5
            symbol = self.symbol_var.get()
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    current_price = (tick.ask + tick.bid) / 2
                else:
                    self.log(f"❌ Cannot get real price for {symbol}")
                    return None, 0, 0
            else:
                self.log("❌ MT5 not available - cannot get real prices")
                return None, 0, 0
            
            # Normal mode uses balance-based TP/SL
            tp_balance_pct = config.TP_PERSEN_BALANCE      # 1% of balance
            sl_balance_pct = config.SL_PERSEN_BALANCE      # 3% of balance
            
            # Generate real signal (placeholder - add real indicator logic)
            signal = "BUY"  # Replace with real signal logic
            tp, sl = self.calculate_proper_tp_sl(signal, current_price, tp_balance_pct, sl_balance_pct)
            
            return signal, tp, sl
            
        except Exception as e:
            self.log(f"❌ Signal check error: {e}")
            return None, 0, 0

class ScalpingTradingBot(BaseTradingBot):
    """Scalping Trading Mode Bot"""
    def __init__(self):
        super().__init__("Scalping")
        self.setup_base_gui()
        self.create_scalping_gui()
        
    def create_scalping_gui(self):
        """Create Scalping Trading GUI"""
        # Main title
        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.pack(pady=10, fill="x")
        
        tk.Label(title_frame, text="⚡ Scalping Trading Mode", 
                font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#e67e22").pack()
        tk.Label(title_frame, text="Quick trades with small profits and tight risk control", 
                font=("Arial", 10), bg="#f0f0f0", fg="gray").pack()
        
        # Similar GUI structure but with scalping-specific settings
        config_frame = tk.LabelFrame(self.root, text="Scalping Configuration", 
                                   font=("Arial", 12, "bold"))
        config_frame.pack(pady=10, padx=20, fill="x")
        
        # Fixed scalping settings
        tk.Label(config_frame, text="Fixed TP: 0.5% | Fixed SL: 2% | Fast Scanning", 
                font=("Arial", 10, "bold"), fg="#e67e22").pack(pady=5)
        
        # Rest of GUI similar to normal mode but with scalping branding
        self.create_common_gui_elements()
        
    def create_common_gui_elements(self):
        """Create common GUI elements"""
        # Control buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=15)
        
        tk.Button(control_frame, text="🟡 Start Scalping", 
                 command=self.start_bot, font=("Arial", 12, "bold"),
                 bg="#f39c12", fg="white", padx=20, pady=5).pack(side="left", padx=10)
        
        tk.Button(control_frame, text="⏹️ Stop Bot", 
                 command=self.stop_bot, font=("Arial", 12, "bold"),
                 bg="#e74c3c", fg="white", padx=20, pady=5).pack(side="left", padx=10)
        
        # Account info and log (same as normal mode)
        info_frame = tk.LabelFrame(self.root, text="Account Information")
        info_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(info_frame, textvariable=self.account_info_var).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.profit_var).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.opportunities_var).pack(anchor="w")
        
        log_frame = tk.LabelFrame(self.root, text="Scalping Log")
        log_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.log_text = ScrolledText(log_frame, height=15, width=80)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def start_bot(self):
        """Start scalping bot"""
        if self.running:
            messagebox.showwarning("Already Running", "Scalping bot is already running!")
            return
            
        self.running = True
        self.log("🟡 Starting Scalping Mode...")
        self.log("⚡ Configuration: TP=0.5%, SL=2%, Fast execution")
        
        self.bot_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.bot_thread.start()
        
    def stop_bot(self):
        """Stop scalping bot"""
        self.running = False
        self.log("⏹️ Scalping bot stopped")
        
    def trading_loop(self):
        """Main trading loop for scalping mode"""
        while self.running:
            try:
                signal, tp, sl = self.check_trading_signal()
                if signal:
                    self.log(f"⚡ Scalping Signal: {signal} | TP: {tp:.5f} | SL: {sl:.5f}")
                    
                # Faster scanning for scalping
                time.sleep(5)  # 5 second intervals for scalping
                
            except Exception as e:
                self.log(f"❌ Scalping loop error: {e}")
                time.sleep(2)
                
    def check_trading_signal(self):
        """Check for trading signals in scalping mode"""
        try:
            # Get REAL price from MT5
            symbol = self.symbol_var.get()
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    current_price = (tick.ask + tick.bid) / 2
                else:
                    self.log(f"❌ Cannot get real price for {symbol}")
                    return None, 0, 0
            else:
                self.log("❌ MT5 not available - cannot get real prices")
                return None, 0, 0
            
            # Scalping mode uses balance-based TP/SL
            tp_balance_pct = config.SCALPING_TP_PERSEN_BALANCE  # 0.5% of balance
            sl_balance_pct = config.SCALPING_SL_PERSEN_BALANCE  # 2% of balance
            
            # Generate real signal (placeholder - add real indicator logic)
            signal = "BUY"  # Replace with real signal logic
            tp, sl = self.calculate_proper_tp_sl(signal, current_price, tp_balance_pct, sl_balance_pct)
            
            return signal, tp, sl
            
        except Exception as e:
            self.log(f"❌ Scalping signal error: {e}")
            return None, 0, 0

class HFTTradingBot(BaseTradingBot):
    """HFT Trading Mode Bot"""
    def __init__(self):
        super().__init__("HFT")
        self.setup_base_gui()
        self.create_hft_gui()
        
    def create_hft_gui(self):
        """Create HFT Trading GUI"""
        # Main title
        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.pack(pady=10, fill="x")
        
        tk.Label(title_frame, text="🚀 HFT (High Frequency Trading)", 
                font=("Arial", 18, "bold"), bg="#f0f0f0", fg="#e74c3c").pack()
        tk.Label(title_frame, text="Ultra-fast automated trading with maximum efficiency", 
                font=("Arial", 10), bg="#f0f0f0", fg="gray").pack()
        
        # HFT specific configuration
        config_frame = tk.LabelFrame(self.root, text="HFT Configuration", 
                                   font=("Arial", 12, "bold"))
        config_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(config_frame, text="Fixed TP: 0.3% | Fixed SL: 1.5% | Ultra-Fast (1s intervals)", 
                font=("Arial", 10, "bold"), fg="#e74c3c").pack(pady=5)
        tk.Label(config_frame, text="Maximum: 100 trades per session", 
                font=("Arial", 10), fg="red").pack()
        
        # Quick Start button prominently displayed
        quick_start_frame = tk.Frame(self.root, bg="#f8e8e8")
        quick_start_frame.pack(pady=15, padx=20, fill="x")
        
        tk.Label(quick_start_frame, text="🚀 HFT Quick Start", 
                font=("Arial", 14, "bold"), bg="#f8e8e8", fg="#e74c3c").pack(pady=5)
        
        tk.Button(quick_start_frame, text="🔴 Activate HFT Now", 
                 command=self.hft_quick_start, font=("Arial", 14, "bold"),
                 bg="#e74c3c", fg="white", padx=30, pady=10).pack(pady=5)
        
        # Control buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="🚀 Start HFT Mode", 
                 command=self.start_bot, font=("Arial", 12, "bold"),
                 bg="#c0392b", fg="white", padx=20, pady=5).pack(side="left", padx=10)
        
        tk.Button(control_frame, text="⏹️ Emergency Stop", 
                 command=self.emergency_stop, font=("Arial", 12, "bold"),
                 bg="#8b0000", fg="white", padx=20, pady=5).pack(side="left", padx=10)
        
        # Account info and log
        info_frame = tk.LabelFrame(self.root, text="HFT Performance Monitor")
        info_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(info_frame, textvariable=self.account_info_var).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.profit_var).pack(anchor="w")
        tk.Label(info_frame, textvariable=self.opportunities_var).pack(anchor="w")
        
        log_frame = tk.LabelFrame(self.root, text="HFT Trading Log")
        log_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.log_text = ScrolledText(log_frame, height=15, width=80)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def hft_quick_start(self):
        """HFT Quick Start function"""
        result = messagebox.askyesno("HFT Quick Start", 
            "Start HFT mode with maximum speed settings?\n\n"
            "⚠️ This will use ultra-fast 1-second scanning\n"
            "⚠️ Maximum 100 trades per session\n"
            "⚠️ TP: 0.3% | SL: 1.5%")
        
        if result:
            self.start_bot()
        
    def start_bot(self):
        """Start HFT bot"""
        if self.running:
            messagebox.showwarning("Already Running", "HFT bot is already running!")
            return
            
        self.running = True
        self.log("🚀 Starting HFT Mode...")
        self.log("🔴 Configuration: TP=0.3%, SL=1.5%, Ultra-fast execution")
        self.log("⚠️ Maximum 100 trades per session")
        
        self.bot_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.bot_thread.start()
        
    def emergency_stop(self):
        """Emergency stop for HFT"""
        self.running = False
        self.log("🛑 EMERGENCY STOP - HFT bot halted immediately")
        messagebox.showinfo("Emergency Stop", "HFT bot has been stopped immediately!")
        
    def stop_bot(self):
        """Stop HFT bot"""
        self.running = False
        self.log("⏹️ HFT bot stopped")
        
    def trading_loop(self):
        """Main trading loop for HFT mode"""
        trade_count = 0
        max_trades = 100
        
        while self.running and trade_count < max_trades:
            try:
                signal, tp, sl = self.check_trading_signal()
                if signal:
                    trade_count += 1
                    self.log(f"🚀 HFT Signal #{trade_count}: {signal} | TP: {tp:.5f} | SL: {sl:.5f}")
                    
                    if trade_count >= max_trades:
                        self.log(f"🛑 Maximum trades reached ({max_trades}). Stopping HFT.")
                        break
                
                # Ultra-fast scanning - 1 second
                time.sleep(1)
                
            except Exception as e:
                self.log(f"❌ HFT loop error: {e}")
                time.sleep(1)
                
        self.running = False
        self.log("🏁 HFT session completed")
        
    def check_trading_signal(self):
        """Check for trading signals in HFT mode"""
        try:
            # Get REAL price from MT5
            symbol = self.symbol_var.get()
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    current_price = (tick.ask + tick.bid) / 2
                else:
                    self.log(f"❌ Cannot get real price for {symbol}")
                    return None, 0, 0
            else:
                self.log("❌ MT5 not available - cannot get real prices")
                return None, 0, 0
            
            # HFT mode uses balance-based TP/SL
            tp_balance_pct = config.HFT_TP_PERSEN_BALANCE  # 0.3% of balance
            sl_balance_pct = config.HFT_SL_PERSEN_BALANCE  # 1.5% of balance
            
            # Generate real signal (placeholder - add real indicator logic)
            signal = "BUY"  # Replace with real signal logic
            tp, sl = self.calculate_proper_tp_sl(signal, current_price, tp_balance_pct, sl_balance_pct)
            
            return signal, tp, sl
            
        except Exception as e:
            self.log(f"❌ HFT signal error: {e}")
            return None, 0, 0

if __name__ == "__main__":
    # Start with mode selector
    selector = TradingModeSelector()
    selector.run()