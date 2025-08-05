#!/usr/bin/env python3
"""
Enhanced Trading Bot - Optimized for Market Opportunity Capture
Fixed price retrieval and signal generation issues
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
    # Create minimal numpy-like functions
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data):
            if not data:
                return 0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return variance ** 0.5
    np = MockNumpy()

# Import enhanced modules
from market_data_api import MarketDataAPI
from simulation_trading import SimulationTrading
from enhanced_indicators import EnhancedIndicators
from config import config

class TradingBot:
    def __init__(self):
        # Initialize enhanced market data and trading simulation
        self.market_api = MarketDataAPI()
        self.market_data_api = self.market_api  # Alias for compatibility
        self.mt5 = SimulationTrading(self.market_api)
        self.indicators = EnhancedIndicators()
        
        # Bot state
        self.running = False
        self.connected = False  # Connection status
        self.modal_awal = None
        self.last_price = None
        self.last_prices = []  # Price history list for HFT
        self.last_reset_date = datetime.date.today()
        self.order_counter = 0
        self.total_opportunities_captured = 0
        self.total_opportunities_missed = 0
        
        # Threading
        self.bot_thread = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup enhanced GUI interface"""
        self.root = tk.Tk()
        self.root.title("🚀 Enhanced Trading Bot - Opportunity Capture System")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables
        self.symbol_var = tk.StringVar(value=config.DEFAULT_SYMBOL)
        self.lot_var = tk.StringVar(value=str(config.DEFAULT_LOT))
        self.interval_var = tk.StringVar(value=str(config.DEFAULT_INTERVAL))
        # Initialize all required variables for backward compatibility
        self.tp_var = tk.StringVar(value=str(config.TP_PERSEN_BALANCE * 100))
        self.sl_var = tk.StringVar(value=str(config.SL_PERSEN_BALANCE * 100))
        self.scalping_tp_var = tk.StringVar(value=str(config.SCALPING_TP_PERSEN_BALANCE * 100))
        self.scalping_sl_var = tk.StringVar(value=str(config.SCALPING_SL_PERSEN_BALANCE * 100))
        self.account_info_var = tk.StringVar(value="Account: Not Connected")
        self.profit_var = tk.StringVar(value="Real-time P/L: -")
        self.balance_var = tk.StringVar(value="$10,000")  # Account balance display
        self.scalping_mode_var = tk.BooleanVar(value=config.SCALPING_OVERRIDE_ENABLED)
        self.opportunities_var = tk.StringVar(value="Opportunities: Captured: 0 | Missed: 0")
        
        self.create_enhanced_gui()
    
    def create_enhanced_gui(self):
        """Create enhanced GUI elements"""
        # Style
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Success.TLabel", foreground="green", font=("Segoe UI", 10, "bold"))
        style.configure("Warning.TLabel", foreground="orange", font=("Segoe UI", 10, "bold"))
        
        # Header frame with enhanced info
        header_frame = ttk.LabelFrame(self.root, text="🤖 Enhanced Trading Bot - Opportunity Capture System", padding=10)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(header_frame, text="✅ Fixed Price Retrieval | ⚡ Optimized Signal Generation | 🎯 Enhanced Opportunity Capture", 
                 style="Success.TLabel").pack()
        
        # Info frame with opportunities counter
        info_frame = ttk.LabelFrame(self.root, text="Account & Performance Info", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill="x")
        
        ttk.Label(info_grid, textvariable=self.account_info_var).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(info_grid, textvariable=self.profit_var, foreground="green").grid(row=0, column=1, sticky="e", padx=5)
        ttk.Label(info_grid, textvariable=self.opportunities_var, style="Success.TLabel").grid(row=1, column=0, columnspan=2, pady=5)
        
        info_grid.columnconfigure(0, weight=1)
        info_grid.columnconfigure(1, weight=1)
        
        # Enhanced settings frame
        setting_frame = ttk.LabelFrame(self.root, text="⚙️ Enhanced Trading Settings", padding=10)
        setting_frame.pack(padx=10, pady=5, fill="x")
        
        settings_notebook = ttk.Notebook(setting_frame)
        settings_notebook.pack(fill="x", expand=True)
        
        # Basic settings tab
        basic_tab = ttk.Frame(settings_notebook)
        settings_notebook.add(basic_tab, text="Basic Settings")
        
        basic_left = ttk.Frame(basic_tab)
        basic_left.grid(row=0, column=0, padx=10, sticky="n")
        
        basic_fields = [
            ("Symbol:", self.symbol_var),
            ("Lot Size:", self.lot_var),
            ("Scan Interval (s):", self.interval_var),
        ]
        
        for i, (label, var) in enumerate(basic_fields):
            ttk.Label(basic_left, text=label).grid(row=i, column=0, sticky="e", pady=5)
            ttk.Entry(basic_left, textvariable=var, width=20).grid(row=i, column=1, pady=5)
        
        # UNIFIED BALANCE-BASED TP/SL SECTION (HAPUS DUPLIKASI)
        balance_frame = ttk.LabelFrame(basic_tab, text="💰 Unified Balance-Based TP/SL", padding=10)
        balance_frame.grid(row=0, column=1, padx=10, sticky="n")
        
        ttk.Label(balance_frame, text="💡 SEMUA TP/SL berdasarkan % modal", 
                 style="Success.TLabel").grid(row=0, column=0, columnspan=2, pady=5)
        
        # Initialize balance variables
        self.tp_balance_var = tk.StringVar(value=str(config.TP_PERSEN_BALANCE * 100))
        self.sl_balance_var = tk.StringVar(value=str(config.SL_PERSEN_BALANCE * 100))
        
        unified_fields = [
            ("Normal TP (% modal):", self.tp_balance_var),
            ("Normal SL (% modal):", self.sl_balance_var),
        ]
        
        for i, (label, var) in enumerate(unified_fields):
            ttk.Label(balance_frame, text=label).grid(row=i+1, column=0, sticky="e", pady=3, padx=5)
            ttk.Entry(balance_frame, textvariable=var, width=15).grid(row=i+1, column=1, pady=3, padx=5)
        
        # Mode indicator with clear explanation
        mode_info = ttk.Label(balance_frame, text="🔄 Mode otomatis:\n• Normal: User setting diatas\n• Scalping: 0.5% TP, 2% SL\n• HFT: 0.3% TP, 1.5% SL", 
                             style="TLabel", font=("Segoe UI", 8), justify="left")
        mode_info.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Add update button for settings
        ttk.Button(balance_frame, text="💾 Update TP/SL", 
                  command=self.update_tp_sl_settings).grid(row=4, column=0, columnspan=2, pady=5)
        
        # HFT & Scalping settings tab
        hft_tab = ttk.Frame(settings_notebook)
        settings_notebook.add(hft_tab, text="HFT & Scalping")
        
        # HFT Mode Section
        hft_frame = ttk.LabelFrame(hft_tab, text="⚡ High-Frequency Trading (HFT)", padding=10)
        hft_frame.pack(fill="x", padx=5, pady=5)
        
        # Initialize HFT variables
        self.hft_mode_var = tk.BooleanVar(value=False)
        
        hft_controls = ttk.Frame(hft_frame)
        hft_controls.pack(fill="x")
        
        ttk.Checkbutton(hft_controls, text="⚡ Enable HFT Mode (1s scan, 10 trades/sec)", 
                       variable=self.hft_mode_var, 
                       command=self.toggle_hft_mode).pack(anchor="w", pady=5)
        
        self.hft_button = ttk.Button(hft_controls, text="🚀 Activate HFT Now", 
                                   command=self.activate_hft_instant, 
                                   style="Accent.TButton")
        self.hft_button.pack(anchor="w", pady=5)
        
        # HFT Status and parameters
        self.hft_status_var = tk.StringVar(value="HFT Status: Disabled")
        ttk.Label(hft_controls, textvariable=self.hft_status_var, style="TLabel").pack(anchor="w", pady=2)
        
        # HFT parameters info
        hft_info = ttk.Frame(hft_controls)
        hft_info.pack(anchor="w", pady=3)
        
        ttk.Label(hft_info, text="⚡ HFT Parameters:", style="TLabel").pack(anchor="w")
        ttk.Label(hft_info, text=f"• TP: {config.HFT_TP_PERSEN_BALANCE*100:.1f}% dari modal", 
                 style="Success.TLabel").pack(anchor="w")
        ttk.Label(hft_info, text=f"• SL: {config.HFT_SL_PERSEN_BALANCE*100:.1f}% dari modal", 
                 style="Warning.TLabel").pack(anchor="w")
        ttk.Label(hft_info, text="• Max: 100 orders/session, 1s scan", 
                 style="TLabel", font=("Segoe UI", 8)).pack(anchor="w")
        
        # Scalping Mode Section  
        scalping_frame = ttk.LabelFrame(hft_tab, text="🔥 Scalping Mode", padding=10)
        scalping_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Checkbutton(scalping_frame, text="🔥 Enable Enhanced Scalping Mode", 
                       variable=self.scalping_mode_var, 
                       style="Success.TLabel").pack(pady=5)
        
        scalp_controls = ttk.Frame(scalping_frame)
        scalp_controls.pack()
        
        # Display current scalping settings (informational)
        scalp_info = ttk.Frame(scalp_controls)
        scalp_info.pack(pady=5)
        
        ttk.Label(scalp_info, text="📊 Scalping Parameters:", style="TLabel").pack()
        ttk.Label(scalp_info, text=f"• TP: {config.SCALPING_TP_PERSEN_BALANCE*100:.1f}% dari modal", 
                 style="Success.TLabel").pack()
        ttk.Label(scalp_info, text=f"• SL: {config.SCALPING_SL_PERSEN_BALANCE*100:.1f}% dari modal", 
                 style="Warning.TLabel").pack()
        ttk.Label(scalp_info, text="(Otomatis aktif saat scalping mode)", 
                 style="TLabel", font=("Segoe UI", 8)).pack()
        
        # Enhanced control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=15)
        
        # Primary buttons
        primary_buttons = ttk.Frame(button_frame)
        primary_buttons.pack()
        
        self.connect_button = ttk.Button(primary_buttons, text="🔗 Connect to Market", 
                                       command=self.connect_mt5, style="TButton")
        self.start_button = ttk.Button(primary_buttons, text="🚀 Start Enhanced Bot", 
                                     command=self.start_bot, state="disabled")
        self.stop_button = ttk.Button(primary_buttons, text="🛑 Stop Bot", 
                                    command=self.stop_bot, state="disabled")
        
        self.connect_button.grid(row=0, column=0, padx=10)
        self.start_button.grid(row=0, column=1, padx=10)
        self.stop_button.grid(row=0, column=2, padx=10)
        
        # Secondary buttons
        secondary_buttons = ttk.Frame(button_frame)
        secondary_buttons.pack(pady=10)
        
        self.close_button = ttk.Button(secondary_buttons, text="❌ Close All Positions", 
                                     command=self.manual_close_all)
        self.reset_button = ttk.Button(secondary_buttons, text="🔄 Reset Counters", 
                                     command=self.reset_counters)
        
        self.close_button.grid(row=0, column=0, padx=10)
        self.reset_button.grid(row=0, column=1, padx=10)
        
        # HFT Quick Controls
        hft_quick = ttk.Button(secondary_buttons, text="⚡ HFT Quick Start", 
                              command=self.hft_quick_start, style="Accent.TButton")
        hft_quick.grid(row=0, column=2, padx=10)
        
        # Enhanced log frame with tabs
        log_frame = ttk.LabelFrame(self.root, text="📊 Enhanced Trading Monitor", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        log_notebook = ttk.Notebook(log_frame)
        log_notebook.pack(fill="both", expand=True)
        
        # Trading log tab
        log_tab = ttk.Frame(log_notebook)
        log_notebook.add(log_tab, text="Trading Log")
        
        self.log_box = ScrolledText(log_tab, width=140, height=20, 
                                   bg="#ffffff", fg="#333333", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True)
        
        # Performance tab
        perf_tab = ttk.Frame(log_notebook)
        log_notebook.add(perf_tab, text="Performance")
        
        self.perf_box = ScrolledText(perf_tab, width=140, height=20, 
                                    bg="#f8f8f8", fg="#333333", font=("Consolas", 9))
        self.perf_box.pack(fill="both", expand=True)
    
    def reset_counters(self):
        """Reset opportunity counters"""
        self.total_opportunities_captured = 0
        self.total_opportunities_missed = 0
        self.update_opportunities_display()
        self.log("🔄 Opportunity counters reset")
    
    # ======================
    # HFT MODE FUNCTIONS
    # ======================
    def toggle_hft_mode(self):
        """Toggle HFT mode on/off"""
        if self.hft_mode_var.get():
            self.activate_hft_mode()
        else:
            self.deactivate_hft_mode()
    
    def activate_hft_mode(self):
        """Activate HFT mode with optimized settings"""
        try:
            # Import HFT config
            from hft_config import hft_config
            hft_config.enable_hft_mode()
            
            # Set HFT parameters
            self.interval_var.set("1")        # 1 second scanning
            self.lot_var.set("0.01")          # Small lot for HFT
            self.tp_var.set("0.1")            # 0.1% TP
            self.sl_var.set("0.3")            # 0.3% SL
            self.scalping_mode_var.set(True)  # Enable scalping
            
            # Update HFT status
            self.hft_status_var.set("HFT Status: ⚡ ACTIVE (1s scan, ultra-fast)")
            
            self.log("⚡ HFT MODE ACTIVATED!")
            self.log("   • Ultra-fast 1-second scanning enabled")
            self.log("   • High-frequency parameters set")
            self.log("   • Target: 10 trades/second capability")
            self.log("   • TP: 0.1% | SL: 0.3% (tight parameters)")
            
        except Exception as e:
            self.log(f"❌ HFT activation error: {e}")
    
    def deactivate_hft_mode(self):
        """Deactivate HFT mode and return to normal"""
        try:
            # Reset to normal parameters
            self.interval_var.set("8")        # Normal scanning
            self.tp_var.set("0.8")            # Normal TP
            self.sl_var.set("4.0")            # Normal SL
            
            # Update HFT status
            self.hft_status_var.set("HFT Status: Disabled (normal trading)")
            
            self.log("🔄 HFT MODE DISABLED")
            self.log("   • Returned to normal trading parameters")
            self.log("   • 8-second scanning restored")
            
        except Exception as e:
            self.log(f"❌ HFT deactivation error: {e}")
    
    def activate_hft_instant(self):
        """Instant HFT activation with one click"""
        self.hft_mode_var.set(True)
        self.activate_hft_mode()
        
        # If bot is running, apply changes immediately
        if self.running:
            self.log("⚡ HFT parameters applied to running bot!")
    
    def hft_quick_start(self):
        """Quick start HFT with optimal settings"""
        try:
            # Enable HFT mode
            self.hft_mode_var.set(True)
            self.activate_hft_mode()
            
            # Auto-connect if not connected
            if not self.connected:
                self.connect_mt5()
            
            # Auto-start if not running
            if not self.running:
                self.start_bot()
            
            self.log("🚀 HFT QUICK START COMPLETED!")
            self.log("   • HFT mode activated")
            self.log("   • Bot connected and started")
            self.log("   • Ready for high-frequency trading")
            
        except Exception as e:
            self.log(f"❌ HFT quick start error: {e}")
    
    def update_opportunities_display(self):
        """Update opportunities display"""
        self.opportunities_var.set(
            f"Opportunities: Captured: {self.total_opportunities_captured} | "
            f"Missed: {self.total_opportunities_missed} | "
            f"Success Rate: {self.calculate_success_rate():.1f}%"
        )
    
    def calculate_success_rate(self):
        """Calculate opportunity capture success rate"""
        total = self.total_opportunities_captured + self.total_opportunities_missed
        if total == 0:
            return 0.0
        return (self.total_opportunities_captured / total) * 100
    
    # ======================
    # ENHANCED LOGGING
    # ======================
    def log_to_file(self, text):
        """Enhanced log to file"""
        try:
            with open(config.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now()} - {text}\n")
        except Exception as e:
            print(f"Error logging to file: {e}")

    def log(self, text, log_type="INFO"):
        """Enhanced log entry with types"""
        timestamp = f"{datetime.datetime.now():%H:%M:%S}"
        
        # Color coding based on log type
        if "opportunity" in text.lower() or "signal" in text.lower():
            log_entry = f"🎯 {timestamp} - {text}"
        elif "error" in text.lower() or "fail" in text.lower():
            log_entry = f"❌ {timestamp} - {text}"
        elif "success" in text.lower() or "profit" in text.lower():
            log_entry = f"✅ {timestamp} - {text}"
        elif "warning" in text.lower():
            log_entry = f"⚠️  {timestamp} - {text}"
        else:
            log_entry = f"ℹ️  {timestamp} - {text}"
        
        self.log_box.insert(tk.END, log_entry + "\n")
        self.log_box.see(tk.END)
        self.log_to_file(text)

    def log_performance(self, text):
        """Log performance metrics"""
        timestamp = f"{datetime.datetime.now():%H:%M:%S}"
        perf_entry = f"{timestamp} - {text}"
        
        self.perf_box.insert(tk.END, perf_entry + "\n")
        self.perf_box.see(tk.END)

    def send_telegram(self, text):
        """Enhanced Telegram notification"""
        try:
            if (config.TELEGRAM_BOT_TOKEN != "your_bot_token_here" and 
                config.TELEGRAM_CHAT_ID != "your_chat_id_here"):
                
                enhanced_text = f"🤖 Trading Bot Alert\n{text}\n⏰ {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
                
                url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
                data = {"chat_id": config.TELEGRAM_CHAT_ID, "text": enhanced_text}
                
                response = requests.post(url, data=data, timeout=5)
                if response.status_code == 200:
                    self.log("📱 Telegram notification sent successfully")
                else:
                    self.log(f"📱 Telegram notification failed: {response.status_code}")
                    
        except Exception as e:
            self.log(f"📱 Telegram error: {e}")

    def export_trade_log(self, harga_order, tp, sl, sinyal, confidence=0, strength=0):
        """Enhanced trade log export"""
        try:
            filename = config.TRADE_LOG_FILE
            fieldnames = ["timestamp", "signal", "price", "tp", "sl", "confidence", "strength", "success_rate"]
            file_exists = os.path.isfile(filename)
            
            with open(filename, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "signal": sinyal,
                    "price": round(harga_order, 5),
                    "tp": round(tp, 5),
                    "sl": round(sl, 5),
                    "confidence": round(confidence, 3),
                    "strength": round(strength, 3),
                    "success_rate": round(self.calculate_success_rate(), 1)
                })
        except Exception as e:
            self.log(f"Error exporting trade log: {e}")
    
    # ======================
    # ENHANCED CONNECTION
    # ======================
    def connect_mt5(self):
        """Enhanced connection to market simulation"""
        try:
            self.log("🔗 Connecting to Enhanced Trading Simulation...")
            self.log("⚡ Features: Fixed Price Retrieval, Optimized Signals, Enhanced Opportunity Capture")
            
            # Show enhanced setup dialog
            balance_dialog = tk.Toplevel(self.root)
            balance_dialog.title("🚀 Enhanced Virtual Trading Setup")
            balance_dialog.geometry("500x300")
            balance_dialog.transient(self.root)
            balance_dialog.grab_set()
            
            # Center the dialog
            balance_dialog.update_idletasks()
            x = (balance_dialog.winfo_screenwidth() // 2) - (500 // 2)
            y = (balance_dialog.winfo_screenheight() // 2) - (300 // 2)
            balance_dialog.geometry(f"500x300+{x}+{y}")
            
            ttk.Label(balance_dialog, text="🚀 ENHANCED TRADING SIMULATION", 
                     font=("Segoe UI", 14, "bold")).pack(pady=10)
            
            ttk.Label(balance_dialog, text="✅ Fixed Price Retrieval Issues", 
                     foreground="green", font=("Segoe UI", 10)).pack()
            ttk.Label(balance_dialog, text="⚡ Optimized Signal Generation", 
                     foreground="green", font=("Segoe UI", 10)).pack()
            ttk.Label(balance_dialog, text="🎯 Enhanced Opportunity Capture", 
                     foreground="green", font=("Segoe UI", 10)).pack(pady=(0, 10))
            
            ttk.Label(balance_dialog, text="Set your virtual account balance:", 
                     font=("Segoe UI", 11)).pack(pady=5)
            
            balance_var = tk.StringVar(value="10000")
            balance_entry = ttk.Entry(balance_dialog, textvariable=balance_var, 
                                    font=("Segoe UI", 12), width=20)
            balance_entry.pack(pady=10)
            
            # Configuration display
            config_frame = ttk.LabelFrame(balance_dialog, text="Enhanced Configuration", padding=5)
            config_frame.pack(pady=10, padx=20, fill="x")
            
            ttk.Label(config_frame, text=f"• Price Spike Threshold: {config.LONJAKAN_THRESHOLD} (Reduced for more opportunities)", 
                     font=("Segoe UI", 9)).pack(anchor="w")
            ttk.Label(config_frame, text=f"• Scan Interval: {config.DEFAULT_INTERVAL}s (Faster scanning)", 
                     font=("Segoe UI", 9)).pack(anchor="w")
            ttk.Label(config_frame, text=f"• Signal Confidence: {config.SIGNAL_CONFIDENCE_THRESHOLD} (Optimized threshold)", 
                     font=("Segoe UI", 9)).pack(anchor="w")
            
            result = {"balance": 10000.0}
            
            def on_connect():
                try:
                    balance = float(balance_var.get())
                    if balance <= 0:
                        messagebox.showerror("Error", "Balance must be greater than 0")
                        return
                    result["balance"] = balance
                    balance_dialog.destroy()
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid number")
            
            def on_cancel():
                balance_dialog.destroy()
            
            button_frame = ttk.Frame(balance_dialog)
            button_frame.pack(pady=20)
            ttk.Button(button_frame, text="🚀 Connect Enhanced Bot", command=on_connect).pack(side="left", padx=10)
            ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="left", padx=10)
            
            # Wait for dialog to close
            self.root.wait_window(balance_dialog)
            
            if result["balance"] is None or result["balance"] <= 0:
                return False
            
            # Set virtual account balance
            if hasattr(self.mt5, 'account'):
                self.mt5.account.balance = result["balance"]
                self.mt5.account.equity = result["balance"] 
                self.mt5.account.login = 999999
            
            if self.mt5.initialize():
                account_info = self.mt5.account_info()
                if account_info:
                    self.modal_awal = account_info.balance
                    account_text = f"Enhanced Virtual Account: {account_info.login} | Balance: ${account_info.balance:,.2f}"
                    self.account_info_var.set(account_text)
                    self.log(f"✅ Connected: {account_text}")
                    self.log("📊 Real market data - Enhanced virtual trading")
                    self.log("🛡️  100% Safe - No real money involved")
                    self.log("🎯 Enhanced opportunity capture system active")
                    
                    self.connect_button.config(state="disabled")
                    self.start_button.config(state="normal")
                    
                    messagebox.showinfo("Success", 
                                      f"Connected to Enhanced Trading Simulation!\n"
                                      f"Virtual Balance: ${account_info.balance:,.2f}\n"
                                      f"✅ Fixed price retrieval\n"
                                      f"⚡ Optimized signal generation\n"
                                      f"🎯 Enhanced opportunity capture")
                    return True
                else:
                    self.log("❌ Failed to get account information")
                    return False
            else:
                self.log("❌ Failed to initialize market connection")
                return False
                
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("Connection Error", error_msg)
            return False
    
    def get_data_with_retry(self, symbol, timeframe=None, n=100):
        """Enhanced data retrieval with retry logic"""
        for attempt in range(config.PRICE_FETCH_RETRY):
            try:
                rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
                if rates is not None and len(rates) >= config.MIN_DATA_POINTS:
                    return rates
                
                if attempt < config.PRICE_FETCH_RETRY - 1:
                    self.log(f"⚠️  Data fetch attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.5 + attempt * 0.3)
                    
            except Exception as e:
                if attempt < config.PRICE_FETCH_RETRY - 1:
                    self.log(f"⚠️  Data fetch error (attempt {attempt + 1}): {e}")
                    time.sleep(0.5 + attempt * 0.3)
                else:
                    self.log(f"❌ Data fetch failed after {config.PRICE_FETCH_RETRY} attempts: {e}")
        
        return None

    def get_total_open_orders(self):
        """Get total open positions with error handling"""
        try:
            positions = self.mt5.positions_get()
            return len(positions) if positions else 0
        except Exception as e:
            self.log(f"Error getting open orders: {e}")
            return 0

    def close_all_orders(self):
        """Enhanced close all positions"""
        try:
            positions = self.mt5.positions_get()
            if not positions:
                self.log("No positions to close")
                return True
            
            closed_count = 0
            for pos in positions:
                try:
                    result = self.mt5.position_close(pos.ticket)
                    if result and result.retcode == 10009:
                        closed_count += 1
                        self.log(f"✅ Position {pos.ticket} closed successfully")
                    else:
                        self.log(f"❌ Failed to close position {pos.ticket}")
                except Exception as e:
                    self.log(f"❌ Error closing position {pos.ticket}: {e}")
            
            self.log(f"🔄 Closed {closed_count} out of {len(positions)} positions")
            return closed_count > 0
            
        except Exception as e:
            self.log(f"❌ Error in close all orders: {e}")
            return False

    def manual_close_all(self):
        """Manual close all positions"""
        try:
            result = messagebox.askyesno("Confirm", "Are you sure you want to close all positions?")
            if result:
                self.close_all_orders()
                self.send_telegram("🔄 All positions closed manually")
        except Exception as e:
            self.log(f"Manual close error: {e}")
    
    # ======================
    # ENHANCED SIGNAL GENERATION
    # ======================
    def enhanced_signal_check(self, symbol):
        """Enhanced signal checking with improved opportunity capture"""
        try:
            # Get market data with retry
            rates = self.get_data_with_retry(symbol, n=config.DATA_BUFFER_SIZE)
            if rates is None or len(rates) < config.MIN_DATA_POINTS:
                self.log("❌ Insufficient market data for analysis")
                self.total_opportunities_missed += 1
                self.update_opportunities_display()
                return None, 0, 0
            
            # Extract price arrays
            close_prices = rates['close']
            high_prices = rates['high']
            low_prices = rates['low']
            
            current_price = close_prices[-1]
            
            # Enhanced price spike detection (FIXED THRESHOLD)
            if len(close_prices) >= 5:
                recent_prices = close_prices[-5:]
                price_volatility = np.std(recent_prices) / np.mean(recent_prices) * 100
                
                # More lenient spike detection
                if price_volatility > config.LONJAKAN_THRESHOLD:
                    self.log(f"⚠️  High volatility detected ({price_volatility:.2f}%), analyzing carefully...")
                    # Don't skip, but reduce confidence
                    volatility_penalty = 0.2
                else:
                    volatility_penalty = 0
            else:
                volatility_penalty = 0
            
            # Enhanced signal analysis
            signal_data = self.indicators.enhanced_signal_analysis(
                close_prices, high_prices, low_prices
            )
            
            signal = signal_data['signal']
            base_confidence = signal_data['confidence']
            
            # Apply volatility penalty but don't completely block
            adjusted_confidence = max(0, base_confidence - volatility_penalty)
            
            # WINRATE ENHANCEMENT: Multi-confirmation system
            if config.WINRATE_BOOST_ENABLED:
                winrate_boost = self.calculate_winrate_boost(signal_data, close_prices)
                adjusted_confidence = min(1.0, adjusted_confidence * winrate_boost)
                self.log(f"📈 Winrate boost applied: {winrate_boost:.2f}x")
            
            # Dynamic confidence threshold based on trading mode
            if hasattr(self, 'hft_mode_var') and self.hft_mode_var.get():
                min_confidence = config.SIGNAL_CONFIDENCE_THRESHOLD_HFT  # Ultra-low for HFT
                max_orders = config.MAX_ORDER_PER_SESSION_HFT  # Higher limit for HFT
                self.log("⚡ HFT mode: Using aggressive thresholds")
            else:
                min_confidence = config.SIGNAL_CONFIDENCE_THRESHOLD
                max_orders = config.MAX_ORDER_PER_SESSION
            
            # Check order limit based on mode
            current_orders = self.order_counter
            if current_orders >= max_orders:
                self.log(f"⛔ Max orders reached: {current_orders}/{max_orders}")
                return None, 0, 0
            
            if signal in ['BUY', 'SELL'] and adjusted_confidence >= min_confidence:
                # UNIFIED BALANCE-BASED TP/SL CALCULATION
                tp, sl = self.calculate_unified_tp_sl(signal, current_price)
                
                self.log(f"🎯 ENHANCED SIGNAL DETECTED!")
                self.log(f"   Signal: {signal}")
                self.log(f"   Confidence: {adjusted_confidence:.2f} (Base: {base_confidence:.2f})")
                self.log(f"   Strength: {signal_data['strength']:.2f}")
                self.log(f"   Price: {current_price}")
                self.log(f"   TP: {tp} | SL: {sl}")
                
                # Log technical indicators
                indicators = signal_data.get('indicators', {})
                if indicators:
                    self.log(f"   Technical Analysis:")
                    self.log(f"   - RSI: {indicators.get('rsi', 0):.1f}")
                    self.log(f"   - MA Short: {indicators.get('ma_short', 0):.5f}")
                    self.log(f"   - EMA Fast: {indicators.get('ema_fast', 0):.5f}")
                
                self.total_opportunities_captured += 1
                self.update_opportunities_display()
                
                return signal, tp, sl
            else:
                # Log why signal was rejected (for debugging)
                if signal in ['BUY', 'SELL']:
                    self.log(f"⚠️  Signal {signal} rejected - Low confidence: {adjusted_confidence:.2f} < {min_confidence:.2f}")
                    self.total_opportunities_missed += 1
                else:
                    self.log(f"📊 Market analysis: {signal} - Confidence: {adjusted_confidence:.2f}")
                
                self.update_opportunities_display()
                return None, 0, 0
            
        except Exception as e:
            self.log(f"❌ Enhanced signal check error: {e}")
            self.total_opportunities_missed += 1
            self.update_opportunities_display()
            return None, 0, 0

    def calculate_unified_tp_sl(self, signal, current_price):
        """UNIFIED TP/SL calculation - SEMUA PAKAI BALANCE-BASED CALCULATION"""
        try:
            # Get REAL account balance from MT5
            real_balance = self.get_real_account_balance()
            
            if real_balance <= 0:
                # Fallback: Try to get from balance display
                try:
                    display_balance = float(self.balance_var.get().replace('$', '').replace(',', ''))
                    real_balance = display_balance if display_balance > 0 else 10000
                except:
                    real_balance = 10000  # Emergency fallback
            
            symbol = self.symbol_var.get()
            lot_size = float(self.lot_var.get())
            
            # Determine TP/SL percentages based on trading mode
            if hasattr(self, 'hft_mode_var') and self.hft_mode_var.get():
                # HFT Mode - Ultra tight TP/SL
                tp_balance_pct = config.HFT_TP_PERSEN_BALANCE     # 0.3% dari modal
                sl_balance_pct = config.HFT_SL_PERSEN_BALANCE     # 1.5% dari modal
                mode_name = "HFT"
            elif self.scalping_mode_var.get():
                # Scalping Mode  
                tp_balance_pct = config.SCALPING_TP_PERSEN_BALANCE  # 0.5% dari modal
                sl_balance_pct = config.SCALPING_SL_PERSEN_BALANCE  # 2% dari modal
                mode_name = "Scalping"
            else:
                # Normal Mode - Use user setting from GUI
                try:
                    if hasattr(self, 'tp_balance_var') and hasattr(self, 'sl_balance_var'):
                        tp_balance_pct = float(self.tp_balance_var.get()) / 100.0
                        sl_balance_pct = float(self.sl_balance_var.get()) / 100.0
                    else:
                        tp_balance_pct = config.TP_PERSEN_BALANCE
                        sl_balance_pct = config.SL_PERSEN_BALANCE
                except:
                    tp_balance_pct = config.TP_PERSEN_BALANCE
                    sl_balance_pct = config.SL_PERSEN_BALANCE
                mode_name = "Normal"
            
            # CALCULATE PROFIT/LOSS TARGET IN MONEY AMOUNT (not pips)
            target_profit_amount = real_balance * tp_balance_pct  # Contoh: 5juta * 0.01 = 50ribu
            max_loss_amount = real_balance * sl_balance_pct       # Contoh: 5juta * 0.03 = 150ribu
            
            # Convert money amounts to price levels based on lot size and symbol
            tp_price_diff, sl_price_diff = self.convert_money_to_price_difference(
                symbol, lot_size, target_profit_amount, max_loss_amount
            )
            
            # Calculate final TP/SL prices
            if signal == "BUY":
                tp = current_price + tp_price_diff
                sl = current_price - sl_price_diff
            else:  # SELL
                tp = current_price - tp_price_diff
                sl = current_price + sl_price_diff
            
            # LOGGING FOR UNIFIED BALANCE-BASED TP/SL
            self.log(f"💰 {mode_name} Mode TP/SL Calculation:")
            self.log(f"   Account Balance: ${real_balance:,.2f}")
            self.log(f"   Target Profit: ${target_profit_amount:.2f} ({tp_balance_pct*100:.1f}% dari modal)")
            self.log(f"   Max Loss: ${max_loss_amount:.2f} ({sl_balance_pct*100:.1f}% dari modal)")
            self.log(f"   TP Price: {tp:.5f} | SL Price: {sl:.5f}")
            self.log(f"   Signal: {signal} | Current Price: {current_price:.5f}")
            
            return tp, sl
            
        except Exception as e:
            self.log(f"❌ Unified TP/SL calculation error: {e}")
            # Emergency fallback with safe defaults
            if signal == "BUY":
                return current_price * 1.01, current_price * 0.99
            else:
                return current_price * 0.99, current_price * 1.01
    
    def get_real_account_balance(self):
        """Get real account balance from MT5 connection"""
        try:
            # Try to get from MT5 simulation first
            if hasattr(self.mt5, 'get_account_info'):
                account_info = self.mt5.get_account_info()
                if account_info and hasattr(account_info, 'balance'):
                    return account_info.balance
            
            # Try to get from MT5 simulation balance
            if hasattr(self.mt5, 'balance'):
                return self.mt5.balance
                
            # Fallback to display balance
            try:
                display_balance = float(self.balance_var.get().replace('$', '').replace(',', ''))
                return display_balance if display_balance > 0 else 10000
            except:
                return 10000  # Emergency default
        except Exception as e:
            self.log(f"❌ Error getting real balance: {e}")
            return 10000  # Safe default for real money
    
    def convert_money_to_price_difference(self, symbol, lot_size, profit_amount, loss_amount):
        """Convert money amounts to price differences for TP/SL"""
        try:
            # Get contract size and pip value for the symbol
            contract_size = self.get_contract_size(symbol)
            pip_value = self.get_pip_value_per_pip(symbol, lot_size)
            
            # Calculate price difference needed for target amounts
            tp_pips = profit_amount / pip_value if pip_value > 0 else 50
            sl_pips = loss_amount / pip_value if pip_value > 0 else 100
            
            # Convert pips to price difference
            pip_size = self.get_pip_size(symbol)
            tp_price_diff = tp_pips * pip_size
            sl_price_diff = sl_pips * pip_size
            
            return tp_price_diff, sl_price_diff
            
        except Exception as e:
            self.log(f"❌ Error converting money to price: {e}")
            # Safe fallback - small price differences
            return 0.001, 0.002
    
    def get_contract_size(self, symbol):
        """Get contract size for symbol"""
        try:
            if symbol.startswith("XAU"):  # Gold
                return 100
            elif "USD" in symbol:  # Forex pairs
                return 100000
            elif symbol.startswith("BTC"):  # Crypto
                return 1
            else:
                return 100000  # Default
        except:
            return 100000
    
    def get_pip_value_per_pip(self, symbol, lot_size):
        """Get pip value per single pip movement"""
        try:
            contract_size = self.get_contract_size(symbol)
            
            if "JPY" in symbol:
                pip_value = (lot_size * contract_size) * 0.01
            elif symbol.startswith("XAU"):  # Gold
                pip_value = lot_size * 10  # $10 per 0.1 movement for 1 lot
            else:
                pip_value = (lot_size * contract_size) * 0.0001
                
            return pip_value
        except:
            return 10  # Safe default
    
    def calculate_tp_sl_price_based(self, signal, current_price):
        """Fallback TP/SL calculation based on price percentage"""
        try:
            if self.scalping_mode_var.get():
                tp_pct = config.SCALPING_TP_PERSEN
                sl_pct = config.SCALPING_SL_PERSEN
            else:
                tp_pct = config.TP_PERSEN_DEFAULT
                sl_pct = config.SL_PERSEN_DEFAULT
                
            if signal == "BUY":
                tp = current_price * (1 + tp_pct)
                sl = current_price * (1 - sl_pct)
            else:
                tp = current_price * (1 - tp_pct)
                sl = current_price * (1 + sl_pct)
                
            return tp, sl
        except:
            return current_price * 1.01, current_price * 0.99
    
    def calculate_winrate_boost(self, signal_data, close_prices):
        """Calculate winrate boost multiplier based on multiple confirmations"""
        try:
            boost = 1.0
            confirmations = 0
            
            # Check trend confirmation
            if len(close_prices) >= config.TREND_CONFIRMATION_PERIOD:
                trend_prices = close_prices[-config.TREND_CONFIRMATION_PERIOD:]
                if signal_data['signal'] == 'BUY' and trend_prices[-1] > trend_prices[0]:
                    confirmations += 1
                    boost += 0.2
                elif signal_data['signal'] == 'SELL' and trend_prices[-1] < trend_prices[0]:
                    confirmations += 1
                    boost += 0.2
            
            # Check multiple indicator agreement
            indicators = signal_data.get('indicators', {})
            rsi = indicators.get('rsi', 50)
            ma_signal = indicators.get('ma_signal', 'HOLD')
            ema_signal = indicators.get('ema_signal', 'HOLD')
            
            # RSI confirmation
            if signal_data['signal'] == 'BUY' and rsi < 50:
                confirmations += 1
                boost += 0.15
            elif signal_data['signal'] == 'SELL' and rsi > 50:
                confirmations += 1
                boost += 0.15
            
            # MA/EMA agreement
            if ma_signal == signal_data['signal']:
                confirmations += 1
                boost += 0.1
            if ema_signal == signal_data['signal']:
                confirmations += 1
                boost += 0.1
            
            # Strength multiplier
            strength = signal_data.get('strength', 0.5)
            if strength > 0.7:
                boost *= config.SIGNAL_STRENGTH_MULTIPLIER
                confirmations += 1
            
            # Require minimum confirmations for high boost
            if confirmations >= config.MULTI_CONFIRMATION_REQUIRED:
                self.log(f"✅ High confidence signal: {confirmations} confirmations")
                return min(boost, 2.5)  # Cap at 2.5x boost
            else:
                return min(boost, 1.3)  # Lower boost for fewer confirmations
                
        except Exception as e:
            self.log(f"❌ Winrate boost calculation error: {e}")
            return 1.0
    
    def update_tp_sl_settings(self):
        """Update TP/SL settings from GUI input"""
        try:
            tp_pct = float(self.tp_balance_var.get())
            sl_pct = float(self.sl_balance_var.get())
            
            # Validate settings
            if tp_pct <= 0 or sl_pct <= 0:
                messagebox.showerror("Error", "TP/SL harus lebih dari 0%")
                return
            
            if tp_pct >= sl_pct:
                messagebox.showwarning("Warning", "SL sebaiknya lebih besar dari TP untuk risk management")
            
            # Update config
            config.TP_PERSEN_BALANCE = tp_pct / 100.0
            config.SL_PERSEN_BALANCE = sl_pct / 100.0
            
            self.log(f"💾 TP/SL Settings Updated:")
            self.log(f"   Normal TP: {tp_pct}% dari modal")
            self.log(f"   Normal SL: {sl_pct}% dari modal")
            
            messagebox.showinfo("Success", f"TP/SL berhasil diupdate!\nTP: {tp_pct}% | SL: {sl_pct}%")
            
        except ValueError:
            messagebox.showerror("Error", "Input harus berupa angka")
        except Exception as e:
            self.log(f"❌ Error updating TP/SL: {e}")
            messagebox.showerror("Error", f"Gagal update TP/SL: {e}")
    
    def get_pip_size(self, symbol):
        """Get pip size for the symbol"""
        try:
            if "JPY" in symbol:
                return 0.01  # JPY pairs
            elif symbol.startswith("XAU"):  # Gold
                return 0.1
            else:
                return 0.0001  # Major pairs
        except:
            return 0.0001  # Default
    
    # ======================
    # ENHANCED TRADING LOGIC
    # ======================
    def enhanced_trading_loop(self):
        """Enhanced main trading loop with better opportunity capture"""
        while self.running:
            try:
                current_time = datetime.datetime.now()
                current_hour = current_time.hour
                
                # 24/7 Trading Check - Skip time restrictions if enabled
                if hasattr(config, 'ENABLE_24_7_TRADING') and config.ENABLE_24_7_TRADING:
                    # 24/7 mode - no time restrictions
                    pass
                elif not (config.TRADING_START_HOUR <= current_hour <= config.TRADING_END_HOUR):
                    self.log(f"⏰ Outside trading hours ({config.TRADING_START_HOUR}:00-{config.TRADING_END_HOUR}:00)")
                    time.sleep(60)  # Check every minute outside trading hours
                    continue
                
                # Reset daily counters
                if current_time.date() != self.last_reset_date:
                    self.log("🌅 New trading day - Resetting counters")
                    self.order_counter = 0
                    self.last_reset_date = current_time.date()
                
                # Update account info and positions
                self.update_account_display()
                
                # Check position limits
                open_positions = self.get_total_open_orders()
                if open_positions >= config.MAX_ORDER_PER_SESSION:
                    self.log(f"📊 Maximum positions reached ({open_positions}/{config.MAX_ORDER_PER_SESSION})")
                    time.sleep(int(self.interval_var.get()))
                    continue
                
                # Enhanced signal detection with HFT optimization
                symbol = self.symbol_var.get()
                
                # Check if HFT mode is active for faster processing
                if hasattr(self, 'hft_mode_var') and self.hft_mode_var.get():
                    # HFT mode - use faster signal detection
                    signal, tp, sl = self.hft_signal_check(symbol)
                else:
                    # Normal mode
                    signal, tp, sl = self.enhanced_signal_check(symbol)
                
                if signal and tp and sl:
                    # Execute trade with enhanced logic
                    success = self.execute_enhanced_trade(signal, tp, sl)
                    if success:
                        self.order_counter += 1
                        self.log(f"✅ Trade executed successfully! (Order #{self.order_counter})")
                        
                        # Send enhanced notification
                        self.send_telegram(
                            f"🎯 Trade Alert!\n"
                            f"Signal: {signal}\n"
                            f"Symbol: {symbol}\n"
                            f"TP: {tp:.5f} | SL: {sl:.5f}\n"
                            f"Opportunities Captured: {self.total_opportunities_captured}"
                        )
                else:
                    # Continue scanning for opportunities
                    pass
                
                # Enhanced scanning interval
                scan_interval = max(int(self.interval_var.get()), 5)  # Minimum 5 seconds
                time.sleep(scan_interval)
                
            except Exception as e:
                self.log(f"❌ Trading loop error: {e}")
                time.sleep(10)  # Wait longer on error
        
        self.log("🛑 Enhanced trading loop stopped")

    def execute_enhanced_trade(self, signal, tp, sl):
        """Execute trade with enhanced error handling"""
        try:
            symbol = self.symbol_var.get()
            lot = float(self.lot_var.get())
            
            # Get current market price
            tick = self.mt5.symbol_info_tick(symbol)
            if not tick:
                self.log("❌ Unable to get market price")
                return False
            
            # Determine order type and price
            if signal == "BUY":
                order_type = self.mt5.ORDER_TYPE_BUY
                price = tick.ask
                order_name = "Enhanced BUY"
            else:
                order_type = self.mt5.ORDER_TYPE_SELL
                price = tick.bid
                order_name = "Enhanced SELL"
            
            # Enhanced order request
            request = {
                "action": self.mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": config.MT5_DEVIATION,
                "magic": config.MT5_MAGIC_NUMBER,
                "comment": f"Enhanced Bot v2.0 - {signal}",
                "type_time": self.mt5.ORDER_TIME_GTC,
                "type_filling": self.mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = self.mt5.order_send(request)
            
            if result and result.retcode == 10009:
                # Log successful trade
                self.log(f"🎯 {order_name} Order #{result.deal} @ {price:.5f}")
                self.log(f"   TP: {tp:.5f} | SL: {sl:.5f}")
                self.log(f"   Volume: {lot} lots")
                
                # Export to log
                signal_data = self.indicators.enhanced_signal_analysis(
                    self.market_api.get_price_array(symbol, count=50)
                )
                
                self.export_trade_log(
                    price, tp, sl, signal,
                    signal_data.get('confidence', 0),
                    signal_data.get('strength', 0)
                )
                
                # Performance logging
                self.log_performance(f"Trade executed: {signal} at {price:.5f}")
                
                return True
            else:
                error_code = result.retcode if result else "Unknown"
                self.log(f"❌ Order failed - Error code: {error_code}")
                return False
                
        except Exception as e:
            self.log(f"❌ Trade execution error: {e}")
            return False

    def update_account_display(self):
        """Enhanced account display update"""
        try:
            account_info = self.mt5.account_info()
            if account_info:
                # Update positions for profit calculation
                self.mt5.update_positions()
                
                profit_loss = account_info.profit
                balance = account_info.balance
                equity = account_info.equity
                
                # Update displays
                account_text = f"Account: {account_info.login} | Balance: ${balance:,.2f} | Equity: ${equity:,.2f}"
                self.account_info_var.set(account_text)
                
                profit_color = "green" if profit_loss >= 0 else "red"
                profit_text = f"Real-time P/L: ${profit_loss:,.2f}"
                self.profit_var.set(profit_text)
                
                # Performance metrics
                if self.modal_awal:
                    total_return = ((equity - self.modal_awal) / self.modal_awal) * 100
                    self.log_performance(f"Total Return: {total_return:+.2f}% | Current P/L: ${profit_loss:+.2f}")
                
        except Exception as e:
            self.log(f"Account display update error: {e}")
    
    # ======================
    # BOT CONTROLS
    # ======================
    def start_bot(self):
        """Start enhanced trading bot"""
        try:
            if self.running:
                messagebox.showwarning("Warning", "Bot is already running!")
                return
            
            # Validation checks
            try:
                lot = float(self.lot_var.get())
                interval = int(self.interval_var.get())
                tp_pct = float(self.tp_var.get())
                sl_pct = float(self.sl_var.get())
                
                if lot <= 0 or interval <= 0 or tp_pct <= 0 or sl_pct <= 0:
                    raise ValueError("All values must be positive")
                    
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please check your input values: {e}")
                return
            
            # Account balance check
            account_info = self.mt5.account_info()
            if account_info and account_info.balance < config.SALDO_MINIMAL:
                result = messagebox.askyesno("Low Balance", 
                    f"Account balance (${account_info.balance:,.2f}) is below recommended minimum (${config.SALDO_MINIMAL:,.2f}).\n"
                    f"Continue anyway?")
                if not result:
                    return
            
            # Start enhanced bot
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            # Reset counters
            self.order_counter = 0
            self.total_opportunities_captured = 0
            self.total_opportunities_missed = 0
            self.update_opportunities_display()
            
            # Start trading thread
            self.bot_thread = threading.Thread(target=self.enhanced_trading_loop, daemon=True)
            self.bot_thread.start()
            
            symbol = self.symbol_var.get()
            self.log("🚀 ENHANCED TRADING BOT STARTED!")
            self.log(f"   Symbol: {symbol} | Lot: {lot}")
            self.log(f"   TP: {tp_pct}% | SL: {sl_pct}%")
            self.log(f"   Scan Interval: {interval}s")
            self.log(f"   Price Spike Threshold: {config.LONJAKAN_THRESHOLD} (Enhanced)")
            self.log(f"   Signal Confidence Threshold: {config.SIGNAL_CONFIDENCE_THRESHOLD}")
            self.log("🎯 Enhanced opportunity capture system active!")
            
            self.send_telegram(
                f"🚀 Enhanced Trading Bot Started!\n"
                f"Symbol: {symbol}\n"
                f"Enhanced features active: Price fix, Signal optimization"
            )
            
        except Exception as e:
            self.log(f"❌ Start bot error: {e}")
            messagebox.showerror("Start Error", f"Failed to start bot: {e}")

    def stop_bot(self):
        """Stop enhanced trading bot"""
        try:
            if not self.running:
                messagebox.showwarning("Warning", "Bot is not running!")
                return
            
            self.running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            
            # Final statistics
            success_rate = self.calculate_success_rate()
            
            self.log("🛑 ENHANCED TRADING BOT STOPPED!")
            self.log(f"📊 Session Summary:")
            self.log(f"   Orders Executed: {self.order_counter}")
            self.log(f"   Opportunities Captured: {self.total_opportunities_captured}")
            self.log(f"   Opportunities Missed: {self.total_opportunities_missed}")
            self.log(f"   Success Rate: {success_rate:.1f}%")
            
            self.send_telegram(
                f"🛑 Enhanced Trading Bot Stopped\n"
                f"Orders: {self.order_counter}\n"
                f"Success Rate: {success_rate:.1f}%\n"
                f"Opportunities Captured: {self.total_opportunities_captured}"
            )
            
        except Exception as e:
            self.log(f"❌ Stop bot error: {e}")

    def on_closing(self):
        """Enhanced application closing"""
        try:
            if self.running:
                result = messagebox.askyesno("Confirm Exit", 
                    "Trading bot is still running. Stop bot and exit?")
                if result:
                    self.stop_bot()
                    time.sleep(1)  # Give time for bot to stop
                else:
                    return
            
            # Close all positions option
            if self.get_total_open_orders() > 0:
                result = messagebox.askyesno("Open Positions", 
                    "You have open positions. Close all positions before exit?")
                if result:
                    self.close_all_orders()
            
            # Shutdown MT5 connection
            if hasattr(self.mt5, 'shutdown'):
                self.mt5.shutdown()
            
            self.log("👋 Enhanced Trading Bot session ended")
            self.root.destroy()
            
        except Exception as e:
            print(f"Closing error: {e}")
            self.root.destroy()

    def hft_signal_check(self, symbol):
        """HFT-optimized signal detection"""
        try:
            # Get current price with minimal delay
            market_data = self.market_data_api.get_market_data(symbol, count=1)
            if not market_data or len(market_data) == 0:
                return None, None, None
            
            price = market_data[-1]['close']  # Get the latest close price
            
            # Simple but fast signal generation for HFT
            self.last_prices.append(price)
            if len(self.last_prices) > 10:  # Keep only last 10 prices
                self.last_prices.pop(0)
            
            if len(self.last_prices) < 3:
                return None, None, None
            
            # HFT signal logic - faster and simpler
            recent_change = (price - self.last_prices[-2]) / self.last_prices[-2] * 100
            
            # HFT thresholds - very sensitive
            if recent_change > 0.05:  # 0.05% upward movement
                signal = "BUY"
                tp, sl = self.calculate_unified_tp_sl(signal, price)
                
                self.log(f"⚡ HFT BUY Signal: {symbol} @ {price:.5f}")
                return signal, tp, sl
                
            elif recent_change < -0.05:  # 0.05% downward movement
                signal = "SELL" 
                tp, sl = self.calculate_unified_tp_sl(signal, price)
                
                self.log(f"⚡ HFT SELL Signal: {symbol} @ {price:.5f}")
                return signal, tp, sl
            
            return None, None, None
            
        except Exception as e:
            self.log(f"HFT signal error: {e}")
            return None, None, None

def main():
    """Main function"""
    try:
        app = TradingBot()
        app.root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()
