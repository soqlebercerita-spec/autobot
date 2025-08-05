#!/usr/bin/env python3
"""
Enhanced Trading Bot for Windows with MT5 - Fixed Price Retrieval & Signal Generation
Optimized for capturing market opportunities effectively with real MetaTrader5 integration
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
from enhanced_indicators import EnhancedIndicators
from config import config

# Import MT5 with enhanced error handling
from mt5_wrapper import mt5, MT5_AVAILABLE

class TradingBotWindows:
    def __init__(self):
        # Enhanced components
        self.indicators = EnhancedIndicators()
        
        # Bot state
        self.running = False
        self.modal_awal = None
        self.last_price = None
        self.last_reset_date = datetime.date.today()
        self.order_counter = 0
        self.total_opportunities_captured = 0
        self.total_opportunities_missed = 0
        self.price_retrieval_failures = 0
        
        # Threading
        self.bot_thread = None
        
        # Price data cache for reliability
        self.price_cache = {}
        self.last_successful_fetch = {}
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup enhanced GUI interface for Windows MT5"""
        self.root = tk.Tk()
        self.root.title("🚀 Enhanced Trading Bot Windows - MT5 Opportunity Capture System")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)  # Make window resizable
        self.root.configure(bg="#f0f0f0")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables - Updated to balance-based system
        self.symbol_var = tk.StringVar(value=config.DEFAULT_SYMBOL)
        self.lot_var = tk.StringVar(value=str(config.DEFAULT_LOT))
        self.interval_var = tk.StringVar(value=str(config.DEFAULT_INTERVAL))
        self.tp_balance_var = tk.StringVar(value=str(config.TP_PERSEN_BALANCE * 100))
        self.sl_balance_var = tk.StringVar(value=str(config.SL_PERSEN_BALANCE * 100))
        self.account_info_var = tk.StringVar(value="Account: Not Connected")
        self.profit_var = tk.StringVar(value="Real-time P/L: -")
        self.balance_var = tk.StringVar(value="$0.00")
        self.scalping_mode_var = tk.BooleanVar(value=config.SCALPING_OVERRIDE_ENABLED)
        self.hft_mode_var = tk.BooleanVar(value=False)
        self.opportunities_var = tk.StringVar(value="Opportunities: Captured: 0 | Missed: 0")
        
        # HFT support
        self.last_prices = []  # Price history for HFT signals
        self.price_status_var = tk.StringVar(value="Price Feed: Disconnected")
        
        self.create_enhanced_gui()
    
    def create_enhanced_gui(self):
        """Create enhanced GUI elements for Windows MT5"""
        # Style
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Success.TLabel", foreground="green", font=("Segoe UI", 10, "bold"))
        style.configure("Warning.TLabel", foreground="orange", font=("Segoe UI", 10, "bold"))
        style.configure("Error.TLabel", foreground="red", font=("Segoe UI", 10, "bold"))
        
        # Header frame with MT5 status
        header_frame = ttk.LabelFrame(self.root, text="🤖 Enhanced Windows MT5 Trading Bot - Fixed & Optimized", padding=10)
        header_frame.pack(fill="x", padx=10, pady=5)
        
        mt5_status_frame = ttk.Frame(header_frame)
        mt5_status_frame.pack(fill="x")
        
        if MT5_AVAILABLE:
            ttk.Label(mt5_status_frame, text="✅ MetaTrader5 Available - Ready for Live Trading", 
                     style="Success.TLabel").pack(side="left")
        else:
            ttk.Label(mt5_status_frame, text="❌ MetaTrader5 Not Found - Install MT5 for live trading", 
                     style="Error.TLabel").pack(side="left")
        
        ttk.Label(header_frame, text="🔧 Fixed: Price Retrieval | ⚡ Optimized: Signal Generation | 🎯 Enhanced: Opportunity Capture", 
                 style="Success.TLabel").pack(pady=5)
        
        # Enhanced info frame
        info_frame = ttk.LabelFrame(self.root, text="📊 Account & Performance Monitor", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill="x")
        
        ttk.Label(info_grid, textvariable=self.account_info_var).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(info_grid, textvariable=self.profit_var, foreground="green").grid(row=0, column=1, sticky="e", padx=5)
        ttk.Label(info_grid, textvariable=self.price_status_var, style="Success.TLabel").grid(row=1, column=0, sticky="w", padx=5)
        ttk.Label(info_grid, textvariable=self.opportunities_var, style="Success.TLabel").grid(row=1, column=1, sticky="e", padx=5)
        
        info_grid.columnconfigure(0, weight=1)
        info_grid.columnconfigure(1, weight=1)
        
        # Enhanced settings with tabs
        setting_frame = ttk.LabelFrame(self.root, text="⚙️ Enhanced Trading Configuration", padding=10)
        setting_frame.pack(padx=10, pady=5, fill="x")
        
        settings_notebook = ttk.Notebook(setting_frame)
        settings_notebook.pack(fill="x", expand=True)
        
        # Basic settings tab
        basic_tab = ttk.Frame(settings_notebook)
        settings_notebook.add(basic_tab, text="Basic Settings")
        
        basic_left = ttk.Frame(basic_tab)
        basic_left.grid(row=0, column=0, padx=10, sticky="nw")
        
        basic_fields = [
            ("Symbol:", self.symbol_var),
            ("Lot Size:", self.lot_var),
            ("Scan Interval (s):", self.interval_var),
        ]
        
        for i, (label, var) in enumerate(basic_fields):
            ttk.Label(basic_left, text=label).grid(row=i, column=0, sticky="e", pady=5)
            ttk.Entry(basic_left, textvariable=var, width=20).grid(row=i, column=1, pady=5)
        
        basic_right = ttk.Frame(basic_tab)
        basic_right.grid(row=0, column=1, padx=10, sticky="nw")
        
        risk_fields = [
            ("TP % dari Modal:", self.tp_balance_var),
            ("SL % dari Modal:", self.sl_balance_var),
        ]
        
        for i, (label, var) in enumerate(risk_fields):
            ttk.Label(basic_right, text=label).grid(row=i, column=0, sticky="e", pady=5)
            ttk.Entry(basic_right, textvariable=var, width=20).grid(row=i, column=1, pady=5)
        
        # Enhanced configuration display
        config_right = ttk.Frame(basic_tab)
        config_right.grid(row=0, column=2, padx=10, sticky="nw")
        
        ttk.Label(config_right, text="🔧 Enhanced Config:", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(config_right, text=f"• Spike Threshold: {config.LONJAKAN_THRESHOLD}", font=("Segoe UI", 8)).grid(row=1, column=0, sticky="w")
        ttk.Label(config_right, text=f"• Signal Confidence: {config.SIGNAL_CONFIDENCE_THRESHOLD}", font=("Segoe UI", 8)).grid(row=2, column=0, sticky="w")
        ttk.Label(config_right, text=f"• Max Orders: {config.MAX_ORDER_PER_SESSION}", font=("Segoe UI", 8)).grid(row=3, column=0, sticky="w")
        
        # Scalping settings tab
        scalping_tab = ttk.Frame(settings_notebook)
        settings_notebook.add(scalping_tab, text="Scalping Mode")
        
        ttk.Checkbutton(scalping_tab, text="🔥 Enable Enhanced Scalping Mode (0.5% TP, 2% SL)", 
                       variable=self.scalping_mode_var).pack(anchor="w", pady=5)
        
        ttk.Checkbutton(scalping_tab, text="⚡ Enable HFT Mode (0.3% TP, 1.5% SL)", 
                       variable=self.hft_mode_var).pack(anchor="w", pady=5)
        
        # HFT mode info
        hft_info = ttk.LabelFrame(scalping_tab, text="⚡ HFT Mode Info", padding=5)
        hft_info.pack(fill="x", pady=5)
        
        ttk.Label(hft_info, text="• Ultra-fast 1s scanning", font=("Segoe UI", 8)).pack(anchor="w")
        ttk.Label(hft_info, text="• 100 orders per session", font=("Segoe UI", 8)).pack(anchor="w")
        ttk.Label(hft_info, text="• 0.15 confidence threshold", font=("Segoe UI", 8)).pack(anchor="w")
        
        # Enhanced control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=15)
        
        # Primary buttons
        primary_buttons = ttk.Frame(button_frame)
        primary_buttons.pack()
        
        self.connect_button = ttk.Button(primary_buttons, text="🔗 Connect MT5", 
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
        self.test_button = ttk.Button(secondary_buttons, text="🧪 Test Price Feed", 
                                    command=self.test_price_feed)
        self.reset_button = ttk.Button(secondary_buttons, text="🔄 Reset Counters", 
                                     command=self.reset_counters)
        
        self.close_button.grid(row=0, column=0, padx=10)
        self.test_button.grid(row=0, column=1, padx=10)
        self.reset_button.grid(row=0, column=2, padx=10)
        
        # Enhanced log frame with multiple tabs
        log_frame = ttk.LabelFrame(self.root, text="📊 Enhanced Trading Monitor & Logs", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        log_notebook = ttk.Notebook(log_frame)
        log_notebook.pack(fill="both", expand=True)
        
        # Trading log tab
        log_tab = ttk.Frame(log_notebook)
        log_notebook.add(log_tab, text="Trading Log")
        
        self.log_box = ScrolledText(log_tab, width=140, height=18, 
                                   bg="#ffffff", fg="#333333", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True)
        
        # Performance tab
        perf_tab = ttk.Frame(log_notebook)
        log_notebook.add(perf_tab, text="Performance Monitor")
        
        self.perf_box = ScrolledText(perf_tab, width=140, height=18, 
                                    bg="#f8f8f8", fg="#333333", font=("Consolas", 9))
        self.perf_box.pack(fill="both", expand=True)
        
        # Price feed tab
        price_tab = ttk.Frame(log_notebook)
        log_notebook.add(price_tab, text="Price Feed Monitor")
        
        self.price_box = ScrolledText(price_tab, width=140, height=18, 
                                     bg="#f0f8ff", fg="#333333", font=("Consolas", 9))
        self.price_box.pack(fill="both", expand=True)
    
    def reset_counters(self):
        """Reset all performance counters"""
        self.total_opportunities_captured = 0
        self.total_opportunities_missed = 0
        self.price_retrieval_failures = 0
        self.order_counter = 0
        self.update_opportunities_display()
        self.log("🔄 All counters reset to zero")
    
    def update_opportunities_display(self):
        """Update opportunities and performance display"""
        total = self.total_opportunities_captured + self.total_opportunities_missed
        success_rate = (self.total_opportunities_captured / total * 100) if total > 0 else 0
        
        self.opportunities_var.set(
            f"Captured: {self.total_opportunities_captured} | "
            f"Missed: {self.total_opportunities_missed} | "
            f"Success: {success_rate:.1f}% | "
            f"Feed Errors: {self.price_retrieval_failures}"
        )
    
    def test_price_feed(self):
        """Test price feed connectivity and reliability"""
        if not MT5_AVAILABLE:
            messagebox.showerror("Error", "MetaTrader5 not available for testing")
            return
        
        self.log("🧪 Testing price feed connectivity...")
        symbol = self.symbol_var.get()
        
        try:
            # Test basic connection
            if not mt5.initialize():
                self.log("❌ MT5 initialization failed for testing")
                return
            
            # Test symbol availability
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.log(f"⚠️  Symbol {symbol} not found, trying to add to Market Watch...")
                if not mt5.symbol_select(symbol, True):
                    self.log(f"❌ Failed to add {symbol} to Market Watch")
                    return
                else:
                    self.log(f"✅ Successfully added {symbol} to Market Watch")
            
            # Test price retrieval
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                self.log(f"✅ Price feed test successful!")
                self.log(f"   Symbol: {symbol}")
                self.log(f"   Bid: {tick.bid:.5f}")
                self.log(f"   Ask: {tick.ask:.5f}")
                self.log(f"   Spread: {tick.ask - tick.bid:.5f}")
                self.log(f"   Time: {datetime.datetime.fromtimestamp(tick.time)}")
                
                self.price_status_var.set("Price Feed: ✅ Connected & Working")
                
                # Test historical data
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
                if rates is not None and len(rates) > 0:
                    self.log(f"✅ Historical data available: {len(rates)} bars")
                    latest_close = rates[-1]['close']
                    self.log(f"   Latest close: {latest_close:.5f}")
                else:
                    self.log("⚠️  Historical data retrieval issue")
                
            else:
                self.log(f"❌ Failed to get price data for {symbol}")
                self.price_status_var.set("Price Feed: ❌ Failed")
                
        except Exception as e:
            self.log(f"❌ Price feed test error: {e}")
            self.price_status_var.set("Price Feed: ❌ Error")
    
    # ======================
    # ENHANCED LOGGING
    # ======================
    def log_to_file(self, text):
        """Enhanced log to file with rotation"""
        try:
            with open(config.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now()} - {text}\n")
        except Exception as e:
            print(f"Error logging to file: {e}")

    def log(self, text, log_type="INFO"):
        """Enhanced logging with visual indicators"""
        timestamp = f"{datetime.datetime.now():%H:%M:%S}"
        
        # Enhanced visual indicators
        if "opportunity" in text.lower() or "signal" in text.lower():
            log_entry = f"🎯 {timestamp} - {text}"
        elif "error" in text.lower() or "fail" in text.lower():
            log_entry = f"❌ {timestamp} - {text}"
        elif "success" in text.lower() or "profit" in text.lower() or "executed" in text.lower():
            log_entry = f"✅ {timestamp} - {text}"
        elif "warning" in text.lower() or "spike" in text.lower():
            log_entry = f"⚠️  {timestamp} - {text}"
        elif "test" in text.lower():
            log_entry = f"🧪 {timestamp} - {text}"
        elif "connect" in text.lower():
            log_entry = f"🔗 {timestamp} - {text}"
        else:
            log_entry = f"ℹ️  {timestamp} - {text}"
        
        self.log_box.insert(tk.END, log_entry + "\n")
        self.log_box.see(tk.END)
        self.log_to_file(text)

    def log_performance(self, text):
        """Log performance metrics"""
        timestamp = f"{datetime.datetime.now():%H:%M:%S}"
        perf_entry = f"📊 {timestamp} - {text}"
        
        self.perf_box.insert(tk.END, perf_entry + "\n")
        self.perf_box.see(tk.END)

    def log_price_feed(self, text):
        """Log price feed information"""
        timestamp = f"{datetime.datetime.now():%H:%M:%S}"
        price_entry = f"💹 {timestamp} - {text}"
        
        self.price_box.insert(tk.END, price_entry + "\n")
        self.price_box.see(tk.END)

    def send_telegram(self, text):
        """Enhanced Telegram notifications"""
        try:
            if (config.TELEGRAM_BOT_TOKEN != "your_bot_token_here" and 
                config.TELEGRAM_CHAT_ID != "your_chat_id_here"):
                
                enhanced_text = f"🤖 Enhanced Windows MT5 Bot\n{text}\n⏰ {datetime.datetime.now():%Y-%m-%d %H:%M:%S}"
                
                url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
                data = {"chat_id": config.TELEGRAM_CHAT_ID, "text": enhanced_text}
                
                response = requests.post(url, data=data, timeout=5)
                if response.status_code == 200:
                    self.log("📱 Telegram notification sent")
                else:
                    self.log(f"📱 Telegram failed: {response.status_code}")
                    
        except Exception as e:
            self.log(f"📱 Telegram error: {e}")

    def export_trade_log(self, harga_order, tp, sl, sinyal, confidence=0, strength=0):
        """Enhanced trade log export with additional metrics"""
        try:
            filename = config.TRADE_LOG_FILE
            fieldnames = ["timestamp", "signal", "price", "tp", "sl", "confidence", "strength", 
                         "success_rate", "feed_errors", "session_orders"]
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
                    "success_rate": round(self.calculate_success_rate(), 1),
                    "feed_errors": self.price_retrieval_failures,
                    "session_orders": self.order_counter
                })
        except Exception as e:
            self.log(f"Error exporting trade log: {e}")
    
    def calculate_success_rate(self):
        """Calculate opportunity capture success rate"""
        total = self.total_opportunities_captured + self.total_opportunities_missed
        if total == 0:
            return 0.0
        return (self.total_opportunities_captured / total) * 100
    
    # ======================
    # ENHANCED MT5 CONNECTION
    # ======================
    def connect_mt5(self):
        """Enhanced MetaTrader5 connection with comprehensive checks"""
        try:
            if not MT5_AVAILABLE:
                messagebox.showerror("MetaTrader5 Not Found", 
                    "MetaTrader5 Python library not installed!\n\n"
                    "Installation steps:\n"
                    "1. Open Command Prompt as Administrator\n"
                    "2. Run: pip install MetaTrader5\n"
                    "3. Restart this application\n\n"
                    "Also ensure:\n"
                    "• MetaTrader5 terminal is installed\n"
                    "• You are logged into your trading account\n"
                    "• Automated trading is enabled in MT5")
                return False
            
            self.log("🔗 Connecting to MetaTrader5 with enhanced validation...")
            
            # Initialize MT5 connection
            if not mt5.initialize():
                error_code, error_msg = mt5.last_error()
                error_details = f"MT5 initialization failed\nError: {error_code} - {error_msg}\n\n"
                
                # Enhanced error diagnostics
                diagnostics = "Common solutions:\n"
                diagnostics += "• Ensure MT5 is running and logged in\n"
                diagnostics += "• Enable 'Allow automated trading' in MT5 Tools→Options→Expert Advisors\n"
                diagnostics += "• Enable 'Allow DLL imports'\n"
                diagnostics += "• Check if MT5 terminal is not busy with manual trading\n"
                diagnostics += "• Restart MT5 terminal and try again\n"
                diagnostics += "• Run this application as Administrator"
                
                self.log(f"❌ {error_details}")
                messagebox.showerror("MT5 Connection Failed", error_details + diagnostics)
                return False
            
            # Validate account information
            account_info = mt5.account_info()
            if not account_info:
                self.log("❌ Failed to retrieve account information")
                messagebox.showerror("Account Error", 
                    "Failed to get account information.\n\n"
                    "Please ensure:\n"
                    "• You are logged into a trading account\n"
                    "• Account is not blocked or suspended\n"
                    "• Connection to trading server is stable")
                mt5.shutdown()
                return False
            
            # Store initial account data
            self.modal_awal = account_info.balance
            account_text = f"Account: {account_info.login} | Balance: ${account_info.balance:,.2f}"
            self.account_info_var.set(account_text)
            
            # Test symbol availability and market data
            symbol = self.symbol_var.get()
            self.log(f"🔍 Validating symbol: {symbol}")
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.log(f"⚠️  Symbol {symbol} not found, attempting to add to Market Watch...")
                if not mt5.symbol_select(symbol, True):
                    self.log(f"❌ Failed to add {symbol} to Market Watch")
                    messagebox.showwarning("Symbol Warning", 
                        f"Symbol {symbol} is not available in your MT5.\n\n"
                        f"Please:\n"
                        f"1. Open MT5 Market Watch\n"
                        f"2. Right-click and select 'Show All'\n"
                        f"3. Find and add {symbol}\n"
                        f"4. Or change symbol in bot settings")
                else:
                    self.log(f"✅ Successfully added {symbol} to Market Watch")
            else:
                self.log(f"✅ Symbol {symbol} is available")
            
            # Test price feed
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                self.log(f"✅ Price feed working - Bid: {tick.bid:.5f}, Ask: {tick.ask:.5f}")
                self.price_status_var.set("Price Feed: ✅ Connected")
                self.log_price_feed(f"Initial price check: {symbol} @ {tick.bid:.5f}/{tick.ask:.5f}")
            else:
                self.log(f"⚠️  Price feed issue for {symbol}")
                self.price_status_var.set("Price Feed: ⚠️  Warning")
            
            # Test historical data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 50)
            if rates is not None and len(rates) >= config.MIN_DATA_POINTS:
                self.log(f"✅ Historical data available: {len(rates)} bars")
                self.log_price_feed(f"Historical data test: {len(rates)} bars retrieved")
            else:
                self.log("⚠️  Limited historical data - may affect signal quality")
            
            # Final connection status
            self.log(f"✅ Enhanced MT5 Connection Successful!")
            self.log(f"   Account: {account_info.login}")
            self.log(f"   Balance: ${account_info.balance:,.2f}")
            self.log(f"   Server: {account_info.server if hasattr(account_info, 'server') else 'Unknown'}")
            self.log(f"   Leverage: 1:{account_info.leverage}")
            self.log(f"   Symbol: {symbol} ✅")
            self.log("🚀 Ready for enhanced opportunity capture!")
            
            # Update UI state
            self.connect_button.config(state="disabled")
            self.start_button.config(state="normal")
            
            # Performance tracking
            self.log_performance(f"MT5 connected successfully - Account: {account_info.login}")
            
            messagebox.showinfo("Connection Successful", 
                f"Enhanced MT5 Connection Established!\n\n"
                f"Account: {account_info.login}\n"
                f"Balance: ${account_info.balance:,.2f}\n"
                f"Symbol: {symbol} ✅\n"
                f"Price Feed: Active ✅\n\n"
                f"Enhanced features active:\n"
                f"• Fixed price retrieval\n"
                f"• Optimized signal generation\n"
                f"• Better opportunity capture")
            
            return True
                
        except Exception as e:
            error_msg = f"Enhanced connection error: {str(e)}"
            self.log(f"❌ {error_msg}")
            messagebox.showerror("Connection Error", 
                f"{error_msg}\n\n"
                f"Please try:\n"
                f"1. Restart MetaTrader5\n"
                f"2. Ensure you're logged in\n"
                f"3. Run as Administrator\n"
                f"4. Check firewall settings")
            return False
    
    def get_data_with_enhanced_retry(self, symbol, timeframe=None, n=100):
        """Enhanced data retrieval with comprehensive retry logic"""
        if not MT5_AVAILABLE:
            return None
        
        # Set default timeframe if not provided
        if timeframe is None:
            timeframe = mt5.TIMEFRAME_M1
        
        for attempt in range(config.PRICE_FETCH_RETRY):
            try:
                # Try to get rates
                rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
                
                if rates is not None and len(rates) >= config.MIN_DATA_POINTS:
                    # Cache successful data
                    self.price_cache[symbol] = rates
                    self.last_successful_fetch[symbol] = datetime.datetime.now()
                    
                    if attempt > 0:
                        self.log(f"✅ Price data recovered on attempt {attempt + 1}")
                        self.log_price_feed(f"Data recovery successful: {len(rates)} bars")
                    
                    return rates
                else:
                    if attempt == 0:
                        self.log_price_feed(f"Price fetch attempt {attempt + 1} - insufficient data")
                
                # Wait with exponential backoff
                if attempt < config.PRICE_FETCH_RETRY - 1:
                    wait_time = 0.5 * (2 ** attempt)  # 0.5, 1.0, 2.0 seconds
                    time.sleep(wait_time)
                    
            except Exception as e:
                self.log_price_feed(f"Fetch error attempt {attempt + 1}: {e}")
                if attempt < config.PRICE_FETCH_RETRY - 1:
                    time.sleep(0.5 + attempt * 0.3)
        
        # If all attempts failed, try to use cached data
        if symbol in self.price_cache:
            cached_data = self.price_cache[symbol]
            cache_age = datetime.datetime.now() - self.last_successful_fetch.get(symbol, datetime.datetime.min)
            
            if cache_age.total_seconds() < 300:  # Use cache if less than 5 minutes old
                self.log_price_feed(f"Using cached data (age: {cache_age.total_seconds():.0f}s)")
                return cached_data[-n:] if len(cached_data) > n else cached_data
        
        # Track failures
        self.price_retrieval_failures += 1
        self.log(f"❌ Price data fetch failed after {config.PRICE_FETCH_RETRY} attempts")
        self.price_status_var.set(f"Price Feed: ❌ Error (Failures: {self.price_retrieval_failures})")
        
        return None

    def get_total_open_orders(self):
        """Get total open positions with enhanced error handling"""
        if not MT5_AVAILABLE:
            return 0
        
        try:
            positions = mt5.positions_get()
            count = len(positions) if positions else 0
            return count
        except Exception as e:
            self.log(f"Error getting open orders: {e}")
            return 0

    def close_all_orders(self):
        """Enhanced close all positions with detailed reporting"""
        if not MT5_AVAILABLE:
            self.log("❌ MT5 not available for closing positions")
            return False
        
        try:
            positions = mt5.positions_get()
            if not positions:
                self.log("ℹ️  No positions to close")
                return True
            
            self.log(f"🔄 Attempting to close {len(positions)} positions...")
            closed_count = 0
            failed_count = 0
            
            for pos in positions:
                try:
                    symbol = pos.symbol
                    order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
                    
                    tick = mt5.symbol_info_tick(symbol)
                    if not tick:
                        self.log(f"❌ No price data for {symbol}")
                        failed_count += 1
                        continue
                    
                    price = tick.ask if order_type == mt5.ORDER_TYPE_SELL else tick.bid
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": order_type,
                        "position": pos.ticket,
                        "price": price,
                        "deviation": config.MT5_DEVIATION,
                        "magic": config.MT5_MAGIC_NUMBER,
                        "comment": "Enhanced Bot - Close All",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request)
                    if result and result.retcode == 10009:
                        self.log(f"✅ Position {pos.ticket} closed @ {price:.5f}")
                        closed_count += 1
                    else:
                        error_code = result.retcode if result else "Unknown"
                        self.log(f"❌ Failed to close {pos.ticket} - Error: {error_code}")
                        failed_count += 1
                        
                except Exception as e:
                    self.log(f"❌ Error closing position {pos.ticket}: {e}")
                    failed_count += 1
            
            self.log(f"📊 Close All Summary: ✅{closed_count} closed, ❌{failed_count} failed")
            self.log_performance(f"Close all executed: {closed_count}/{len(positions)} successful")
            
            return closed_count > 0
            
        except Exception as e:
            self.log(f"❌ Close all error: {e}")
            return False

    def manual_close_all(self):
        """Manual close all with confirmation"""
        try:
            open_count = self.get_total_open_orders()
            if open_count == 0:
                messagebox.showinfo("No Positions", "No open positions to close")
                return
            
            result = messagebox.askyesno("Confirm Close All", 
                f"Close all {open_count} open positions?\n\n"
                f"This action cannot be undone.")
            
            if result:
                success = self.close_all_orders()
                if success:
                    self.send_telegram(f"🔄 All positions closed manually - {open_count} positions")
                else:
                    messagebox.showwarning("Close Failed", "Some positions could not be closed. Check logs.")
                    
        except Exception as e:
            self.log(f"Manual close error: {e}")
    
    # ======================
    # ENHANCED SIGNAL GENERATION
    # ======================
    def enhanced_signal_check(self, symbol):
        """Enhanced signal checking with improved opportunity capture for MT5"""
        try:
            # Get market data with enhanced retry
            rates = self.get_data_with_enhanced_retry(symbol, n=config.DATA_BUFFER_SIZE)
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
            
            # Enhanced price spike detection - FIXED THRESHOLD
            if len(close_prices) >= 5:
                recent_prices = close_prices[-5:]
                price_volatility = np.std(recent_prices) / np.mean(recent_prices) * 100
                
                # More lenient spike detection (key fix)
                if price_volatility > config.LONJAKAN_THRESHOLD:
                    self.log(f"⚠️  Moderate volatility detected ({price_volatility:.2f}%), proceeding with analysis...")
                    self.log_price_feed(f"Volatility: {price_volatility:.2f}% (threshold: {config.LONJAKAN_THRESHOLD}%)")
                    # Reduce confidence but don't skip
                    volatility_penalty = 0.15
                else:
                    volatility_penalty = 0
            else:
                volatility_penalty = 0
            
            # Enhanced signal analysis using indicators
            signal_data = self.indicators.enhanced_signal_analysis(
                close_prices, high_prices, low_prices
            )
            
            signal = signal_data['signal']
            base_confidence = signal_data['confidence']
            
            # Apply volatility penalty but don't completely block signals
            adjusted_confidence = max(0, base_confidence - volatility_penalty)
            
            # More lenient threshold for better opportunity capture
            min_confidence = config.SIGNAL_CONFIDENCE_THRESHOLD * 0.75  # 25% more lenient
            
            if signal in ['BUY', 'SELL'] and adjusted_confidence >= min_confidence:
                # Calculate TP/SL
                tp, sl = self.calculate_unified_tp_sl(signal, current_price)
                
                self.log(f"🎯 ENHANCED SIGNAL DETECTED!")
                self.log(f"   Signal: {signal}")
                self.log(f"   Confidence: {adjusted_confidence:.3f} (Base: {base_confidence:.3f})")
                self.log(f"   Strength: {signal_data['strength']:.3f}")
                self.log(f"   Current Price: {current_price:.5f}")
                self.log(f"   TP: {tp:.5f} | SL: {sl:.5f}")
                
                # Log detailed technical analysis
                indicators = signal_data.get('indicators', {})
                scores = signal_data.get('scores', {})
                
                if indicators:
                    self.log(f"   📊 Technical Analysis:")
                    self.log(f"      RSI: {indicators.get('rsi', 0):.1f}")
                    self.log(f"      MA Short: {indicators.get('ma_short', 0):.5f}")
                    self.log(f"      EMA Fast: {indicators.get('ema_fast', 0):.5f}")
                    self.log(f"      BB Upper: {indicators.get('bb_upper', 0):.5f}")
                    self.log(f"      BB Lower: {indicators.get('bb_lower', 0):.5f}")
                
                if scores:
                    self.log(f"   📈 Signal Scores:")
                    self.log(f"      Buy Score: {scores.get('buy_score', 0)}")
                    self.log(f"      Sell Score: {scores.get('sell_score', 0)}")
                
                # Price feed logging
                self.log_price_feed(f"Signal generated: {signal} @ {current_price:.5f} (Conf: {adjusted_confidence:.3f})")
                
                self.total_opportunities_captured += 1
                self.update_opportunities_display()
                
                return signal, tp, sl
            else:
                # Enhanced rejection logging
                if signal in ['BUY', 'SELL']:
                    self.log(f"⚠️  Signal {signal} rejected - Confidence {adjusted_confidence:.3f} < {min_confidence:.3f}")
                    self.log_price_feed(f"Signal rejected: {signal} - Low confidence")
                    self.total_opportunities_missed += 1
                else:
                    self.log(f"📊 Market analysis: {signal} - Confidence: {adjusted_confidence:.3f}")
                
                self.update_opportunities_display()
                return None, 0, 0
            
        except Exception as e:
            self.log(f"❌ Enhanced signal check error: {e}")
            self.price_retrieval_failures += 1
            self.total_opportunities_missed += 1
            self.update_opportunities_display()
            return None, 0, 0

    def calculate_unified_tp_sl(self, signal, current_price):
        """Calculate TP/SL based on ACCOUNT BALANCE percentage for REAL TRADING"""
        try:
            # Get REAL account balance from MT5
            if not MT5_AVAILABLE:
                self.log("❌ MT5 not available - cannot get real account balance")
                return 0, 0
            
            account_info = mt5.account_info()
            if account_info is None:
                self.log("❌ Cannot connect to MT5 for real account info")
                return 0, 0
            
            balance = account_info.balance
            symbol = self.symbol_var.get()
            lot_size = float(self.lot_var.get())
            
            # Determine TP/SL as PERCENTAGE OF ACCOUNT BALANCE
            if self.hft_mode_var.get():
                tp_balance_pct = config.HFT_TP_PERSEN_BALANCE     # 0.3% of balance
                sl_balance_pct = config.HFT_SL_PERSEN_BALANCE     # 1.5% of balance
                mode = "HFT"
            elif self.scalping_mode_var.get():
                tp_balance_pct = config.SCALPING_TP_PERSEN_BALANCE # 0.5% of balance
                sl_balance_pct = config.SCALPING_SL_PERSEN_BALANCE # 2% of balance
                mode = "Scalping"
            else:
                tp_balance_pct = config.TP_PERSEN_BALANCE         # 1% of balance
                sl_balance_pct = config.SL_PERSEN_BALANCE         # 3% of balance
                mode = "Normal"
            
            # Calculate target profit/loss in MONEY (not price)
            tp_money_target = balance * tp_balance_pct
            sl_money_limit = balance * sl_balance_pct
            
            # Convert money amounts to price levels using lot size
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.log(f"❌ Cannot get symbol info for {symbol}")
                return 0, 0
            
            # Calculate contract value
            contract_size = symbol_info.trade_contract_size
            point = symbol_info.point
            
            # For most symbols, 1 pip = 10 * point
            pip_size = 10 * point
            
            # Calculate pip value per lot
            pip_value_per_lot = contract_size * pip_size
            
            # Calculate pip value for our lot size
            pip_value = pip_value_per_lot * lot_size
            
            # Convert money targets to pips
            tp_pips = tp_money_target / pip_value if pip_value > 0 else 50
            sl_pips = sl_money_limit / pip_value if pip_value > 0 else 100
            
            # Convert pips to price levels
            if signal == "BUY":
                tp = current_price + (tp_pips * pip_size)
                sl = current_price - (sl_pips * pip_size)
            else:  # SELL
                tp = current_price - (tp_pips * pip_size)
                sl = current_price + (sl_pips * pip_size)
            
            # Validate results
            if tp <= 0 or sl <= 0:
                raise ValueError("Invalid TP/SL calculation")
            
            self.log(f"💰 {mode} BALANCE-BASED TP/SL (REAL TRADING):")
            self.log(f"   Account Balance: ${balance:,.2f}")
            self.log(f"   TP Money Target: ${tp_money_target:,.2f} ({tp_balance_pct*100:.1f}% of balance)")
            self.log(f"   SL Money Limit: ${sl_money_limit:,.2f} ({sl_balance_pct*100:.1f}% of balance)")
            self.log(f"   Current Price: {current_price:.5f}")
            self.log(f"   TP Price: {tp:.5f}")
            self.log(f"   SL Price: {sl:.5f}")
            self.log(f"   Contract Size: {contract_size}, Pip Value: ${pip_value:.2f}")
            
            return tp, sl
            
        except Exception as e:
            self.log(f"❌ Balance-based TP/SL calculation error: {e}")
            # Emergency fallback - minimal risk
            if signal == "BUY":
                tp = current_price * 1.005  # 0.5% profit
                sl = current_price * 0.995  # 0.5% loss
            else:
                tp = current_price * 0.995  # 0.5% profit
                sl = current_price * 1.005  # 0.5% loss
            return tp, sl
    
    def hft_signal_check(self, symbol):
        """HFT-optimized signal detection for Windows MT5"""
        try:
            # Get current price from MT5
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None, None, None
            
            price = (tick.ask + tick.bid) / 2  # Use mid price
            
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

    # Old function removed - now using unified TP/SL system
    
    # ======================
    # ENHANCED TRADING LOOP
    # ======================
    def enhanced_trading_loop(self):
        """Enhanced main trading loop optimized for Windows MT5"""
        self.log("🚀 Enhanced trading loop started with MT5 integration")
        
        while self.running:
            try:
                current_time = datetime.datetime.now()
                current_hour = current_time.hour
                
                # Enhanced trading hours check
                if not (config.TRADING_START_HOUR <= current_hour <= config.TRADING_END_HOUR):
                    self.log(f"⏰ Outside trading hours ({config.TRADING_START_HOUR}:00-{config.TRADING_END_HOUR}:00)")
                    time.sleep(60)  # Check every minute
                    continue
                
                # Daily reset logic
                if current_time.date() != self.last_reset_date:
                    self.log("🌅 New trading day - Resetting session counters")
                    self.order_counter = 0
                    self.last_reset_date = current_time.date()
                
                # Update account and position information
                self.update_account_display()
                
                # Check maximum positions limit (will be overridden below)
                open_positions = self.get_total_open_orders()
                
                # Enhanced signal detection with HFT support
                symbol = self.symbol_var.get()
                
                # Choose signal method based on mode
                if self.hft_mode_var.get():
                    signal, tp, sl = self.hft_signal_check(symbol)
                    max_orders = config.MAX_ORDER_PER_SESSION_HFT
                    interval_sleep = config.HFT_INTERVAL
                else:
                    signal, tp, sl = self.enhanced_signal_check(symbol)
                    max_orders = config.MAX_ORDER_PER_SESSION
                    interval_sleep = int(self.interval_var.get())
                
                # Update order limit check
                if open_positions >= max_orders:
                    mode_name = "HFT" if self.hft_mode_var.get() else "Normal"
                    self.log(f"📊 Maximum {mode_name} positions reached ({open_positions}/{max_orders})")
                    time.sleep(interval_sleep)
                    continue
                
                if signal and tp and sl:
                    # Execute trade with enhanced MT5 integration
                    success = self.execute_enhanced_trade(signal, tp, sl)
                    if success:
                        self.order_counter += 1
                        self.log(f"✅ Trade #{self.order_counter} executed successfully!")
                        
                        # Enhanced notification
                        self.send_telegram(
                            f"🎯 Enhanced MT5 Trade Alert!\n"
                            f"Signal: {signal}\n"
                            f"Symbol: {symbol}\n"
                            f"TP: {tp:.5f} | SL: {sl:.5f}\n"
                            f"Session Orders: {self.order_counter}\n"
                            f"Success Rate: {self.calculate_success_rate():.1f}%"
                        )
                        
                        # Performance logging
                        self.log_performance(f"Trade executed: {signal} @ TP:{tp:.5f} SL:{sl:.5f}")
                
                # Enhanced scanning with dynamic intervals and HFT support
                if self.hft_mode_var.get():
                    scan_interval = config.HFT_INTERVAL  # 1 second for HFT
                else:
                    base_interval = int(self.interval_var.get())
                    # Adjust interval based on market conditions and failures
                    if self.price_retrieval_failures > 5:
                        scan_interval = max(base_interval * 1.5, 15)  # Slower if many failures
                        self.log_price_feed(f"Increased scan interval due to feed issues: {scan_interval}s")
                    else:
                        scan_interval = max(base_interval, 5)  # Minimum 5 seconds
                
                time.sleep(scan_interval)
                
            except Exception as e:
                self.log(f"❌ Trading loop error: {e}")
                self.log_performance(f"Trading loop error: {str(e)[:100]}")
                time.sleep(10)  # Wait longer on error
        
        self.log("🛑 Enhanced trading loop stopped")

    def execute_enhanced_trade(self, signal, tp, sl):
        """Execute trade with enhanced MT5 integration and error handling"""
        if not MT5_AVAILABLE:
            self.log("❌ MT5 not available for trade execution")
            return False
        
        try:
            symbol = self.symbol_var.get()
            lot = float(self.lot_var.get())
            
            # Get current market price with validation
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.log("❌ Unable to get current market price")
                self.price_retrieval_failures += 1
                return False
            
            # Determine order parameters
            if signal == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                order_name = "Enhanced BUY"
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                order_name = "Enhanced SELL"
            
            # Validate TP/SL levels
            spread = tick.ask - tick.bid
            min_distance = spread * 3  # Minimum distance should be 3x spread
            
            if signal == "BUY":
                if tp - price < min_distance or price - sl < min_distance:
                    self.log(f"⚠️  Adjusting TP/SL due to spread constraints")
                    tp = max(tp, price + min_distance)
                    sl = min(sl, price - min_distance)
            else:
                if price - tp < min_distance or sl - price < min_distance:
                    self.log(f"⚠️  Adjusting TP/SL due to spread constraints")
                    tp = min(tp, price - min_distance)
                    sl = max(sl, price + min_distance)
            
            # Create enhanced order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": config.MT5_DEVIATION,
                "magic": config.MT5_MAGIC_NUMBER,
                "comment": f"Enhanced Bot v2.0 - {signal}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Log order details before execution
            self.log(f"📤 Sending {order_name} order:")
            self.log(f"   Symbol: {symbol}")
            self.log(f"   Volume: {lot} lots")
            self.log(f"   Price: {price:.5f}")
            self.log(f"   TP: {tp:.5f}")
            self.log(f"   SL: {sl:.5f}")
            self.log(f"   Spread: {spread:.5f}")
            
            # Send order to MT5
            result = mt5.order_send(request)
            
            if result and result.retcode == 10009:  # TRADE_RETCODE_DONE
                # Successful order execution
                self.log(f"🎯 {order_name} Order SUCCESSFUL!")
                self.log(f"   Ticket: #{result.deal}")
                self.log(f"   Executed @ {price:.5f}")
                self.log(f"   TP: {tp:.5f} | SL: {sl:.5f}")
                self.log(f"   Volume: {lot} lots")
                
                # Update price feed status
                self.price_status_var.set("Price Feed: ✅ Active & Trading")
                
                # Export detailed trade log
                signal_data = self.indicators.enhanced_signal_analysis(
                    self.get_price_array_from_rates(
                        self.get_data_with_enhanced_retry(symbol, n=50)
                    )
                )
                
                self.export_trade_log(
                    price, tp, sl, signal,
                    signal_data.get('confidence', 0),
                    signal_data.get('strength', 0)
                )
                
                # Performance and price feed logging
                self.log_performance(f"Order executed: {signal} #{result.deal} @ {price:.5f}")
                self.log_price_feed(f"Trade executed: {signal} #{result.deal} @ {price:.5f}")
                
                return True
            else:
                # Order execution failed
                error_code = result.retcode if result else "Unknown"
                error_description = self.get_error_description(error_code)
                
                self.log(f"❌ {order_name} Order FAILED!")
                self.log(f"   Error Code: {error_code}")
                self.log(f"   Error: {error_description}")
                self.log(f"   Price: {price:.5f}")
                self.log(f"   Volume: {lot}")
                
                self.log_performance(f"Order failed: {signal} - Error {error_code}")
                
                return False
                
        except Exception as e:
            self.log(f"❌ Trade execution error: {e}")
            self.log_performance(f"Trade execution error: {str(e)[:100]}")
            return False

    def get_price_array_from_rates(self, rates):
        """Extract close prices from rates data"""
        if rates is None:
            return np.array([])
        return rates['close']

    def get_error_description(self, error_code):
        """Get human-readable error description for MT5 error codes"""
        error_descriptions = {
            10004: "Requote",
            10006: "Request rejected",
            10007: "Request canceled by trader",
            10008: "Order placed",
            10009: "Request completed",
            10010: "Only part of the request was completed",
            10011: "Request processing error",
            10012: "Request canceled by timeout",
            10013: "Invalid request",
            10014: "Invalid volume in the request",
            10015: "Invalid price in the request",
            10016: "Invalid stops in the request",
            10017: "Trade is disabled",
            10018: "Market is closed",
            10019: "There is not enough money to complete the request",
            10020: "Prices changed",
            10021: "There are no quotes to process the request",
            10022: "Invalid order expiration date in the request",
            10023: "Order state changed",
            10024: "Too frequent requests",
            10025: "No changes in request",
            10026: "Autotrading disabled by server",
            10027: "Autotrading disabled by client terminal",
            10028: "Request locked for processing",
            10029: "Order or position frozen",
            10030: "Invalid order filling type",
            10031: "No connection with the trade server"
        }
        return error_descriptions.get(error_code, f"Unknown error code: {error_code}")

    def update_account_display(self):
        """Enhanced account display update with position monitoring"""
        try:
            if not MT5_AVAILABLE:
                return
            
            account_info = mt5.account_info()
            if not account_info:
                self.log("⚠️  Unable to retrieve account information")
                return
            
            # Calculate current profit/loss from positions
            positions = mt5.positions_get()
            total_profit = 0
            if positions:
                for pos in positions:
                    try:
                        current_price = mt5.symbol_info_tick(pos.symbol)
                        if current_price:
                            if pos.type == 0:  # BUY position
                                profit = (current_price.bid - pos.price_open) * pos.volume * 100000
                            else:  # SELL position
                                profit = (pos.price_open - current_price.ask) * pos.volume * 100000
                            total_profit += profit
                    except:
                        pass
            
            # Update display variables
            balance = account_info.balance
            equity = balance + total_profit
            
            account_text = f"MT5 Account: {account_info.login} | Balance: ${balance:,.2f} | Equity: ${equity:,.2f}"
            self.account_info_var.set(account_text)
            
            profit_color = "green" if total_profit >= 0 else "red"
            profit_text = f"Real-time P/L: ${total_profit:,.2f}"
            self.profit_var.set(profit_text)
            
            # Calculate performance metrics
            if self.modal_awal and self.modal_awal > 0:
                total_return = ((equity - self.modal_awal) / self.modal_awal) * 100
                daily_pnl = total_profit
                
                # Log performance metrics periodically
                if self.order_counter % 5 == 0:  # Every 5 orders
                    self.log_performance(f"Performance Update:")
                    self.log_performance(f"  Total Return: {total_return:+.2f}%")
                    self.log_performance(f"  Current P/L: ${total_profit:+.2f}")
                    self.log_performance(f"  Open Positions: {len(positions) if positions else 0}")
                    self.log_performance(f"  Success Rate: {self.calculate_success_rate():.1f}%")
                
        except Exception as e:
            self.log(f"Account display update error: {e}")
    
    # ======================
    # BOT CONTROLS
    # ======================
    def start_bot(self):
        """Start enhanced trading bot with comprehensive validation"""
        try:
            if self.running:
                messagebox.showwarning("Already Running", "Enhanced bot is already running!")
                return
            
            # Enhanced input validation
            try:
                lot = float(self.lot_var.get())
                interval = int(self.interval_var.get())
                tp_pct = float(self.tp_balance_var.get())
                sl_pct = float(self.sl_balance_var.get())
                
                if lot <= 0 or lot > 10:
                    raise ValueError("Lot size must be between 0.01 and 10")
                if interval < 5 or interval > 300:
                    raise ValueError("Interval must be between 5 and 300 seconds")
                if tp_pct <= 0 or tp_pct > 50:
                    raise ValueError("Take Profit must be between 0.1% and 50%")
                if sl_pct <= 0 or sl_pct > 50:
                    raise ValueError("Stop Loss must be between 0.1% and 50%")
                    
            except ValueError as e:
                messagebox.showerror("Invalid Input", f"Please check your settings:\n{e}")
                return
            
            # MT5 connection validation
            if not MT5_AVAILABLE:
                messagebox.showerror("MT5 Required", 
                    "MetaTrader5 is required for live trading.\n"
                    "Please install MT5 and the Python library.")
                return
            
            # Account balance validation
            if MT5_AVAILABLE:
                account_info = mt5.account_info()
                if account_info and account_info.balance < config.SALDO_MINIMAL:
                    result = messagebox.askyesno("Low Balance Warning", 
                        f"Account balance (${account_info.balance:,.2f}) is below "
                        f"recommended minimum (${config.SALDO_MINIMAL:,.2f}).\n\n"
                        f"Trading with low balance increases risk.\n"
                        f"Continue anyway?")
                    if not result:
                        return
            
            # Symbol validation
            symbol = self.symbol_var.get()
            if MT5_AVAILABLE:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    result = messagebox.askyesno("Symbol Warning", 
                        f"Symbol {symbol} may not be available.\n"
                        f"This could cause trading errors.\n\n"
                        f"Continue anyway?")
                    if not result:
                        return
            
            # Start the enhanced bot
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            # Reset session counters
            self.order_counter = 0
            self.total_opportunities_captured = 0
            self.total_opportunities_missed = 0
            self.price_retrieval_failures = 0
            self.update_opportunities_display()
            
            # Start the trading thread
            self.bot_thread = threading.Thread(target=self.enhanced_trading_loop, daemon=True)
            self.bot_thread.start()
            
            # Enhanced startup logging
            self.log("🚀 ENHANCED WINDOWS MT5 TRADING BOT STARTED!")
            self.log(f"   ═══════════════════════════════════════")
            self.log(f"   🎯 Symbol: {symbol}")
            self.log(f"   💰 Lot Size: {lot}")
            self.log(f"   ⏱️  Scan Interval: {interval}s")
            self.log(f"   📈 Take Profit: {tp_pct}%")
            self.log(f"   📉 Stop Loss: {sl_pct}%")
            self.log(f"   🔥 Scalping Mode: {'✅ Enabled' if self.scalping_mode_var.get() else '❌ Disabled'}")
            self.log(f"   ═══════════════════════════════════════")
            self.log(f"   🔧 Enhanced Configuration:")
            self.log(f"      Price Spike Threshold: {config.LONJAKAN_THRESHOLD} (Fixed)")
            self.log(f"      Signal Confidence: {config.SIGNAL_CONFIDENCE_THRESHOLD}")
            self.log(f"      Max Positions: {config.MAX_ORDER_PER_SESSION}")
            self.log(f"      Trading Hours: {config.TRADING_START_HOUR}:00-{config.TRADING_END_HOUR}:00")
            self.log(f"   ═══════════════════════════════════════")
            self.log("🎯 Enhanced features active:")
            self.log("   ✅ Fixed price retrieval mechanism")
            self.log("   ⚡ Optimized signal generation")
            self.log("   🔄 Enhanced retry logic")
            self.log("   📊 Improved opportunity capture")
            self.log("   🛡️  Advanced error handling")
            
            # Update price feed status
            self.price_status_var.set("Price Feed: 🔄 Monitoring")
            
            # Performance logging
            self.log_performance("Enhanced MT5 trading session started")
            self.log_performance(f"Configuration: {symbol} | {lot} lots | {interval}s interval")
            
            # Send enhanced startup notification
            self.send_telegram(
                f"🚀 Enhanced Windows MT5 Bot Started!\n"
                f"Symbol: {symbol}\n"
                f"Lot: {lot} | Interval: {interval}s\n"
                f"TP: {tp_pct}% | SL: {sl_pct}%\n"
                f"Enhanced features: ✅ Active\n"
                f"Spike threshold: {config.LONJAKAN_THRESHOLD} (Fixed)"
            )
            
        except Exception as e:
            self.log(f"❌ Start bot error: {e}")
            messagebox.showerror("Start Error", f"Failed to start enhanced bot:\n{e}")

    def stop_bot(self):
        """Stop enhanced trading bot with comprehensive reporting"""
        try:
            if not self.running:
                messagebox.showwarning("Not Running", "Enhanced bot is not currently running!")
                return
            
            self.running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            
            # Calculate final session statistics
            success_rate = self.calculate_success_rate()
            total_opportunities = self.total_opportunities_captured + self.total_opportunities_missed
            
            # Enhanced stop logging
            self.log("🛑 ENHANCED WINDOWS MT5 TRADING BOT STOPPED!")
            self.log(f"   ═══════════════════════════════════════")
            self.log(f"   📊 SESSION SUMMARY:")
            self.log(f"      Orders Executed: {self.order_counter}")
            self.log(f"      Opportunities Captured: {self.total_opportunities_captured}")
            self.log(f"      Opportunities Missed: {self.total_opportunities_missed}")
            self.log(f"      Total Opportunities: {total_opportunities}")
            self.log(f"      Success Rate: {success_rate:.1f}%")
            self.log(f"      Price Feed Errors: {self.price_retrieval_failures}")
            self.log(f"   ═══════════════════════════════════════")
            
            # Calculate performance if available
            if MT5_AVAILABLE:
                try:
                    account_info = mt5.account_info()
                    if account_info and self.modal_awal:
                        total_return = ((account_info.balance - self.modal_awal) / self.modal_awal) * 100
                        self.log(f"   💰 PERFORMANCE:")
                        self.log(f"      Initial Balance: ${self.modal_awal:,.2f}")
                        self.log(f"      Final Balance: ${account_info.balance:,.2f}")
                        self.log(f"      Total Return: {total_return:+.2f}%")
                        self.log(f"   ═══════════════════════════════════════")
                except:
                    pass
            
            # Update price feed status
            self.price_status_var.set("Price Feed: ⏸️  Stopped")
            
            # Final performance logging
            self.log_performance("Enhanced trading session ended")
            self.log_performance(f"Final stats: {self.order_counter} orders, {success_rate:.1f}% success rate")
            
            # Send enhanced stop notification
            self.send_telegram(
                f"🛑 Enhanced Windows MT5 Bot Stopped\n"
                f"Session Summary:\n"
                f"• Orders: {self.order_counter}\n"
                f"• Captured: {self.total_opportunities_captured}\n"
                f"• Missed: {self.total_opportunities_missed}\n"
                f"• Success Rate: {success_rate:.1f}%\n"
                f"• Feed Errors: {self.price_retrieval_failures}"
            )
            
        except Exception as e:
            self.log(f"❌ Stop bot error: {e}")

    def on_closing(self):
        """Enhanced application closing with cleanup"""
        try:
            if self.running:
                result = messagebox.askyesno("Confirm Exit", 
                    "Enhanced trading bot is still running.\n"
                    "Stop bot and exit application?")
                if result:
                    self.stop_bot()
                    time.sleep(2)  # Give time for bot to stop properly
                else:
                    return
            
            # Check for open positions
            if MT5_AVAILABLE:
                try:
                    open_count = self.get_total_open_orders()
                    if open_count > 0:
                        result = messagebox.askyesno("Open Positions", 
                            f"You have {open_count} open positions.\n"
                            f"Close all positions before exit?")
                        if result:
                            self.close_all_orders()
                            time.sleep(1)
                except:
                    pass
            
            # Cleanup MT5 connection
            if MT5_AVAILABLE:
                try:
                    mt5.shutdown()
                    self.log("🔌 MT5 connection closed")
                except:
                    pass
            
            # Final session log
            self.log("👋 Enhanced Windows MT5 Trading Bot session ended")
            self.log("💾 All logs saved to files")
            
            # Performance summary to file
            try:
                with open("session_summary.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n=== Session Summary {datetime.datetime.now()} ===\n")
                    f.write(f"Orders: {self.order_counter}\n")
                    f.write(f"Opportunities Captured: {self.total_opportunities_captured}\n")
                    f.write(f"Opportunities Missed: {self.total_opportunities_missed}\n")
                    f.write(f"Success Rate: {self.calculate_success_rate():.1f}%\n")
                    f.write(f"Feed Errors: {self.price_retrieval_failures}\n")
                    f.write("=======================================\n")
            except:
                pass
            
            self.root.destroy()
            
        except Exception as e:
            print(f"Enhanced closing error: {e}")
            self.root.destroy()

def main():
    """Main function to start the enhanced Windows MT5 trading bot"""
    try:
        print("🚀 Starting Enhanced Windows MT5 Trading Bot...")
        print("✅ Fixed price retrieval issues")
        print("⚡ Optimized signal generation")
        print("🎯 Enhanced opportunity capture")
        print("🔧 Improved error handling")
        
        app = TradingBotWindows()
        app.root.mainloop()
        
    except Exception as e:
        print(f"Application startup error: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application:\n{e}")

if __name__ == "__main__":
    main()
