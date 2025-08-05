
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
import platform
import sys

# Import MT5 library if available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    # Provide a mock object if MT5 is not installed
    class MockMT5:
        def initialize(self): return False
        def account_info(self): return None
        def symbol_info_tick(self, symbol): return None
        def symbol_info(self, symbol): return None
        def symbol_select(self, symbol, select): return False
        def order_send(self, request): return None
        def positions_get(self): return []
        def position_close(self, ticket): return None
        def shutdown(self): pass
        def last_error(self): return (1, "MT5 not available")
        def copy_rates_from_pos(self, symbol, timeframe, start, count): return None
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        TRADE_ACTION_DEAL = 0
        TRADE_RETCODE_DONE = 10009
        ORDER_TIME_GTC = 0
        ORDER_FILLING_IOC = 0
        TIMEFRAME_M1 = 1
    
    mt5 = MockMT5()

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

# Import market data with fallback
try:
    from tradebot.market_data_api import MarketDataAPI
except ImportError:
    # Create simple fallback if market_data_api not available
    class MarketDataAPI:
        def get_price(self, symbol):
            return None
        def get_market_data(self, symbol, count=1):
            # Basic simulation if market_data_api is missing
            return [{'close': 1.1, 'bid': 1.1, 'ask': 1.1, 'spread': 0, 'time': time.time(), 'source': 'Simulation'}] * count
        def get_price_array(self, symbol, count=50):
            return [1.1] * count

# Import other modules
try:
    from tradebot.simulation_trading import SimulationTrading
except ImportError:
    class SimulationTrading:
        def __init__(self, market_api):
            self.market_api = market_api
        def get_current_price(self, symbol):
            return {'price': 1.1, 'bid': 1.1, 'ask': 1.1, 'spread': 0, 'time': time.time(), 'source': 'Simulation'}
        def positions_get(self): return []
        def symbol_info_tick(self, symbol): 
            return type('tick', (), {'ask': 1.1, 'bid': 1.1})
        def order_send(self, request): 
            return type('result', (), {'retcode': 10009, 'deal': 12345})
        def position_close(self, ticket): return True
        def account_info(self):
            return type('account', (), {'balance': 10000, 'equity': 10000, 'profit': 0, 'login': 'DEMO'})
        def update_positions(self): pass
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        TRADE_ACTION_DEAL = 0
        TRADE_RETCODE_DONE = 10009
        ORDER_TIME_GTC = 0
        ORDER_FILLING_IOC = 0

try:
    from tradebot.enhanced_indicators import EnhancedIndicators
except ImportError:
    class EnhancedIndicators:
        def enhanced_signal_analysis(self, close_prices, high_prices=None, low_prices=None):
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'strength': 0.5,
                'indicators': {},
                'scores': {}
            }

try:
    from tradebot.config import config
except ImportError:
    # Create default config
    class Config:
        DEFAULT_SYMBOL = "EURUSD"
        DEFAULT_LOT = 0.01
        DEFAULT_INTERVAL = 10
        TP_PERSEN_BALANCE = 0.01  # 1%
        SL_PERSEN_BALANCE = 0.03  # 3%
        HFT_TP_PERSEN_BALANCE = 0.003  # 0.3%
        HFT_SL_PERSEN_BALANCE = 0.015  # 1.5%
        SCALPING_TP_PERSEN_BALANCE = 0.005  # 0.5%
        SCALPING_SL_PERSEN_BALANCE = 0.02   # 2%
        SCALPING_OVERRIDE_ENABLED = True
        LONJAKAN_THRESHOLD = 2.0
        SIGNAL_CONFIDENCE_THRESHOLD = 0.6
        SIGNAL_CONFIDENCE_THRESHOLD_HFT = 0.15
        MAX_ORDER_PER_SESSION = 50
        MAX_ORDER_PER_SESSION_HFT = 100
        HFT_INTERVAL = 1
        DATA_BUFFER_SIZE = 100
        MIN_DATA_POINTS = 20
        PRICE_FETCH_RETRY = 3
        TRADING_START_HOUR = 0
        TRADING_END_HOUR = 23
        SALDO_MINIMAL = 1000
        MT5_DEVIATION = 20
        MT5_MAGIC_NUMBER = 123456
        LOG_FILE = "trading_log.txt"
        TRADE_LOG_FILE = "trade_log.csv"
        TELEGRAM_BOT_TOKEN = "your_bot_token_here"
        TELEGRAM_CHAT_ID = "your_chat_id_here"
        WINRATE_BOOST_ENABLED = True
        TREND_CONFIRMATION_PERIOD = 10
        MULTI_CONFIRMATION_REQUIRED = 2
        SIGNAL_STRENGTH_MULTIPLIER = 1.2
        TP_PERSEN_DEFAULT = 0.01
        SL_PERSEN_DEFAULT = 0.03
        SCALPING_TP_PERSEN = 0.005
        SCALPING_SL_PERSEN = 0.02
        ENABLE_24_7_TRADING = True
    
    config = Config()

class TradingBot:
    def __init__(self):
        # Initialize basic variables first
        self.running = False
        self.connected = False
        self.bot_thread = None
        self.market_api = None
        self.simulation = None
        self.indicators = None
        
        # Initialize state variables
        self.modal_awal = None
        self.last_price = None
        self.last_prices = []
        self.last_reset_date = datetime.date.today()
        self.order_counter = 0
        self.total_opportunities_captured = 0
        self.total_opportunities_missed = 0
        
        # Initialize GUI components to None first
        self.root = None
        self.log_box = None
        self.perf_box = None
        self.connect_button = None
        self.start_button = None
        self.stop_button = None
        self.close_button = None
        self.reset_button = None
        
        try:
            # Initialize simulation trading
            self.market_api = MarketDataAPI()
            self.simulation = SimulationTrading(self.market_api)

            # Initialize platform detection
            self.is_windows = platform.system() == "Windows"
            
            print(f"🖥️  Platform: {platform.system()}")

            # Initialize MT5 related objects (but don't connect yet)
            if MT5_AVAILABLE and self.is_windows:
                self.mt5 = mt5  # Use actual MT5
                print("✅ MT5 library available for Windows")
            else:
                # Use simulation trading for compatibility
                self.mt5 = self.simulation
                if not self.is_windows:
                    print("⚠️  MT5 requires Windows platform - using simulation mode")
                else:
                    print("⚠️  MT5 not available - using simulation mode")

            self.indicators = EnhancedIndicators()
            
            # Setup GUI
            self.setup_gui()
            
            # Now try to auto-connect MT5 after GUI is ready
            if MT5_AVAILABLE and self.is_windows:
                try:
                    self.auto_connect_mt5()
                except Exception as e:
                    self.log(f"MT5 auto-connection failed: {e}")
            
            self.log_performance("Session started")
            
        except Exception as e:
            print(f"Initialization error: {e}")
            if hasattr(self, 'root') and self.root:
                try:
                    self.root.destroy()
                except:
                    pass

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
        self.tp_var = tk.StringVar(value=str(config.TP_PERSEN_BALANCE * 100))
        self.sl_var = tk.StringVar(value=str(config.SL_PERSEN_BALANCE * 100))
        self.scalping_tp_var = tk.StringVar(value=str(config.SCALPING_TP_PERSEN_BALANCE * 100))
        self.scalping_sl_var = tk.StringVar(value=str(config.SCALPING_SL_PERSEN_BALANCE * 100))
        self.account_info_var = tk.StringVar(value="Account: Not Connected")
        self.profit_var = tk.StringVar(value="Real-time P/L: -")
        self.balance_var = tk.StringVar(value="$10,000")
        self.scalping_mode_var = tk.BooleanVar(value=config.SCALPING_OVERRIDE_ENABLED)
        self.opportunities_var = tk.StringVar(value="Opportunities: Captured: 0 | Missed: 0")

        self.create_enhanced_gui()

        # Initial setup message
        self.log("🚀 Enhanced Integrated Trading Bot Started")
        self.log("   Features: Technical Analysis + Risk Management + Windows MT5")
        self.log("   Version: v2.0 Enhanced for Windows MT5")

        # Platform-specific messages
        if self.is_windows:
            if MT5_AVAILABLE:
                self.log("✅ Windows MT5 Platform Ready")
                self.log("💡 For live trading: Ensure MT5 is running and logged in")
            else:
                self.log("⚠️  Install MetaTrader5: pip install MetaTrader5")
        else:
            self.log("ℹ️  Non-Windows platform - Simulation mode only")

    def create_enhanced_gui(self):
        """Create enhanced GUI elements"""
        # Style
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Success.TLabel", foreground="green", font=("Segoe UI", 10, "bold"))
        style.configure("Warning.TLabel", foreground="orange", font=("Segoe UI", 10, "bold"))
        style.configure("Accent.TButton", background="#007bff", foreground="white", font=("Segoe UI", 10, "bold"))

        # Header frame
        header_frame = ttk.LabelFrame(self.root, text="🤖 Enhanced Trading Bot - Opportunity Capture System", padding=10)
        header_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(header_frame, text="✅ Fixed Price Retrieval | ⚡ Optimized Signal Generation | 🎯 Enhanced Opportunity Capture",
                 style="Success.TLabel").pack()

        # Info frame
        info_frame = ttk.LabelFrame(self.root, text="Account & Performance Info", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)

        info_grid = ttk.Frame(info_frame)
        info_grid.pack(fill="x")

        ttk.Label(info_grid, textvariable=self.account_info_var).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(info_grid, textvariable=self.profit_var, foreground="green").grid(row=0, column=1, sticky="e", padx=5)
        ttk.Label(info_grid, textvariable=self.opportunities_var, style="Success.TLabel").grid(row=1, column=0, columnspan=2, pady=5)

        info_grid.columnconfigure(0, weight=1)
        info_grid.columnconfigure(1, weight=1)

        # Settings frame
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

        # Balance-based TP/SL section
        balance_frame = ttk.LabelFrame(basic_tab, text="💰 Balance-Based TP/SL", padding=10)
        balance_frame.grid(row=0, column=1, padx=10, sticky="n")

        ttk.Label(balance_frame, text="💡 TP/SL berdasarkan % modal",
                 style="Success.TLabel").grid(row=0, column=0, columnspan=2, pady=5)

        self.tp_balance_var = tk.StringVar(value=str(config.TP_PERSEN_BALANCE * 100))
        self.sl_balance_var = tk.StringVar(value=str(config.SL_PERSEN_BALANCE * 100))

        unified_fields = [
            ("Normal TP (% modal):", self.tp_balance_var),
            ("Normal SL (% modal):", self.sl_balance_var),
        ]

        for i, (label, var) in enumerate(unified_fields):
            ttk.Label(balance_frame, text=label).grid(row=i+1, column=0, sticky="e", pady=3, padx=5)
            ttk.Entry(balance_frame, textvariable=var, width=15).grid(row=i+1, column=1, pady=3, padx=5)

        # Mode indicator
        mode_info = ttk.Label(balance_frame, text="🔄 Mode otomatis:\n• Normal: User setting diatas\n• Scalping: 0.5% TP, 2% SL\n• HFT: 0.3% TP, 1.5% SL",
                             style="TLabel", font=("Segoe UI", 8), justify="left")
        mode_info.grid(row=3, column=0, columnspan=2, pady=5)

        # HFT & Scalping settings tab
        hft_tab = ttk.Frame(settings_notebook)
        settings_notebook.add(hft_tab, text="HFT & Scalping")

        # HFT Mode Section
        hft_frame = ttk.LabelFrame(hft_tab, text="⚡ High-Frequency Trading (HFT)", padding=10)
        hft_frame.pack(fill="x", padx=5, pady=5)

        self.hft_mode_var = tk.BooleanVar(value=False)

        hft_controls = ttk.Frame(hft_frame)
        hft_controls.pack(fill="x")

        ttk.Checkbutton(hft_controls, text="⚡ Enable HFT Mode (1s scan, fast trading)",
                       variable=self.hft_mode_var,
                       command=self.toggle_hft_mode).pack(anchor="w", pady=5)

        # Scalping Mode Section
        scalping_frame = ttk.LabelFrame(hft_tab, text="🔥 Scalping Mode", padding=10)
        scalping_frame.pack(fill="x", padx=5, pady=5)

        ttk.Checkbutton(scalping_frame, text="🔥 Enable Enhanced Scalping Mode",
                       variable=self.scalping_mode_var,
                       style="Success.TLabel").pack(pady=5)

        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=15)

        # Primary buttons
        primary_buttons = ttk.Frame(button_frame)
        primary_buttons.pack()

        self.connect_button = ttk.Button(primary_buttons, text="🔗 Connect to Market",
                                       command=self.connect_mt5, style="TButton")
        self.start_button = ttk.Button(primary_buttons, text="🚀 Start Enhanced Bot",
                                     command=self.start_bot, state="normal")
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

    def toggle_hft_mode(self):
        """Toggle HFT mode on/off"""
        if self.hft_mode_var.get():
            self.log("⚡ HFT MODE ACTIVATED!")
            self.interval_var.set("1")  # 1 second scanning
        else:
            self.log("🔄 HFT MODE DISABLED")
            self.interval_var.set("10")  # Normal scanning

    def update_opportunities_display(self):
        """Update opportunities display"""
        try:
            if hasattr(self, 'opportunities_var') and self.opportunities_var:
                self.opportunities_var.set(
                    f"Opportunities: Captured: {self.total_opportunities_captured} | "
                    f"Missed: {self.total_opportunities_missed} | "
                    f"Success Rate: {self.calculate_success_rate():.1f}%"
                )
        except (tk.TclError, RuntimeError, AttributeError):
            pass  # GUI variable no longer valid

    def calculate_success_rate(self):
        """Calculate opportunity capture success rate"""
        total = self.total_opportunities_captured + self.total_opportunities_missed
        if total == 0:
            return 0.0
        return (self.total_opportunities_captured / total) * 100

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
        elif "error" in text.lower() or "fail" in text.lower() or "❌" in text:
            log_entry = f"❌ {timestamp} - {text}"
        elif "success" in text.lower() or "profit" in text.lower() or "✅" in text:
            log_entry = f"✅ {timestamp} - {text}"
        elif "warning" in text.lower() or "⚠️" in text:
            log_entry = f"⚠️  {timestamp} - {text}"
        else:
            log_entry = f"ℹ️  {timestamp} - {text}"

        # Safe GUI logging with thread safety
        try:
            if hasattr(self, 'log_box') and self.log_box and hasattr(self, 'root') and self.root and self.root.winfo_exists():
                # Use after_idle to ensure thread safety
                self.root.after_idle(lambda: self._safe_log_insert(log_entry))
            else:
                print(log_entry)  # Fallback to console if GUI not ready
        except (tk.TclError, RuntimeError, AttributeError):
            print(log_entry)  # Fallback to console on any GUI error
        
        self.log_to_file(text)

    def _safe_log_insert(self, log_entry):
        """Safely insert log entry into GUI"""
        try:
            if self.log_box and self.log_box.winfo_exists():
                self.log_box.insert(tk.END, log_entry + "\n")
                self.log_box.see(tk.END)
        except (tk.TclError, AttributeError):
            pass  # GUI already destroyed
    
    def _safe_perf_insert(self, perf_entry):
        """Safely insert performance entry into GUI"""
        try:
            if self.perf_box and self.perf_box.winfo_exists():
                self.perf_box.insert(tk.END, perf_entry + "\n")
                self.perf_box.see(tk.END)
        except (tk.TclError, AttributeError):
            pass  # GUI already destroyed

    def log_performance(self, text):
        """Log performance metrics"""
        timestamp = f"{datetime.datetime.now():%H:%M:%S}"
        perf_entry = f"{timestamp} - {text}"

        # Safe GUI logging with thread safety
        try:
            if hasattr(self, 'perf_box') and self.perf_box and hasattr(self, 'root') and self.root and self.root.winfo_exists():
                # Use after_idle to ensure thread safety
                self.root.after_idle(lambda: self._safe_perf_insert(perf_entry))
            else:
                print(f"PERF: {perf_entry}")  # Fallback to console
        except (tk.TclError, RuntimeError, AttributeError):
            print(f"PERF: {perf_entry}")  # Fallback to console on any GUI error

    def auto_connect_mt5(self):
        """Auto-connect MT5 during initialization (safe version)"""
        if not MT5_AVAILABLE:
            self.log("❌ MT5 library not available")
            return False

        if not self.is_windows:
            self.log("❌ MT5 requires Windows platform")
            return False

        try:
            self.log("🔄 Initializing MT5 connection...")
            if not mt5.initialize():
                error_code, error_msg = mt5.last_error()
                self.log(f"❌ MT5 initialization failed: {error_code} - {error_msg}")
                return False

            # Get account information
            account_info = mt5.account_info()
            if account_info is None:
                self.log("❌ Failed to get MT5 account info")
                return False

            # Success!
            self.log("✅ Windows MT5 Connection Successful!")
            self.log(f"   Account: {account_info.login}")
            self.log(f"   Balance: ${account_info.balance:,.2f}")
            self.log(f"   Leverage: 1:{account_info.leverage}")
            self.connected = True

            # Update account display
            self.account_info_var.set(f"Account: {account_info.login} | Balance: ${account_info.balance:,.2f}")

            # Safely update GUI buttons if they exist
            if hasattr(self, 'connect_button') and self.connect_button:
                self.connect_button.config(state="disabled")
            if hasattr(self, 'start_button') and self.start_button:
                self.start_button.config(state="normal")

            return True

        except Exception as e:
            self.log(f"❌ Windows MT5 connection error: {e}")
            self.connected = False
            return False

    def connect_mt5(self):
        """Enhanced MT5 connection for Windows (manual connection)"""
        if not MT5_AVAILABLE:
            self.log("❌ MT5 library not available")
            messagebox.showerror("Error", "MetaTrader5 library not installed.\nInstall with: pip install MetaTrader5")
            return False

        if not self.is_windows:
            self.log("❌ MT5 requires Windows platform")
            messagebox.showerror("Error", "MT5 only works on Windows platform")
            return False

        try:
            self.log("🔄 Initializing MT5 connection...")
            if not mt5.initialize():
                error_code, error_msg = mt5.last_error()
                self.log(f"❌ MT5 initialization failed: {error_code} - {error_msg}")
                messagebox.showerror("MT5 Error", f"MT5 initialization failed: {error_code} - {error_msg}\n\nMake sure:\n• MT5 is running\n• You are logged in\n• Auto trading is enabled")
                return False

            # Get account information
            account_info = mt5.account_info()
            if account_info is None:
                self.log("❌ Failed to get MT5 account info")
                messagebox.showerror("MT5 Error", "Failed to get account info.\nPlease check your MT5 login status.")
                return False

            # Success!
            self.log("✅ Windows MT5 Connection Successful!")
            self.log(f"   Account: {account_info.login}")
            self.log(f"   Balance: ${account_info.balance:,.2f}")
            self.log(f"   Leverage: 1:{account_info.leverage}")
            self.connected = True

            # Update account display
            self.account_info_var.set(f"Account: {account_info.login} | Balance: ${account_info.balance:,.2f}")

            # Update GUI buttons
            if self.connect_button:
                self.connect_button.config(state="disabled")
            if self.start_button:
                self.start_button.config(state="normal")

            messagebox.showinfo("Success", f"Connected to MT5!\nAccount: {account_info.login}\nBalance: ${account_info.balance:,.2f}")
            return True

        except Exception as e:
            self.log(f"❌ Windows MT5 connection error: {e}")
            messagebox.showerror("Connection Error", f"Failed to connect to MT5: {e}")
            self.connected = False
            return False

    def get_total_open_orders(self):
        """Get total open positions with error handling"""
        try:
            positions = self.mt5.positions_get()
            return len(positions) if positions else 0
        except Exception as e:
            self.log(f"Error getting open orders: {e}")
            return 0

    def manual_close_all(self):
        """Manual close all positions"""
        try:
            result = messagebox.askyesno("Confirm", "Are you sure you want to close all positions?")
            if result:
                if not self.connected or not MT5_AVAILABLE:
                    self.log("❌ MT5 not connected")
                    return
                
                positions = mt5.positions_get()
                if not positions:
                    self.log("ℹ️ No open positions to close")
                    messagebox.showinfo("Info", "No open positions found")
                    return
                
                closed_count = 0
                for position in positions:
                    # Prepare close request
                    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    
                    # Get current price
                    tick = mt5.symbol_info_tick(position.symbol)
                    if not tick:
                        continue
                    
                    close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                    
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": position.symbol,
                        "volume": position.volume,
                        "type": close_type,
                        "position": position.ticket,
                        "price": close_price,
                        "deviation": 20,
                        "magic": 123456,
                        "comment": "Manual close all",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    # Close position
                    close_result = mt5.order_send(close_request)
                    if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
                        closed_count += 1
                        self.log(f"✅ Closed position: {position.symbol} ticket {position.ticket}")
                    else:
                        self.log(f"❌ Failed to close position {position.ticket}")
                
                self.log(f"🔄 Closed {closed_count} out of {len(positions)} positions")
                messagebox.showinfo("Success", f"Closed {closed_count} out of {len(positions)} positions")
                
        except Exception as e:
            self.log(f"❌ Manual close error: {e}")
            messagebox.showerror("Error", f"Failed to close positions: {e}")

    def enhanced_trading_loop(self):
        """Enhanced main trading loop with real MT5 trading"""
        try:
            while self.running:
                try:
                    # Check if GUI still exists before logging
                    if not self.running:
                        break
                        
                    self.log("📊 Scanning market for opportunities...")
                    
                    # Get real market data and analyze
                    symbol = self.symbol_var.get()
                    opportunity = self.analyze_market_opportunity(symbol)
                    
                    if opportunity:
                        # Execute real trade
                        if self.execute_trade(opportunity):
                            self.total_opportunities_captured += 1
                            self.update_opportunities_display()
                            self.log(f"🎯 Trade executed: {opportunity['action']} {symbol}")
                        else:
                            self.total_opportunities_missed += 1
                            self.update_opportunities_display()
                            self.log("❌ Trade execution failed")
                    else:
                        self.log("📈 No trading opportunity found")
                    
                    # Enhanced scanning interval with interruption check
                    scan_interval = max(int(self.interval_var.get()), 1)
                    for _ in range(scan_interval):
                        if not self.running:
                            break
                        time.sleep(1)

                except Exception as e:
                    if self.running:  # Only log if still running
                        self.log(f"❌ Trading loop error: {e}")
                    time.sleep(10)
                    
        except Exception as e:
            print(f"Trading loop fatal error: {e}")
        finally:
            # Safe final log
            try:
                if self.running:
                    self.log("🛑 Enhanced trading loop stopped")
            except:
                print("🛑 Enhanced trading loop stopped")

    def start_bot(self):
        """Start enhanced trading bot"""
        try:
            if self.running:
                messagebox.showwarning("Warning", "Bot is already running!")
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
            lot = self.lot_var.get()
            interval = self.interval_var.get()
            
            self.log("🚀 ENHANCED TRADING BOT STARTED!")
            self.log(f"   Symbol: {symbol} | Lot: {lot}")
            self.log(f"   Scan Interval: {interval}s")
            self.log("🎯 Enhanced opportunity capture system active!")

        except Exception as e:
            self.log(f"❌ Start bot error: {e}")
            messagebox.showerror("Start Error", f"Failed to start bot: {e}")

    def stop_bot(self):
        """Stop enhanced trading bot"""
        try:
            if not self.running:
                messagebox.showwarning("Warning", "Bot is not running!")
                return

            # Stop the trading loop first
            self.running = False
            
            # Wait for thread to finish gracefully
            if self.bot_thread and self.bot_thread.is_alive():
                self.bot_thread.join(timeout=5)  # Wait max 5 seconds
            
            # Update GUI buttons safely
            try:
                if hasattr(self, 'start_button') and self.start_button:
                    self.start_button.config(state="normal")
                if hasattr(self, 'stop_button') and self.stop_button:
                    self.stop_button.config(state="disabled")
            except tk.TclError:
                pass  # GUI already destroyed

            # Final statistics
            success_rate = self.calculate_success_rate()

            self.log("🛑 ENHANCED TRADING BOT STOPPED!")
            self.log(f"📊 Session Summary:")
            self.log(f"   Orders Executed: {self.order_counter}")
            self.log(f"   Opportunities Captured: {self.total_opportunities_captured}")
            self.log(f"   Opportunities Missed: {self.total_opportunities_missed}")
            self.log(f"   Success Rate: {success_rate:.1f}%")

        except Exception as e:
            print(f"❌ Stop bot error: {e}")

    def analyze_market_opportunity(self, symbol):
        """Analyze market for trading opportunities"""
        try:
            if not self.connected or not MT5_AVAILABLE:
                return None
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.log(f"❌ No tick data for {symbol}")
                return None
            
            # Get recent price data for analysis
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 20)
            if rates is None or len(rates) < 10:
                self.log(f"❌ Insufficient price data for {symbol}")
                return None
            
            # Simple moving average analysis
            close_prices = [rate[4] for rate in rates]  # Close prices
            ma_short = np.mean(close_prices[-5:])  # 5-period MA
            ma_long = np.mean(close_prices[-10:])   # 10-period MA
            
            current_price = tick.bid
            spread = tick.ask - tick.bid
            
            # Check if spread is acceptable
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return None
            
            spread_pips = spread / symbol_info.point
            
            # Trading conditions
            if spread_pips > 50:  # Spread too wide
                return None
            
            # Generate signal
            signal = None
            if ma_short > ma_long * 1.001:  # Upward momentum
                signal = {
                    'action': 'BUY',
                    'price': tick.ask,
                    'sl': tick.ask - (20 * symbol_info.point),  # 20 point SL
                    'tp': tick.ask + (30 * symbol_info.point),  # 30 point TP
                    'reason': 'MA crossover bullish'
                }
            elif ma_short < ma_long * 0.999:  # Downward momentum
                signal = {
                    'action': 'SELL',
                    'price': tick.bid,
                    'sl': tick.bid + (20 * symbol_info.point),  # 20 point SL
                    'tp': tick.bid - (30 * symbol_info.point),  # 30 point TP
                    'reason': 'MA crossover bearish'
                }
            
            return signal
            
        except Exception as e:
            self.log(f"❌ Market analysis error: {e}")
            return None
    
    def execute_trade(self, opportunity):
        """Execute real trade through MT5"""
        try:
            if not self.connected or not MT5_AVAILABLE:
                self.log("❌ MT5 not connected")
                return False
            
            symbol = self.symbol_var.get()
            lot_size = float(self.lot_var.get())
            
            # Check if we already have positions
            positions = mt5.positions_get(symbol=symbol)
            if positions and len(positions) >= 3:  # Max 3 positions per symbol
                self.log("⚠️ Maximum positions reached")
                return False
            
            # Prepare order request
            order_type = mt5.ORDER_TYPE_BUY if opportunity['action'] == 'BUY' else mt5.ORDER_TYPE_SELL
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": opportunity['price'],
                "sl": opportunity['sl'],
                "tp": opportunity['tp'],
                "deviation": 20,
                "magic": 123456,
                "comment": f"AuraTrade-{opportunity['reason']}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.log(f"✅ Order successful: {opportunity['action']} {lot_size} {symbol}")
                self.log(f"   Price: {opportunity['price']}, SL: {opportunity['sl']}, TP: {opportunity['tp']}")
                self.log(f"   Reason: {opportunity['reason']}")
                self.order_counter += 1
                return True
            else:
                error_msg = f"Order failed: {result.comment if result else 'Unknown error'}"
                self.log(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            self.log(f"❌ Trade execution error: {e}")
            return False

    def on_closing(self):
        """Enhanced application closing"""
        try:
            # Stop trading bot if running
            if self.running:
                try:
                    result = messagebox.askyesno("Confirm Exit",
                        "Trading bot is still running. Stop bot and exit?")
                    if result:
                        self.running = False  # Stop immediately
                        
                        # Wait for thread to finish
                        if hasattr(self, 'bot_thread') and self.bot_thread and self.bot_thread.is_alive():
                            self.bot_thread.join(timeout=3)
                        
                        time.sleep(1)
                    else:
                        return
                except tk.TclError:
                    # GUI already destroyed, just stop
                    self.running = False

            # Shutdown MT5 connection if it was established
            if hasattr(self, 'connected') and self.connected and MT5_AVAILABLE:
                try:
                    mt5.shutdown()
                    print("🔌 MT5 connection closed.")
                except Exception as e:
                    print(f"Error shutting down MT5: {e}")

            print("👋 Enhanced Trading Bot session ended")
            
            # Destroy GUI safely
            if hasattr(self, 'root') and self.root:
                try:
                    self.root.quit()  # Exit mainloop
                    self.root.destroy()
                except tk.TclError:
                    pass  # Already destroyed

        except Exception as e:
            print(f"Closing error: {e}")
            # Force destroy if needed
            try:
                if hasattr(self, 'root') and self.root:
                    self.root.destroy()
            except:
                pass

def main():
    """Main function to run the bot"""
    try:
        print("🚀 Starting Enhanced Trading Bot...")
        app = TradingBot()
        app.root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
