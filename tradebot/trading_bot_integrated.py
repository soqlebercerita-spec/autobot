#!/usr/bin/env python3
"""
Enhanced Trading Bot - Fixed Version with Advanced Features
24/7 Auto Trading • HFT • Scalping • Auto TP/SL • High Win Rate
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
import queue
import traceback

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
        @staticmethod
        def corrcoef(x, y): return [[1, 0], [0, 1]]
        @staticmethod
        def diff(data): return [data[i+1] - data[i] for i in range(len(data)-1)]
        @staticmethod
        def polyfit(x, y, deg): return [0] * (deg + 1)
        @staticmethod
        def percentile(data, pct): 
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * pct / 100
            return sorted_data[int(k)]
    np = MockNumpy()

# Import MT5 library if available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    class MockMT5:
        def initialize(self): return False
        def account_info(self): return None
        def symbol_info_tick(self, symbol): return None
        def symbol_info(self, symbol): return None
        def symbol_select(self, symbol, select): return False
        def order_send(self, request): return None
        def positions_get(self, symbol=None): return []
        def position_close(self, ticket): return None
        def shutdown(self): pass
        def last_error(self): return (1, "MT5 not available")
        def copy_rates_from_pos(self, symbol, timeframe, start, count): return None
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        POSITION_TYPE_BUY = 0
        POSITION_TYPE_SELL = 1  
        TRADE_ACTION_DEAL = 0
        TRADE_RETCODE_DONE = 10009
        ORDER_TIME_GTC = 0
        ORDER_FILLING_IOC = 0
        TIMEFRAME_M1 = 1
    mt5 = MockMT5()

# Enhanced Indicators Class
class EnhancedIndicators:
    def __init__(self):
        self.cache = {}

    def calculate_sma(self, prices, period):
        """Simple Moving Average"""
        if len(prices) < period:
            return None
        return np.mean(prices[-period:])

    def calculate_ema(self, prices, period):
        """Exponential Moving Average"""
        if len(prices) < period:
            return None
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema

    def calculate_rsi(self, prices, period=14):
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50

        gains = []
        losses = []
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices):
        """MACD Indicator"""
        if len(prices) < 26:
            return {'macd': 0, 'signal': 0, 'histogram': 0}

        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)

        if not ema12 or not ema26:
            return {'macd': 0, 'signal': 0, 'histogram': 0}

        macd = ema12 - ema26
        signal = macd * 0.9  # Simplified signal line
        histogram = macd - signal

        return {'macd': macd, 'signal': signal, 'histogram': histogram}

    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Bollinger Bands"""
        if len(prices) < period:
            return None, None, None

        middle = self.calculate_sma(prices, period)
        std = np.std(prices[-period:])

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, lower, middle

    def enhanced_signal_analysis(self, prices, symbol="EURUSD"):
        """Enhanced signal generation with multiple confirmations"""
        if len(prices) < 50:
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'strength': 0.5,
                'indicators': {},
                'reason': 'Insufficient data'
            }

        # Calculate all indicators
        sma20 = self.calculate_sma(prices, 20)
        sma50 = self.calculate_sma(prices, 50)
        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(prices)

        current_price = prices[-1]

        # Signal scoring system
        buy_score = 0
        sell_score = 0
        reasons = []

        # Trend Analysis
        if sma20 and sma50 and sma20 > sma50:
            buy_score += 2
            reasons.append("Bullish trend (SMA20 > SMA50)")
        elif sma20 and sma50 and sma20 < sma50:
            sell_score += 2
            reasons.append("Bearish trend (SMA20 < SMA50)")

        # EMA Crossover
        if ema12 and ema26:
            if ema12 > ema26:
                buy_score += 1.5
                reasons.append("EMA bullish crossover")
            else:
                sell_score += 1.5
                reasons.append("EMA bearish crossover")

        # RSI Analysis
        if rsi:
            if rsi < 30:
                buy_score += 2
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                sell_score += 2
                reasons.append(f"RSI overbought ({rsi:.1f})")
            elif 40 < rsi < 60:
                buy_score += 0.5
                sell_score += 0.5

        # MACD Analysis
        if macd['histogram'] > 0:
            buy_score += 1
            reasons.append("MACD bullish")
        elif macd['histogram'] < 0:
            sell_score += 1
            reasons.append("MACD bearish")

        # Bollinger Bands
        if bb_upper and bb_lower and bb_middle:
            if current_price < bb_lower:
                buy_score += 1.5
                reasons.append("Price below lower BB")
            elif current_price > bb_upper:
                sell_score += 1.5
                reasons.append("Price above upper BB")

        # Volatility Analysis
        volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
        if volatility > 0.02:  # High volatility
            buy_score *= 1.2
            sell_score *= 1.2
            reasons.append("High volatility boost")

        # Determine signal
        total_score = buy_score + sell_score
        if total_score > 0:
            buy_confidence = buy_score / total_score
            sell_confidence = sell_score / total_score
        else:
            buy_confidence = sell_confidence = 0.5

        if buy_score > sell_score and buy_confidence > 0.65:
            signal = 'BUY'
            confidence = buy_confidence
            strength = min(buy_score / 6, 1.0)
        elif sell_score > buy_score and sell_confidence > 0.65:
            signal = 'SELL' 
            confidence = sell_confidence
            strength = min(sell_score / 6, 1.0)
        else:
            signal = 'HOLD'
            confidence = 0.5
            strength = 0.5

        return {
            'signal': signal,
            'confidence': confidence,
            'strength': strength,
            'indicators': {
                'sma20': sma20,
                'sma50': sma50,
                'ema12': ema12,
                'ema26': ema26,
                'rsi': rsi,
                'macd': macd,
                'bb_upper': bb_upper,
                'bb_lower': bb_lower,
                'volatility': volatility
            },
            'scores': {
                'buy_score': buy_score,
                'sell_score': sell_score
            },
            'reasons': reasons[:3]  # Top 3 reasons
        }

# Enhanced Market Data API
class MarketDataAPI:
    def __init__(self):
        self.cache = {}
        self.last_update = {}

    def get_price(self, symbol):
        """Get current price with enhanced error handling"""
        try:
            # Try MT5 first if available
            if MT5_AVAILABLE:
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    return {
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'price': (tick.bid + tick.ask) / 2,
                        'spread': tick.ask - tick.bid,
                        'time': time.time(),
                        'source': 'MT5'
                    }

            # Fallback to simulation data
            base_price = self._get_base_price(symbol)
            spread = base_price * 0.0001  # 1 pip spread

            return {
                'bid': base_price - spread/2,
                'ask': base_price + spread/2,
                'price': base_price,
                'spread': spread,
                'time': time.time(),
                'source': 'Simulation'
            }

        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return None

    def _get_base_price(self, symbol):
        """Generate realistic price simulation"""
        price_map = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2650,
            'USDJPY': 149.50,
            'USDCAD': 1.3750,
            'AUDUSD': 0.6550,
            'NZDUSD': 0.5950,
            'USDCHF': 0.8950,
            'XAUUSDm': 2025.50,
            'XAUUSD': 2025.50,
            'BTCUSD': 42500.00,
            'SPX500': 4750.00
        }

        base = price_map.get(symbol, 1.0000)

        # Add realistic price movement
        current_time = time.time()
        movement = np.sin(current_time / 100) * 0.01 + (np.random.random() - 0.5) * 0.02

        return base * (1 + movement)

    def get_recent_prices(self, symbol, count=50):
        """Get recent price history"""
        try:
            if MT5_AVAILABLE:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
                if rates is not None:
                    return [rate[4] for rate in rates]  # Close prices

            # Simulation mode
            base_price = self._get_base_price(symbol)
            prices = []

            for i in range(count):
                movement = (np.random.random() - 0.5) * 0.01
                price = base_price * (1 + movement * (count - i) / count)
                prices.append(price)

            return prices

        except Exception as e:
            print(f"Error getting recent prices: {e}")
            return []

# Enhanced Config
class Config:
    # Basic Settings
    DEFAULT_SYMBOL = "EURUSD"
    DEFAULT_LOT = 0.01
    DEFAULT_INTERVAL = 5

    # Balance-based TP/SL (percentage of balance)
    TP_PERSEN_BALANCE = 0.01      # 1% of balance
    SL_PERSEN_BALANCE = 0.005     # 0.5% of balance

    # HFT Settings  
    HFT_TP_PERSEN_BALANCE = 0.003  # 0.3% for HFT
    HFT_SL_PERSEN_BALANCE = 0.002  # 0.2% for HFT
    HFT_INTERVAL = 1               # 1 second

    # Scalping Settings
    SCALPING_TP_PERSEN_BALANCE = 0.005  # 0.5% for scalping
    SCALPING_SL_PERSEN_BALANCE = 0.003  # 0.3% for scalping
    SCALPING_OVERRIDE_ENABLED = True

    # Risk Management
    MAX_ORDER_PER_SESSION = 100
    MAX_ORDER_PER_SESSION_HFT = 300
    MAX_DRAWDOWN = 10  # 10%
    SALDO_MINIMAL = 100
    MAX_RISK_PER_TRADE = 2  # 2% per trade

    # Signal Thresholds
    SIGNAL_CONFIDENCE_THRESHOLD = 0.65
    SIGNAL_CONFIDENCE_THRESHOLD_HFT = 0.55
    SIGNAL_STRENGTH_MULTIPLIER = 1.5

    # Trading Hours
    TRADING_START_HOUR = 0
    TRADING_END_HOUR = 23
    ENABLE_24_7_TRADING = True

    # Files
    LOG_FILE = "trading_log.txt"
    TRADE_LOG_FILE = "trade_log.csv"

config = Config()

class TradingBot:
    def __init__(self):
        # Thread-safe GUI update queue
        self.gui_queue = queue.Queue()

        # Initialize basic variables
        self.running = False
        self.connected = False
        self.bot_thread = None
        self.gui_update_thread = None

        # State variables
        self.order_counter = 0
        self.total_opportunities_captured = 0
        self.total_opportunities_missed = 0
        self.session_profit = 0.0
        self.session_trades = 0
        self.last_reset_date = datetime.date.today()

        # Initialize components
        self.market_api = MarketDataAPI()
        self.indicators = EnhancedIndicators()

        # Platform detection
        self.is_windows = platform.system() == "Windows"

        # GUI components (initialized in setup_gui)
        self.root = None
        self.log_box = None
        self.perf_box = None

        try:
            self.setup_gui()
            self.start_gui_update_thread()

            # Auto-connect MT5 if Windows
            if MT5_AVAILABLE and self.is_windows:
                threading.Thread(target=self.auto_connect_mt5, daemon=True).start()

        except Exception as e:
            print(f"Initialization error: {e}")
            traceback.print_exc()

    def setup_gui(self):
        """Setup enhanced GUI interface"""
        self.root = tk.Tk()
        self.root.title("🚀 AuraTrade Bot - 24/7 Auto Trading System")
        self.root.geometry("1600x1000")
        self.root.configure(bg="#1e1e1e")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Theme colors
        bg_color = "#1e1e1e"
        fg_color = "#ffffff"
        accent_color = "#00ff88"

        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background=bg_color, foreground=fg_color, font=("Consolas", 10))
        style.configure("TButton", font=("Consolas", 10, "bold"))
        style.configure("TFrame", background=bg_color)
        style.configure("TLabelFrame", background=bg_color, foreground=fg_color, font=("Consolas", 12, "bold"))

        # GUI Variables
        self.symbol_var = tk.StringVar(value=config.DEFAULT_SYMBOL)
        self.lot_var = tk.StringVar(value=str(config.DEFAULT_LOT))
        self.interval_var = tk.StringVar(value=str(config.DEFAULT_INTERVAL))
        self.tp_balance_var = tk.StringVar(value=str(config.TP_PERSEN_BALANCE * 100))
        self.sl_balance_var = tk.StringVar(value=str(config.SL_PERSEN_BALANCE * 100))
        self.account_info_var = tk.StringVar(value="🔌 Not Connected")
        self.profit_var = tk.StringVar(value="💰 P/L: $0.00")
        self.opportunities_var = tk.StringVar(value="🎯 Opportunities: 0 Captured | 0 Missed")
        self.status_var = tk.StringVar(value="⏸️ Stopped")
        self.hft_mode_var = tk.BooleanVar(value=False)
        self.scalping_mode_var = tk.BooleanVar(value=config.SCALPING_OVERRIDE_ENABLED)
        self.auto_tp_sl_var = tk.BooleanVar(value=True)
        self.trading_24_7_var = tk.BooleanVar(value=config.ENABLE_24_7_TRADING)

        self.create_gui_components()

        # Initial messages
        self.safe_log("🚀 AuraTrade Bot Initialized", "SUCCESS")
        self.safe_log("💡 Advanced Features: HFT • Scalping • Auto TP/SL • 24/7 Trading", "INFO")
        self.safe_log(f"🖥️ Platform: {platform.system()}", "INFO")

        if self.is_windows and MT5_AVAILABLE:
            self.safe_log("✅ MT5 Library Available", "SUCCESS")
        else:
            self.safe_log("⚠️ Using Simulation Mode", "WARNING")

    def create_gui_components(self):
        """Create all GUI components"""

        # Header Frame
        header_frame = ttk.LabelFrame(self.root, text="🤖 AuraTrade - Professional Trading Bot", padding=15)
        header_frame.pack(fill="x", padx=10, pady=5)

        # Status indicators
        status_frame = ttk.Frame(header_frame)
        status_frame.pack(fill="x")

        ttk.Label(status_frame, textvariable=self.status_var, font=("Consolas", 12, "bold")).pack(side="left", padx=10)
        ttk.Label(status_frame, textvariable=self.account_info_var).pack(side="left", padx=10)
        ttk.Label(status_frame, textvariable=self.profit_var, foreground="#00ff88").pack(side="right", padx=10)

        # Opportunities display
        ttk.Label(header_frame, textvariable=self.opportunities_var, 
                 font=("Consolas", 10, "bold"), foreground="#ffaa00").pack(pady=5)

        # Settings Notebook
        settings_frame = ttk.LabelFrame(self.root, text="⚙️ Trading Configuration", padding=10)
        settings_frame.pack(padx=10, pady=5, fill="x")

        notebook = ttk.Notebook(settings_frame)
        notebook.pack(fill="x", expand=True)

        # Basic Settings Tab
        basic_tab = ttk.Frame(notebook)
        notebook.add(basic_tab, text="Basic Settings")

        basic_left = ttk.Frame(basic_tab)
        basic_left.pack(side="left", fill="y", padx=10)

        # Symbol setting
        symbol_frame = ttk.Frame(basic_left)
        symbol_frame.pack(fill="x", pady=5)
        ttk.Label(symbol_frame, text="Symbol:").pack(side="left", padx=(0,5))
        ttk.Entry(symbol_frame, textvariable=self.symbol_var, width=15).pack(side="right")

        # Lot size setting
        lot_frame = ttk.Frame(basic_left)
        lot_frame.pack(fill="x", pady=5)
        ttk.Label(lot_frame, text="Lot Size:").pack(side="left", padx=(0,5))
        ttk.Entry(lot_frame, textvariable=self.lot_var, width=15).pack(side="right")

        # Interval setting
        interval_frame = ttk.Frame(basic_left)
        interval_frame.pack(fill="x", pady=5)
        ttk.Label(interval_frame, text="Scan Interval (s):").pack(side="left", padx=(0,5))
        ttk.Entry(interval_frame, textvariable=self.interval_var, width=15).pack(side="right")

        # TP/SL Settings based on balance percentage
        tpsl_frame = ttk.LabelFrame(basic_tab, text="💰 Auto TP/SL (% of Balance)", padding=10)
        tpsl_frame.pack(side="right", fill="both", expand=True, padx=10)

        ttk.Checkbutton(tpsl_frame, text="🎯 Enable Auto TP/SL", 
                       variable=self.auto_tp_sl_var).pack(anchor="w", pady=5)

        # Create sub-frames for proper layout using pack only
        tp_frame = ttk.Frame(tpsl_frame)
        tp_frame.pack(fill="x", pady=3)
        ttk.Label(tp_frame, text="Take Profit (%):").pack(side="left")
        ttk.Entry(tp_frame, textvariable=self.tp_balance_var, width=10).pack(side="right")

        sl_frame = ttk.Frame(tpsl_frame)
        sl_frame.pack(fill="x", pady=3)
        ttk.Label(sl_frame, text="Stop Loss (%):").pack(side="left")
        ttk.Entry(sl_frame, textvariable=self.sl_balance_var, width=10).pack(side="right")

        # Advanced Settings Tab
        advanced_tab = ttk.Frame(notebook)
        notebook.add(advanced_tab, text="Advanced Settings")

        # HFT Settings
        hft_frame = ttk.LabelFrame(advanced_tab, text="⚡ High-Frequency Trading", padding=10)
        hft_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(hft_frame, text="⚡ Enable HFT Mode (Ultra-Fast Trading)", 
                       variable=self.hft_mode_var, command=self.toggle_hft_mode).pack(anchor="w")

        # Scalping Settings
        scalping_frame = ttk.LabelFrame(advanced_tab, text="🔥 Scalping Mode", padding=10)
        scalping_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(scalping_frame, text="🔥 Enable Scalping Mode", 
                       variable=self.scalping_mode_var).pack(anchor="w")

        # 24/7 Trading
        always_frame = ttk.LabelFrame(advanced_tab, text="🌐 24/7 Trading", padding=10)
        always_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(always_frame, text="🌐 Enable 24/7 Non-Stop Trading", 
                       variable=self.trading_24_7_var).pack(anchor="w")

        # Control Buttons
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=15)

        # Primary buttons
        primary_frame = ttk.Frame(control_frame)
        primary_frame.pack()

        self.connect_button = ttk.Button(primary_frame, text="🔗 Connect MT5", 
                                       command=self.connect_mt5, width=15)
        self.start_button = ttk.Button(primary_frame, text="🚀 Start Auto Trading", 
                                     command=self.start_bot, width=20)
        self.stop_button = ttk.Button(primary_frame, text="🛑 Stop Trading", 
                                    command=self.stop_bot, state="disabled", width=15)

        self.connect_button.pack(side="left", padx=5)
        self.start_button.pack(side="left", padx=5)
        self.stop_button.pack(side="left", padx=5)

        # Secondary buttons
        secondary_frame = ttk.Frame(control_frame)
        secondary_frame.pack(pady=10)

        ttk.Button(secondary_frame, text="❌ Close All Positions", 
                  command=self.close_all_positions, width=20).pack(side="left", padx=5)
        ttk.Button(secondary_frame, text="🔄 Reset Counters", 
                  command=self.reset_counters, width=15).pack(side="left", padx=5)
        ttk.Button(secondary_frame, text="💾 Save Settings", 
                  command=self.save_settings, width=15).pack(side="left", padx=5)

        # Log Notebook
        log_frame = ttk.LabelFrame(self.root, text="📊 Trading Monitor", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        log_notebook = ttk.Notebook(log_frame)
        log_notebook.pack(fill="both", expand=True)

        # Trading Log
        log_tab = ttk.Frame(log_notebook)
        log_notebook.add(log_tab, text="📈 Trading Log")

        self.log_box = ScrolledText(log_tab, width=120, height=25, 
                                   bg="#0a0a0a", fg="#00ff88", 
                                   font=("Consolas", 9), insertbackground="#00ff88")
        self.log_box.pack(fill="both", expand=True)

        # Performance Log
        perf_tab = ttk.Frame(log_notebook)
        log_notebook.add(perf_tab, text="📊 Performance")

        self.perf_box = ScrolledText(perf_tab, width=120, height=25,
                                    bg="#0a0a0a", fg="#ffaa00",
                                    font=("Consolas", 9), insertbackground="#ffaa00")
        self.perf_box.pack(fill="both", expand=True)

    def start_gui_update_thread(self):
        """Start thread-safe GUI update system"""
        def update_gui():
            while True:
                try:
                    # Process GUI updates from queue
                    while not self.gui_queue.empty():
                        update_func = self.gui_queue.get_nowait()
                        update_func()
                    time.sleep(0.1)
                except Exception as e:
                    print(f"GUI update error: {e}")
                    time.sleep(1)

        self.gui_update_thread = threading.Thread(target=update_gui, daemon=True)
        self.gui_update_thread.start()

    def safe_log(self, message, log_type="INFO"):
        """Thread-safe logging"""
        def update_log():
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")

            # Color and icon based on type
            if log_type == "SUCCESS":
                icon = "✅"
                color = "#00ff88"
            elif log_type == "ERROR":
                icon = "❌"
                color = "#ff4444"
            elif log_type == "WARNING":
                icon = "⚠️"
                color = "#ffaa00"
            elif log_type == "TRADE":
                icon = "💰"
                color = "#00aaff"
            else:
                icon = "ℹ️"
                color = "#ffffff"

            log_entry = f"{icon} {timestamp} - {message}\n"

            try:
                if self.log_box and hasattr(self.log_box, 'winfo_exists') and self.log_box.winfo_exists():
                    self.log_box.insert(tk.END, log_entry)
                    self.log_box.see(tk.END)

                    # Color the last line
                    try:
                        self.log_box.tag_add(log_type, "end-2l", "end-1l")
                        self.log_box.tag_config(log_type, foreground=color)
                    except:
                        pass  # Skip coloring if there's an issue
                else:
                    print(f"{log_entry.strip()}")
            except Exception as e:
                print(f"{log_entry.strip()}")

        # Add to queue for thread-safe update
        self.gui_queue.put(update_log)

        # Also log to file
        try:
            with open(config.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now()} - {message}\n")
        except:
            pass

    def safe_performance_log(self, message):
        """Thread-safe performance logging"""
        def update_perf():
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            perf_entry = f"📊 {timestamp} - {message}\n"

            try:
                if self.perf_box and hasattr(self.perf_box, 'winfo_exists') and self.perf_box.winfo_exists():
                    self.perf_box.insert(tk.END, perf_entry)
                    self.perf_box.see(tk.END)
                else:
                    print(f"PERF: {perf_entry.strip()}")
            except Exception as e:
                print(f"PERF: {perf_entry.strip()}")

        self.gui_queue.put(update_perf)

    def update_status_display(self):
        """Update status displays"""
        def update():
            try:
                # Update status
                if self.running:
                    mode_text = ""
                    if self.hft_mode_var.get():
                        mode_text = " (HFT)"
                    elif self.scalping_mode_var.get():
                        mode_text = " (Scalping)"

                    self.status_var.set(f"🚀 Running{mode_text}")
                else:
                    self.status_var.set("⏸️ Stopped")

                # Update opportunities
                success_rate = self.calculate_success_rate()
                self.opportunities_var.set(
                    f"🎯 Opportunities: {self.total_opportunities_captured} Captured | "
                    f"{self.total_opportunities_missed} Missed | "
                    f"Success: {success_rate:.1f}%"
                )

                # Update profit
                self.profit_var.set(f"💰 Session P/L: ${self.session_profit:.2f} ({self.session_trades} trades)")

            except:
                pass

        self.gui_queue.put(update)

    def calculate_success_rate(self):
        """Calculate success rate"""
        total = self.total_opportunities_captured + self.total_opportunities_missed
        if total == 0:
            return 0.0
        return (self.total_opportunities_captured / total) * 100

    def toggle_hft_mode(self):
        """Toggle HFT mode"""
        if self.hft_mode_var.get():
            self.interval_var.set("1")
            self.safe_log("⚡ HFT MODE ACTIVATED! Ultra-fast trading enabled", "SUCCESS")
        else:
            self.interval_var.set("5")
            self.safe_log("🔄 HFT Mode disabled", "INFO")

    def auto_connect_mt5(self):
        """Auto-connect to MT5"""
        if not self.is_windows or not MT5_AVAILABLE:
            return

        try:
            self.safe_log("🔄 Auto-connecting to MT5...", "INFO")

            if not mt5.initialize():
                error_code, error_msg = mt5.last_error()
                self.safe_log(f"❌ MT5 auto-connect failed: {error_code} - {error_msg}", "ERROR")
                return

            account_info = mt5.account_info()
            if account_info is None:
                self.safe_log("❌ Failed to get MT5 account info", "ERROR")
                return

            self.connected = True
            self.safe_log("✅ MT5 Connected Successfully!", "SUCCESS")
            self.safe_log(f"Account: {account_info.login} | Balance: ${account_info.balance:,.2f}", "INFO")

            # Update GUI
            def update_gui():
                self.account_info_var.set(f"🔗 MT5: {account_info.login} | ${account_info.balance:,.2f}")
                if hasattr(self, 'connect_button'):
                    self.connect_button.configure(state="disabled")

            self.gui_queue.put(update_gui)

        except Exception as e:
            self.safe_log(f"❌ MT5 auto-connect error: {e}", "ERROR")

    def connect_mt5(self):
        """Manual MT5 connection"""
        if not self.is_windows or not MT5_AVAILABLE:
            messagebox.showerror("Error", "MT5 is only available on Windows platform")
            return

        threading.Thread(target=self.auto_connect_mt5, daemon=True).start()

    def start_bot(self):
        """Start the trading bot"""
        if self.running:
            messagebox.showwarning("Warning", "Bot is already running!")
            return

        try:
            self.running = True
            self.order_counter = 0
            self.session_profit = 0.0
            self.session_trades = 0
            self.total_opportunities_captured = 0
            self.total_opportunities_missed = 0

            # Update GUI
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")

            # Start trading thread
            self.bot_thread = threading.Thread(target=self.enhanced_trading_loop, daemon=True)
            self.bot_thread.start()

            # Log startup info
            symbol = self.symbol_var.get()
            lot = self.lot_var.get()
            interval = self.interval_var.get()

            self.safe_log("🚀 AURA TRADE BOT STARTED!", "SUCCESS")
            self.safe_log(f"Symbol: {symbol} | Lot: {lot} | Interval: {interval}s", "INFO")

            if self.hft_mode_var.get():
                self.safe_log("⚡ HFT Mode: ON - Ultra-fast trading enabled", "SUCCESS")
            if self.scalping_mode_var.get():
                self.safe_log("🔥 Scalping Mode: ON", "SUCCESS")  
            if self.trading_24_7_var.get():
                self.safe_log("🌐 24/7 Trading: ON", "SUCCESS")
            if self.auto_tp_sl_var.get():
                self.safe_log("🎯 Auto TP/SL: ON", "SUCCESS")

        except Exception as e:
            self.safe_log(f"❌ Failed to start bot: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to start bot: {e}")

    def stop_bot(self):
        """Stop the trading bot"""
        try:
            self.running = False

            # Update GUI
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

            # Wait for thread to finish
            if self.bot_thread and self.bot_thread.is_alive():
                self.bot_thread.join(timeout=3)

            # Final statistics
            success_rate = self.calculate_success_rate()

            self.safe_log("🛑 TRADING BOT STOPPED!", "WARNING")
            self.safe_log("📊 Session Summary:", "INFO")
            self.safe_log(f"   Total Trades: {self.session_trades}", "INFO")
            self.safe_log(f"   Session P/L: ${self.session_profit:.2f}", "INFO")
            self.safe_log(f"   Opportunities Captured: {self.total_opportunities_captured}", "INFO")
            self.safe_log(f"   Success Rate: {success_rate:.1f}%", "INFO")

        except Exception as e:
            self.safe_log(f"❌ Error stopping bot: {e}", "ERROR")

    def enhanced_trading_loop(self):
        """Enhanced main trading loop with improved signal processing"""
        try:
            self.safe_log("🔄 Enhanced trading loop started", "INFO")

            while self.running:
                try:
                    # Check trading hours
                    if not self.is_trading_hours():
                        time.sleep(60)
                        continue

                    # Get trading parameters
                    symbol = self.symbol_var.get()
                    scan_interval = max(int(self.interval_var.get()), 1)

                    # Market analysis
                    self.safe_log(f"📊 Analyzing {symbol}...", "INFO")

                    # Get market data with retry logic
                    prices = self.market_api.get_recent_prices(symbol, count=100)
                    if not prices or len(prices) < 20:
                        self.safe_log("⚠️ Insufficient market data, retrying...", "WARNING")
                        time.sleep(min(scan_interval, 10))
                        continue

                    # Enhanced signal analysis
                    signal_data = self.indicators.enhanced_signal_analysis(prices)

                    signal = signal_data.get('signal', 'HOLD')
                    confidence = signal_data.get('confidence', 0.0)
                    strength = signal_data.get('strength', 0.0)

                    # Determine trading thresholds based on mode
                    if self.hft_mode_var.get():
                        confidence_threshold = 0.55
                        strength_threshold = 0.4
                        max_orders = 200
                    elif self.scalping_mode_var.get():
                        confidence_threshold = 0.60
                        strength_threshold = 0.5
                        max_orders = 100
                    else:
                        confidence_threshold = 0.65
                        strength_threshold = 0.6
                        max_orders = 50

                    # Check if we can trade
                    if self.session_trades >= max_orders:
                        self.safe_log(f"📊 Maximum trades reached ({self.session_trades}/{max_orders})", "INFO")
                        time.sleep(scan_interval)
                        continue

                    # Process signal
                    if signal in ['BUY', 'SELL'] and confidence >= confidence_threshold and strength >= strength_threshold:
                        # Calculate TP/SL
                        current_price = prices[-1] if prices else 0
                        tp, sl = self.calculate_unified_tp_sl(signal, current_price)

                        if tp > 0 and sl > 0:
                            # Execute trade
                            success = self.execute_enhanced_trade(symbol, {
                                'signal': signal,
                                'confidence': confidence,
                                'strength': strength,
                                'price': current_price,
                                'tp': tp,
                                'sl': sl
                            })

                            if success:
                                self.total_opportunities_captured += 1
                                self.session_trades += 1
                                self.safe_log(f"🎯 Trade executed: {signal} {symbol}", "TRADE")
                                self.safe_log(f"   Confidence: {confidence:.3f}, Strength: {strength:.3f}", "INFO")

                                # Log reasons if available
                                reasons = signal_data.get('reasons', [])
                                if reasons:
                                    self.safe_log(f"   Reasons: {', '.join(reasons[:2])}", "INFO")

                            else:
                                self.total_opportunities_missed += 1
                                self.safe_log("❌ Trade execution failed", "ERROR")
                        else:
                            self.safe_log("❌ Invalid TP/SL calculation", "ERROR")
                            self.total_opportunities_missed += 1

                    elif signal in ['BUY', 'SELL']:
                        self.safe_log(f"📈 Signal {signal} - Conf:{confidence:.3f}, Str:{strength:.3f} (Below threshold)", "INFO")
                        self.total_opportunities_missed += 1
                    else:
                        self.safe_log("📊 Market analysis: No trading signal", "INFO")

                    # Update displays
                    self.update_status_display()

                    # Dynamic sleep based on market activity
                    if signal in ['BUY', 'SELL']:
                        sleep_time = max(scan_interval // 2, 2)  # Faster after signal
                    else:
                        sleep_time = scan_interval

                    # Sleep with interruption check
                    for _ in range(sleep_time):
                        if not self.running:
                            break
                        time.sleep(1)

                except Exception as e:
                    if self.running:
                        self.safe_log(f"❌ Trading loop error: {e}", "ERROR")
                    time.sleep(min(scan_interval * 2, 30))  # Wait longer on error

        except Exception as e:
            self.safe_log(f"❌ Trading loop fatal error: {e}", "ERROR")
        finally:
            self.safe_log("🛑 Enhanced trading loop stopped", "INFO")

    def is_trading_hours(self):
        """Check if within trading hours"""
        if self.trading_24_7_var.get():
            return True

        current_hour = datetime.datetime.now().hour
        return config.TRADING_START_HOUR <= current_hour <= config.TRADING_END_HOUR

    def calculate_unified_tp_sl(self, signal, current_price):
        """Calculate TP/SL based on current config and market conditions."""
        if not current_price or current_price == 0:
            return 0, 0

        try:
            if MT5_AVAILABLE and self.connected:
                account_info = mt5.account_info()
                if not account_info:
                    return 0, 0
                balance = account_info.balance
            else:
                balance = 10000  # Simulation balance

            if self.auto_tp_sl_var.get():
                if self.hft_mode_var.get():
                    tp_percent = config.HFT_TP_PERSEN_BALANCE
                    sl_percent = config.HFT_SL_PERSEN_BALANCE
                elif self.scalping_mode_var.get():
                    tp_percent = config.SCALPING_TP_PERSEN_BALANCE
                    sl_percent = config.SCALPING_SL_PERSEN_BALANCE
                else:
                    tp_percent = float(self.tp_balance_var.get()) / 100
                    sl_percent = float(self.sl_balance_var.get()) / 100
            else:
                tp_percent = 0.01  # Default TP 1%
                sl_percent = 0.005 # Default SL 0.5%

            # Get current lot size
            try:
                lot_size = float(self.lot_var.get())
                if lot_size <= 0: lot_size = config.DEFAULT_LOT
            except ValueError:
                lot_size = config.DEFAULT_LOT
                self.safe_log("⚠️ Invalid lot size, using default.", "WARNING")

            # Calculate pip value
            # This is a simplification and ideally should be fetched from the broker for accuracy
            # For simulation and general use, we estimate based on common pairs
            pip_value = current_price * 0.0001  # For pairs like EURUSD

            # Calculate TP/SL in price points
            tp_points = (balance * tp_percent) / (lot_size * 10000) # Assuming 100k lot size
            sl_points = (balance * sl_percent) / (lot_size * 10000)

            if signal == 'BUY':
                tp_price = current_price + tp_points
                sl_price = current_price - sl_points
            else:  # SELL
                tp_price = current_price - tp_points
                sl_price = current_price + sl_points
                
            return tp_price, sl_price

        except Exception as e:
            self.safe_log(f"❌ Error calculating TP/SL: {e}", "ERROR")
            return 0, 0


    def execute_enhanced_trade(self, symbol, signal_data):
        """Execute trade with enhanced features"""
        try:
            if not self.connected and MT5_AVAILABLE:
                self.safe_log("❌ MT5 not connected", "ERROR")
                return False

            # Get account balance
            if MT5_AVAILABLE and self.connected:
                account_info = mt5.account_info()
                if not account_info:
                    return False
                balance = account_info.balance
            else:
                balance = 10000  # Simulation balance

            # Calculate position size
            lot_size = float(self.lot_var.get())

            # Get TP/SL prices from signal_data which is already calculated
            tp_price = signal_data.get('tp', 0)
            sl_price = signal_data.get('sl', 0)

            # Get current price for execution
            price_data = self.market_api.get_price(symbol)
            if not price_data:
                return False
            
            if signal_data['signal'] == 'BUY':
                order_type = mt5.ORDER_TYPE_BUY if MT5_AVAILABLE else 0
                execution_price = price_data['ask']
            else:  # SELL
                order_type = mt5.ORDER_TYPE_SELL if MT5_AVAILABLE else 1
                execution_price = price_data['bid']

            # Execute order
            if MT5_AVAILABLE and self.connected:
                # Real MT5 trading
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": lot_size,
                    "type": order_type,
                    "price": execution_price,
                    "sl": sl_price,
                    "tp": tp_price,
                    "deviation": 10,
                    "magic": 123456,
                    "comment": f"AuraTrade-{signal_data['signal'][:3]}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }

                result = mt5.order_send(request)

                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self.order_counter += 1
                    self.session_trades += 1

                    # Log trade details
                    self.safe_log(f"✅ Order executed successfully!", "SUCCESS")
                    self.safe_log(f"   Ticket: {result.deal}", "INFO")
                    self.safe_log(f"   Type: {signal_data['signal']}", "INFO")
                    self.safe_log(f"   Price: {execution_price:.5f}", "INFO")
                    self.safe_log(f"   TP: {tp_price:.5f} | SL: {sl_price:.5f}", "INFO")
                    self.safe_log(f"   Lot Size: {lot_size}", "INFO")

                    return True
                else:
                    error_msg = result.comment if result else "Unknown error"
                    self.safe_log(f"❌ Order failed: {error_msg}", "ERROR")
                    return False
            else:
                # Simulation mode
                self.order_counter += 1
                self.session_trades += 1

                # Simulate trade result
                import random
                is_profitable = random.random() < 0.65  # 65% win rate

                if is_profitable:
                    profit = (tp_price - execution_price) * lot_size * 100000 if signal_data['signal'] == 'BUY' else (execution_price - tp_price) * lot_size * 100000
                    self.session_profit += profit
                    self.safe_log(f"✅ Simulated trade PROFIT: ${profit:.2f}", "SUCCESS")
                else:
                    loss = (execution_price - sl_price) * lot_size * 100000 if signal_data['signal'] == 'BUY' else (sl_price - execution_price) * lot_size * 100000
                    self.session_profit += loss
                    self.safe_log(f"❌ Simulated trade LOSS: ${loss:.2f}", "ERROR")

                self.safe_log(f"📊 Simulation trade executed:", "TRADE")
                self.safe_log(f"   Type: {signal_data['signal']}", "INFO")
                self.safe_log(f"   Price: {execution_price:.5f}", "INFO")
                self.safe_log(f"   TP Target: {tp_price:.5f} | SL Risk: {sl_price:.5f}", "INFO")

                return True

        except Exception as e:
            self.safe_log(f"❌ Trade execution error: {e}", "ERROR")
            return False

    def close_all_positions(self):
        """Close all open positions"""
        try:
            if not messagebox.askyesno("Confirm", "Close all open positions?"):
                return

            if not self.connected or not MT5_AVAILABLE:
                self.safe_log("❌ MT5 not connected", "ERROR")
                return

            positions = mt5.positions_get()
            if not positions:
                self.safe_log("ℹ️ No open positions to close", "INFO")
                return

            closed_count = 0
            for position in positions:
                try:
                    # Get current price
                    tick = mt5.symbol_info_tick(position.symbol)
                    if not tick:
                        continue

                    # Determine close order type and price
                    if position.type == mt5.POSITION_TYPE_BUY:
                        close_type = mt5.ORDER_TYPE_SELL
                        close_price = tick.bid
                    else:
                        close_type = mt5.ORDER_TYPE_BUY
                        close_price = tick.ask

                    # Close request
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": position.symbol,
                        "volume": position.volume,
                        "type": close_type,
                        "position": position.ticket,
                        "price": close_price,
                        "deviation": 10,
                        "magic": 123456,
                        "comment": "AuraTrade Close All",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }

                    # Execute close
                    result = mt5.order_send(close_request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        closed_count += 1
                        self.safe_log(f"✅ Closed position: {position.symbol} #{position.ticket}", "SUCCESS")
                    else:
                        self.safe_log(f"❌ Failed to close position #{position.ticket}", "ERROR")

                except Exception as e:
                    self.safe_log(f"❌ Error closing position: {e}", "ERROR")

            self.safe_log(f"🔄 Closed {closed_count} out of {len(positions)} positions", "INFO")

        except Exception as e:
            self.safe_log(f"❌ Close all error: {e}", "ERROR")

    def reset_counters(self):
        """Reset all counters"""
        self.order_counter = 0
        self.total_opportunities_captured = 0
        self.total_opportunities_missed = 0 
        self.session_profit = 0.0
        self.session_trades = 0
        self.update_status_display()
        self.safe_log("🔄 All counters reset", "INFO")

    def save_settings(self):
        """Save current settings to file"""
        try:
            settings = {
                'symbol': self.symbol_var.get(),
                'lot_size': self.lot_var.get(),
                'interval': self.interval_var.get(),
                'tp_balance': self.tp_balance_var.get(),
                'sl_balance': self.sl_balance_var.get(),
                'hft_mode': self.hft_mode_var.get(),
                'scalping_mode': self.scalping_mode_var.get(),
                'auto_tp_sl': self.auto_tp_sl_var.get(),
                'trading_24_7': self.trading_24_7_var.get()
            }

            with open('trading_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)

            self.safe_log("💾 Settings saved successfully", "SUCCESS")

        except Exception as e:
            self.safe_log(f"❌ Failed to save settings: {e}", "ERROR")

    def on_closing(self):
        """Enhanced application closing"""
        try:
            if self.running:
                if messagebox.askyesno("Confirm Exit", "Trading bot is running. Stop and exit?"):
                    self.running = False
                    time.sleep(2)
                else:
                    return

            # Shutdown MT5 connection
            if self.connected and MT5_AVAILABLE:
                try:
                    mt5.shutdown()
                    self.safe_log("🔌 MT5 connection closed", "INFO")
                except:
                    pass

            # Close GUI
            try:
                self.root.quit()
                self.root.destroy()  
            except:
                pass

        except Exception as e:
            print(f"Closing error: {e}")

def main():
    """Main function"""
    try:
        print("🚀 Starting AuraTrade Bot...")
        app = TradingBot()
        app.root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()