#!/usr/bin/env python3
"""
Trading Bot Terintegrasi - Auto Trading & Auto Scalping
Bot trading otomatis dengan indikator teknikal, manajemen risiko, dan GUI
"""

import time
import datetime
import numpy as np
import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.scrolledtext import ScrolledText
import tkinter.messagebox as messagebox
import requests
import csv
import os
import json

# Import modul simulasi
from market_data_api import MarketDataAPI
from simulation_trading import SimulationTrading

# Konfigurasi default
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "CHAT_ID")

TP_PERSEN_DEFAULT = 0.01
SL_PERSEN_DEFAULT = 0.05
SCALPING_TP_PERSEN = 0.005
SCALPING_SL_PERSEN = 0.01
SCALPING_OVERRIDE_ENABLED = True

MAX_ORDER_PER_SESSION = 10
SALDO_MINIMAL = 500000
TARGET_PROFIT_PERSEN = 10
LONJAKAN_THRESHOLD = 10
RESET_ORDER_HOUR = 0
SKOR_MINIMAL = 4
TRADING_START_HOUR = 7
TRADING_END_HOUR = 21
TRAILING_STOP_PIPS = 50

class TradingBot:
    def __init__(self):
        # Initialize market data and trading simulation
        self.market_api = MarketDataAPI()
        self.mt5 = SimulationTrading(self.market_api)
        
        # Bot state
        self.running = False
        self.modal_awal = None
        self.last_price = None
        self.last_reset_date = datetime.date.today()
        self.order_counter = 0
        
        # Threading
        self.bot_thread = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup GUI interface"""
        self.root = tk.Tk()
        self.root.title("Trading Bot Aman Cuan - Auto Trading & Scalping")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables
        self.symbol_var = tk.StringVar(value="XAUUSDm")
        self.lot_var = tk.StringVar(value="0.01")
        self.interval_var = tk.StringVar(value="10")
        self.tp_var = tk.StringVar(value=str(TP_PERSEN_DEFAULT * 100))
        self.sl_var = tk.StringVar(value=str(SL_PERSEN_DEFAULT * 100))
        self.scalping_tp_var = tk.StringVar(value=str(SCALPING_TP_PERSEN * 100))
        self.scalping_sl_var = tk.StringVar(value=str(SCALPING_SL_PERSEN * 100))
        self.account_info_var = tk.StringVar(value="Account: Not Connected")
        self.profit_var = tk.StringVar(value="Real-time P/L: -")
        self.scalping_mode_var = tk.BooleanVar(value=SCALPING_OVERRIDE_ENABLED)
        
        self.create_gui_elements()
    
    def create_gui_elements(self):
        """Create GUI elements"""
        # Style
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        
        # Info frame
        info_frame = ttk.LabelFrame(self.root, text="Info Akun", padding=10)
        info_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(info_frame, textvariable=self.account_info_var).grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.profit_var, foreground="green").grid(row=0, column=1, sticky="e")
        
        # Settings frame
        setting_frame = ttk.LabelFrame(self.root, text="Pengaturan Trading", padding=10)
        setting_frame.pack(padx=10, pady=5, fill="x")
        
        # Left settings
        left_frame = ttk.Frame(setting_frame)
        left_frame.grid(row=0, column=0, padx=10, sticky="n")
        
        fields_left = [
            ("Symbol:", self.symbol_var),
            ("Lot:", self.lot_var),
            ("Interval (s):", self.interval_var),
            ("TP (%):", self.tp_var),
            ("SL (%):", self.sl_var)
        ]
        
        for i, (label, var) in enumerate(fields_left):
            ttk.Label(left_frame, text=label).grid(row=i, column=0, sticky="e", pady=5)
            ttk.Entry(left_frame, textvariable=var, width=20).grid(row=i, column=1, pady=5)
        
        # Right settings
        right_frame = ttk.Frame(setting_frame)
        right_frame.grid(row=0, column=1, padx=10, sticky="n")
        
        fields_right = [
            ("Scalp TP (%):", self.scalping_tp_var),
            ("Scalp SL (%):", self.scalping_sl_var)
        ]
        
        for i, (label, var) in enumerate(fields_right):
            ttk.Label(right_frame, text=label).grid(row=i, column=0, sticky="e", pady=5)
            ttk.Entry(right_frame, textvariable=var, width=20).grid(row=i, column=1, pady=5)
        
        # Scalping mode checkbox
        ttk.Checkbutton(right_frame, text="Enable Scalping Mode", 
                       variable=self.scalping_mode_var).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Buttons frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.connect_button = ttk.Button(button_frame, text="🔗 Connect", command=self.connect_mt5)
        self.start_button = ttk.Button(button_frame, text="🚀 Start", command=self.start_bot)
        self.stop_button = ttk.Button(button_frame, text="🛑 Stop", command=self.stop_bot, state="disabled")
        self.close_button = ttk.Button(button_frame, text="❌ Close All", command=self.manual_close_all)
        
        self.connect_button.grid(row=0, column=0, padx=10)
        self.start_button.grid(row=0, column=1, padx=10)
        self.stop_button.grid(row=0, column=2, padx=10)
        self.close_button.grid(row=0, column=3, padx=10)
        
        # Log frame
        log_frame = ttk.LabelFrame(self.root, text="Trading Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_box = ScrolledText(log_frame, width=120, height=25, 
                                   bg="#ffffff", fg="#333333", font=("Consolas", 9))
        self.log_box.pack(fill="both", expand=True)
    
    # ======================
    # LOGGING & TELEGRAM
    # ======================
    def log_to_file(self, text):
        """Log to file"""
        try:
            with open("log_trading.txt", "a", encoding="utf-8") as f:
                f.write(f"{datetime.datetime.now()} - {text}\n")
        except Exception as e:
            print(f"Error logging to file: {e}")

    def log(self, text):
        """Add log entry"""
        timestamp = f"{datetime.datetime.now():%H:%M:%S}"
        log_entry = f"{timestamp} - {text}"
        
        self.log_box.insert(tk.END, log_entry + "\n")
        self.log_box.see(tk.END)
        self.log_to_file(text)

    def send_telegram(self, text):
        """Send Telegram notification"""
        try:
            if TELEGRAM_BOT_TOKEN != "TOKEN" and TELEGRAM_CHAT_ID != "CHAT_ID":
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                data = {"chat_id": TELEGRAM_CHAT_ID, "text": text}
                requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"Telegram error: {e}")

    def export_trade_log(self, harga_order, tp, sl, sinyal):
        """Export trade to CSV"""
        try:
            filename = "trade_log.csv"
            fieldnames = ["timestamp", "signal", "price", "tp", "sl"]
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
                    "sl": round(sl, 5)
                })
        except Exception as e:
            self.log(f"Error exporting trade log: {e}")
    
    # ======================
    # CONNECTION & SETUP
    # ======================
    def connect_mt5(self):
        """Connect to market data simulation"""
        try:
            self.log("🔗 Connecting to Trading Simulation...")
            self.log("⚠️  NOTE: This is SIMULATION mode - No real money at risk!")
            
            # Show dialog to set virtual account balance
            balance_dialog = tk.Toplevel(self.root)
            balance_dialog.title("Setup Virtual Account")
            balance_dialog.geometry("400x200")
            balance_dialog.transient(self.root)
            balance_dialog.grab_set()
            
            # Center the dialog
            balance_dialog.update_idletasks()
            x = (balance_dialog.winfo_screenwidth() // 2) - (400 // 2)
            y = (balance_dialog.winfo_screenheight() // 2) - (200 // 2)
            balance_dialog.geometry(f"400x200+{x}+{y}")
            
            ttk.Label(balance_dialog, text="🤖 TRADING SIMULATION SETUP", 
                     font=("Segoe UI", 12, "bold")).pack(pady=10)
            ttk.Label(balance_dialog, text="Set your virtual account balance:", 
                     font=("Segoe UI", 10)).pack(pady=5)
            
            balance_var = tk.StringVar(value="10000")
            balance_entry = ttk.Entry(balance_dialog, textvariable=balance_var, 
                                    font=("Segoe UI", 12), width=20)
            balance_entry.pack(pady=10)
            
            result = {"balance": None}
            
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
            ttk.Button(button_frame, text="Connect", command=on_connect).pack(side="left", padx=10)
            ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="left", padx=10)
            
            # Wait for dialog to close
            self.root.wait_window(balance_dialog)
            
            if result["balance"] is None:
                return False
            
            # Set virtual account balance
            if hasattr(self.mt5, 'account'):
                self.mt5.account.balance = result["balance"]
                self.mt5.account.equity = result["balance"] 
                self.mt5.account.login = 999999  # Virtual account number
            
            if self.mt5.initialize():
                account_info = self.mt5.account_info()
                if account_info:
                    self.modal_awal = account_info.balance
                    account_text = f"Virtual Account: {account_info.login} | Balance: ${account_info.balance:,.2f}"
                    self.account_info_var.set(account_text)
                    self.log(f"✅ Connected: {account_text}")
                    self.log("📊 Real market data - Virtual trading")
                    self.log("🛡️  100% Safe - No real money involved")
                    
                    self.connect_button.config(state="disabled")
                    self.start_button.config(state="normal")
                    
                    messagebox.showinfo("Success", 
                                      f"Connected to Trading Simulation!\n"
                                      f"Virtual Balance: ${account_info.balance:,.2f}\n"
                                      f"Real market data, safe simulation")
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
    
    def get_data(self, symbol, timeframe=None, n=50):
        """Get market data"""
        try:
            rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
            if rates is None or len(rates) < n:
                return None
            return rates
        except Exception as e:
            self.log(f"Error getting data: {e}")
            return None

    def get_total_open_orders(self):
        """Get total open positions"""
        try:
            positions = self.mt5.positions_get()
            return len(positions) if positions else 0
        except:
            return 0

    def close_all_orders(self):
        """Close all open positions"""
        try:
            return self.mt5.close_all_positions()
        except Exception as e:
            self.log(f"Error closing all orders: {e}")
            return False

    def detect_lonjakan(self, data, threshold=LONJAKAN_THRESHOLD):
        """Detect price spike"""
        try:
            current_price = data['close'][-1]
            if self.last_price is None:
                self.last_price = current_price
                return False
            
            change = abs(current_price - self.last_price)
            self.last_price = current_price
            return change > threshold
        except:
            return False
    
    # ======================
    # TECHNICAL INDICATORS
    # ======================
    def calculate_ma(self, data, period=10):
        """Calculate Simple Moving Average"""
        try:
            return np.mean(data['close'][-period:])
        except:
            return 0

    def calculate_ema(self, data, period=9):
        """Calculate Exponential Moving Average"""
        try:
            prices = data['close']
            weights = np.exp(np.linspace(-1., 0., period))
            weights /= weights.sum()
            return np.convolve(prices, weights, mode='valid')[-1]
        except:
            return 0

    def calculate_wma(self, data, period=5):
        """Calculate Weighted Moving Average"""
        try:
            weights = np.arange(1, period + 1)
            return np.convolve(data['close'], weights / weights.sum(), mode='valid')[-1]
        except:
            return 0

    def calculate_rsi(self, data, period=14):
        """Calculate RSI"""
        try:
            deltas = np.diff(data['close'])
            seed = deltas[:period]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return 50

    def get_bollinger_bands(self, data, period=20, dev=2):
        """Calculate Bollinger Bands"""
        try:
            closes = data['close']
            sma = np.mean(closes[-period:])
            std = np.std(closes[-period:])
            upper = sma + dev * std
            lower = sma - dev * std
            return upper, lower, sma
        except:
            current_price = data['close'][-1] if len(data['close']) > 0 else 0
            return current_price, current_price, current_price
    
    # ======================
    # TRAILING STOP
    # ======================
    def apply_trailing_stop(self):
        """Apply trailing stop to positions"""
        try:
            positions = self.mt5.positions_get()
            if positions is None:
                return
            
            for pos in positions:
                symbol = pos.symbol
                tick = self.mt5.symbol_info_tick(symbol)
                if not tick:
                    continue
                
                symbol_info = self.mt5.symbol_info(symbol)
                if not symbol_info:
                    continue
                
                point = symbol_info.point
                sl_new = 0.0
                
                if pos.type == 0:  # BUY
                    sl_new = tick.bid - TRAILING_STOP_PIPS * point
                    if pos.sl == 0 or sl_new > pos.sl:
                        self.modify_sl(pos, sl_new)
                else:  # SELL
                    sl_new = tick.ask + TRAILING_STOP_PIPS * point
                    if pos.sl == 0 or sl_new < pos.sl:
                        self.modify_sl(pos, sl_new)
                        
        except Exception as e:
            self.log(f"Error applying trailing stop: {e}")

    def modify_sl(self, position, new_sl):
        """Modify stop loss"""
        try:
            request = {
                "action": self.mt5.TRADE_ACTION_SLTP,
                "position": position.ticket,
                "sl": new_sl,
                "tp": position.tp
            }
            result = self.mt5.order_send(request)
            if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                self.log(f"Trailing stop updated for {position.symbol}: {new_sl:.5f}")
        except Exception as e:
            self.log(f"Error modifying SL: {e}")
    
    # ======================
    # BOT CONTROLS
    # ======================
    def start_bot(self):
        """Start trading bot"""
        if not self.mt5.terminal_info():
            messagebox.showerror("Error", "Please connect to market data first!")
            return
        
        if self.running:
            return
        
        self.running = True
        self.order_counter = 0
        
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        
        # Start bot in separate thread
        self.bot_thread = threading.Thread(target=self.trading_bot, daemon=True)
        self.bot_thread.start()
        
        self.log("🚀 Bot started!")
        self.send_telegram("✅ Trading Bot started!")

    def stop_bot(self):
        """Stop trading bot"""
        self.running = False
        
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
        self.log("🛑 Bot stopped!")
        self.send_telegram("🚫 Trading Bot stopped!")

    def manual_close_all(self):
        """Manually close all positions"""
        try:
            if self.close_all_orders():
                self.log("❌ All positions closed manually")
                self.send_telegram("❌ All positions closed manually")
            else:
                self.log("Error closing positions")
        except Exception as e:
            self.log(f"Error in manual close: {e}")
    
    # ======================
    # MAIN TRADING LOGIC
    # ======================
    def trading_bot(self):
        """Main trading bot logic"""
        symbol = self.symbol_var.get()
        lot = float(self.lot_var.get())
        interval = int(self.interval_var.get())
        tp_persen = float(self.tp_var.get()) / 100
        sl_persen = float(self.sl_var.get()) / 100
        scalp_tp = float(self.scalping_tp_var.get()) / 100
        scalp_sl = float(self.scalping_sl_var.get()) / 100
        scalping_enabled = self.scalping_mode_var.get()

        self.log(f"Trading started - Symbol: {symbol}, Lot: {lot}")
        
        while self.running:
            try:
                # Check connection
                if not self.mt5.terminal_info():
                    self.log("Connection lost, trying to reconnect...")
                    if not self.mt5.initialize():
                        self.log("Reconnection failed. Bot stopped.")
                        break
                
                # Check trading hours
                now = datetime.datetime.now()
                if now.hour < TRADING_START_HOUR or now.hour >= TRADING_END_HOUR:
                    self.log("Outside trading hours, waiting...")
                    time.sleep(interval)
                    continue
                
                # Check order limits
                if self.get_total_open_orders() >= MAX_ORDER_PER_SESSION:
                    self.log("Order limit reached, waiting...")
                    time.sleep(interval)
                    continue
                
                # Get market data
                data = self.get_data(symbol, n=50)
                if data is None:
                    self.log("Failed to get market data!")
                    time.sleep(interval)
                    continue
                
                # Calculate indicators
                harga = data['close'][-1]
                ma10 = self.calculate_ma(data, 10)
                ema9 = self.calculate_ema(data, 9)
                ema21 = self.calculate_ema(data, 21)
                ema50 = self.calculate_ema(data, 50)
                wma5 = self.calculate_wma(data, 5)
                wma10 = self.calculate_wma(data, 10)
                rsi = self.calculate_rsi(data, 14)
                bb_upper, bb_lower, bb_middle = self.get_bollinger_bands(data, 20)
                
                # Update P/L display
                account_info = self.mt5.account_info()
                if account_info and self.modal_awal:
                    profit_real = account_info.equity - self.modal_awal
                    self.profit_var.set(f"Real-time P/L: {profit_real:,.2f}")
                
                # Less sensitive price spike detection
                # Comment out for now to get more signals
                # if self.detect_lonjakan(data):
                #     self.log("Price spike detected, skipping...")
                #     time.sleep(interval)
                #     continue
                
                # Generate trading signals
                sinyal = None
                order_type = None
                tp = None
                sl = None
                
                # Logging untuk debug
                self.log(f"Debug - Price: {harga:.5f} | MA10: {ma10:.5f if ma10 else 'None'} | RSI: {rsi:.1f}")
                self.log(f"Debug - BB: Upper {bb_upper:.5f}, Lower {bb_lower:.5f}")
                
                # Scoring system untuk signal strength
                buy_score = 0
                sell_score = 0
                
                # BUY scoring
                if ma10 and harga < ma10:
                    buy_score += 1
                if rsi and rsi < 40:  # Lebih fleksibel dari 20
                    buy_score += 1
                if harga <= bb_lower * 1.001:  # Small buffer
                    buy_score += 1
                if wma5 and wma10 and wma5 < wma10:
                    buy_score += 1
                if ema9 and ema21 and ema9 < ema21:
                    buy_score += 1
                
                # SELL scoring  
                if ma10 and harga > ma10:
                    sell_score += 1
                if rsi and rsi > 60:  # Lebih fleksibel dari 80
                    sell_score += 1
                if harga >= bb_upper * 0.999:  # Small buffer
                    sell_score += 1
                if wma5 and wma10 and wma5 > wma10:
                    sell_score += 1
                if ema9 and ema21 and ema9 > ema21:
                    sell_score += 1
                
                self.log(f"Debug - BUY Score: {buy_score}/5 | SELL Score: {sell_score}/5")
                
                # Signal generation dengan score minimal 3/5
                if buy_score >= 3:
                    sinyal = "BUY"
                    order_type = self.mt5.ORDER_TYPE_BUY
                    
                    if scalping_enabled:
                        tp = harga * (1 + scalp_tp)
                        sl = harga * (1 - scalp_sl)
                    else:
                        tp = harga * (1 + tp_persen)
                        sl = harga * (1 - sl_persen)
                
                elif sell_score >= 3:
                    sinyal = "SELL"
                    order_type = self.mt5.ORDER_TYPE_SELL
                    
                    if scalping_enabled:
                        tp = harga * (1 - scalp_tp)
                        sl = harga * (1 + scalp_sl)
                    else:
                        tp = harga * (1 - tp_persen)
                        sl = harga * (1 + sl_persen)
                
                # Execute trade
                if sinyal and tp is not None and sl is not None:
                    tick = self.mt5.symbol_info_tick(symbol)
                    if not tick:
                        self.log("No tick data available")
                        time.sleep(interval)
                        continue
                    
                    harga_order = tick.ask if order_type == self.mt5.ORDER_TYPE_BUY else tick.bid
                    
                    request = {
                        "action": self.mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot,
                        "type": order_type,
                        "price": harga_order,
                        "sl": sl,
                        "tp": tp,
                        "deviation": 20,
                        "magic": 123456,
                        "comment": "AutoBot Entry",
                        "type_time": self.mt5.ORDER_TIME_GTC,
                        "type_filling": self.mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = self.mt5.order_send(request)
                    if result.retcode == self.mt5.TRADE_RETCODE_DONE:
                        self.order_counter += 1
                        
                        mode_text = "Scalping" if scalping_enabled else "Normal"
                        log_msg = f"{sinyal} {mode_text} Order #{self.order_counter} @ {harga_order:.5f} | TP: {tp:.5f} | SL: {sl:.5f}"
                        self.log(log_msg)
                        
                        telegram_msg = f"🎯 {sinyal} {mode_text}\nPrice: {harga_order:.5f}\nTP: {tp:.5f}\nSL: {sl:.5f}\nLot: {lot}"
                        self.send_telegram(telegram_msg)
                        
                        self.export_trade_log(harga_order, tp, sl, sinyal)
                    else:
                        self.log(f"Order failed: {result.retcode} - {result.comment}")
                else:
                    self.log("No valid signal, waiting...")
                
                # Apply trailing stops
                self.apply_trailing_stop()
                
                # Wait for next cycle
                time.sleep(interval)
                
            except Exception as e:
                self.log(f"Error in trading loop: {e}")
                time.sleep(interval)
        
        self.log("Trading bot stopped")
    
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            if messagebox.askokcancel("Quit", "Bot is running. Stop bot and exit?"):
                self.stop_bot()
                time.sleep(1)  # Give time for bot to stop
                self.root.destroy()
        else:
            self.root.destroy()
    
    def run(self):
        """Run the application"""
        self.log("🤖 TRADING BOT SIMULASI - 100% AMAN!")
        self.log("=" * 60)
        self.log("⚠️  PENTING: Ini adalah SIMULASI TRADING")
        self.log("💰 Tidak menggunakan uang asli - Hanya untuk belajar")
        self.log("📊 Data pasar REAL-TIME - Trading VIRTUAL")
        self.log("=" * 60)
        self.log("✨ Features: Auto Trading, Auto Scalping, Technical Indicators")
        self.log("🛡️ Risk Management: TP/SL, Position Limits, Trading Hours") 
        self.log("📱 Telegram Notifications Available")
        self.log("👆 Click 'Connect' to setup virtual account!")
        
        self.root.mainloop()

if __name__ == "__main__":
    try:
        bot = TradingBot()
        bot.run()
    except Exception as e:
        print(f"Error starting bot: {e}")
        input("Press Enter to exit...")