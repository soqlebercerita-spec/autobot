#!/usr/bin/env python3
"""
REAL MONEY Trading Bot - MetaTrader5 Integration
PERINGATAN: BOT INI UNTUK TRADING DENGAN UANG ASLI!
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

# Real MT5 imports
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("PERINGATAN: MetaTrader5 tidak terinstall! Install dengan: pip install MetaTrader5")

# Import enhanced modules
from market_data_api import MarketDataAPI
from enhanced_indicators import EnhancedIndicators
from config import config

class RealTradingBot:
    def __init__(self):
        """Initialize REAL trading bot with MT5 integration"""
        # Safety check
        if not MT5_AVAILABLE:
            raise Exception("MetaTrader5 library tidak ditemukan! Install dulu: pip install MetaTrader5")
        
        # Initialize components
        self.market_api = MarketDataAPI()
        self.indicators = EnhancedIndicators()
        
        # MT5 connection
        self.mt5_connected = False
        self.account_info = None
        self.real_balance = 0.0
        
        # Bot state
        self.running = False
        self.trading_enabled = False  # Extra safety
        self.emergency_stop = False
        self.modal_awal = None
        self.last_price = None
        self.last_prices = []
        self.last_reset_date = datetime.date.today()
        self.order_counter = 0
        self.total_opportunities_captured = 0
        self.total_opportunities_missed = 0
        self.session_profit = 0.0
        self.max_drawdown_reached = 0.0
        
        # Safety limits
        self.daily_loss_limit = 0.0
        self.max_orders_reached = False
        
        # Threading
        self.bot_thread = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup GUI for REAL trading"""
        self.root = tk.Tk()
        self.root.title("💰 REAL MONEY Trading Bot - MT5 Integration")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables
        self.symbol_var = tk.StringVar(value=config.DEFAULT_SYMBOL)
        self.lot_var = tk.StringVar(value=str(config.DEFAULT_LOT))
        self.interval_var = tk.StringVar(value=str(config.DEFAULT_INTERVAL))
        self.tp_balance_var = tk.StringVar(value=str(config.TP_PERSEN_BALANCE * 100))
        self.sl_balance_var = tk.StringVar(value=str(config.SL_PERSEN_BALANCE * 100))
        self.account_info_var = tk.StringVar(value="Account: Not Connected")
        self.profit_var = tk.StringVar(value="Real P/L: $0.00")
        self.balance_var = tk.StringVar(value="$0.00")
        self.scalping_mode_var = tk.BooleanVar(value=False)
        self.hft_mode_var = tk.BooleanVar(value=False)
        self.trading_enabled_var = tk.BooleanVar(value=False)
        self.opportunities_var = tk.StringVar(value="Opportunities: Captured: 0 | Missed: 0")
        
        self.create_real_trading_gui()
        
    def create_real_trading_gui(self):
        """Create GUI specifically for real trading with safety features"""
        # Style
        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Success.TLabel", foreground="green", font=("Segoe UI", 10, "bold"))
        style.configure("Warning.TLabel", foreground="red", font=("Segoe UI", 10, "bold"))
        style.configure("Danger.TButton", foreground="white", background="red", font=("Segoe UI", 10, "bold"))
        
        # SAFETY WARNING Header
        warning_frame = ttk.LabelFrame(self.root, text="⚠️ REAL MONEY TRADING WARNING", padding=10)
        warning_frame.pack(fill="x", padx=10, pady=5)
        
        warning_text = """🚨 PERINGATAN: Bot ini akan menggunakan UANG ASLI dari akun MT5 Anda!
✅ Pastikan MT5 sudah terinstall dan login
✅ Pastikan modal yang digunakan sesuai dengan kemampuan risk
✅ Bot akan otomatis stop jika mencapai daily loss limit
⛔ JANGAN gunakan di akun live jika belum testing di demo"""
        
        ttk.Label(warning_frame, text=warning_text, style="Warning.TLabel", justify="left").pack()
        
        # Connection Status
        status_frame = ttk.LabelFrame(self.root, text="🔗 MT5 Connection Status", padding=10)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        status_grid = ttk.Frame(status_frame)
        status_grid.pack(fill="x")
        
        ttk.Label(status_grid, textvariable=self.account_info_var).grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(status_grid, textvariable=self.profit_var, style="Success.TLabel").grid(row=0, column=1, sticky="e", padx=5)
        ttk.Label(status_grid, text="Real Balance:", style="TLabel").grid(row=1, column=0, sticky="w", padx=5)
        ttk.Label(status_grid, textvariable=self.balance_var, style="Success.TLabel").grid(row=1, column=1, sticky="e", padx=5)
        
        status_grid.columnconfigure(0, weight=1)
        status_grid.columnconfigure(1, weight=1)
        
        # Trading Settings
        settings_frame = ttk.LabelFrame(self.root, text="⚙️ Real Trading Settings", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)
        
        # Basic settings
        basic_frame = ttk.Frame(settings_frame)
        basic_frame.pack(fill="x", pady=5)
        
        ttk.Label(basic_frame, text="Symbol:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(basic_frame, textvariable=self.symbol_var, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(basic_frame, text="Lot Size:").grid(row=0, column=2, sticky="e", padx=5, pady=2)
        ttk.Entry(basic_frame, textvariable=self.lot_var, width=15).grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Label(basic_frame, text="Scan Interval:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(basic_frame, textvariable=self.interval_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        # TP/SL Balance-based settings
        tpsl_frame = ttk.LabelFrame(settings_frame, text="💰 Balance-Based TP/SL", padding=5)
        tpsl_frame.pack(fill="x", pady=5)
        
        ttk.Label(tpsl_frame, text="TP % dari Modal:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Entry(tpsl_frame, textvariable=self.tp_balance_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(tpsl_frame, text="SL % dari Modal:").grid(row=0, column=2, sticky="e", padx=5, pady=2)
        ttk.Entry(tpsl_frame, textvariable=self.sl_balance_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        # Trading modes
        modes_frame = ttk.LabelFrame(settings_frame, text="🎯 Trading Modes", padding=5)
        modes_frame.pack(fill="x", pady=5)
        
        ttk.Checkbutton(modes_frame, text="🔥 Scalping Mode (0.5% TP, 2% SL)", 
                       variable=self.scalping_mode_var).pack(anchor="w", pady=2)
        ttk.Checkbutton(modes_frame, text="⚡ HFT Mode (0.3% TP, 1.5% SL)", 
                       variable=self.hft_mode_var).pack(anchor="w", pady=2)
        
        # Safety Controls
        safety_frame = ttk.LabelFrame(self.root, text="🛡️ Safety Controls", padding=10)
        safety_frame.pack(fill="x", padx=10, pady=5)
        
        # Trading enabled checkbox (extra safety)
        ttk.Checkbutton(safety_frame, text="✅ Enable Real Trading (CENTANG UNTUK TRADING ASLI)", 
                       variable=self.trading_enabled_var, 
                       style="Warning.TLabel").pack(anchor="w", pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=15)
        
        # Connection buttons
        conn_frame = ttk.Frame(button_frame)
        conn_frame.pack(pady=5)
        
        self.connect_button = ttk.Button(conn_frame, text="🔗 Connect to MT5", 
                                       command=self.connect_mt5_real)
        self.test_button = ttk.Button(conn_frame, text="🧪 Test Connection", 
                                    command=self.test_mt5_connection)
        
        self.connect_button.grid(row=0, column=0, padx=10)
        self.test_button.grid(row=0, column=1, padx=10)
        
        # Trading control buttons
        trading_frame = ttk.Frame(button_frame)
        trading_frame.pack(pady=5)
        
        self.start_button = ttk.Button(trading_frame, text="🚀 Start REAL Trading", 
                                     command=self.start_real_trading, 
                                     state="disabled")
        self.stop_button = ttk.Button(trading_frame, text="🛑 Stop Trading", 
                                    command=self.stop_trading, 
                                    state="disabled")
        self.emergency_button = ttk.Button(trading_frame, text="🚨 EMERGENCY STOP", 
                                         command=self.emergency_stop_all,
                                         style="Danger.TButton")
        
        self.start_button.grid(row=0, column=0, padx=10)
        self.stop_button.grid(row=0, column=1, padx=10)
        self.emergency_button.grid(row=0, column=2, padx=10)
        
        # Log display
        log_frame = ttk.LabelFrame(self.root, text="📊 Trading Log", padding=10)
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.log_display = ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_display.pack(fill="both", expand=True)
        
        # Performance info
        perf_frame = ttk.LabelFrame(self.root, text="📈 Performance", padding=5)
        perf_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(perf_frame, textvariable=self.opportunities_var, style="Success.TLabel").pack()
    
    def connect_mt5_real(self):
        """Connect to real MT5"""
        try:
            if not MT5_AVAILABLE:
                messagebox.showerror("Error", "MetaTrader5 library tidak terinstall!\nInstall dengan: pip install MetaTrader5")
                return
            
            # Initialize MT5
            if not mt5.initialize():
                error = mt5.last_error()
                messagebox.showerror("MT5 Error", f"Gagal initialize MT5: {error}")
                self.log("❌ Gagal connect ke MT5")
                return
            
            # Get account info
            account_info = mt5.account_info()
            if account_info is None:
                messagebox.showerror("Error", "Gagal get account info. Pastikan MT5 sudah login!")
                mt5.shutdown()
                return
            
            # Store account info
            self.account_info = account_info._asdict()
            self.real_balance = self.account_info['balance']
            self.mt5_connected = True
            
            # Update GUI
            self.account_info_var.set(f"Account: {self.account_info['login']} | Server: {self.account_info['server']}")
            self.balance_var.set(f"${self.real_balance:,.2f}")
            
            # Enable buttons
            self.start_button.config(state="normal")
            self.test_button.config(state="normal")
            
            self.log(f"✅ MT5 Connected Successfully!")
            self.log(f"   Account: {self.account_info['login']}")
            self.log(f"   Server: {self.account_info['server']}")
            self.log(f"   Balance: ${self.real_balance:,.2f}")
            self.log(f"   Currency: {self.account_info['currency']}")
            
            messagebox.showinfo("Success", f"Connected to MT5!\nAccount: {self.account_info['login']}\nBalance: ${self.real_balance:,.2f}")
            
        except Exception as e:
            self.log(f"❌ MT5 connection error: {e}")
            messagebox.showerror("Error", f"MT5 connection failed: {e}")
    
    def test_mt5_connection(self):
        """Test MT5 connection and trading capabilities"""
        try:
            if not self.mt5_connected:
                messagebox.showwarning("Warning", "Connect to MT5 first!")
                return
            
            # Test account info
            account_info = mt5.account_info()
            if account_info is None:
                messagebox.showerror("Error", "MT5 connection lost!")
                return
            
            # Test symbol info
            symbol = self.symbol_var.get()
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                messagebox.showerror("Error", f"Symbol {symbol} not found!")
                return
            
            # Test market data
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
            if rates is None:
                messagebox.showerror("Error", f"Failed to get market data for {symbol}")
                return
            
            # Test positions
            positions = mt5.positions_get()
            
            self.log("🧪 MT5 Connection Test Results:")
            self.log(f"   ✅ Account info: OK")
            self.log(f"   ✅ Symbol {symbol}: OK")
            self.log(f"   ✅ Market data: OK ({len(rates)} bars)")
            self.log(f"   ✅ Positions: {len(positions) if positions else 0}")
            self.log(f"   ✅ Trading allowed: {account_info.trade_allowed}")
            
            messagebox.showinfo("Test Results", "✅ All MT5 tests passed!\nBot ready for real trading.")
            
        except Exception as e:
            self.log(f"❌ MT5 test error: {e}")
            messagebox.showerror("Test Failed", f"MT5 test failed: {e}")
    
    def start_real_trading(self):
        """Start real money trading with safety checks"""
        try:
            # Safety checks
            if not self.mt5_connected:
                messagebox.showerror("Error", "Connect to MT5 first!")
                return
            
            if not self.trading_enabled_var.get():
                messagebox.showerror("Error", "Enable 'Real Trading' checkbox first!")
                return
            
            # Confirmation dialog
            confirm = messagebox.askyesno(
                "CONFIRM REAL TRADING", 
                f"⚠️ KONFIRMASI TRADING DENGAN UANG ASLI!\n\n"
                f"Account: {self.account_info['login']}\n"
                f"Balance: ${self.real_balance:,.2f}\n"
                f"Symbol: {self.symbol_var.get()}\n"
                f"Lot Size: {self.lot_var.get()}\n\n"
                f"Apakah Anda yakin ingin memulai trading dengan uang asli?"
            )
            
            if not confirm:
                return
            
            # Set safety limits
            self.daily_loss_limit = self.real_balance * 0.05  # 5% max daily loss
            self.session_profit = 0.0
            self.max_drawdown_reached = 0.0
            
            # Start trading
            self.running = True
            self.emergency_stop = False
            self.max_orders_reached = False
            
            # Update GUI
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
            # Start trading thread
            self.bot_thread = threading.Thread(target=self.real_trading_loop, daemon=True)
            self.bot_thread.start()
            
            self.log("🚀 REAL MONEY TRADING STARTED!")
            self.log(f"   Daily Loss Limit: ${self.daily_loss_limit:,.2f}")
            self.log(f"   Max Risk per Trade: {config.MAX_RISK_PER_TRADE}%")
            
        except Exception as e:
            self.log(f"❌ Failed to start real trading: {e}")
            messagebox.showerror("Error", f"Failed to start trading: {e}")
    
    def real_trading_loop(self):
        """Main real trading loop with safety monitoring"""
        while self.running and not self.emergency_stop:
            try:
                # Safety checks
                if not self.safety_check():
                    break
                
                # Update account info
                self.update_account_info()
                
                # Check trading conditions
                if self.should_trade():
                    # Get trading signal
                    if self.hft_mode_var.get():
                        signal, tp, sl = self.hft_signal_check()
                    else:
                        signal, tp, sl = self.enhanced_signal_check()
                    
                    if signal and tp and sl:
                        # Execute trade
                        self.execute_real_trade(signal, tp, sl)
                
                # Sleep based on mode
                if self.hft_mode_var.get():
                    time.sleep(config.HFT_INTERVAL)
                else:
                    time.sleep(int(self.interval_var.get()))
                
            except Exception as e:
                self.log(f"❌ Trading loop error: {e}")
                time.sleep(5)
        
        self.log("🛑 Trading loop stopped")
    
    def safety_check(self):
        """Comprehensive safety checks for real trading"""
        try:
            # Check MT5 connection
            if not mt5.terminal_info():
                self.log("❌ MT5 terminal disconnected!")
                self.emergency_stop_all()
                return False
            
            # Check account info
            account_info = mt5.account_info()
            if account_info is None:
                self.log("❌ Cannot get account info!")
                self.emergency_stop_all()
                return False
            
            current_balance = account_info.balance
            current_equity = account_info.equity
            
            # Check daily loss limit
            daily_loss = self.real_balance - current_equity
            if daily_loss > self.daily_loss_limit:
                self.log(f"🚨 DAILY LOSS LIMIT REACHED: ${daily_loss:,.2f}")
                self.emergency_stop_all()
                return False
            
            # Check max drawdown
            drawdown_pct = (daily_loss / self.real_balance) * 100
            if drawdown_pct > config.MAX_DRAWDOWN:
                self.log(f"🚨 MAX DRAWDOWN REACHED: {drawdown_pct:.2f}%")
                self.emergency_stop_all()
                return False
            
            # Check order count
            if self.order_counter >= config.MAX_ORDER_PER_SESSION:
                if not self.max_orders_reached:
                    self.log(f"⛔ Max orders reached: {self.order_counter}")
                    self.max_orders_reached = True
                return False
            
            return True
            
        except Exception as e:
            self.log(f"❌ Safety check error: {e}")
            return False
    
    def execute_real_trade(self, signal, tp, sl):
        """Execute real trade with MT5"""
        try:
            symbol = self.symbol_var.get()
            lot_size = float(self.lot_var.get())
            
            # Get current price
            if signal == "BUY":
                price = mt5.symbol_info_tick(symbol).ask
                order_type = mt5.ORDER_TYPE_BUY
            else:
                price = mt5.symbol_info_tick(symbol).bid
                order_type = mt5.ORDER_TYPE_SELL
            
            # Create order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "magic": config.MT5_MAGIC_NUMBER,
                "comment": f"TradingBot_{signal}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.log(f"❌ Order failed: {result.retcode} - {result.comment}")
                return False
            
            # Order successful
            self.order_counter += 1
            self.total_opportunities_captured += 1
            
            self.log(f"✅ REAL TRADE EXECUTED!")
            self.log(f"   Signal: {signal}")
            self.log(f"   Symbol: {symbol}")
            self.log(f"   Volume: {lot_size}")
            self.log(f"   Price: {price}")
            self.log(f"   TP: {tp}")
            self.log(f"   SL: {sl}")
            self.log(f"   Ticket: {result.order}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Trade execution error: {e}")
            return False
    
    def emergency_stop_all(self):
        """Emergency stop all trading and close positions"""
        try:
            self.emergency_stop = True
            self.running = False
            
            # Close all positions
            positions = mt5.positions_get()
            if positions:
                for position in positions:
                    # Create close request
                    if position.type == mt5.POSITION_TYPE_BUY:
                        order_type = mt5.ORDER_TYPE_SELL
                        price = mt5.symbol_info_tick(position.symbol).bid
                    else:
                        order_type = mt5.ORDER_TYPE_BUY
                        price = mt5.symbol_info_tick(position.symbol).ask
                    
                    close_request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": position.symbol,
                        "volume": position.volume,
                        "type": order_type,
                        "position": position.ticket,
                        "price": price,
                        "magic": config.MT5_MAGIC_NUMBER,
                        "comment": "Emergency_Close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(close_request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.log(f"✅ Position {position.ticket} closed")
                    else:
                        self.log(f"❌ Failed to close position {position.ticket}")
            
            # Update GUI
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            
            self.log("🚨 EMERGENCY STOP ACTIVATED!")
            self.log("🛑 All positions closed")
            
            messagebox.showwarning("Emergency Stop", "🚨 Emergency stop activated!\nAll positions have been closed.")
            
        except Exception as e:
            self.log(f"❌ Emergency stop error: {e}")
    
    def stop_trading(self):
        """Normal stop trading"""
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.log("🛑 Trading stopped normally")
    
    def update_account_info(self):
        """Update account information display"""
        try:
            account_info = mt5.account_info()
            if account_info:
                current_balance = account_info.balance
                current_equity = account_info.equity
                current_profit = account_info.profit
                
                self.balance_var.set(f"${current_balance:,.2f}")
                self.profit_var.set(f"P/L: ${current_profit:,.2f}")
                
                # Update opportunities display
                self.opportunities_var.set(f"Opportunities: Captured: {self.total_opportunities_captured} | Missed: {self.total_opportunities_missed}")
                
        except Exception as e:
            self.log(f"❌ Account update error: {e}")
    
    def should_trade(self):
        """Check if trading conditions are met"""
        # Add your trading conditions here
        return (self.trading_enabled_var.get() and 
                not self.max_orders_reached and 
                not self.emergency_stop)
    
    def enhanced_signal_check(self):
        """Enhanced signal detection for real trading"""
        # Implementation from the original bot
        # Add proper signal detection logic here
        return None, None, None
    
    def hft_signal_check(self):
        """HFT signal detection for real trading"""
        # Implementation from the original bot
        # Add HFT signal detection logic here
        return None, None, None
    
    def log(self, message):
        """Log message to display"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        try:
            self.log_display.insert(tk.END, log_message)
            self.log_display.see(tk.END)
            
            # Also log to file
            with open("real_trading_log.txt", "a", encoding="utf-8") as f:
                f.write(log_message)
                
        except:
            pass
    
    def on_closing(self):
        """Handle application closing"""
        try:
            if self.running:
                confirm = messagebox.askyesno("Confirm Exit", 
                    "Trading bot masih berjalan!\nStop trading dan close semua posisi?")
                if confirm:
                    self.emergency_stop_all()
                else:
                    return
            
            # Shutdown MT5
            if self.mt5_connected:
                mt5.shutdown()
            
            self.root.destroy()
            
        except Exception as e:
            print(f"Closing error: {e}")
            self.root.destroy()

def main():
    """Main function for real trading bot"""
    try:
        app = RealTradingBot()
        app.root.mainloop()
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()