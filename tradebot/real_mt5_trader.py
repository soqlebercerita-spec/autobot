#!/usr/bin/env python3
"""
Real MT5 Trading Bot - For LIVE Trading with Real Account Balance Calculations
All TP/SL calculations based on account balance percentage, not market price
"""

import time
import datetime
import threading
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.scrolledtext import ScrolledText
import tkinter.messagebox as messagebox

from mt5_wrapper import mt5, MT5_AVAILABLE
from enhanced_indicators import EnhancedIndicators
from config import config

class RealMT5TradingBot:
    """Real MT5 Trading Bot for Live Trading"""
    
    def __init__(self):
        self.indicators = EnhancedIndicators()
        self.running = False
        self.connected = False
        self.order_counter = 0
        self.total_profit = 0.0
        self.initial_balance = 0.0
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup GUI for real trading"""
        self.root = tk.Tk()
        self.root.title("🎯 REAL MT5 Trading Bot - Live Account")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f8f8f8")
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables
        self.symbol_var = tk.StringVar(value=config.DEFAULT_SYMBOL)
        self.lot_var = tk.StringVar(value=str(config.DEFAULT_LOT))
        self.mode_var = tk.StringVar(value="Normal")
        self.account_info_var = tk.StringVar(value="Account: Not Connected")
        self.balance_var = tk.StringVar(value="$0.00")
        self.equity_var = tk.StringVar(value="$0.00")
        self.profit_var = tk.StringVar(value="$0.00")
        self.connection_status_var = tk.StringVar(value="❌ Not Connected")
        self.trading_enabled_var = tk.BooleanVar(value=False)
        
        self.create_real_trading_gui()
        
    def create_real_trading_gui(self):
        """Create GUI for real trading"""
        
        # Title
        title_frame = tk.Frame(self.root, bg="#f8f8f8")
        title_frame.pack(pady=10, fill="x")
        
        tk.Label(title_frame, text="🎯 REAL MT5 TRADING BOT", 
                font=("Arial", 20, "bold"), bg="#f8f8f8", fg="#c0392b").pack()
        tk.Label(title_frame, text="LIVE ACCOUNT - Balance-Based TP/SL System", 
                font=("Arial", 12), bg="#f8f8f8", fg="gray").pack()
        
        # Connection status
        status_frame = tk.LabelFrame(self.root, text="MT5 Connection Status", 
                                   font=("Arial", 12, "bold"))
        status_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Label(status_frame, textvariable=self.connection_status_var, 
                font=("Arial", 14, "bold")).pack(pady=5)
        
        btn_frame = tk.Frame(status_frame)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="🔌 Connect to MT5", 
                 command=self.connect_mt5, font=("Arial", 12, "bold"),
                 bg="#27ae60", fg="white", padx=20).pack(side="left", padx=10)
        
        tk.Button(btn_frame, text="🔍 Test Connection", 
                 command=self.test_connection, font=("Arial", 12, "bold"),
                 bg="#3498db", fg="white", padx=20).pack(side="left", padx=10)
        
        # Account Information
        account_frame = tk.LabelFrame(self.root, text="Real Account Information", 
                                    font=("Arial", 12, "bold"))
        account_frame.pack(pady=10, padx=20, fill="x")
        
        info_grid = tk.Frame(account_frame)
        info_grid.pack(fill="x", padx=10, pady=10)
        
        tk.Label(info_grid, text="Account:", font=("Arial", 10, "bold"), width=12).grid(row=0, column=0, sticky="w")
        tk.Label(info_grid, textvariable=self.account_info_var, font=("Arial", 10)).grid(row=0, column=1, sticky="w")
        
        tk.Label(info_grid, text="Balance:", font=("Arial", 10, "bold"), width=12).grid(row=1, column=0, sticky="w")
        tk.Label(info_grid, textvariable=self.balance_var, font=("Arial", 10, "bold"), fg="blue").grid(row=1, column=1, sticky="w")
        
        tk.Label(info_grid, text="Equity:", font=("Arial", 10, "bold"), width=12).grid(row=2, column=0, sticky="w")
        tk.Label(info_grid, textvariable=self.equity_var, font=("Arial", 10)).grid(row=2, column=1, sticky="w")
        
        tk.Label(info_grid, text="Profit:", font=("Arial", 10, "bold"), width=12).grid(row=3, column=0, sticky="w")
        tk.Label(info_grid, textvariable=self.profit_var, font=("Arial", 10, "bold")).grid(row=3, column=1, sticky="w")
        
        # Trading Configuration
        config_frame = tk.LabelFrame(self.root, text="Trading Configuration", 
                                   font=("Arial", 12, "bold"))
        config_frame.pack(pady=10, padx=20, fill="x")
        
        config_grid = tk.Frame(config_frame)
        config_grid.pack(fill="x", padx=10, pady=10)
        
        tk.Label(config_grid, text="Symbol:", width=12).grid(row=0, column=0, sticky="w")
        tk.Entry(config_grid, textvariable=self.symbol_var, width=15).grid(row=0, column=1, padx=5)
        
        tk.Label(config_grid, text="Lot Size:", width=12).grid(row=0, column=2, sticky="w", padx=(20,0))
        tk.Entry(config_grid, textvariable=self.lot_var, width=10).grid(row=0, column=3, padx=5)
        
        tk.Label(config_grid, text="Mode:", width=12).grid(row=1, column=0, sticky="w")
        mode_combo = ttk.Combobox(config_grid, textvariable=self.mode_var, 
                                 values=["Normal", "Scalping", "HFT"], width=12)
        mode_combo.grid(row=1, column=1, padx=5)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_change)
        
        # TP/SL Information based on mode
        self.tpsl_info_var = tk.StringVar(value="Normal: TP=1% of balance, SL=3% of balance")
        tk.Label(config_frame, textvariable=self.tpsl_info_var, 
                font=("Arial", 10, "bold"), fg="#e67e22").pack(pady=5)
        
        # Safety Controls
        safety_frame = tk.LabelFrame(self.root, text="🛡️ Safety Controls", 
                                   font=("Arial", 12, "bold"), fg="#c0392b")
        safety_frame.pack(pady=10, padx=20, fill="x")
        
        tk.Checkbutton(safety_frame, text="✅ I understand this is REAL MONEY trading", 
                      variable=self.trading_enabled_var, font=("Arial", 11, "bold"),
                      fg="#c0392b").pack(pady=5)
        
        # Control Buttons
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=15)
        
        tk.Button(control_frame, text="🎯 Start REAL Trading", 
                 command=self.start_real_trading, font=("Arial", 14, "bold"),
                 bg="#c0392b", fg="white", padx=30, pady=8).pack(side="left", padx=10)
        
        tk.Button(control_frame, text="🛑 EMERGENCY STOP", 
                 command=self.emergency_stop, font=("Arial", 14, "bold"),
                 bg="#8b0000", fg="white", padx=30, pady=8).pack(side="left", padx=10)
        
        # Trading Log
        log_frame = tk.LabelFrame(self.root, text="Real Trading Log")
        log_frame.pack(pady=10, padx=20, fill="both", expand=True)
        
        self.log_text = ScrolledText(log_frame, height=15, width=80)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
    def connect_mt5(self):
        """Connect to real MT5"""
        try:
            if not MT5_AVAILABLE:
                messagebox.showerror("MT5 Not Available", 
                    "MetaTrader5 library is not available.\n"
                    "Please install MT5 and the Python library for real trading.")
                return
            
            self.log("🔌 Attempting to connect to MT5...")
            
            if mt5.initialize():
                self.connected = True
                self.connection_status_var.set("✅ Connected to MT5")
                self.log("✅ Successfully connected to MT5")
                self.update_account_info()
            else:
                error = mt5.last_error()
                self.log(f"❌ MT5 connection failed: {error}")
                self.connection_status_var.set("❌ Connection Failed")
                
        except Exception as e:
            self.log(f"❌ Connection error: {e}")
            self.connection_status_var.set("❌ Connection Error")
    
    def test_connection(self):
        """Test MT5 connection and account access"""
        try:
            if not self.connected:
                self.log("⚠️ Not connected to MT5. Connecting first...")
                self.connect_mt5()
                return
            
            self.log("🔍 Testing MT5 connection...")
            
            # Test account info
            account_info = mt5.account_info()
            if account_info:
                self.log("✅ Account info retrieved successfully")
                self.log(f"   Login: {account_info.login}")
                self.log(f"   Server: {account_info.server}")
                self.log(f"   Balance: ${account_info.balance:,.2f}")
                self.log(f"   Equity: ${account_info.equity:,.2f}")
                self.log(f"   Margin: ${account_info.margin:,.2f}")
            else:
                self.log("❌ Cannot retrieve account info")
            
            # Test symbol access
            symbol = self.symbol_var.get()
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info:
                self.log(f"✅ Symbol {symbol} is available")
                
                # Test price retrieval
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    self.log(f"✅ Price data: Bid={tick.bid:.5f}, Ask={tick.ask:.5f}")
                else:
                    self.log(f"⚠️ Cannot get price for {symbol}")
            else:
                self.log(f"❌ Symbol {symbol} not available")
                
        except Exception as e:
            self.log(f"❌ Connection test error: {e}")
    
    def update_account_info(self):
        """Update account information display"""
        try:
            if not self.connected:
                return
            
            account_info = mt5.account_info()
            if account_info:
                self.account_info_var.set(f"Login: {account_info.login} | Server: {account_info.server}")
                self.balance_var.set(f"${account_info.balance:,.2f}")
                self.equity_var.set(f"${account_info.equity:,.2f}")
                
                profit = account_info.equity - account_info.balance
                color = "green" if profit >= 0 else "red"
                self.profit_var.set(f"${profit:+,.2f}")
                
                if self.initial_balance == 0:
                    self.initial_balance = account_info.balance
                    
        except Exception as e:
            self.log(f"❌ Account update error: {e}")
    
    def on_mode_change(self, event=None):
        """Update TP/SL info when mode changes"""
        mode = self.mode_var.get()
        if mode == "Normal":
            self.tpsl_info_var.set("Normal: TP=1% of balance, SL=3% of balance")
        elif mode == "Scalping":
            self.tpsl_info_var.set("Scalping: TP=0.5% of balance, SL=2% of balance")
        elif mode == "HFT":
            self.tpsl_info_var.set("HFT: TP=0.3% of balance, SL=1.5% of balance")
    
    def calculate_balance_based_tp_sl(self, signal, current_price):
        """Calculate TP/SL based on account balance percentage"""
        try:
            if not self.connected:
                self.log("❌ Not connected to MT5 for balance calculation")
                return 0, 0
            
            account_info = mt5.account_info()
            if not account_info:
                self.log("❌ Cannot get account info for TP/SL calculation")
                return 0, 0
            
            balance = account_info.balance
            mode = self.mode_var.get()
            symbol = self.symbol_var.get()
            lot_size = float(self.lot_var.get())
            
            # Get TP/SL percentages based on mode
            if mode == "HFT":
                tp_pct = config.HFT_TP_PERSEN_BALANCE
                sl_pct = config.HFT_SL_PERSEN_BALANCE
            elif mode == "Scalping":
                tp_pct = config.SCALPING_TP_PERSEN_BALANCE
                sl_pct = config.SCALPING_SL_PERSEN_BALANCE
            else:  # Normal
                tp_pct = config.TP_PERSEN_BALANCE
                sl_pct = config.SL_PERSEN_BALANCE
            
            # Calculate money targets
            tp_money = balance * tp_pct
            sl_money = balance * sl_pct
            
            # Get symbol specifications
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                self.log(f"❌ Cannot get symbol info for {symbol}")
                return 0, 0
            
            # Calculate pip value
            contract_size = symbol_info.trade_contract_size
            point = symbol_info.point
            pip_size = 10 * point
            pip_value = contract_size * pip_size * lot_size
            
            # Convert money to pips
            tp_pips = tp_money / pip_value if pip_value > 0 else 50
            sl_pips = sl_money / pip_value if pip_value > 0 else 100
            
            # Calculate price levels
            if signal == "BUY":
                tp = current_price + (tp_pips * pip_size)
                sl = current_price - (sl_pips * pip_size)
            else:  # SELL
                tp = current_price - (tp_pips * pip_size)
                sl = current_price + (sl_pips * pip_size)
            
            self.log(f"💰 {mode} Balance-Based TP/SL:")
            self.log(f"   Balance: ${balance:,.2f}")
            self.log(f"   TP Target: ${tp_money:,.2f} ({tp_pct*100:.1f}%)")
            self.log(f"   SL Limit: ${sl_money:,.2f} ({sl_pct*100:.1f}%)")
            self.log(f"   TP Price: {tp:.5f} | SL Price: {sl:.5f}")
            
            return tp, sl
            
        except Exception as e:
            self.log(f"❌ TP/SL calculation error: {e}")
            return 0, 0
    
    def start_real_trading(self):
        """Start real trading with safety checks"""
        try:
            # Safety checks
            if not self.trading_enabled_var.get():
                messagebox.showerror("Safety Check", 
                    "Please confirm you understand this is REAL MONEY trading!")
                return
            
            if not self.connected:
                messagebox.showerror("Not Connected", 
                    "Please connect to MT5 first!")
                return
            
            if self.running:
                messagebox.showwarning("Already Running", 
                    "Real trading bot is already running!")
                return
            
            # Final confirmation
            result = messagebox.askyesno("REAL TRADING CONFIRMATION", 
                f"⚠️ WARNING: You are about to start REAL MONEY trading!\n\n"
                f"Mode: {self.mode_var.get()}\n"
                f"Symbol: {self.symbol_var.get()}\n"
                f"Lot Size: {self.lot_var.get()}\n\n"
                f"All TP/SL will be calculated based on your account balance.\n"
                f"Do you want to proceed?")
            
            if not result:
                return
            
            self.running = True
            self.log("🎯 STARTING REAL TRADING BOT...")
            self.log(f"⚠️ Mode: {self.mode_var.get()}")
            self.log(f"⚠️ Symbol: {self.symbol_var.get()}")
            self.log(f"⚠️ Lot Size: {self.lot_var.get()}")
            self.log("💰 Using balance-based TP/SL calculations")
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self.real_trading_loop, daemon=True)
            self.trading_thread.start()
            
        except Exception as e:
            self.log(f"❌ Start trading error: {e}")
    
    def real_trading_loop(self):
        """Main real trading loop"""
        while self.running:
            try:
                # Update account info
                self.update_account_info()
                
                # Check for trading signals
                signal, tp, sl = self.check_real_signal()
                
                if signal and tp > 0 and sl > 0:
                    self.log(f"🎯 Real Signal: {signal} | TP: {tp:.5f} | SL: {sl:.5f}")
                    # Execute real trade here
                    # self.execute_real_trade(signal, tp, sl)
                
                # Determine sleep interval based on mode
                mode = self.mode_var.get()
                if mode == "HFT":
                    time.sleep(1)  # 1 second for HFT
                elif mode == "Scalping":
                    time.sleep(5)  # 5 seconds for Scalping
                else:
                    time.sleep(10)  # 10 seconds for Normal
                
            except Exception as e:
                self.log(f"❌ Trading loop error: {e}")
                time.sleep(10)
    
    def check_real_signal(self):
        """Check for real trading signals"""
        try:
            symbol = self.symbol_var.get()
            
            # Get real price from MT5
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return None, 0, 0
            
            current_price = (tick.ask + tick.bid) / 2
            
            # Simple signal generation (replace with real indicator logic)
            signal = "BUY"  # Placeholder
            
            # Calculate balance-based TP/SL
            tp, sl = self.calculate_balance_based_tp_sl(signal, current_price)
            
            return signal, tp, sl
            
        except Exception as e:
            self.log(f"❌ Signal check error: {e}")
            return None, 0, 0
    
    def emergency_stop(self):
        """Emergency stop all trading"""
        self.running = False
        self.log("🛑 EMERGENCY STOP ACTIVATED")
        messagebox.showinfo("Emergency Stop", "Real trading bot stopped immediately!")
    
    def log(self, message):
        """Log message to display"""
        if hasattr(self, 'log_text'):
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"{timestamp} - {message}\n")
            self.log_text.see(tk.END)
            self.root.update_idletasks()
    
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            result = messagebox.askyesno("Exit Confirmation", 
                "Real trading bot is running. Stop and exit?")
            if result:
                self.running = False
            else:
                return
        
        if self.connected:
            mt5.shutdown()
        self.root.destroy()
    
    def run(self):
        """Run the real trading bot"""
        self.root.mainloop()

if __name__ == "__main__":
    bot = RealMT5TradingBot()
    bot.run()