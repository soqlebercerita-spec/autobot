#!/usr/bin/env python3
"""
MetaTrader5 Automated Trading Bot
Advanced forex/commodity trading bot with technical indicators and risk management
"""

import MetaTrader5 as mt5
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
from indicators import TechnicalIndicators
from risk_manager import RiskManager
from telegram_notifier import TelegramNotifier
from trade_logger import TradeLogger
from config import TradingConfig

class TradingBot:
    def __init__(self):
        self.config = TradingConfig()
        self.indicators = TechnicalIndicators()
        self.risk_manager = RiskManager(self.config)
        self.telegram = TelegramNotifier(self.config)
        self.logger = TradeLogger()
        
        # Bot state
        self.running = False
        self.modal_awal = None
        self.last_price = None
        self.last_reset_date = datetime.date.today()
        self.order_counter = 0
        self.positions_data = {}
        
        # Threading
        self.bot_thread = None
        self.gui_update_thread = None
        
        self.setup_gui()
        self.load_settings()

    def setup_gui(self):
        """Initialize the GUI components"""
        self.root = tk.Tk()
        self.root.title("Advanced MetaTrader5 Trading Bot")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2c3e50")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Variables
        self.symbol_var = tk.StringVar(value=self.config.DEFAULT_SYMBOL)
        self.lot_var = tk.StringVar(value=str(self.config.DEFAULT_LOT))
        self.interval_var = tk.StringVar(value=str(self.config.DEFAULT_INTERVAL))
        self.tp_var = tk.StringVar(value=str(self.config.TP_PERSEN_DEFAULT * 100))
        self.sl_var = tk.StringVar(value=str(self.config.SL_PERSEN_DEFAULT * 100))
        self.scalping_tp_var = tk.StringVar(value=str(self.config.SCALPING_TP_PERSEN * 100))
        self.scalping_sl_var = tk.StringVar(value=str(self.config.SCALPING_SL_PERSEN * 100))
        self.account_info_var = tk.StringVar(value="Account: Not Connected")
        self.profit_var = tk.StringVar(value="Real-time P/L: 0.00")
        self.status_var = tk.StringVar(value="Status: Stopped")
        self.scalping_mode_var = tk.BooleanVar(value=self.config.SCALPING_OVERRIDE_ENABLED)
        self.auto_close_var = tk.BooleanVar(value=True)
        
        self.create_gui_elements()
        
    def create_gui_elements(self):
        """Create all GUI elements"""
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"), foreground="#ecf0f1")
        style.configure("Header.TLabel", font=("Segoe UI", 10, "bold"), foreground="#3498db")
        style.configure("Info.TLabel", font=("Segoe UI", 9), foreground="#ecf0f1")
        style.configure("Success.TLabel", font=("Segoe UI", 9), foreground="#27ae60")
        style.configure("Error.TLabel", font=("Segoe UI", 9), foreground="#e74c3c")
        
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Top frame for connection and status
        top_frame = ttk.LabelFrame(main_container, text="Connection & Status", padding=10)
        top_frame.pack(fill="x", pady=(0, 10))
        
        # Connection controls
        conn_frame = ttk.Frame(top_frame)
        conn_frame.pack(fill="x")
        
        ttk.Button(conn_frame, text="Connect MT5", command=self.connect_mt5, 
                  style="Accent.TButton").pack(side="left", padx=(0, 10))
        ttk.Button(conn_frame, text="Disconnect", command=self.disconnect_mt5,
                  style="Accent.TButton").pack(side="left", padx=(0, 10))
        
        # Status display
        status_frame = ttk.Frame(top_frame)
        status_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Label(status_frame, textvariable=self.account_info_var, 
                 style="Info.TLabel").pack(side="left")
        ttk.Label(status_frame, textvariable=self.profit_var, 
                 style="Success.TLabel").pack(side="right")
        ttk.Label(status_frame, textvariable=self.status_var, 
                 style="Header.TLabel").pack(anchor="center")
        
        # Middle frame for settings
        middle_frame = ttk.Frame(main_container)
        middle_frame.pack(fill="x", pady=(0, 10))
        
        # Trading settings
        settings_frame = ttk.LabelFrame(middle_frame, text="Trading Settings", padding=10)
        settings_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.create_settings_section(settings_frame)
        
        # Risk management settings
        risk_frame = ttk.LabelFrame(middle_frame, text="Risk Management", padding=10)
        risk_frame.pack(side="right", fill="both", expand=True)
        
        self.create_risk_section(risk_frame)
        
        # Control buttons
        control_frame = ttk.LabelFrame(main_container, text="Bot Controls", padding=10)
        control_frame.pack(fill="x", pady=(0, 10))
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack()
        
        self.start_button = ttk.Button(button_frame, text="Start Bot", 
                                      command=self.start_bot, style="Accent.TButton")
        self.start_button.pack(side="left", padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Bot", 
                                     command=self.stop_bot, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 10))
        
        ttk.Button(button_frame, text="Close All Positions", 
                  command=self.manual_close_all).pack(side="left", padx=(0, 10))
        
        ttk.Button(button_frame, text="Save Settings", 
                  command=self.save_settings).pack(side="left", padx=(0, 10))
        
        ttk.Button(button_frame, text="Export Logs", 
                  command=self.export_logs).pack(side="left", padx=(0, 10))
        
        # Log display
        log_frame = ttk.LabelFrame(main_container, text="Trading Log", padding=10)
        log_frame.pack(fill="both", expand=True)
        
        self.log_box = ScrolledText(log_frame, height=15, bg="#34495e", fg="#ecf0f1",
                                   font=("Consolas", 9), wrap=tk.WORD)
        self.log_box.pack(fill="both", expand=True)
        
    def create_settings_section(self, parent):
        """Create trading settings section"""
        # Basic settings
        basic_frame = ttk.Frame(parent)
        basic_frame.pack(fill="x", pady=(0, 10))
        
        settings = [
            ("Symbol:", self.symbol_var, 15),
            ("Lot Size:", self.lot_var, 10),
            ("Interval (s):", self.interval_var, 10),
        ]
        
        for i, (label, var, width) in enumerate(settings):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(basic_frame, text=label, style="Info.TLabel").grid(
                row=row, column=col, sticky="e", padx=(0, 5), pady=5)
            ttk.Entry(basic_frame, textvariable=var, width=width).grid(
                row=row, column=col+1, padx=(0, 20), pady=5)
        
        # TP/SL settings
        tp_sl_frame = ttk.LabelFrame(parent, text="Take Profit / Stop Loss", padding=5)
        tp_sl_frame.pack(fill="x", pady=(0, 10))
        
        tp_sl_settings = [
            ("Normal TP (%):", self.tp_var),
            ("Normal SL (%):", self.sl_var),
            ("Scalp TP (%):", self.scalping_tp_var),
            ("Scalp SL (%):", self.scalping_sl_var),
        ]
        
        for i, (label, var) in enumerate(tp_sl_settings):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(tp_sl_frame, text=label, style="Info.TLabel").grid(
                row=row, column=col, sticky="e", padx=(0, 5), pady=5)
            ttk.Entry(tp_sl_frame, textvariable=var, width=10).grid(
                row=row, column=col+1, padx=(0, 20), pady=5)
        
        # Mode settings
        mode_frame = ttk.Frame(parent)
        mode_frame.pack(fill="x")
        
        ttk.Checkbutton(mode_frame, text="Enable Scalping Mode", 
                       variable=self.scalping_mode_var).pack(side="left", padx=(0, 20))
        ttk.Checkbutton(mode_frame, text="Auto Close at End", 
                       variable=self.auto_close_var).pack(side="left")
        
    def create_risk_section(self, parent):
        """Create risk management section"""
        risk_settings = [
            ("Max Orders per Session:", str(self.config.MAX_ORDER_PER_SESSION)),
            ("Min Balance Required:", str(self.config.SALDO_MINIMAL)),
            ("Target Profit (%):", str(self.config.TARGET_PROFIT_PERSEN)),
            ("Trading Start Hour:", str(self.config.TRADING_START_HOUR)),
            ("Trading End Hour:", str(self.config.TRADING_END_HOUR)),
            ("Trailing Stop (pips):", str(self.config.TRAILING_STOP_PIPS)),
        ]
        
        self.risk_vars = {}
        for i, (label, default_value) in enumerate(risk_settings):
            var = tk.StringVar(value=default_value)
            self.risk_vars[label] = var
            
            ttk.Label(parent, text=label, style="Info.TLabel").grid(
                row=i, column=0, sticky="e", padx=(0, 5), pady=5)
            ttk.Entry(parent, textvariable=var, width=15).grid(
                row=i, column=1, padx=(0, 5), pady=5)

    def log(self, text, level="INFO"):
        """Add log entry with timestamp"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {text}"
        
        # Add to GUI
        self.log_box.insert(tk.END, log_entry + "\n")
        self.log_box.see(tk.END)
        
        # Save to file
        self.logger.log_to_file(text, level)
        
        # Send important messages to Telegram
        if level in ["ERROR", "TRADE", "ALERT"]:
            self.telegram.send_message(f"{level}: {text}")

    def connect_mt5(self):
        """Connect to MetaTrader5"""
        try:
            if not mt5.initialize():
                error_msg = f"MT5 initialization failed: {mt5.last_error()}"
                self.log(error_msg, "ERROR")
                messagebox.showerror("Connection Error", error_msg)
                return False
            
            account_info = mt5.account_info()
            if account_info:
                self.modal_awal = account_info.balance
                account_text = f"Account: {account_info.login} | Balance: {account_info.balance:,.2f}"
                self.account_info_var.set(account_text)
                self.log(f"Connected to MT5: {account_text}", "INFO")
                
                # Verify symbol exists
                symbol_info = mt5.symbol_info(self.symbol_var.get())
                if symbol_info is None:
                    self.log(f"Symbol {self.symbol_var.get()} not found", "ERROR")
                    return False
                    
                return True
            else:
                self.log("Failed to get account information", "ERROR")
                return False
                
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            self.log(error_msg, "ERROR")
            messagebox.showerror("Connection Error", error_msg)
            return False

    def disconnect_mt5(self):
        """Disconnect from MetaTrader5"""
        if self.running:
            self.stop_bot()
        
        mt5.shutdown()
        self.account_info_var.set("Account: Disconnected")
        self.log("Disconnected from MT5", "INFO")

    def get_market_data(self, symbol, timeframe=mt5.TIMEFRAME_M1, count=100):
        """Get market data for analysis"""
        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) < count:
                self.log(f"Failed to get market data for {symbol}", "ERROR")
                return None
            return rates
        except Exception as e:
            self.log(f"Error getting market data: {str(e)}", "ERROR")
            return None

    def analyze_market(self, data):
        """Analyze market conditions and generate signals"""
        try:
            # Calculate technical indicators
            ma10 = self.indicators.calculate_ma(data, 10)
            ma20 = self.indicators.calculate_ma(data, 20)
            ema9 = self.indicators.calculate_ema(data, 9)
            ema21 = self.indicators.calculate_ema(data, 21)
            ema50 = self.indicators.calculate_ema(data, 50)
            wma5 = self.indicators.calculate_wma(data, 5)
            wma10 = self.indicators.calculate_wma(data, 10)
            rsi = self.indicators.calculate_rsi(data, 14)
            bb_upper, bb_lower, bb_middle = self.indicators.get_bollinger_bands(data, 20)
            
            current_price = data['close'][-1]
            
            # Generate signals
            signal = None
            confidence = 0
            
            # Bullish conditions
            bullish_conditions = [
                current_price > ma10,  # Price above MA10
                ema9 > ema21,  # Short EMA above long EMA
                ema21 > ema50,  # Medium EMA above long EMA
                rsi < 70,  # RSI not overbought
                current_price > bb_middle,  # Price above BB middle
                wma5 > wma10,  # Short WMA above long WMA
            ]
            
            # Bearish conditions
            bearish_conditions = [
                current_price < ma10,  # Price below MA10
                ema9 < ema21,  # Short EMA below long EMA
                ema21 < ema50,  # Medium EMA below long EMA
                rsi > 30,  # RSI not oversold
                current_price < bb_middle,  # Price below BB middle
                wma5 < wma10,  # Short WMA below long WMA
            ]
            
            bullish_score = sum(bullish_conditions)
            bearish_score = sum(bearish_conditions)
            
            # Determine signal based on conditions
            if bullish_score >= self.config.SKOR_MINIMAL and bullish_score > bearish_score:
                signal = "BUY"
                confidence = bullish_score / len(bullish_conditions) * 100
            elif bearish_score >= self.config.SKOR_MINIMAL and bearish_score > bullish_score:
                signal = "SELL"
                confidence = bearish_score / len(bearish_conditions) * 100
            
            return {
                'signal': signal,
                'confidence': confidence,
                'price': current_price,
                'indicators': {
                    'ma10': ma10, 'ma20': ma20,
                    'ema9': ema9, 'ema21': ema21, 'ema50': ema50,
                    'wma5': wma5, 'wma10': wma10,
                    'rsi': rsi,
                    'bb_upper': bb_upper, 'bb_lower': bb_lower, 'bb_middle': bb_middle
                }
            }
            
        except Exception as e:
            self.log(f"Error in market analysis: {str(e)}", "ERROR")
            return None

    def calculate_position_size(self, symbol, risk_percent=1.0):
        """Calculate optimal position size based on risk management"""
        try:
            account_info = mt5.account_info()
            if not account_info:
                return float(self.lot_var.get())
            
            balance = account_info.balance
            risk_amount = balance * (risk_percent / 100)
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return float(self.lot_var.get())
            
            # Calculate based on stop loss distance
            sl_pips = self.config.SL_PERSEN_DEFAULT * 100 * 100  # Convert to pips
            pip_value = symbol_info.trade_tick_value
            
            if pip_value > 0 and sl_pips > 0:
                position_size = risk_amount / (sl_pips * pip_value)
                # Ensure it's within allowed range
                min_lot = symbol_info.volume_min
                max_lot = min(symbol_info.volume_max, float(self.lot_var.get()) * 2)
                
                return max(min_lot, min(max_lot, round(position_size, 2)))
            
            return float(self.lot_var.get())
            
        except Exception as e:
            self.log(f"Error calculating position size: {str(e)}", "ERROR")
            return float(self.lot_var.get())

    def execute_trade(self, signal, analysis_data):
        """Execute a trade based on signal"""
        try:
            symbol = self.symbol_var.get()
            
            # Check if we can place more orders
            if not self.risk_manager.can_place_order():
                self.log("Order limit reached or risk conditions not met", "WARNING")
                return False
            
            # Get current price
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                self.log(f"Failed to get tick data for {symbol}", "ERROR")
                return False
            
            # Calculate lot size
            lot_size = self.calculate_position_size(symbol)
            
            # Determine order type and prices
            if signal == "BUY":
                order_type = mt5.ORDER_TYPE_BUY
                price = tick.ask
                
                # Calculate TP and SL
                if self.scalping_mode_var.get():
                    tp = price * (1 + self.config.SCALPING_TP_PERSEN)
                    sl = price * (1 - self.config.SCALPING_SL_PERSEN)
                else:
                    tp = price * (1 + float(self.tp_var.get()) / 100)
                    sl = price * (1 - float(self.sl_var.get()) / 100)
                    
            elif signal == "SELL":
                order_type = mt5.ORDER_TYPE_SELL
                price = tick.bid
                
                # Calculate TP and SL
                if self.scalping_mode_var.get():
                    tp = price * (1 - self.config.SCALPING_TP_PERSEN)
                    sl = price * (1 + self.config.SCALPING_SL_PERSEN)
                else:
                    tp = price * (1 - float(self.tp_var.get()) / 100)
                    sl = price * (1 + float(self.sl_var.get()) / 100)
            else:
                return False
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": 20,
                "magic": 234000,
                "comment": f"AutoBot_{signal}_{int(time.time())}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Execute order
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.order_counter += 1
                trade_info = f"{signal} Order Executed: {price:.5f} | TP: {tp:.5f} | SL: {sl:.5f} | Lot: {lot_size}"
                self.log(trade_info, "TRADE")
                
                # Log trade details
                self.logger.export_trade_log(price, tp, sl, signal, lot_size, analysis_data['confidence'])
                
                # Update risk manager
                self.risk_manager.add_position(result.order, signal, lot_size, price, tp, sl)
                
                return True
            else:
                error_msg = f"Order failed: {result.retcode} - {result.comment}"
                self.log(error_msg, "ERROR")
                return False
                
        except Exception as e:
            self.log(f"Error executing trade: {str(e)}", "ERROR")
            return False

    def manage_positions(self):
        """Manage existing positions with trailing stops"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return
            
            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                
                # Get current price
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    continue
                
                current_price = tick.bid if position.type == 0 else tick.ask
                
                # Calculate trailing stop
                new_sl = self.risk_manager.calculate_trailing_stop(position, current_price)
                
                if new_sl and new_sl != position.sl:
                    # Modify stop loss
                    request = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "sl": new_sl,
                        "tp": position.tp
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        self.log(f"Trailing stop updated for {symbol}: {new_sl:.5f}", "INFO")
                    
        except Exception as e:
            self.log(f"Error managing positions: {str(e)}", "ERROR")

    def update_profit_display(self):
        """Update real-time profit display"""
        try:
            account_info = mt5.account_info()
            if account_info and self.modal_awal:
                current_equity = account_info.equity
                profit = current_equity - self.modal_awal
                profit_percent = (profit / self.modal_awal) * 100 if self.modal_awal > 0 else 0
                
                profit_text = f"P/L: {profit:,.2f} ({profit_percent:.2f}%)"
                self.profit_var.set(profit_text)
                
                # Check profit target
                if profit_percent >= self.config.TARGET_PROFIT_PERSEN:
                    self.log(f"Profit target reached: {profit_percent:.2f}%", "ALERT")
                    if self.auto_close_var.get():
                        self.close_all_positions()
                        
        except Exception as e:
            self.log(f"Error updating profit display: {str(e)}", "ERROR")

    def trading_loop(self):
        """Main trading loop"""
        self.log("Trading bot started", "INFO")
        self.telegram.send_message("🤖 Trading bot started")
        
        while self.running:
            try:
                # Check MT5 connection
                if not mt5.terminal_info():
                    self.log("MT5 connection lost, attempting reconnection...", "WARNING")
                    if not self.connect_mt5():
                        self.log("Reconnection failed, stopping bot", "ERROR")
                        break
                
                # Check trading hours
                current_hour = datetime.datetime.now().hour
                if current_hour < self.config.TRADING_START_HOUR or current_hour >= self.config.TRADING_END_HOUR:
                    if current_hour == self.config.TRADING_END_HOUR and self.auto_close_var.get():
                        self.close_all_positions()
                    time.sleep(60)  # Check every minute during off hours
                    continue
                
                # Get market data
                symbol = self.symbol_var.get()
                data = self.get_market_data(symbol)
                if data is None:
                    time.sleep(int(self.interval_var.get()))
                    continue
                
                # Analyze market
                analysis = self.analyze_market(data)
                if analysis is None:
                    time.sleep(int(self.interval_var.get()))
                    continue
                
                # Update profit display
                self.update_profit_display()
                
                # Manage existing positions
                self.manage_positions()
                
                # Check for new trade signals
                if analysis['signal'] and analysis['confidence'] >= 70:  # Minimum 70% confidence
                    if self.execute_trade(analysis['signal'], analysis):
                        # Wait a bit longer after placing a trade
                        time.sleep(int(self.interval_var.get()) * 2)
                    else:
                        time.sleep(int(self.interval_var.get()))
                else:
                    time.sleep(int(self.interval_var.get()))
                
            except Exception as e:
                self.log(f"Error in trading loop: {str(e)}", "ERROR")
                time.sleep(int(self.interval_var.get()))
        
        self.log("Trading bot stopped", "INFO")
        self.telegram.send_message("🛑 Trading bot stopped")

    def start_bot(self):
        """Start the trading bot"""
        if not mt5.terminal_info():
            messagebox.showerror("Error", "Please connect to MT5 first")
            return
        
        if self.running:
            messagebox.showwarning("Warning", "Bot is already running")
            return
        
        # Validate settings
        try:
            float(self.lot_var.get())
            int(self.interval_var.get())
            float(self.tp_var.get())
            float(self.sl_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please check your trading settings")
            return
        
        # Reset counters
        self.order_counter = 0
        self.running = True
        
        # Update UI
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_var.set("Status: Running")
        
        # Start trading thread
        self.bot_thread = threading.Thread(target=self.trading_loop, daemon=True)
        self.bot_thread.start()
        
        # Start GUI update thread
        self.gui_update_thread = threading.Thread(target=self.gui_update_loop, daemon=True)
        self.gui_update_thread.start()

    def stop_bot(self):
        """Stop the trading bot"""
        self.running = False
        
        # Update UI
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Status: Stopped")
        
        self.log("Bot stop requested", "INFO")

    def gui_update_loop(self):
        """Update GUI elements periodically"""
        while self.running:
            try:
                self.root.after(0, self.update_profit_display)
                time.sleep(5)  # Update every 5 seconds
            except:
                break

    def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = mt5.positions_get()
            if not positions:
                self.log("No positions to close", "INFO")
                return
            
            closed_count = 0
            for position in positions:
                symbol = position.symbol
                ticket = position.ticket
                volume = position.volume
                
                if position.type == 0:  # Buy position
                    order_type = mt5.ORDER_TYPE_SELL
                    price = mt5.symbol_info_tick(symbol).bid
                else:  # Sell position
                    order_type = mt5.ORDER_TYPE_BUY
                    price = mt5.symbol_info_tick(symbol).ask
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "position": ticket,
                    "symbol": symbol,
                    "volume": volume,
                    "type": order_type,
                    "price": price,
                    "deviation": 20,
                    "magic": 234000,
                    "comment": "CloseAll",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    closed_count += 1
                    
            self.log(f"Closed {closed_count} positions", "INFO")
            self.telegram.send_message(f"❌ Closed {closed_count} positions")
            
        except Exception as e:
            self.log(f"Error closing positions: {str(e)}", "ERROR")

    def manual_close_all(self):
        """Manual close all positions"""
        if messagebox.askyesno("Confirm", "Close all open positions?"):
            self.close_all_positions()

    def save_settings(self):
        """Save current settings to file"""
        try:
            settings = {
                'symbol': self.symbol_var.get(),
                'lot': self.lot_var.get(),
                'interval': self.interval_var.get(),
                'tp': self.tp_var.get(),
                'sl': self.sl_var.get(),
                'scalping_tp': self.scalping_tp_var.get(),
                'scalping_sl': self.scalping_sl_var.get(),
                'scalping_mode': self.scalping_mode_var.get(),
                'auto_close': self.auto_close_var.get(),
            }
            
            with open('bot_settings.json', 'w') as f:
                json.dump(settings, f, indent=2)
            
            self.log("Settings saved", "INFO")
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            self.log(f"Error saving settings: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists('bot_settings.json'):
                with open('bot_settings.json', 'r') as f:
                    settings = json.load(f)
                
                self.symbol_var.set(settings.get('symbol', 'XAUUSDm'))
                self.lot_var.set(settings.get('lot', '0.01'))
                self.interval_var.set(settings.get('interval', '10'))
                self.tp_var.set(settings.get('tp', '1.0'))
                self.sl_var.set(settings.get('sl', '5.0'))
                self.scalping_tp_var.set(settings.get('scalping_tp', '0.5'))
                self.scalping_sl_var.set(settings.get('scalping_sl', '1.0'))
                self.scalping_mode_var.set(settings.get('scalping_mode', True))
                self.auto_close_var.set(settings.get('auto_close', True))
                
                self.log("Settings loaded", "INFO")
                
        except Exception as e:
            self.log(f"Error loading settings: {str(e)}", "ERROR")

    def export_logs(self):
        """Export trading logs"""
        try:
            self.logger.export_all_logs()
            self.log("Logs exported", "INFO")
            messagebox.showinfo("Success", "Logs exported successfully")
        except Exception as e:
            self.log(f"Error exporting logs: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Failed to export logs: {str(e)}")

    def on_closing(self):
        """Handle window closing"""
        if self.running:
            if messagebox.askyesno("Confirm", "Bot is running. Stop and exit?"):
                self.stop_bot()
                time.sleep(1)  # Give time for threads to stop
                self.disconnect_mt5()
                self.root.destroy()
        else:
            self.disconnect_mt5()
            self.root.destroy()

    def run(self):
        """Start the application"""
        self.log("Trading Bot initialized", "INFO")
        self.root.mainloop()

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
