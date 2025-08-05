"""
Enhanced Configuration Manager for Windows Trading Bot
Manages all bot settings with GUI interface
"""

import os
import json
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from config import config

class ConfigManager:
    def __init__(self):
        self.config_file = "bot_config.json"
        self.backup_file = "bot_config_backup.json"
        self.settings = self.load_settings()
        
    def load_settings(self):
        """Load settings from file or create defaults"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    settings = json.load(f)
                print("✅ Configuration loaded")
                return settings
            else:
                # Create default settings
                default_settings = self.get_default_settings()
                self.save_settings(default_settings)
                return default_settings
                
        except Exception as e:
            print(f"⚠️ Error loading config: {e}")
            return self.get_default_settings()
    
    def get_default_settings(self):
        """Get default configuration settings"""
        return {
            "trading": {
                "tp_percent": config.TP_PERSEN_DEFAULT,
                "sl_percent": config.SL_PERSEN_DEFAULT,
                "scalping_tp": config.SCALPING_TP_PERSEN,
                "scalping_sl": config.SCALPING_SL_PERSEN,
                "max_orders": config.MAX_ORDER_PER_SESSION,
                "minimum_balance": config.SALDO_MINIMAL,
                "risk_per_trade": config.MAX_RISK_PER_TRADE,
                "max_drawdown": config.MAX_DRAWDOWN,
                "default_symbol": config.DEFAULT_SYMBOL,
                "default_lot": config.DEFAULT_LOT,
                "scan_interval": config.DEFAULT_INTERVAL
            },
            "trading_hours": {
                "start_hour": config.TRADING_START_HOUR,
                "end_hour": config.TRADING_END_HOUR,
                "reset_hour": config.RESET_ORDER_HOUR
            },
            "indicators": {
                "ma_periods": config.MA_PERIODS,
                "ema_periods": config.EMA_PERIODS,
                "rsi_period": config.RSI_PERIOD,
                "rsi_oversold": config.RSI_OVERSOLD,
                "rsi_overbought": config.RSI_OVERBOUGHT,
                "bb_period": config.BB_PERIOD,
                "bb_deviation": config.BB_DEVIATION
            },
            "signals": {
                "confidence_threshold": config.SIGNAL_CONFIDENCE_THRESHOLD,
                "minimum_score": config.SKOR_MINIMAL,
                "spike_threshold": config.LONJAKAN_THRESHOLD,
                "trend_strength": config.TREND_STRENGTH_MIN
            },
            "telegram": {
                "bot_token": config.TELEGRAM_BOT_TOKEN,
                "chat_id": config.TELEGRAM_CHAT_ID,
                "enabled": False
            },
            "advanced": {
                "mt5_magic": config.MT5_MAGIC_NUMBER,
                "mt5_deviation": config.MT5_DEVIATION,
                "mt5_timeout": config.MT5_TIMEOUT,
                "price_fetch_retry": config.PRICE_FETCH_RETRY,
                "data_buffer_size": config.DATA_BUFFER_SIZE
            },
            "ui": {
                "auto_scroll": True,
                "sound_alerts": True,
                "desktop_notifications": True,
                "theme": "default"
            },
            "meta": {
                "created": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
    
    def save_settings(self, settings=None):
        """Save settings to file with backup"""
        try:
            if settings is None:
                settings = self.settings
            
            # Create backup of existing config
            if os.path.exists(self.config_file):
                import shutil
                shutil.copy2(self.config_file, self.backup_file)
            
            # Update metadata
            settings["meta"]["last_modified"] = datetime.now().isoformat()
            
            # Save new settings
            with open(self.config_file, 'w') as f:
                json.dump(settings, f, indent=2)
            
            self.settings = settings
            print("✅ Configuration saved")
            return True
            
        except Exception as e:
            print(f"❌ Error saving config: {e}")
            return False
    
    def restore_backup(self):
        """Restore configuration from backup"""
        try:
            if os.path.exists(self.backup_file):
                import shutil
                shutil.copy2(self.backup_file, self.config_file)
                self.settings = self.load_settings()
                print("✅ Configuration restored from backup")
                return True
            else:
                print("⚠️ No backup file found")
                return False
                
        except Exception as e:
            print(f"❌ Error restoring backup: {e}")
            return False
    
    def get_setting(self, category, key, default=None):
        """Get specific setting value"""
        try:
            return self.settings.get(category, {}).get(key, default)
        except:
            return default
    
    def set_setting(self, category, key, value):
        """Set specific setting value"""
        try:
            if category not in self.settings:
                self.settings[category] = {}
            self.settings[category][key] = value
            return True
        except Exception as e:
            print(f"❌ Error setting config: {e}")
            return False
    
    def export_settings(self, filename):
        """Export settings to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.settings, f, indent=2)
            print(f"✅ Settings exported to {filename}")
            return True
        except Exception as e:
            print(f"❌ Export error: {e}")
            return False
    
    def import_settings(self, filename):
        """Import settings from file"""
        try:
            with open(filename, 'r') as f:
                imported_settings = json.load(f)
            
            # Validate imported settings
            if self.validate_settings(imported_settings):
                self.settings = imported_settings
                self.save_settings()
                print(f"✅ Settings imported from {filename}")
                return True
            else:
                print("❌ Invalid settings file")
                return False
                
        except Exception as e:
            print(f"❌ Import error: {e}")
            return False
    
    def validate_settings(self, settings):
        """Validate settings structure"""
        try:
            required_categories = ["trading", "trading_hours", "indicators", "signals"]
            
            for category in required_categories:
                if category not in settings:
                    return False
            
            # Basic validation for critical values
            trading = settings.get("trading", {})
            if trading.get("max_orders", 0) <= 0:
                return False
            if trading.get("minimum_balance", 0) <= 0:
                return False
                
            return True
            
        except:
            return False
    
    def open_config_gui(self):
        """Open configuration GUI"""
        ConfigGUI(self)

class ConfigGUI:
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.root = tk.Toplevel()
        self.root.title("🔧 Trading Bot Configuration")
        self.root.geometry("800x700")
        self.root.transient()
        
        self.setup_gui()
        self.load_current_settings()
        
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
    
    def setup_gui(self):
        """Setup the configuration GUI"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Trading Settings Tab
        self.trading_frame = ttk.Frame(notebook)
        notebook.add(self.trading_frame, text="Trading")
        self.setup_trading_tab()
        
        # Indicators Tab
        self.indicators_frame = ttk.Frame(notebook)
        notebook.add(self.indicators_frame, text="Indicators")
        self.setup_indicators_tab()
        
        # Signals Tab
        self.signals_frame = ttk.Frame(notebook)
        notebook.add(self.signals_frame, text="Signals")
        self.setup_signals_tab()
        
        # Telegram Tab
        self.telegram_frame = ttk.Frame(notebook)
        notebook.add(self.telegram_frame, text="Telegram")
        self.setup_telegram_tab()
        
        # Advanced Tab
        self.advanced_frame = ttk.Frame(notebook)
        notebook.add(self.advanced_frame, text="Advanced")
        self.setup_advanced_tab()
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.root)
        buttons_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Button(buttons_frame, text="Save", command=self.save_settings).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=self.root.destroy).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side="left", padx=5)
        ttk.Button(buttons_frame, text="Export", command=self.export_config).pack(side="right", padx=5)
        ttk.Button(buttons_frame, text="Import", command=self.import_config).pack(side="right", padx=5)
    
    def setup_trading_tab(self):
        """Setup trading settings tab"""
        # Create scrollable frame
        canvas = tk.Canvas(self.trading_frame)
        scrollbar = ttk.Scrollbar(self.trading_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Risk Management
        risk_frame = ttk.LabelFrame(scrollable_frame, text="Risk Management", padding=10)
        risk_frame.pack(fill="x", padx=5, pady=5)
        
        self.tp_percent = self.create_setting_entry(risk_frame, "Take Profit %:", 0, 0)
        self.sl_percent = self.create_setting_entry(risk_frame, "Stop Loss %:", 0, 1)
        self.risk_per_trade = self.create_setting_entry(risk_frame, "Risk per Trade %:", 1, 0)
        self.max_drawdown = self.create_setting_entry(risk_frame, "Max Drawdown %:", 1, 1)
        
        # Position Management
        position_frame = ttk.LabelFrame(scrollable_frame, text="Position Management", padding=10)
        position_frame.pack(fill="x", padx=5, pady=5)
        
        self.max_orders = self.create_setting_entry(position_frame, "Max Orders per Session:", 0, 0)
        self.minimum_balance = self.create_setting_entry(position_frame, "Minimum Balance:", 0, 1)
        self.default_lot = self.create_setting_entry(position_frame, "Default Lot Size:", 1, 0)
        self.scan_interval = self.create_setting_entry(position_frame, "Scan Interval (seconds):", 1, 1)
        
        # Symbol Settings
        symbol_frame = ttk.LabelFrame(scrollable_frame, text="Symbol Settings", padding=10)
        symbol_frame.pack(fill="x", padx=5, pady=5)
        
        self.default_symbol = self.create_setting_entry(symbol_frame, "Default Symbol:", 0, 0, width=15)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_indicators_tab(self):
        """Setup indicators settings tab"""
        # RSI Settings
        rsi_frame = ttk.LabelFrame(self.indicators_frame, text="RSI Settings", padding=10)
        rsi_frame.pack(fill="x", padx=5, pady=5)
        
        self.rsi_period = self.create_setting_entry(rsi_frame, "RSI Period:", 0, 0)
        self.rsi_oversold = self.create_setting_entry(rsi_frame, "Oversold Level:", 0, 1)
        self.rsi_overbought = self.create_setting_entry(rsi_frame, "Overbought Level:", 1, 0)
        
        # Bollinger Bands
        bb_frame = ttk.LabelFrame(self.indicators_frame, text="Bollinger Bands", padding=10)
        bb_frame.pack(fill="x", padx=5, pady=5)
        
        self.bb_period = self.create_setting_entry(bb_frame, "BB Period:", 0, 0)
        self.bb_deviation = self.create_setting_entry(bb_frame, "BB Deviation:", 0, 1)
    
    def setup_signals_tab(self):
        """Setup signals settings tab"""
        signals_frame = ttk.LabelFrame(self.signals_frame, text="Signal Generation", padding=10)
        signals_frame.pack(fill="x", padx=5, pady=5)
        
        self.confidence_threshold = self.create_setting_entry(signals_frame, "Confidence Threshold:", 0, 0)
        self.minimum_score = self.create_setting_entry(signals_frame, "Minimum Score:", 0, 1)
        self.spike_threshold = self.create_setting_entry(signals_frame, "Spike Threshold %:", 1, 0)
        self.trend_strength = self.create_setting_entry(signals_frame, "Trend Strength:", 1, 1)
    
    def setup_telegram_tab(self):
        """Setup telegram settings tab"""
        telegram_frame = ttk.LabelFrame(self.telegram_frame, text="Telegram Notifications", padding=10)
        telegram_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(telegram_frame, text="Bot Token:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.bot_token = ttk.Entry(telegram_frame, width=50, show="*")
        self.bot_token.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(telegram_frame, text="Chat ID:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.chat_id = ttk.Entry(telegram_frame, width=30)
        self.chat_id.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        self.telegram_enabled = tk.BooleanVar()
        ttk.Checkbutton(telegram_frame, text="Enable Telegram Notifications", 
                       variable=self.telegram_enabled).grid(row=2, column=0, columnspan=2, pady=10)
    
    def setup_advanced_tab(self):
        """Setup advanced settings tab"""
        mt5_frame = ttk.LabelFrame(self.advanced_frame, text="MetaTrader5 Settings", padding=10)
        mt5_frame.pack(fill="x", padx=5, pady=5)
        
        self.mt5_magic = self.create_setting_entry(mt5_frame, "Magic Number:", 0, 0)
        self.mt5_deviation = self.create_setting_entry(mt5_frame, "Deviation:", 0, 1)
        self.mt5_timeout = self.create_setting_entry(mt5_frame, "Timeout (ms):", 1, 0)
    
    def create_setting_entry(self, parent, label, row, col, width=10):
        """Create a labeled entry widget"""
        ttk.Label(parent, text=label).grid(row=row, column=col*2, sticky="w", padx=5, pady=2)
        entry = ttk.Entry(parent, width=width)
        entry.grid(row=row, column=col*2+1, padx=5, pady=2, sticky="w")
        return entry
    
    def load_current_settings(self):
        """Load current settings into GUI"""
        settings = self.config_manager.settings
        
        # Trading settings
        self.tp_percent.insert(0, str(settings["trading"]["tp_percent"]))
        self.sl_percent.insert(0, str(settings["trading"]["sl_percent"]))
        self.risk_per_trade.insert(0, str(settings["trading"]["risk_per_trade"]))
        self.max_drawdown.insert(0, str(settings["trading"]["max_drawdown"]))
        self.max_orders.insert(0, str(settings["trading"]["max_orders"]))
        self.minimum_balance.insert(0, str(settings["trading"]["minimum_balance"]))
        self.default_lot.insert(0, str(settings["trading"]["default_lot"]))
        self.scan_interval.insert(0, str(settings["trading"]["scan_interval"]))
        self.default_symbol.insert(0, str(settings["trading"]["default_symbol"]))
        
        # Indicators
        self.rsi_period.insert(0, str(settings["indicators"]["rsi_period"]))
        self.rsi_oversold.insert(0, str(settings["indicators"]["rsi_oversold"]))
        self.rsi_overbought.insert(0, str(settings["indicators"]["rsi_overbought"]))
        self.bb_period.insert(0, str(settings["indicators"]["bb_period"]))
        self.bb_deviation.insert(0, str(settings["indicators"]["bb_deviation"]))
        
        # Signals
        self.confidence_threshold.insert(0, str(settings["signals"]["confidence_threshold"]))
        self.minimum_score.insert(0, str(settings["signals"]["minimum_score"]))
        self.spike_threshold.insert(0, str(settings["signals"]["spike_threshold"]))
        self.trend_strength.insert(0, str(settings["signals"]["trend_strength"]))
        
        # Telegram
        self.bot_token.insert(0, settings["telegram"]["bot_token"])
        self.chat_id.insert(0, settings["telegram"]["chat_id"])
        self.telegram_enabled.set(settings["telegram"]["enabled"])
        
        # Advanced
        self.mt5_magic.insert(0, str(settings["advanced"]["mt5_magic"]))
        self.mt5_deviation.insert(0, str(settings["advanced"]["mt5_deviation"]))
        self.mt5_timeout.insert(0, str(settings["advanced"]["mt5_timeout"]))
    
    def save_settings(self):
        """Save settings from GUI"""
        try:
            settings = self.config_manager.settings
            
            # Update settings from GUI
            settings["trading"]["tp_percent"] = float(self.tp_percent.get())
            settings["trading"]["sl_percent"] = float(self.sl_percent.get())
            settings["trading"]["risk_per_trade"] = float(self.risk_per_trade.get())
            settings["trading"]["max_drawdown"] = float(self.max_drawdown.get())
            settings["trading"]["max_orders"] = int(self.max_orders.get())
            settings["trading"]["minimum_balance"] = float(self.minimum_balance.get())
            settings["trading"]["default_lot"] = float(self.default_lot.get())
            settings["trading"]["scan_interval"] = int(self.scan_interval.get())
            settings["trading"]["default_symbol"] = self.default_symbol.get()
            
            # Save and close
            if self.config_manager.save_settings(settings):
                messagebox.showinfo("Success", "Settings saved successfully!")
                self.root.destroy()
            else:
                messagebox.showerror("Error", "Failed to save settings!")
                
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please check your input values: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving settings: {e}")
    
    def reset_defaults(self):
        """Reset to default settings"""
        if messagebox.askyesno("Reset", "Reset all settings to defaults?"):
            self.config_manager.settings = self.config_manager.get_default_settings()
            self.root.destroy()
            self.config_manager.open_config_gui()
    
    def export_config(self):
        """Export configuration"""
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            if self.config_manager.export_settings(filename):
                messagebox.showinfo("Success", f"Settings exported to {filename}")
            else:
                messagebox.showerror("Error", "Export failed!")
    
    def import_config(self):
        """Import configuration"""
        from tkinter import filedialog
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            if self.config_manager.import_settings(filename):
                messagebox.showinfo("Success", "Settings imported successfully!")
                self.root.destroy()
                self.config_manager.open_config_gui()
            else:
                messagebox.showerror("Error", "Import failed or invalid file!")

# Global instance
config_manager = ConfigManager()