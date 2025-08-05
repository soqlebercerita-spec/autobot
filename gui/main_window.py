"""
Main GUI Window for AuraTrade Bot
Tkinter-based graphical user interface
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional
import queue

from core.trading_engine import TradingEngine
from core.mt5_connector import MT5Connector
from gui.dashboard import TradingDashboard
from gui.charts import ChartWidget
from utils.logger import Logger
from utils.notifier import Notifier

class MainWindow:
    """Main application window"""
    
    def __init__(self, trading_engine: TradingEngine, mt5_connector: MT5Connector):
        self.trading_engine = trading_engine
        self.mt5_connector = mt5_connector
        self.logger = Logger()
        self.notifier = Notifier()
        
        # GUI state
        self.root = None
        self.running = False
        self.update_queue = queue.Queue()
        
        # GUI components
        self.dashboard = None
        self.chart_widget = None
        self.log_text = None
        self.status_bar = None
        
        # Update intervals
        self.fast_update_interval = 1000  # 1 second for critical data
        self.slow_update_interval = 5000  # 5 seconds for charts
        
        # Style configuration
        self.style_config = {
            'bg_color': '#1e1e1e',
            'fg_color': '#ffffff',
            'accent_color': '#0078d4',
            'success_color': '#107c10',
            'warning_color': '#ffb900',
            'error_color': '#d83b01',
            'font_family': 'Segoe UI',
            'font_size': 9
        }
        
        self.logger.info("Main Window initialized")
    
    def create_window(self):
        """Create and configure main window"""
        try:
            self.root = tk.Tk()
            self.root.title("AuraTrade Bot - Professional Trading System")
            self.root.geometry("1400x900")
            self.root.minsize(1200, 800)
            
            # Configure window icon (if available)
            try:
                self.root.iconbitmap("assets/icon.ico")
            except:
                pass  # Icon not available
            
            # Configure style
            self._configure_style()
            
            # Create menu bar
            self._create_menu_bar()
            
            # Create main layout
            self._create_main_layout()
            
            # Configure window close event
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            
            # Start update loops
            self._start_update_loops()
            
            self.logger.info("Main window created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating main window: {str(e)}")
            raise
    
    def _configure_style(self):
        """Configure GUI styling"""
        try:
            # Configure root window
            self.root.configure(bg=self.style_config['bg_color'])
            
            # Configure ttk style
            style = ttk.Style()
            style.theme_use('clam')  # Use clam theme as base
            
            # Configure colors for dark theme
            style.configure('TFrame', background=self.style_config['bg_color'])
            style.configure('TLabel', 
                          background=self.style_config['bg_color'], 
                          foreground=self.style_config['fg_color'],
                          font=(self.style_config['font_family'], self.style_config['font_size']))
            
            style.configure('TButton',
                          font=(self.style_config['font_family'], self.style_config['font_size']))
            
            style.configure('Treeview',
                          background='#2d2d2d',
                          foreground=self.style_config['fg_color'],
                          fieldbackground='#2d2d2d',
                          font=(self.style_config['font_family'], self.style_config['font_size']))
            
            style.configure('Treeview.Heading',
                          background='#404040',
                          foreground=self.style_config['fg_color'],
                          font=(self.style_config['font_family'], self.style_config['font_size'], 'bold'))
            
        except Exception as e:
            self.logger.error(f"Error configuring style: {str(e)}")
    
    def _create_menu_bar(self):
        """Create application menu bar"""
        try:
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            
            # File menu
            file_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="File", menu=file_menu)
            file_menu.add_command(label="Export Data", command=self._export_data)
            file_menu.add_command(label="Import Settings", command=self._import_settings)
            file_menu.add_command(label="Export Settings", command=self._export_settings)
            file_menu.add_separator()
            file_menu.add_command(label="Exit", command=self._on_closing)
            
            # Trading menu
            trading_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Trading", menu=trading_menu)
            trading_menu.add_command(label="Start Trading", command=self._start_trading)
            trading_menu.add_command(label="Stop Trading", command=self._stop_trading)
            trading_menu.add_command(label="Emergency Stop", command=self._emergency_stop)
            trading_menu.add_separator()
            trading_menu.add_command(label="Close All Positions", command=self._close_all_positions)
            
            # Tools menu
            tools_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Tools", menu=tools_menu)
            tools_menu.add_command(label="Market Analysis", command=self._show_market_analysis)
            tools_menu.add_command(label="Performance Report", command=self._show_performance_report)
            tools_menu.add_command(label="Risk Assessment", command=self._show_risk_assessment)
            tools_menu.add_separator()
            tools_menu.add_command(label="Settings", command=self._show_settings)
            
            # Help menu
            help_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Help", menu=help_menu)
            help_menu.add_command(label="User Guide", command=self._show_user_guide)
            help_menu.add_command(label="About", command=self._show_about)
            
        except Exception as e:
            self.logger.error(f"Error creating menu bar: {str(e)}")
    
    def _create_main_layout(self):
        """Create main window layout"""
        try:
            # Create main container
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Create paned window for resizable sections
            main_paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
            main_paned.pack(fill=tk.BOTH, expand=True)
            
            # Left panel (Dashboard and Controls)
            left_frame = ttk.Frame(main_paned)
            main_paned.add(left_frame, weight=2)
            
            # Right panel (Charts and Analysis)
            right_frame = ttk.Frame(main_paned)
            main_paned.add(right_frame, weight=3)
            
            # Create dashboard in left panel
            self.dashboard = TradingDashboard(left_frame, self.trading_engine, self.mt5_connector)
            
            # Create chart widget in right panel
            self.chart_widget = ChartWidget(right_frame, self.mt5_connector)
            
            # Create bottom panel for logs
            bottom_frame = ttk.Frame(self.root)
            bottom_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
            
            # Create log panel
            self._create_log_panel(bottom_frame)
            
            # Create status bar
            self._create_status_bar()
            
        except Exception as e:
            self.logger.error(f"Error creating main layout: {str(e)}")
    
    def _create_log_panel(self, parent):
        """Create log display panel"""
        try:
            # Log frame
            log_frame = ttk.LabelFrame(parent, text="System Log")
            log_frame.pack(fill=tk.BOTH, expand=True)
            
            # Create text widget with scrollbar
            text_frame = ttk.Frame(log_frame)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            self.log_text = scrolledtext.ScrolledText(
                text_frame,
                height=8,
                bg='#1e1e1e',
                fg='#ffffff',
                font=(self.style_config['font_family'], 8),
                wrap=tk.WORD
            )
            self.log_text.pack(fill=tk.BOTH, expand=True)
            
            # Log control buttons
            button_frame = ttk.Frame(log_frame)
            button_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
            
            ttk.Button(button_frame, text="Clear Log", 
                      command=self._clear_log).pack(side=tk.LEFT, padx=(0, 5))
            ttk.Button(button_frame, text="Save Log", 
                      command=self._save_log).pack(side=tk.LEFT, padx=(0, 5))
            
            # Auto-scroll checkbox
            self.auto_scroll_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(button_frame, text="Auto Scroll", 
                           variable=self.auto_scroll_var).pack(side=tk.RIGHT)
            
        except Exception as e:
            self.logger.error(f"Error creating log panel: {str(e)}")
    
    def _create_status_bar(self):
        """Create status bar"""
        try:
            self.status_bar = ttk.Frame(self.root)
            self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
            
            # Status labels
            self.connection_status = ttk.Label(self.status_bar, text="Disconnected", 
                                             foreground=self.style_config['error_color'])
            self.connection_status.pack(side=tk.LEFT, padx=5)
            
            ttk.Separator(self.status_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
            
            self.trading_status = ttk.Label(self.status_bar, text="Stopped", 
                                          foreground=self.style_config['error_color'])
            self.trading_status.pack(side=tk.LEFT, padx=5)
            
            ttk.Separator(self.status_bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
            
            self.time_label = ttk.Label(self.status_bar, text="")
            self.time_label.pack(side=tk.RIGHT, padx=5)
            
        except Exception as e:
            self.logger.error(f"Error creating status bar: {str(e)}")
    
    def _start_update_loops(self):
        """Start GUI update loops"""
        try:
            # Start fast update loop
            self._fast_update()
            
            # Start slow update loop
            self._slow_update()
            
            # Start log update loop  
            self._update_log()
            
        except Exception as e:
            self.logger.error(f"Error starting update loops: {str(e)}")
    
    def _fast_update(self):
        """Fast update loop for critical data"""
        try:
            if self.root and self.running:
                # Update connection status
                self._update_connection_status()
                
                # Update trading status
                self._update_trading_status()
                
                # Update time
                self._update_time()
                
                # Update dashboard
                if self.dashboard:
                    self.dashboard.update_fast()
                
                # Schedule next update
                self.root.after(self.fast_update_interval, self._fast_update)
                
        except Exception as e:
            self.logger.error(f"Error in fast update loop: {str(e)}")
            if self.root:
                self.root.after(self.fast_update_interval, self._fast_update)
    
    def _slow_update(self):
        """Slow update loop for charts and analysis"""
        try:
            if self.root and self.running:
                # Update charts
                if self.chart_widget:
                    self.chart_widget.update_data()
                
                # Update dashboard slow components
                if self.dashboard:
                    self.dashboard.update_slow()
                
                # Schedule next update
                self.root.after(self.slow_update_interval, self._slow_update)
                
        except Exception as e:
            self.logger.error(f"Error in slow update loop: {str(e)}")
            if self.root:
                self.root.after(self.slow_update_interval, self._slow_update)
    
    def _update_log(self):
        """Update log display"""
        try:
            if self.root and self.running and self.log_text:
                # Process log messages from queue
                try:
                    while True:
                        log_message = self.update_queue.get_nowait()
                        self._add_log_message(log_message)
                except queue.Empty:
                    pass
                
                # Schedule next update
                self.root.after(1000, self._update_log)  # 1 second
                
        except Exception as e:
            if self.root:
                self.root.after(1000, self._update_log)
    
    def _add_log_message(self, message: str):
        """Add message to log display"""
        try:
            if self.log_text:
                # Add timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] {message}\n"
                
                # Insert at end
                self.log_text.insert(tk.END, formatted_message)
                
                # Auto-scroll if enabled
                if self.auto_scroll_var.get():
                    self.log_text.see(tk.END)
                
                # Limit log size (keep last 1000 lines)
                lines = self.log_text.get("1.0", tk.END).split('\n')
                if len(lines) > 1000:
                    # Remove old lines
                    self.log_text.delete("1.0", f"{len(lines) - 1000}.0")
                
        except Exception as e:
            pass  # Avoid recursive logging errors
    
    def _update_connection_status(self):
        """Update MT5 connection status"""
        try:
            if self.mt5_connector and self.mt5_connector.is_connected():
                self.connection_status.config(
                    text="Connected", 
                    foreground=self.style_config['success_color']
                )
            else:
                self.connection_status.config(
                    text="Disconnected", 
                    foreground=self.style_config['error_color']
                )
        except Exception:
            pass
    
    def _update_trading_status(self):
        """Update trading engine status"""
        try:
            if self.trading_engine:
                engine_status = self.trading_engine.get_status()
                if engine_status.get('running', False):
                    if engine_status.get('emergency_stop', False):
                        self.trading_status.config(
                            text="Emergency Stop",
                            foreground=self.style_config['error_color']
                        )
                    else:
                        self.trading_status.config(
                            text="Running",
                            foreground=self.style_config['success_color']
                        )
                else:
                    self.trading_status.config(
                        text="Stopped",
                        foreground=self.style_config['warning_color']
                    )
        except Exception:
            pass
    
    def _update_time(self):
        """Update time display"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.time_label.config(text=current_time)
        except Exception:
            pass
    
    def run(self):
        """Start the GUI application"""
        try:
            self.create_window()
            self.running = True
            
            # Start trading engine if configured
            if self.trading_engine:
                self.trading_engine.start()
            
            # Start notification system
            if self.notifier:
                self.notifier.start()
            
            # Run main loop
            self.root.mainloop()
            
        except Exception as e:
            self.logger.error(f"Error running main window: {str(e)}")
            messagebox.showerror("Error", f"Failed to start application: {str(e)}")
    
    def _on_closing(self):
        """Handle window closing event"""
        try:
            result = messagebox.askyesno("Confirm Exit", 
                                       "Are you sure you want to exit AuraTrade Bot?")
            if result:
                self.running = False
                
                # Stop trading engine
                if self.trading_engine:
                    self.trading_engine.stop()
                
                # Stop notification system
                if self.notifier:
                    self.notifier.stop()
                
                # Destroy window
                if self.root:
                    self.root.destroy()
                
        except Exception as e:
            self.logger.error(f"Error closing application: {str(e)}")
            if self.root:
                self.root.destroy()
    
    # Menu command handlers
    def _start_trading(self):
        """Start trading engine"""
        try:
            if self.trading_engine:
                if self.trading_engine.start():
                    self._add_log_message("Trading engine started")
                    messagebox.showinfo("Success", "Trading engine started successfully")
                else:
                    messagebox.showerror("Error", "Failed to start trading engine")
        except Exception as e:
            messagebox.showerror("Error", f"Error starting trading: {str(e)}")
    
    def _stop_trading(self):
        """Stop trading engine"""
        try:
            if self.trading_engine:
                self.trading_engine.stop()
                self._add_log_message("Trading engine stopped")
                messagebox.showinfo("Success", "Trading engine stopped")
        except Exception as e:
            messagebox.showerror("Error", f"Error stopping trading: {str(e)}")
    
    def _emergency_stop(self):
        """Emergency stop all trading"""
        try:
            result = messagebox.askyesno("Emergency Stop", 
                                       "This will immediately stop all trading and close positions. Continue?")
            if result and self.trading_engine:
                self.trading_engine.force_emergency_stop("User requested emergency stop")
                self._add_log_message("EMERGENCY STOP ACTIVATED")
                messagebox.showwarning("Emergency Stop", "Emergency stop activated")
        except Exception as e:
            messagebox.showerror("Error", f"Error during emergency stop: {str(e)}")
    
    def _close_all_positions(self):
        """Close all open positions"""
        try:
            result = messagebox.askyesno("Close Positions", 
                                       "Close all open positions?")
            if result and self.mt5_connector:
                positions = self.mt5_connector.get_positions()
                closed_count = 0
                
                for position in positions:
                    if self.mt5_connector.close_position(position.ticket):
                        closed_count += 1
                
                self._add_log_message(f"Closed {closed_count} positions")
                messagebox.showinfo("Success", f"Closed {closed_count} positions")
        except Exception as e:
            messagebox.showerror("Error", f"Error closing positions: {str(e)}")
    
    def _export_data(self):
        """Export trading data"""
        try:
            # This would open a dialog for data export options
            messagebox.showinfo("Export Data", "Data export functionality coming soon")
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting data: {str(e)}")
    
    def _import_settings(self):
        """Import settings from file"""
        try:
            messagebox.showinfo("Import Settings", "Settings import functionality coming soon")
        except Exception as e:
            messagebox.showerror("Error", f"Error importing settings: {str(e)}")
    
    def _export_settings(self):
        """Export settings to file"""
        try:
            messagebox.showinfo("Export Settings", "Settings export functionality coming soon")
        except Exception as e:
            messagebox.showerror("Error", f"Error exporting settings: {str(e)}")
    
    def _show_market_analysis(self):
        """Show market analysis window"""
        try:
            messagebox.showinfo("Market Analysis", "Market analysis window coming soon")
        except Exception as e:
            messagebox.showerror("Error", f"Error showing market analysis: {str(e)}")
    
    def _show_performance_report(self):
        """Show performance report"""
        try:
            if self.trading_engine:
                status = self.trading_engine.get_status()
                performance = status.get('performance_metrics', {})
                
                report = f"""Performance Report
                
Total Trades: {performance.get('total_trades', 0)}
Winning Trades: {performance.get('winning_trades', 0)}
Win Rate: {performance.get('win_rate', 0):.1f}%
Total P&L: ${performance.get('total_profit', 0):.2f}
Max Drawdown: {performance.get('max_drawdown', 0):.2%}
                """
                
                messagebox.showinfo("Performance Report", report)
        except Exception as e:
            messagebox.showerror("Error", f"Error showing performance report: {str(e)}")
    
    def _show_risk_assessment(self):
        """Show risk assessment"""
        try:
            messagebox.showinfo("Risk Assessment", "Risk assessment window coming soon")
        except Exception as e:
            messagebox.showerror("Error", f"Error showing risk assessment: {str(e)}")
    
    def _show_settings(self):
        """Show settings window"""
        try:
            messagebox.showinfo("Settings", "Settings window coming soon")
        except Exception as e:
            messagebox.showerror("Error", f"Error showing settings: {str(e)}")
    
    def _show_user_guide(self):
        """Show user guide"""
        try:
            guide_text = """AuraTrade Bot User Guide

1. Connection:
   - Ensure MetaTrader 5 is installed and running
   - Configure MT5 credentials in settings
   - Check connection status in the status bar

2. Trading:
   - Use Trading menu to start/stop the bot
   - Monitor positions in the dashboard
   - Emergency stop available for immediate halt

3. Monitoring:
   - Dashboard shows real-time data
   - Charts display price action and indicators
   - System log shows all activities

4. Safety:
   - Always use stop losses
   - Monitor drawdown limits
   - Keep emergency stop accessible
            """
            
            messagebox.showinfo("User Guide", guide_text)
        except Exception as e:
            messagebox.showerror("Error", f"Error showing user guide: {str(e)}")
    
    def _show_about(self):
        """Show about dialog"""
        try:
            about_text = """AuraTrade Bot v1.0

Professional Algorithmic Trading System

Features:
• High-Frequency Trading (HFT)
• Multiple Trading Strategies
• Advanced Risk Management
• Real-time Market Analysis
• Custom Technical Indicators

Developed with Python and MetaTrader 5

⚠️ Trading involves risk. Past performance 
does not guarantee future results.
            """
            
            messagebox.showinfo("About AuraTrade Bot", about_text)
        except Exception as e:
            messagebox.showerror("Error", f"Error showing about dialog: {str(e)}")
    
    def _clear_log(self):
        """Clear log display"""
        try:
            if self.log_text:
                self.log_text.delete("1.0", tk.END)
        except Exception as e:
            self.logger.error(f"Error clearing log: {str(e)}")
    
    def _save_log(self):
        """Save log to file"""
        try:
            if self.log_text:
                from tkinter import filedialog
                
                filename = filedialog.asksaveasfilename(
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                )
                
                if filename:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(self.log_text.get("1.0", tk.END))
                    
                    messagebox.showinfo("Success", f"Log saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving log: {str(e)}")
    
    def add_log_message(self, message: str):
        """Public method to add log message from external sources"""
        try:
            self.update_queue.put(message)
        except Exception:
            pass  # Queue might be full, ignore
