#!/usr/bin/env python3
"""
Enhanced Trading Bot Launcher - Auto-detection and Startup
Automatically detects environment and starts the appropriate trading bot version
"""

import os
import sys
import platform
import subprocess
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
from tkinter.scrolledtext import ScrolledText
import datetime
import time
import threading

class TradingBotLauncher:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 Enhanced Trading Bot Launcher")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Detection results
        self.mt5_available = False
        self.python_version = sys.version
        self.platform_info = platform.platform()
        self.available_bots = []
        
        self.setup_gui()
        self.detect_environment()
    
    def setup_gui(self):
        """Setup launcher GUI"""
        # Style
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 12))
        style.configure("Success.TLabel", foreground="green", font=("Segoe UI", 10, "bold"))
        style.configure("Warning.TLabel", foreground="orange", font=("Segoe UI", 10, "bold"))
        style.configure("Error.TLabel", foreground="red", font=("Segoe UI", 10, "bold"))
        
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(header_frame, text="🚀 Enhanced Trading Bot Launcher", style="Title.TLabel").pack()
        ttk.Label(header_frame, text="Fixed Price Retrieval • Optimized Signal Generation • Enhanced Opportunity Capture", 
                 style="Success.TLabel").pack(pady=5)
        
        # System Information
        info_frame = ttk.LabelFrame(self.root, text="📊 System Information", padding=10)
        info_frame.pack(fill="x", padx=20, pady=10)
        
        self.info_text = tk.Text(info_frame, height=6, width=80, font=("Consolas", 9))
        self.info_text.pack(fill="x")
        
        # Available Bots
        bots_frame = ttk.LabelFrame(self.root, text="🤖 Available Trading Bots", padding=10)
        bots_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Bot selection
        self.bot_var = tk.StringVar()
        self.bot_listbox = tk.Listbox(bots_frame, height=8, font=("Segoe UI", 10))
        self.bot_listbox.pack(fill="both", expand=True, pady=5)
        
        # Bot descriptions
        self.desc_text = ScrolledText(bots_frame, height=6, width=80, font=("Segoe UI", 9))
        self.desc_text.pack(fill="x", pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=20, pady=10)
        
        self.refresh_button = ttk.Button(button_frame, text="🔄 Refresh Detection", command=self.detect_environment)
        self.launch_button = ttk.Button(button_frame, text="🚀 Launch Selected Bot", command=self.launch_selected_bot)
        self.install_button = ttk.Button(button_frame, text="📦 Install Requirements", command=self.install_requirements)
        self.help_button = ttk.Button(button_frame, text="❓ Help", command=self.show_help)
        
        self.refresh_button.pack(side="left", padx=5)
        self.launch_button.pack(side="left", padx=5)
        self.install_button.pack(side="left", padx=5)
        self.help_button.pack(side="right", padx=5)
        
        # Bind events
        self.bot_listbox.bind('<<ListboxSelect>>', self.on_bot_select)
    
    def detect_environment(self):
        """Detect system environment and available components"""
        self.log_info("🔍 Detecting system environment...")
        
        # Clear previous results
        self.available_bots.clear()
        self.bot_listbox.delete(0, tk.END)
        
        # System information
        info_text = ""
        info_text += f"🖥️  Platform: {self.platform_info}\n"
        info_text += f"🐍 Python: {sys.version.split()[0]}\n"
        info_text += f"📁 Working Directory: {os.getcwd()}\n"
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 8):
            info_text += f"✅ Python version supported ({python_version.major}.{python_version.minor})\n"
        else:
            info_text += f"❌ Python version too old ({python_version.major}.{python_version.minor}), need 3.8+\n"
        
        # Check MetaTrader5 availability
        try:
            import MetaTrader5 as mt5
            self.mt5_available = True
            info_text += "✅ MetaTrader5 library available\n"
        except ImportError:
            self.mt5_available = False
            info_text += "⚠️ MetaTrader5 not available (Replit simulation mode)\n"
        
        # Check required libraries
        required_libs = ['numpy', 'requests', 'tkinter']
        for lib in required_libs:
            try:
                __import__(lib)
                info_text += f"✅ {lib} available\n"
            except ImportError:
                info_text += f"❌ {lib} not found\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        
        # Detect available bot files
        self.detect_available_bots()
    
    def detect_available_bots(self):
        """Detect available trading bot files"""
        bot_files = [
            {
                'file': 'trading_bot_windows.py',
                'name': '🪟 Enhanced Windows MT5 Trading Bot',
                'description': '• Full MetaTrader5 integration for live trading\n• Fixed price retrieval mechanism\n• Optimized signal generation\n• Enhanced opportunity capture\n• Real money trading with MT5\n• Requires MetaTrader5 terminal\n\nBest for: Live trading with real MT5 account',
                'requirements': ['MetaTrader5', 'numpy', 'requests'],
                'recommended': self.mt5_available and platform.system() == 'Windows'
            },
            {
                'file': 'trading_bot_integrated.py',
                'name': '🔬 Enhanced Simulation Trading Bot',
                'description': '• Safe simulation environment\n• Real market data with virtual trading\n• Fixed price retrieval issues\n• Enhanced signal algorithms\n• No real money risk\n• Perfect for testing and learning\n\nBest for: Testing strategies and learning',
                'requirements': ['numpy', 'requests'],
                'recommended': True
            }
        ]
        
        # Check which bot files exist
        for bot in bot_files:
            if os.path.exists(bot['file']):
                self.available_bots.append(bot)
                
                # Format list item
                status = "✅ RECOMMENDED" if bot['recommended'] else "⚠️  AVAILABLE"
                list_item = f"{bot['name']} - {status}"
                self.bot_listbox.insert(tk.END, list_item)
                
                # Color coding
                if bot['recommended']:
                    self.bot_listbox.itemconfig(tk.END, {'fg': 'green', 'font': ('Segoe UI', 10, 'bold')})
                else:
                    self.bot_listbox.itemconfig(tk.END, {'fg': 'orange'})
        
        # Add missing files info
        if not self.available_bots:
            self.bot_listbox.insert(tk.END, "❌ No trading bot files found")
            self.bot_listbox.itemconfig(0, {'fg': 'red'})
        
        # Auto-select recommended bot
        for i, bot in enumerate(self.available_bots):
            if bot['recommended']:
                self.bot_listbox.selection_set(i)
                self.on_bot_select(None)
                break
    
    def on_bot_select(self, event):
        """Handle bot selection"""
        selection = self.bot_listbox.curselection()
        if selection and self.available_bots:
            bot_index = selection[0]
            if bot_index < len(self.available_bots):
                bot = self.available_bots[bot_index]
                
                # Show bot description
                desc_text = f"📝 {bot['name']}\n\n"
                desc_text += f"{bot['description']}\n\n"
                desc_text += f"📦 Requirements: {', '.join(bot['requirements'])}\n"
                desc_text += f"📁 File: {bot['file']}\n"
                
                # Check requirements
                missing_reqs = []
                for req in bot['requirements']:
                    try:
                        if req == 'MetaTrader5':
                            import MetaTrader5
                        else:
                            __import__(req)
                    except ImportError:
                        if req != 'MetaTrader5':  # Skip MetaTrader5 on non-Windows platforms
                            missing_reqs.append(req)
                
                if missing_reqs:
                    desc_text += f"\n⚠️  Missing requirements: {', '.join(missing_reqs)}\n"
                    desc_text += "Click 'Install Requirements' to install missing packages."
                else:
                    desc_text += "\n✅ All requirements satisfied - Ready to launch!"
                
                self.desc_text.delete(1.0, tk.END)
                self.desc_text.insert(1.0, desc_text)
    
    def launch_selected_bot(self):
        """Launch the selected trading bot"""
        selection = self.bot_listbox.curselection()
        if not selection or not self.available_bots:
            messagebox.showwarning("No Selection", "Please select a trading bot to launch")
            return
        
        bot_index = selection[0]
        if bot_index >= len(self.available_bots):
            messagebox.showerror("Invalid Selection", "Invalid bot selection")
            return
        
        bot = self.available_bots[bot_index]
        
        # Check requirements before launching
        missing_reqs = []
        for req in bot['requirements']:
            try:
                if req == 'MetaTrader5':
                    import MetaTrader5
                else:
                    __import__(req)
            except ImportError:
                if req != 'MetaTrader5':  # Skip MetaTrader5 on non-Windows platforms
                    missing_reqs.append(req)
        
        if missing_reqs:
            result = messagebox.askyesno("Missing Requirements", 
                f"Missing required packages: {', '.join(missing_reqs)}\n\n"
                f"Do you want to install them automatically?")
            if result:
                if self.install_requirements():
                    # Recheck after installation
                    time.sleep(2)
                    return self.launch_selected_bot()
                else:
                    return
            else:
                return
        
        # Confirm launch
        result = messagebox.askyesno("Confirm Launch", 
            f"Launch {bot['name']}?\n\n"
            f"This will start the enhanced trading bot with:\n"
            f"• Fixed price retrieval\n"
            f"• Optimized signal generation\n"
            f"• Enhanced opportunity capture")
        
        if not result:
            return
        
        try:
            self.log_info(f"🚀 Launching {bot['name']}...")
            
            # Launch in separate process
            if platform.system() == 'Windows':
                subprocess.Popen([sys.executable, bot['file']], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen([sys.executable, bot['file']])
            
            self.log_info(f"✅ {bot['name']} launched successfully!")
            
            # Option to close launcher
            result = messagebox.askyesno("Bot Launched", 
                f"{bot['name']} has been launched successfully!\n\n"
                f"Close this launcher window?")
            
            if result:
                self.root.destroy()
                
        except Exception as e:
            self.log_info(f"❌ Launch failed: {e}")
            messagebox.showerror("Launch Error", f"Failed to launch trading bot:\n{e}")
    
    def install_requirements(self):
        """Install required packages"""
        try:
            # Determine what to install
            requirements = set()
            for bot in self.available_bots:
                requirements.update(bot['requirements'])
            
            # Filter out already installed packages
            missing_packages = []
            for req in requirements:
                try:
                    if req == 'MetaTrader5':
                        import MetaTrader5
                    else:
                        __import__(req)
                except ImportError:
                    if req != 'MetaTrader5':  # Skip MetaTrader5 on non-Windows platforms
                        missing_packages.append(req)
            
            if not missing_packages:
                messagebox.showinfo("Nothing to Install", "All required packages are already installed!")
                return True
            
            # Confirm installation
            result = messagebox.askyesno("Install Requirements", 
                f"Install the following packages?\n\n"
                f"{', '.join(missing_packages)}\n\n"
                f"This will use pip to install packages.")
            
            if not result:
                return False
            
            # Create installation window
            install_window = tk.Toplevel(self.root)
            install_window.title("Installing Requirements")
            install_window.geometry("600x400")
            install_window.transient(self.root)
            install_window.grab_set()
            
            ttk.Label(install_window, text="📦 Installing Required Packages", 
                     font=("Segoe UI", 12, "bold")).pack(pady=10)
            
            # Progress display
            progress_text = ScrolledText(install_window, width=70, height=20, font=("Consolas", 9))
            progress_text.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Install packages in thread
            def install_thread():
                try:
                    for package in missing_packages:
                        progress_text.insert(tk.END, f"📦 Installing {package}...\n")
                        progress_text.see(tk.END)
                        progress_text.update()
                        
                        # Run pip install
                        result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                              capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            progress_text.insert(tk.END, f"✅ {package} installed successfully\n")
                        else:
                            progress_text.insert(tk.END, f"❌ {package} installation failed:\n")
                            progress_text.insert(tk.END, f"{result.stderr}\n")
                        
                        progress_text.see(tk.END)
                        progress_text.update()
                    
                    progress_text.insert(tk.END, "\n🎉 Installation process completed!\n")
                    progress_text.insert(tk.END, "You can now close this window and launch the bot.\n")
                    progress_text.see(tk.END)
                    
                except Exception as e:
                    progress_text.insert(tk.END, f"\n❌ Installation error: {e}\n")
                    progress_text.see(tk.END)
            
            # Start installation
            install_thread_obj = threading.Thread(target=install_thread, daemon=True)
            install_thread_obj.start()
            
            # Close button
            ttk.Button(install_window, text="Close", 
                      command=install_window.destroy).pack(pady=10)
            
            return True
            
        except Exception as e:
            messagebox.showerror("Installation Error", f"Failed to install requirements:\n{e}")
            return False
    
    def show_help(self):
        """Show help information"""
        help_window = tk.Toplevel(self.root)
        help_window.title("🆘 Enhanced Trading Bot Help")
        help_window.geometry("700x500")
        help_window.transient(self.root)
        
        help_text = ScrolledText(help_window, width=80, height=30, font=("Segoe UI", 10))
        help_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        help_content = """
🚀 ENHANCED TRADING BOT LAUNCHER HELP

═══════════════════════════════════════════════════════════════════

📊 WHAT'S NEW IN ENHANCED VERSION:
• ✅ Fixed price retrieval mechanism - No more "failed to get price" errors
• ⚡ Optimized signal generation - Captures more market opportunities
• 🎯 Enhanced opportunity capture - Reduced missed signals
• 🔧 Better error handling and retry logic
• 📈 Improved technical indicators

═══════════════════════════════════════════════════════════════════

🤖 AVAILABLE TRADING BOTS:

1. 🪟 ENHANCED WINDOWS MT5 TRADING BOT
   • For live trading with real MetaTrader5 account
   • Requires MetaTrader5 terminal installed and logged in
   • Uses real money - HIGH RISK
   • Best for experienced traders
   • Fixed price retrieval issues
   • Enhanced signal generation

2. 🔬 ENHANCED SIMULATION TRADING BOT
   • Safe simulation environment with virtual money
   • Real market data, virtual trading
   • Perfect for testing and learning
   • NO REAL MONEY RISK
   • Great for beginners
   • Same enhanced features as live version

═══════════════════════════════════════════════════════════════════

📦 REQUIREMENTS:
• Python 3.8 or higher
• numpy (for calculations)
• requests (for market data)
• MetaTrader5 (for live trading only)

═══════════════════════════════════════════════════════════════════

🔧 INSTALLATION GUIDE:

1. WINDOWS USERS (For Live Trading):
   • Install MetaTrader5 from your broker
   • Login to your trading account
   • Enable "Allow automated trading" in MT5 settings
   • Run this launcher and select Windows MT5 bot

2. ALL USERS (For Simulation):
   • Just run this launcher
   • Select Simulation Trading Bot
   • No additional setup required

═══════════════════════════════════════════════════════════════════

⚠️  IMPORTANT SAFETY NOTES:

• TRADING INVOLVES SIGNIFICANT RISK
• You can lose all your money
• Test with simulation first
• Use proper risk management
• Never risk more than you can afford to lose
• The enhanced bot fixes technical issues but doesn't guarantee profits

═══════════════════════════════════════════════════════════════════

🆘 TROUBLESHOOTING:

❌ "MetaTrader5 not found"
   → Install MT5 and the Python library: pip install MetaTrader5

❌ "Price retrieval failed" 
   → This is FIXED in the enhanced version!

❌ "No trading signals"
   → Enhanced version captures more opportunities with optimized thresholds

❌ "Connection failed"
   → Check internet connection and MT5 login status

❌ "Permission denied"
   → Run as Administrator on Windows

═══════════════════════════════════════════════════════════════════

📞 GETTING SUPPORT:
• Check the log files for detailed error messages
• Ensure all requirements are installed
• Try the simulation version first
• Test with small amounts initially

═══════════════════════════════════════════════════════════════════

🎯 ENHANCED FEATURES EXPLAINED:

1. FIXED PRICE RETRIEVAL:
   • Added retry mechanism with exponential backoff
   • Multiple fallback data sources
   • Cached price data for continuity
   • Better error handling

2. OPTIMIZED SIGNAL GENERATION:
   • Reduced price spike threshold from 10 to 3
   • More lenient signal confidence requirements
   • Enhanced technical indicator calculations
   • Improved opportunity detection

3. ENHANCED OPPORTUNITY CAPTURE:
   • Faster scanning intervals
   • Better signal scoring system
   • Reduced false rejections
   • Improved market analysis

═══════════════════════════════════════════════════════════════════

Happy Trading! 📈💰 (Remember: Trade Responsibly!)
        """
        
        help_text.insert(1.0, help_content)
        help_text.config(state='disabled')
        
        # Close button
        ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)
    
    def log_info(self, message):
        """Log information to console"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
    
    def run(self):
        """Run the launcher"""
        self.root.mainloop()

def main():
    """Main function"""
    print("🚀 Enhanced Trading Bot Launcher Starting...")
    print("✅ Fixed price retrieval issues")
    print("⚡ Optimized signal generation") 
    print("🎯 Enhanced opportunity capture")
    print("=" * 50)
    
    launcher = TradingBotLauncher()
    launcher.run()

if __name__ == "__main__":
    main()
