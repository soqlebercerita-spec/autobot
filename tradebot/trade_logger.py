"""
Enhanced Trade Logging Module for Trading Bot
Comprehensive logging system for Windows MT5 integration
"""

import csv
import os
import json
import datetime
import threading
from typing import Dict, List, Any

class TradeLogger:
    def __init__(self):
        self.log_dir = "logs"
        self.ensure_log_directory()
        
        # File paths
        self.trade_log_file = os.path.join(self.log_dir, "trade_log.csv")
        self.general_log_file = os.path.join(self.log_dir, "general.log")
        self.error_log_file = os.path.join(self.log_dir, "error.log")
        self.performance_log_file = os.path.join(self.log_dir, "performance.json")
        self.daily_summary_file = os.path.join(self.log_dir, "daily_summary.json")
        
        # Thread lock for file writing
        self.lock = threading.Lock()
        
        # Initialize log files
        self.initialize_trade_log()
        self.initialize_performance_log()
        
        print(f"✅ Trade logger initialized - Logs in: {self.log_dir}")
        
    def ensure_log_directory(self):
        """Create logs directory if it doesn't exist"""
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
        except Exception as e:
            print(f"❌ Error creating log directory: {e}")
    
    def initialize_trade_log(self):
        """Initialize trade log CSV with headers if file doesn't exist"""
        try:
            if not os.path.exists(self.trade_log_file):
                fieldnames = [
                    "timestamp", "date", "time", "signal", "symbol", "action",
                    "price", "volume", "tp", "sl", "confidence", "entry_time", 
                    "exit_time", "exit_price", "profit", "status", "ticket",
                    "magic_number", "comment", "indicators", "market_condition"
                ]
                
                with open(self.trade_log_file, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
        except Exception as e:
            print(f"❌ Error initializing trade log: {e}")
    
    def initialize_performance_log(self):
        """Initialize performance tracking"""
        try:
            if not os.path.exists(self.performance_log_file):
                initial_data = {
                    "created": datetime.datetime.now().isoformat(),
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_profit": 0.0,
                    "total_loss": 0.0,
                    "best_trade": 0.0,
                    "worst_trade": 0.0,
                    "avg_profit": 0.0,
                    "avg_loss": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "last_updated": datetime.datetime.now().isoformat()
                }
                
                with open(self.performance_log_file, "w") as f:
                    json.dump(initial_data, f, indent=2)
                    
        except Exception as e:
            print(f"❌ Error initializing performance log: {e}")
    
    def log_to_file(self, message: str, level: str = "INFO", log_type: str = "general"):
        """Log message to appropriate file"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}\n"
            
            with self.lock:
                if log_type == "error":
                    with open(self.error_log_file, "a", encoding="utf-8") as f:
                        f.write(log_entry)
                else:
                    with open(self.general_log_file, "a", encoding="utf-8") as f:
                        f.write(log_entry)
                        
        except Exception as e:
            print(f"❌ Error writing to log file: {e}")
    
    def log_trade_signal(self, signal_data: Dict[str, Any]):
        """Log trading signal data"""
        try:
            self.log_to_file(
                f"SIGNAL - {signal_data.get('symbol', 'UNKNOWN')} "
                f"{signal_data.get('action', 'UNKNOWN')} "
                f"Price: {signal_data.get('price', 0)} "
                f"Confidence: {signal_data.get('confidence', 0):.2%}"
            )
        except Exception as e:
            print(f"❌ Error logging signal: {e}")
    
    def log_trade_entry(self, trade_data: Dict[str, Any]):
        """Log trade entry to CSV and performance tracking"""
        try:
            timestamp = datetime.datetime.now()
            
            # Prepare trade data for CSV
            csv_data = {
                "timestamp": timestamp.isoformat(),
                "date": timestamp.strftime("%Y-%m-%d"),
                "time": timestamp.strftime("%H:%M:%S"),
                "signal": trade_data.get("signal", ""),
                "symbol": trade_data.get("symbol", ""),
                "action": trade_data.get("action", ""),
                "price": trade_data.get("price", 0),
                "volume": trade_data.get("volume", 0),
                "tp": trade_data.get("tp", 0),
                "sl": trade_data.get("sl", 0),
                "confidence": trade_data.get("confidence", 0),
                "entry_time": timestamp.strftime("%H:%M:%S"),
                "exit_time": "",
                "exit_price": "",
                "profit": "",
                "status": "OPEN",
                "ticket": trade_data.get("ticket", ""),
                "magic_number": trade_data.get("magic_number", ""),
                "comment": trade_data.get("comment", ""),
                "indicators": json.dumps(trade_data.get("indicators", {})),
                "market_condition": trade_data.get("market_condition", "")
            }
            
            # Write to CSV
            with self.lock:
                with open(self.trade_log_file, "a", newline="", encoding="utf-8") as csvfile:
                    fieldnames = csv_data.keys()
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(csv_data)
            
            # Log to general log
            self.log_to_file(
                f"TRADE OPENED - {trade_data.get('symbol')} "
                f"{trade_data.get('action')} {trade_data.get('volume')} "
                f"@ {trade_data.get('price')}"
            )
            
        except Exception as e:
            print(f"❌ Error logging trade entry: {e}")
            self.log_to_file(f"Error logging trade entry: {e}", "ERROR", "error")
    
    def log_trade_exit(self, ticket: str, exit_price: float, profit: float, reason: str = ""):
        """Log trade exit and update performance metrics"""
        try:
            timestamp = datetime.datetime.now()
            
            # Update the existing trade record
            self._update_trade_exit(ticket, exit_price, profit, reason)
            
            # Update performance metrics
            self._update_performance_metrics(profit)
            
            # Log to general log
            profit_text = f"+${profit:.2f}" if profit > 0 else f"-${abs(profit):.2f}"
            self.log_to_file(
                f"TRADE CLOSED - Ticket: {ticket} "
                f"Exit: {exit_price} P&L: {profit_text} "
                f"Reason: {reason}"
            )
            
        except Exception as e:
            print(f"❌ Error logging trade exit: {e}")
            self.log_to_file(f"Error logging trade exit: {e}", "ERROR", "error")
    
    def _update_trade_exit(self, ticket: str, exit_price: float, profit: float, reason: str):
        """Update trade record with exit information"""
        try:
            # Read existing trades
            trades = []
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("ticket") == str(ticket) and row.get("status") == "OPEN":
                            row["exit_time"] = datetime.datetime.now().strftime("%H:%M:%S")
                            row["exit_price"] = exit_price
                            row["profit"] = profit
                            row["status"] = "CLOSED"
                            row["comment"] = f"{row.get('comment', '')} {reason}".strip()
                        trades.append(row)
            
            # Write back updated trades
            if trades:
                with self.lock:
                    with open(self.trade_log_file, "w", newline="", encoding="utf-8") as f:
                        fieldnames = trades[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(trades)
                        
        except Exception as e:
            print(f"❌ Error updating trade exit: {e}")
    
    def _update_performance_metrics(self, profit: float):
        """Update performance metrics with new trade result"""
        try:
            # Load existing performance data
            perf_data = {}
            if os.path.exists(self.performance_log_file):
                with open(self.performance_log_file, "r") as f:
                    perf_data = json.load(f)
            
            # Update metrics
            perf_data["total_trades"] = perf_data.get("total_trades", 0) + 1
            
            if profit > 0:
                perf_data["winning_trades"] = perf_data.get("winning_trades", 0) + 1
                perf_data["total_profit"] = perf_data.get("total_profit", 0) + profit
                perf_data["best_trade"] = max(perf_data.get("best_trade", 0), profit)
            else:
                perf_data["losing_trades"] = perf_data.get("losing_trades", 0) + 1
                perf_data["total_loss"] = perf_data.get("total_loss", 0) + abs(profit)
                perf_data["worst_trade"] = min(perf_data.get("worst_trade", 0), profit)
            
            # Calculate derived metrics
            total_trades = perf_data["total_trades"]
            winning_trades = perf_data["winning_trades"]
            total_profit = perf_data["total_profit"]
            total_loss = perf_data["total_loss"]
            
            perf_data["win_rate"] = winning_trades / total_trades if total_trades > 0 else 0
            perf_data["avg_profit"] = total_profit / winning_trades if winning_trades > 0 else 0
            perf_data["avg_loss"] = total_loss / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
            perf_data["profit_factor"] = total_profit / total_loss if total_loss > 0 else float('inf')
            perf_data["net_profit"] = total_profit - total_loss
            perf_data["last_updated"] = datetime.datetime.now().isoformat()
            
            # Save updated performance data
            with self.lock:
                with open(self.performance_log_file, "w") as f:
                    json.dump(perf_data, f, indent=2)
                    
        except Exception as e:
            print(f"❌ Error updating performance metrics: {e}")
    
    def log_error(self, error_message: str, context: str = ""):
        """Log error message"""
        full_message = f"{context}: {error_message}" if context else error_message
        self.log_to_file(full_message, "ERROR", "error")
        print(f"❌ {full_message}")
    
    def log_info(self, info_message: str):
        """Log info message"""
        self.log_to_file(info_message, "INFO")
        print(f"ℹ️ {info_message}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        try:
            if os.path.exists(self.performance_log_file):
                with open(self.performance_log_file, "r") as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"❌ Error getting performance summary: {e}")
            return {}
    
    def get_daily_trades(self, date: str = None) -> List[Dict[str, Any]]:
        """Get trades for specific date"""
        try:
            if date is None:
                date = datetime.date.today().strftime("%Y-%m-%d")
            
            trades = []
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("date") == date:
                            trades.append(row)
            
            return trades
            
        except Exception as e:
            print(f"❌ Error getting daily trades: {e}")
            return []

# Global instance
trade_logger = TradeLogger()