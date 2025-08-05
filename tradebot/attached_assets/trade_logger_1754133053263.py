"""
Trade Logging Module for Trading Bot
"""

import csv
import os
import json
import datetime
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
        
        # Initialize log files
        self.initialize_trade_log()
        
    def ensure_log_directory(self):
        """Create logs directory if it doesn't exist"""
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
        except Exception as e:
            print(f"Error creating log directory: {e}")
    
    def initialize_trade_log(self):
        """Initialize trade log CSV with headers if file doesn't exist"""
        try:
            if not os.path.exists(self.trade_log_file):
                fieldnames = [
                    "timestamp", "signal", "symbol", "price", "volume", 
                    "tp", "sl", "confidence", "entry_time", "exit_time", 
                    "exit_price", "profit", "status", "magic_number", "comment"
                ]
                
                with open(self.trade_log_file, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
        except Exception as e:
            print(f"Error initializing trade log: {e}")
    
    def log_to_file(self, message: str, level: str = "INFO"):
        """Log general messages to file"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] [{level}] {message}\n"
            
            # Write to general log
            with open(self.general_log_file, "a", encoding="utf-8") as f:
                f.write(log_entry)
            
            # Write errors to separate error log
            if level in ["ERROR", "CRITICAL"]:
                with open(self.error_log_file, "a", encoding="utf-8") as f:
                    f.write(log_entry)
                    
        except Exception as e:
            print(f"Error writing to log file: {e}")
    
    def export_trade_log(self, price: float, tp: float, sl: float, signal: str, 
                        volume: float = 0.01, confidence: float = 0, 
                        symbol: str = "XAUUSDm", magic_number: int = 234000):
        """Export trade entry to CSV log"""
        try:
            fieldnames = [
                "timestamp", "signal", "symbol", "price", "volume", 
                "tp", "sl", "confidence", "entry_time", "exit_time", 
                "exit_price", "profit", "status", "magic_number", "comment"
            ]
            
            file_exists = os.path.isfile(self.trade_log_file)
            
            with open(self.trade_log_file, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "signal": signal,
                    "symbol": symbol,
                    "price": round(price, 5),
                    "volume": volume,
                    "tp": round(tp, 5),
                    "sl": round(sl, 5),
                    "confidence": round(confidence, 2),
                    "entry_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_time": "",
                    "exit_price": "",
                    "profit": "",
                    "status": "OPEN",
                    "magic_number": magic_number,
                    "comment": f"AutoBot_{signal}"
                })
                
        except Exception as e:
            print(f"Error exporting trade log: {e}")
            self.log_to_file(f"Error exporting trade log: {e}", "ERROR")
    
    def update_trade_exit(self, entry_time: str, exit_price: float, profit: float, status: str = "CLOSED"):
        """Update trade log with exit information"""
        try:
            # Read existing data
            trades = []
            fieldnames = [
                "timestamp", "signal", "symbol", "price", "volume", 
                "tp", "sl", "confidence", "entry_time", "exit_time", 
                "exit_price", "profit", "status", "magic_number", "comment"
            ]
            
            if os.path.exists(self.trade_log_file):
                with open(self.trade_log_file, "r", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    trades = list(reader)
            
            # Update the matching trade
            updated = False
            for trade in trades:
                if trade["entry_time"] == entry_time and trade["status"] == "OPEN":
                    trade["exit_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    trade["exit_price"] = round(exit_price, 5)
                    trade["profit"] = round(profit, 2)
                    trade["status"] = status
                    updated = True
                    break
            
            if updated:
                # Rewrite the file with updated data
                with open(self.trade_log_file, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(trades)
                    
        except Exception as e:
            print(f"Error updating trade exit: {e}")
            self.log_to_file(f"Error updating trade exit: {e}", "ERROR")
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics to JSON file"""
        try:
            timestamp = datetime.datetime.now().isoformat()
            
            # Read existing data
            performance_data = []
            if os.path.exists(self.performance_log_file):
                with open(self.performance_log_file, "r", encoding="utf-8") as f:
                    try:
                        performance_data = json.load(f)
                    except json.JSONDecodeError:
                        performance_data = []
            
            # Add new metrics
            metrics["timestamp"] = timestamp
            performance_data.append(metrics)
            
            # Keep only last 1000 entries
            if len(performance_data) > 1000:
                performance_data = performance_data[-1000:]
            
            # Write back to file
            with open(self.performance_log_file, "w", encoding="utf-8") as f:
                json.dump(performance_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error logging performance metrics: {e}")
            self.log_to_file(f"Error logging performance metrics: {e}", "ERROR")
    
    def get_trade_statistics(self, days: int = 30) -> Dict[str, Any]:
        """Get trading statistics for specified number of days"""
        try:
            if not os.path.exists(self.trade_log_file):
                return {}
            
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days)
            
            trades = []
            with open(self.trade_log_file, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        trade_date = datetime.datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                        if trade_date >= cutoff_date:
                            trades.append(row)
                    except ValueError:
                        continue
            
            if not trades:
                return {}
            
            # Calculate statistics
            total_trades = len(trades)
            winning_trades = 0
            losing_trades = 0
            total_profit = 0
            winning_profit = 0
            losing_profit = 0
            
            for trade in trades:
                try:
                    if trade["profit"] and trade["status"] == "CLOSED":
                        profit = float(trade["profit"])
                        total_profit += profit
                        
                        if profit > 0:
                            winning_trades += 1
                            winning_profit += profit
                        else:
                            losing_trades += 1
                            losing_profit += profit
                except (ValueError, KeyError):
                    continue
            
            completed_trades = winning_trades + losing_trades
            win_rate = (winning_trades / completed_trades * 100) if completed_trades > 0 else 0
            avg_win = winning_profit / winning_trades if winning_trades > 0 else 0
            avg_loss = losing_profit / losing_trades if losing_trades > 0 else 0
            profit_factor = abs(winning_profit / losing_profit) if losing_profit != 0 else 0
            
            return {
                "period_days": days,
                "total_trades": total_trades,
                "completed_trades": completed_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": round(win_rate, 2),
                "total_profit": round(total_profit, 2),
                "average_win": round(avg_win, 2),
                "average_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2),
                "calculated_at": datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Error calculating trade statistics: {e}")
            self.log_to_file(f"Error calculating trade statistics: {e}", "ERROR")
            return {}
    
    def export_all_logs(self, export_dir: str = "exports"):
        """Export all logs to specified directory"""
        try:
            # Create export directory
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            full_export_dir = os.path.join(export_dir, f"trading_logs_{timestamp}")
            
            if not os.path.exists(full_export_dir):
                os.makedirs(full_export_dir)
            
            # Copy all log files
            import shutil
            
            if os.path.exists(self.log_dir):
                shutil.copytree(self.log_dir, os.path.join(full_export_dir, "logs"))
            
            # Generate summary report
            stats = self.get_trade_statistics(30)
            if stats:
                with open(os.path.join(full_export_dir, "summary_report.json"), "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
            
            self.log_to_file(f"Logs exported to {full_export_dir}", "INFO")
            return full_export_dir
            
        except Exception as e:
            print(f"Error exporting logs: {e}")
            self.log_to_file(f"Error exporting logs: {e}", "ERROR")
            return None
    
    def clean_old_logs(self, days_to_keep: int = 90):
        """Clean log files older than specified days"""
        try:
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_to_keep)
            
            # Clean general log file
            if os.path.exists(self.general_log_file):
                self._clean_text_log_file(self.general_log_file, cutoff_date)
            
            # Clean error log file
            if os.path.exists(self.error_log_file):
                self._clean_text_log_file(self.error_log_file, cutoff_date)
            
            # Clean trade log file
            if os.path.exists(self.trade_log_file):
                self._clean_csv_log_file(self.trade_log_file, cutoff_date)
            
            self.log_to_file(f"Cleaned logs older than {days_to_keep} days", "INFO")
            
        except Exception as e:
            print(f"Error cleaning old logs: {e}")
            self.log_to_file(f"Error cleaning old logs: {e}", "ERROR")
    
    def _clean_text_log_file(self, file_path: str, cutoff_date: datetime.datetime):
        """Clean text log file entries older than cutoff date"""
        try:
            if not os.path.exists(file_path):
                return
            
            temp_file = file_path + ".temp"
            
            with open(file_path, "r", encoding="utf-8") as infile, \
                 open(temp_file, "w", encoding="utf-8") as outfile:
                
                for line in infile:
                    try:
                        # Extract timestamp from log line
                        if line.startswith("["):
                            timestamp_str = line.split("]")[0][1:]
                            log_date = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                            
                            if log_date >= cutoff_date:
                                outfile.write(line)
                    except (ValueError, IndexError):
                        # Keep lines that don't match expected format
                        outfile.write(line)
            
            # Replace original file with cleaned version
            os.replace(temp_file, file_path)
            
        except Exception as e:
            print(f"Error cleaning text log file {file_path}: {e}")
            # Remove temp file if it exists
            if os.path.exists(file_path + ".temp"):
                os.remove(file_path + ".temp")
    
    def _clean_csv_log_file(self, file_path: str, cutoff_date: datetime.datetime):
        """Clean CSV log file entries older than cutoff date"""
        try:
            if not os.path.exists(file_path):
                return
            
            temp_file = file_path + ".temp"
            
            with open(file_path, "r", encoding="utf-8") as infile:
                reader = csv.DictReader(infile)
                fieldnames = reader.fieldnames
                
                with open(temp_file, "w", newline="", encoding="utf-8") as outfile:
                    if fieldnames is not None:
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        writer.writeheader()
                        
                        for row in reader:
                            try:
                                log_date = datetime.datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                                if log_date >= cutoff_date:
                                    writer.writerow(row)
                            except (ValueError, KeyError):
                                # Keep rows that don't match expected format
                                writer.writerow(row)
            
            # Replace original file with cleaned version
            os.replace(temp_file, file_path)
            
        except Exception as e:
            print(f"Error cleaning CSV log file {file_path}: {e}")
            # Remove temp file if it exists
            if os.path.exists(file_path + ".temp"):
                os.remove(file_path + ".temp")
    
    def get_recent_trades(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get most recent trades"""
        try:
            if not os.path.exists(self.trade_log_file):
                return []
            
            trades = []
            with open(self.trade_log_file, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                trades = list(reader)
            
            # Sort by timestamp (most recent first) and return last N trades
            trades.sort(key=lambda x: x["timestamp"], reverse=True)
            return trades[:count]
            
        except Exception as e:
            print(f"Error getting recent trades: {e}")
            return []
