"""
High-Frequency Trading (HFT) Configuration
Ultra-fast trading parameters for maximum opportunity capture
"""

from config import config

class HFTConfig:
    def __init__(self):
        # HFT Core Settings
        self.HFT_ENABLED = True
        self.HFT_MODE_ACTIVE = False
        
        # Ultra-fast execution settings
        self.HFT_SCAN_INTERVAL = 1          # 1 second scanning
        self.HFT_MAX_EXECUTION_TIME_MS = 1  # 1ms max execution
        self.HFT_MIN_PROFIT_PIPS = 0.3      # 0.3 pips minimum
        self.HFT_MAX_TRADES_PER_SECOND = 10 # 10 trades/second max
        self.HFT_MAX_TRADES_PER_MINUTE = 200 # 200 trades/minute max
        
        # HFT Risk Settings
        self.HFT_MAX_POSITION_SIZE = 0.01   # Small positions
        self.HFT_TP_PERSEN = 0.001          # 0.1% TP (very tight)
        self.HFT_SL_PERSEN = 0.003          # 0.3% SL (tight)
        self.HFT_MAX_CONCURRENT_TRADES = 20 # 20 simultaneous trades
        
        # HFT Signal Settings
        self.HFT_SIGNAL_CONFIDENCE = 0.2    # Lower confidence for speed
        self.HFT_PRICE_SPIKE_THRESHOLD = 1  # 1% spike threshold
        self.HFT_VOLATILITY_FACTOR = 2.0    # Higher volatility tolerance
        
        # HFT Symbols (most liquid)
        self.HFT_SYMBOLS = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF',
            'AUDUSD', 'USDCAD', 'NZDUSD', 'XAUUSDm'
        ]
        
        # Performance tracking
        self.hft_stats = {
            'trades_per_second': 0,
            'trades_per_minute': 0,
            'avg_execution_time': 0,
            'total_hft_trades': 0,
            'hft_profit': 0.0,
            'last_trade_time': None
        }
        
        print("🚀 HFT Configuration loaded")
        print(f"   • Max Speed: {self.HFT_MAX_TRADES_PER_SECOND} trades/second")
        print(f"   • Scan Interval: {self.HFT_SCAN_INTERVAL}s")
        print(f"   • Min Profit: {self.HFT_MIN_PROFIT_PIPS} pips")
    
    def enable_hft_mode(self):
        """Enable HFT mode"""
        self.HFT_MODE_ACTIVE = True
        print("⚡ HFT MODE ACTIVATED!")
        print("   • Ultra-fast execution enabled")
        print("   • High-frequency scanning active")
        print("   • Tight profit targets set")
    
    def disable_hft_mode(self):
        """Disable HFT mode"""
        self.HFT_MODE_ACTIVE = False
        print("🔄 HFT MODE DISABLED")
        print("   • Returning to normal trading mode")
    
    def get_hft_settings(self):
        """Get current HFT settings"""
        return {
            'enabled': self.HFT_MODE_ACTIVE,
            'scan_interval': self.HFT_SCAN_INTERVAL,
            'max_trades_per_second': self.HFT_MAX_TRADES_PER_SECOND,
            'tp_percent': self.HFT_TP_PERSEN,
            'sl_percent': self.HFT_SL_PERSEN,
            'symbols': self.HFT_SYMBOLS
        }

# Global HFT config instance
hft_config = HFTConfig()