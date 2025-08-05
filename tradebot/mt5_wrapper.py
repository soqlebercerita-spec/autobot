"""
MT5 Wrapper for Cross-Platform Compatibility
Provides fallback functionality when MetaTrader5 is not available
"""

# Try to import MetaTrader5, fallback to simulation if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    print("MetaTrader5 library loaded successfully")
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 not available - using simulation mode")
    
    # Create mock mt5 module for compatibility
    class MockMT5:
        TIMEFRAME_M1 = 1
        TIMEFRAME_M5 = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_H1 = 60
        
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        
        @staticmethod
        def initialize():
            return False
            
        @staticmethod
        def shutdown():
            pass
            
        @staticmethod
        def account_info():
            return None
            
        @staticmethod
        def symbol_info(symbol):
            return None
            
        @staticmethod
        def symbol_info_tick(symbol):
            return None
            
        @staticmethod
        def symbol_select(symbol, enable):
            return False
            
        @staticmethod
        def positions_get(symbol=None):
            return []
            
        @staticmethod
        def orders_get(symbol=None):
            return []
            
        @staticmethod
        def order_send(request):
            return None
            
        @staticmethod
        def copy_rates_from_pos(symbol, timeframe, start_pos, count):
            return None
            
        @staticmethod
        def last_error():
            return (1, "MetaTrader5 not available")
    
    mt5 = MockMT5()

# Export the availability status and mt5 instance
__all__ = ['mt5', 'MT5_AVAILABLE']