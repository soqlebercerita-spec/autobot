
"""
Market Data API for Windows MT5 Integration
Provides market data with fallback mechanisms
"""

import sys
import platform
import time
from typing import Optional, Dict, Any

# Import MT5 with Windows check
IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    try:
        import MetaTrader5 as mt5
        MT5_AVAILABLE = True
    except ImportError:
        MT5_AVAILABLE = False
        print("⚠️  MetaTrader5 not installed on Windows - install with: pip install MetaTrader5")
else:
    MT5_AVAILABLE = False
    print("ℹ️  MT5 requires Windows platform")

class MarketDataAPI:
    """Windows MT5 Market Data Provider with simulation fallback"""
    
    def __init__(self):
        self.is_windows = IS_WINDOWS
        self.mt5_available = MT5_AVAILABLE
        self.connected = False
        self.symbol_cache = {}
        
        # Initialize MT5 if available on Windows
        if self.mt5_available and self.is_windows:
            self.connect_mt5()
    
    def connect_mt5(self) -> bool:
        """Connect to MetaTrader5 on Windows"""
        if not self.mt5_available or not self.is_windows:
            return False
        
        try:
            if not mt5.initialize():
                print(f"⚠️  MT5 connection failed: {mt5.last_error()}")
                return False
            
            account = mt5.account_info()
            if account:
                print(f"✅ MT5 Market Data Connected - Account: {account.login}")
                self.connected = True
                return True
            else:
                print("⚠️  MT5 account info not available")
                return False
                
        except Exception as e:
            print(f"❌ MT5 connection error: {e}")
            return False
    
    def get_price(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current price from MT5 or simulation"""
        
        if self.connected and self.mt5_available:
            # Real MT5 data on Windows
            try:
                # Ensure symbol is in Market Watch
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    if not mt5.symbol_select(symbol, True):
                        print(f"⚠️  Could not add {symbol} to Market Watch")
                        return self._get_simulation_price(symbol)
                
                # Get tick data
                tick = mt5.symbol_info_tick(symbol)
                if tick:
                    return {
                        'symbol': symbol,
                        'bid': tick.bid,
                        'ask': tick.ask,
                        'price': (tick.bid + tick.ask) / 2,
                        'spread': tick.ask - tick.bid,
                        'time': tick.time,
                        'source': 'MT5_Windows',
                        'success': True
                    }
                else:
                    print(f"⚠️  No tick data for {symbol}")
                    return self._get_simulation_price(symbol)
                    
            except Exception as e:
                print(f"❌ MT5 price error for {symbol}: {e}")
                return self._get_simulation_price(symbol)
        
        # Fallback to simulation
        return self._get_simulation_price(symbol)
    
    def _get_simulation_price(self, symbol: str) -> Dict[str, Any]:
        """Generate simulation price data"""
        import random
        
        # Base prices for common symbols
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.2500,
            'USDJPY': 150.00,
            'USDCHF': 0.9000,
            'AUDUSD': 0.6500,
            'USDCAD': 1.3500,
            'NZDUSD': 0.6000,
            'XAUUSD': 2000.00,
            'XAGUSD': 25.00,
            'BTCUSD': 45000.00,
            'ETHUSD': 3000.00
        }
        
        # Get base price or use default
        base_price = base_prices.get(symbol, 1.0000)
        
        # Add some random variation (±0.1%)
        variation = random.uniform(-0.001, 0.001)
        current_price = base_price * (1 + variation)
        
        # Calculate spread (0.1% of price)
        spread = current_price * 0.001
        bid = current_price - spread/2
        ask = current_price + spread/2
        
        return {
            'symbol': symbol,
            'bid': round(bid, 5),
            'ask': round(ask, 5),
            'price': round(current_price, 5),
            'spread': round(spread, 5),
            'time': int(time.time()),
            'source': 'Simulation',
            'success': True
        }
    
    def get_historical_data(self, symbol: str, timeframe: str = 'M1', count: int = 100):
        """Get historical price data"""
        if self.connected and self.mt5_available:
            try:
                # Convert timeframe
                tf_map = {
                    'M1': mt5.TIMEFRAME_M1,
                    'M5': mt5.TIMEFRAME_M5,
                    'M15': mt5.TIMEFRAME_M15,
                    'H1': mt5.TIMEFRAME_H1,
                    'D1': mt5.TIMEFRAME_D1
                }
                
                mt5_timeframe = tf_map.get(timeframe, mt5.TIMEFRAME_M1)
                
                # Get rates
                rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
                if rates is not None and len(rates) > 0:
                    return {
                        'success': True,
                        'data': rates,
                        'count': len(rates),
                        'source': 'MT5_Windows'
                    }
                    
            except Exception as e:
                print(f"❌ Historical data error: {e}")
        
        # Return simulation data
        return {
            'success': False,
            'data': None,
            'count': 0,
            'source': 'Simulation',
            'message': 'Historical data not available in simulation'
        }
    
    def disconnect(self):
        """Disconnect from MT5"""
        if self.connected and self.mt5_available:
            try:
                mt5.shutdown()
                self.connected = False
                print("🔌 MT5 Market Data disconnected")
            except:
                pass
    
    def __del__(self):
        """Cleanup on destruction"""
        self.disconnect()

# Global instance
market_data_api = MarketDataAPI()

def get_price(symbol: str):
    """Convenience function to get price"""
    return market_data_api.get_price(symbol)

if __name__ == "__main__":
    # Test the API
    print("🧪 Testing Market Data API...")
    
    test_symbols = ['EURUSD', 'XAUUSD', 'BTCUSD']
    
    for symbol in test_symbols:
        price_data = market_data_api.get_price(symbol)
        if price_data and price_data['success']:
            print(f"✅ {symbol}: {price_data['price']:.5f} ({price_data['source']})")
        else:
            print(f"❌ Failed to get price for {symbol}")
    
    print("🧪 Market Data API test completed")
