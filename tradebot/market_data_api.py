
"""
Enhanced Market Data API for AuraTrade Bot
Multiple data sources with fallback mechanisms
"""

import time
import requests
import json
import threading
from datetime import datetime, timedelta
import random

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class MockNumpy:
        @staticmethod
        def random(): return random.random()
        @staticmethod
        def sin(x): return random.uniform(-1, 1)
    np = MockNumpy()

class MarketDataAPI:
    def __init__(self):
        self.cache = {}
        self.last_update = {}
        self.cache_duration = 1  # 1 second cache
        self.session = requests.Session()
        self.session.timeout = 5
        
        # Price simulation parameters
        self.price_base = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2650, 
            'USDJPY': 149.50,
            'USDCAD': 1.3750,
            'AUDUSD': 0.6550,
            'NZDUSD': 0.5950,
            'USDCHF': 0.8950,
            'XAUUSDm': 2025.50,
            'XAUUSD': 2025.50,
            'GOLD': 2025.50,
            'BTCUSD': 42500.00,
            'ETHUSD': 2400.00,
            'SPX500': 4750.00,
            'NAS100': 15800.00,
            'GER40': 16500.00,
            'UK100': 7600.00,
            'US30': 35200.00
        }
        
        self.volatility = {
            'EURUSD': 0.0008,
            'GBPUSD': 0.0012,
            'USDJPY': 0.008,
            'USDCAD': 0.0009,
            'AUDUSD': 0.0011,
            'NZDUSD': 0.0013,
            'USDCHF': 0.0008,
            'XAUUSDm': 1.5,
            'XAUUSD': 1.5,
            'GOLD': 1.5,
            'BTCUSD': 850.0,
            'ETHUSD': 48.0,
            'SPX500': 12.0,
            'NAS100': 85.0,
            'GER40': 45.0,
            'UK100': 25.0,
            'US30': 120.0
        }
        
        print("📊 Enhanced Market Data API initialized")
    
    def get_price(self, symbol):
        """Get current price with multiple fallback sources"""
        try:
            # Check cache first
            if self._is_cached(symbol):
                return self.cache[symbol]
            
            price_data = None
            
            # Try MT5 first (if available and connected)
            if MT5_AVAILABLE:
                price_data = self._get_mt5_price(symbol)
            
            # Fallback to external APIs
            if not price_data:
                price_data = self._get_external_price(symbol)
            
            # Ultimate fallback to simulation
            if not price_data:
                price_data = self._get_simulated_price(symbol)
            
            # Cache the result
            if price_data:
                self.cache[symbol] = price_data
                self.last_update[symbol] = time.time()
            
            return price_data
            
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return self._get_simulated_price(symbol)
    
    def _is_cached(self, symbol):
        """Check if price is cached and still valid"""
        if symbol not in self.cache or symbol not in self.last_update:
            return False
        
        return (time.time() - self.last_update[symbol]) < self.cache_duration
    
    def _get_mt5_price(self, symbol):
        """Get price from MT5"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'price': (tick.bid + tick.ask) / 2,
                    'spread': tick.ask - tick.bid,
                    'time': time.time(),
                    'volume': getattr(tick, 'volume_real', 0),
                    'source': 'MT5'
                }
        except Exception as e:
            print(f"MT5 price error for {symbol}: {e}")
        
        return None
    
    def _get_external_price(self, symbol):
        """Get price from external API sources"""
        try:
            # Try different external sources
            sources = [
                self._get_fixer_price,
                self._get_exchangerate_price,
                self._get_forex_price
            ]
            
            for source_func in sources:
                try:
                    price_data = source_func(symbol)
                    if price_data:
                        return price_data
                except:
                    continue
                    
        except Exception as e:
            print(f"External API error for {symbol}: {e}")
        
        return None
    
    def _get_fixer_price(self, symbol):
        """Get price from Fixer.io (for forex pairs)"""
        try:
            if len(symbol) == 6 and symbol[:3] != symbol[3:]:  # Forex pair
                base = symbol[:3]
                quote = symbol[3:]
                
                # Free tier URL (limited requests)
                url = f"http://data.fixer.io/api/latest?access_key=YOUR_API_KEY&base={base}&symbols={quote}"
                
                response = self.session.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success') and quote in data.get('rates', {}):
                        rate = data['rates'][quote]
                        spread = rate * 0.0001  # 1 pip spread
                        
                        return {
                            'bid': rate - spread/2,
                            'ask': rate + spread/2,
                            'price': rate,
                            'spread': spread,
                            'time': time.time(),
                            'source': 'Fixer.io'
                        }
        except:
            pass
        
        return None
    
    def _get_exchangerate_price(self, symbol):
        """Get price from ExchangeRate-API"""
        try:
            if len(symbol) == 6:  # Forex pair
                base = symbol[:3]
                quote = symbol[3:]
                
                url = f"https://api.exchangerate-api.com/v4/latest/{base}"
                
                response = self.session.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if quote in data.get('rates', {}):
                        rate = data['rates'][quote]
                        spread = rate * 0.0001
                        
                        return {
                            'bid': rate - spread/2,
                            'ask': rate + spread/2,
                            'price': rate,
                            'spread': spread,
                            'time': time.time(),
                            'source': 'ExchangeRate-API'
                        }
        except:
            pass
        
        return None
    
    def _get_forex_price(self, symbol):
        """Get price from Alpha Vantage or similar"""
        try:
            # This would require API key setup
            # Placeholder for additional external sources
            pass
        except:
            pass
        
        return None
    
    def _get_simulated_price(self, symbol):
        """Generate realistic simulated price"""
        try:
            base_price = self.price_base.get(symbol, 1.0000)
            volatility = self.volatility.get(symbol, 0.001)
            
            # Time-based price movement
            current_time = time.time()
            
            # Daily trend component
            daily_trend = np.sin(current_time / 86400 * 2 * 3.14159) * 0.005
            
            # Hourly volatility
            hourly_vol = np.sin(current_time / 3600 * 2 * 3.14159) * 0.002
            
            # Random walk component
            random_movement = (random.random() - 0.5) * volatility * 2
            
            # Market hours multiplier (higher volatility during market hours)
            hour = datetime.now().hour
            if 8 <= hour <= 17:  # Market hours
                activity_multiplier = 1.5
            elif 17 <= hour <= 22:  # Overlap hours
                activity_multiplier = 2.0
            else:  # Quiet hours
                activity_multiplier = 0.5
            
            # Calculate final price
            price_movement = (daily_trend + hourly_vol + random_movement) * activity_multiplier
            current_price = base_price * (1 + price_movement)
            
            # Calculate spread based on symbol type
            if 'USD' in symbol and len(symbol) == 6:  # Forex
                spread_pips = random.uniform(0.8, 2.5)
                if 'JPY' in symbol:
                    spread = spread_pips * 0.01
                else:
                    spread = spread_pips * 0.0001
            elif 'XAU' in symbol or 'GOLD' in symbol:  # Gold
                spread = random.uniform(0.3, 0.8)
            elif 'BTC' in symbol or 'ETH' in symbol:  # Crypto
                spread = current_price * random.uniform(0.001, 0.003)
            else:  # Indices
                spread = current_price * random.uniform(0.0001, 0.0005)
            
            return {
                'bid': current_price - spread/2,
                'ask': current_price + spread/2,
                'price': current_price,
                'spread': spread,
                'time': current_time,
                'volume': random.randint(50, 500),
                'source': 'Simulation'
            }
            
        except Exception as e:
            print(f"Simulation price error for {symbol}: {e}")
            # Ultimate fallback
            return {
                'bid': 1.0850,
                'ask': 1.0852,
                'price': 1.0851,
                'spread': 0.0002,
                'time': time.time(),
                'source': 'Fallback'
            }
    
    def get_recent_prices(self, symbol, count=50):
        """Get recent price history"""
        try:
            # Try MT5 first
            if MT5_AVAILABLE:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, count)
                if rates is not None and len(rates) > 0:
                    return [float(rate[4]) for rate in rates]  # Close prices
            
            # Generate simulated price history
            current_price = self.get_price(symbol)
            if not current_price:
                return []
            
            base_price = current_price['price']
            volatility = self.volatility.get(symbol, 0.001)
            
            prices = []
            price = base_price
            
            # Generate realistic price history using random walk
            for i in range(count):
                # Random walk with mean reversion
                change = random.uniform(-volatility, volatility)
                
                # Add some mean reversion
                if abs(price - base_price) > base_price * 0.01:  # If moved more than 1%
                    change *= 0.5  # Reduce momentum
                    if price > base_price:
                        change -= volatility * 0.3  # Slight downward bias
                    else:
                        change += volatility * 0.3  # Slight upward bias
                
                price = price * (1 + change)
                prices.append(price)
            
            # Reverse to get chronological order (oldest first)
            prices.reverse()
            
            return prices
            
        except Exception as e:
            print(f"Error getting recent prices for {symbol}: {e}")
            return []
    
    def get_price_array(self, symbol, count=50):
        """Get price array (alias for get_recent_prices)"""
        return self.get_recent_prices(symbol, count)
    
    def get_market_data(self, symbol, count=1):
        """Get comprehensive market data"""
        try:
            price_data = self.get_price(symbol)
            if not price_data:
                return []
            
            # Return array of market data points
            data_points = []
            for i in range(count):
                # Add some variation for multiple data points
                variation = random.uniform(-0.0001, 0.0001)
                
                data_point = {
                    'close': price_data['price'] * (1 + variation),
                    'bid': price_data['bid'] * (1 + variation),
                    'ask': price_data['ask'] * (1 + variation),
                    'spread': price_data['spread'],
                    'time': price_data['time'] - i,
                    'source': price_data['source'],
                    'volume': random.randint(10, 100)
                }
                data_points.append(data_point)
            
            return data_points
            
        except Exception as e:
            print(f"Error getting market data for {symbol}: {e}")
            return []
    
    def get_spread(self, symbol):
        """Get current spread for symbol"""
        try:
            price_data = self.get_price(symbol)
            if price_data and 'spread' in price_data:
                return price_data['spread']
            return None
        except Exception as e:
            print(f"Error getting spread for {symbol}: {e}")
            return None
    
    def is_market_open(self, symbol):
        """Check if market is open for trading"""
        try:
            current_hour = datetime.now().hour
            current_weekday = datetime.now().weekday()
            
            # Weekend check
            if current_weekday >= 5:  # Saturday = 5, Sunday = 6
                return False
            
            # Forex markets (24/5)
            if len(symbol) == 6 and symbol[:3] != symbol[3:]:
                return current_weekday < 5  # Monday to Friday
            
            # Gold and other commodities
            if 'XAU' in symbol or 'GOLD' in symbol:
                return current_weekday < 5 and 1 <= current_hour <= 23
            
            # Stock indices (limited hours)
            if symbol in ['SPX500', 'NAS100', 'US30']:
                return current_weekday < 5 and 14 <= current_hour <= 21  # NYSE hours (UTC)
            
            # Crypto (24/7)
            if 'BTC' in symbol or 'ETH' in symbol:
                return True
            
            # Default: assume 24/5 like forex
            return current_weekday < 5
            
        except:
            return True  # Default to open
    
    def get_trading_session(self):
        """Get current trading session"""
        try:
            current_hour = datetime.now().hour
            
            if 22 <= current_hour or current_hour < 8:
                return "Asian"
            elif 8 <= current_hour < 16:
                return "European" 
            elif 16 <= current_hour < 22:
                return "American"
            else:
                return "Overlap"
                
        except:
            return "Unknown"
    
    def cleanup_cache(self):
        """Clean up old cache entries"""
        try:
            current_time = time.time()
            symbols_to_remove = []
            
            for symbol, last_update in self.last_update.items():
                if current_time - last_update > self.cache_duration * 10:  # Remove if older than 10x cache duration
                    symbols_to_remove.append(symbol)
            
            for symbol in symbols_to_remove:
                if symbol in self.cache:
                    del self.cache[symbol]
                if symbol in self.last_update:
                    del self.last_update[symbol]
                    
        except Exception as e:
            print(f"Cache cleanup error: {e}")

# Test function
def test_market_data():
    """Test market data API"""
    api = MarketDataAPI()
    
    test_symbols = ['EURUSD', 'GBPUSD', 'XAUUSD', 'BTCUSD']
    
    print("🧪 Testing Market Data API")
    print("=" * 50)
    
    for symbol in test_symbols:
        print(f"\n📊 Testing {symbol}:")
        
        # Test current price
        price = api.get_price(symbol)
        if price:
            print(f"   Price: {price['price']:.5f} | Spread: {price['spread']:.5f} | Source: {price['source']}")
        else:
            print("   ❌ Failed to get price")
        
        # Test price history
        prices = api.get_recent_prices(symbol, 10)
        if prices:
            print(f"   History: {len(prices)} prices | Latest: {prices[-1]:.5f}")
        else:
            print("   ❌ Failed to get price history")
        
        # Test market status
        is_open = api.is_market_open(symbol)
        print(f"   Market Open: {'✅ Yes' if is_open else '❌ No'}")
    
    print(f"\n🌐 Current Session: {api.get_trading_session()}")
    print("=" * 50)

if __name__ == "__main__":
    test_market_data()
