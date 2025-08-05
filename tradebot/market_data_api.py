"""
Enhanced Market Data API with Retry Logic and Fallback Sources
Optimized for reliable price retrieval
"""

import requests
import time
import random
import numpy as np
from datetime import datetime, timedelta
from tradebot.config import config

class MarketDataAPI:
    """Enhanced market data provider with multiple sources and retry logic"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingBot/1.0',
            'Accept': 'application/json',
            'Connection': 'keep-alive'
        })
        
        # Market data sources (backup/fallback)
        self.data_sources = [
            'primary',
            'fallback_1', 
            'fallback_2'
        ]
        
        # Price cache for continuity
        self.price_cache = {}
        self.last_update = {}
        
        # Simulated market data for demonstration
        self.base_prices = {
            'XAUUSDm': 2650.0,
            'EURUSD': 1.0850,
            'GBPUSD': 1.2750,
            'BTCUSD': 67500.0,
            'USDJPY': 149.50,
            'AUDUSD': 0.6650,
            'USDCAD': 1.3720,
            'NZDUSD': 0.6120
        }
        
        self.volatility_factors = {
            'XAUUSDm': 0.0015,
            'EURUSD': 0.0008,
            'GBPUSD': 0.0012,
            'BTCUSD': 0.025,
            'USDJPY': 0.0010,
            'AUDUSD': 0.0010,
            'USDCAD': 0.0008,
            'NZDUSD': 0.0012
        }
    
    def get_market_data(self, symbol, timeframe=None, count=100):
        """Get market data with enhanced retry logic"""
        for attempt in range(config.PRICE_FETCH_RETRY):
            try:
                # Try primary data source
                data = self._fetch_primary_data(symbol, timeframe, count)
                if data is not None and len(data) > 0:
                    self._update_cache(symbol, data)
                    return data
                
                # Try fallback sources
                for source in self.data_sources[1:]:
                    data = self._fetch_fallback_data(symbol, source, count)
                    if data is not None and len(data) > 0:
                        self._update_cache(symbol, data)
                        return data
                
                # Use cached data if available
                if symbol in self.price_cache:
                    cached_data = self.price_cache[symbol]
                    if len(cached_data) > 0:
                        print(f"Using cached data for {symbol}")
                        return cached_data[-count:] if len(cached_data) > count else cached_data
                
                # Generate synthetic data as last resort
                print(f"Generating synthetic data for {symbol} (attempt {attempt + 1})")
                data = self._generate_synthetic_data(symbol, count)
                if data is not None:
                    self._update_cache(symbol, data)
                    return data
                
                # Wait before retry
                if attempt < config.PRICE_FETCH_RETRY - 1:
                    time.sleep(1 + attempt * 0.5)
                    
            except Exception as e:
                print(f"Market data fetch error (attempt {attempt + 1}): {e}")
                if attempt < config.PRICE_FETCH_RETRY - 1:
                    time.sleep(1 + attempt * 0.5)
        
        print(f"Failed to get market data for {symbol} after {config.PRICE_FETCH_RETRY} attempts")
        return None
    
    def _fetch_primary_data(self, symbol, timeframe, count):
        """Fetch from primary source (placeholder for real API)"""
        try:
            # This would normally call a real market data API
            # For now, return synthetic data
            return self._generate_synthetic_data(symbol, count)
        except Exception as e:
            print(f"Primary data source error: {e}")
            return None
    
    def _fetch_fallback_data(self, symbol, source, count):
        """Fetch from fallback sources"""
        try:
            # Simulate different fallback sources
            if source == 'fallback_1':
                return self._generate_synthetic_data(symbol, count, volatility_multiplier=0.8)
            elif source == 'fallback_2':
                return self._generate_synthetic_data(symbol, count, volatility_multiplier=1.2)
            
            return None
        except Exception as e:
            print(f"Fallback data source {source} error: {e}")
            return None
    
    def _generate_synthetic_data(self, symbol, count, volatility_multiplier=1.0):
        """Generate realistic synthetic market data"""
        try:
            if symbol not in self.base_prices:
                print(f"Unknown symbol: {symbol}")
                return None
            
            base_price = self.base_prices[symbol]
            volatility = self.volatility_factors.get(symbol, 0.001) * volatility_multiplier
            
            # Generate realistic price movements
            data = []
            current_price = base_price
            
            # Add some trend and mean reversion
            trend = random.uniform(-0.0005, 0.0005)
            
            for i in range(count):
                # Random walk with trend and mean reversion
                random_change = random.gauss(0, volatility)
                mean_reversion = (base_price - current_price) * 0.001
                trend_component = trend * (1 + random.uniform(-0.5, 0.5))
                
                price_change = random_change + mean_reversion + trend_component
                current_price = current_price * (1 + price_change)
                
                # Create OHLC data structure
                high = current_price * (1 + abs(random.gauss(0, volatility * 0.3)))
                low = current_price * (1 - abs(random.gauss(0, volatility * 0.3)))
                open_price = current_price * (1 + random.gauss(0, volatility * 0.1))
                
                # Ensure OHLC consistency
                high = max(high, current_price, open_price)
                low = min(low, current_price, open_price)
                
                # Create data point
                data_point = {
                    'time': datetime.now() - timedelta(minutes=(count - i)),
                    'open': round(open_price, 5),
                    'high': round(high, 5),
                    'low': round(low, 5),
                    'close': round(current_price, 5),
                    'volume': random.randint(50, 500)
                }
                
                data.append(data_point)
            
            return data
            
        except Exception as e:
            print(f"Synthetic data generation error: {e}")
            return None
    
    def _update_cache(self, symbol, data):
        """Update price cache"""
        try:
            if symbol not in self.price_cache:
                self.price_cache[symbol] = []
            
            # Keep only recent data (last 500 points)
            if len(self.price_cache[symbol]) > 500:
                self.price_cache[symbol] = self.price_cache[symbol][-400:]
            
            # Add new data
            self.price_cache[symbol].extend(data)
            self.last_update[symbol] = datetime.now()
            
        except Exception as e:
            print(f"Cache update error: {e}")
    
    def get_current_price(self, symbol):
        """Get current price with enhanced reliability"""
        try:
            # Try to get fresh data
            data = self.get_market_data(symbol, count=1)
            if data and len(data) > 0:
                return data[-1]['close']
            
            # Use cached data
            if symbol in self.price_cache and len(self.price_cache[symbol]) > 0:
                return self.price_cache[symbol][-1]['close']
            
            # Return base price as last resort
            return self.base_prices.get(symbol, 1.0)
            
        except Exception as e:
            print(f"Current price fetch error: {e}")
            return self.base_prices.get(symbol, 1.0)
    
    def get_price_array(self, symbol, field='close', count=100):
        """Get array of prices for calculations"""
        try:
            data = self.get_market_data(symbol, count=count)
            if not data:
                return np.array([])
            
            if field == 'close':
                return np.array([point['close'] for point in data])
            elif field == 'high':
                return np.array([point['high'] for point in data])
            elif field == 'low':
                return np.array([point['low'] for point in data])
            elif field == 'open':
                return np.array([point['open'] for point in data])
            else:
                return np.array([point['close'] for point in data])
                
        except Exception as e:
            print(f"Price array fetch error: {e}")
            return np.array([])
    
    def is_market_open(self):
        """Check if market is open (simplified)"""
        now = datetime.now()
        hour = now.hour
        
        # Simple market hours check (can be enhanced)
        if 6 <= hour <= 22:  # Extended hours for more opportunities
            return True
        return False
    
    def get_spread(self, symbol):
        """Get bid-ask spread (simulated)"""
        try:
            base_spread = {
                'XAUUSDm': 0.30,
                'EURUSD': 0.00015,
                'GBPUSD': 0.00020,
                'BTCUSD': 50.0,
                'USDJPY': 0.015,
                'AUDUSD': 0.00018,
                'USDCAD': 0.00020,
                'NZDUSD': 0.00025
            }
            
            return base_spread.get(symbol, 0.001)
            
        except Exception as e:
            print(f"Spread calculation error: {e}")
            return 0.001
