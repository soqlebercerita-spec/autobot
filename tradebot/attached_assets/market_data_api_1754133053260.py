"""
Market Data API Module for Trading Bot
Provides real market data from free APIs for simulation
"""

import requests
import time
import numpy as np
from datetime import datetime, timedelta

class MarketDataAPI:
    def __init__(self):
        self.base_urls = {
            'forex': 'https://api.exchangerate-api.com/v4/latest/',
            'crypto': 'https://api.coindesk.com/v1/bpi/currentprice.json',
            'gold': 'https://api.metals.live/v1/spot/gold'
        }
        self.last_prices = {}
        self.price_history = {}
        
    def get_current_price(self, symbol="XAUUSD"):
        """Get current market price for given symbol"""
        try:
            if symbol in ["XAUUSD", "XAUUSDm", "GOLD"]:
                return self._get_gold_price()
            elif symbol in ["EURUSD", "GBPUSD", "USDJPY"]:
                return self._get_forex_price(symbol)
            elif symbol in ["BTCUSD", "ETHUSD"]:
                return self._get_crypto_price(symbol)
            else:
                # Generate realistic price for other symbols
                return self._generate_realistic_price(symbol)
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
            return self._generate_realistic_price(symbol)
    
    def _get_gold_price(self):
        """Get current gold price in USD per ounce"""
        try:
            response = requests.get(self.base_urls['gold'], timeout=5)
            if response.status_code == 200:
                data = response.json()
                price = float(data[0]['price'])
                self.last_prices['XAUUSD'] = price
                return price
        except:
            pass
        
        # Fallback to simulated gold price around current market value
        base_price = 2650.0
        return self._simulate_price_movement(base_price, 'XAUUSD')
    
    def _get_forex_price(self, symbol):
        """Get forex price from exchange rate API"""
        try:
            base_currency = symbol[:3]
            quote_currency = symbol[3:]
            
            url = f"{self.base_urls['forex']}{base_currency}"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if quote_currency in data['rates']:
                    price = data['rates'][quote_currency]
                    self.last_prices[symbol] = price
                    return price
        except:
            pass
        
        # Fallback prices for major pairs
        fallback_prices = {
            'EURUSD': 1.0850,
            'GBPUSD': 1.2750,
            'USDJPY': 149.50
        }
        base_price = fallback_prices.get(symbol, 1.0000)
        return self._simulate_price_movement(base_price, symbol)
    
    def _get_crypto_price(self, symbol):
        """Get cryptocurrency price"""
        try:
            if symbol == "BTCUSD":
                response = requests.get(self.base_urls['crypto'], timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    price = float(data['bpi']['USD']['rate'].replace(',', ''))
                    self.last_prices[symbol] = price
                    return price
        except:
            pass
        
        # Fallback crypto prices
        fallback_prices = {
            'BTCUSD': 42000.0,
            'ETHUSD': 2500.0
        }
        base_price = fallback_prices.get(symbol, 1000.0)
        return self._simulate_price_movement(base_price, symbol)
    
    def _generate_realistic_price(self, symbol):
        """Generate realistic price movement for any symbol"""
        # Default starting prices for different asset types
        default_prices = {
            'XAUUSD': 2650.0,
            'XAGUSD': 30.50,
            'EURUSD': 1.0850,
            'GBPUSD': 1.2750,
            'USDJPY': 149.50,
            'USDCAD': 1.3450,
            'AUDUSD': 0.6750,
            'NZDUSD': 0.6150,
            'USDCHF': 0.8950,
            'BTCUSD': 42000.0,
            'ETHUSD': 2500.0,
            'LTCUSD': 70.0,
            'XRPUSD': 0.50
        }
        
        base_price = default_prices.get(symbol, 100.0)
        return self._simulate_price_movement(base_price, symbol)
    
    def _simulate_price_movement(self, base_price, symbol):
        """Simulate realistic price movement"""
        if symbol not in self.last_prices:
            self.last_prices[symbol] = base_price
        
        current_price = self.last_prices[symbol]
        
        # Generate price movement (random walk with trend)
        volatility = self._get_symbol_volatility(symbol)
        
        # Random component
        random_change = np.random.normal(0, volatility)
        
        # Small trend component
        trend = np.random.normal(0, volatility * 0.1)
        
        # Calculate new price
        price_change = (random_change + trend) * current_price
        new_price = current_price + price_change
        
        # Ensure price doesn't go negative or change too dramatically
        if new_price <= 0:
            new_price = current_price * 0.99
        elif abs(price_change / current_price) > 0.05:  # Max 5% change at once
            new_price = current_price * (1 + 0.05 * np.sign(price_change))
        
        self.last_prices[symbol] = new_price
        return new_price
    
    def _get_symbol_volatility(self, symbol):
        """Get typical volatility for different asset types"""
        volatilities = {
            # Forex pairs (typically lower volatility)
            'EURUSD': 0.0003,
            'GBPUSD': 0.0004,
            'USDJPY': 0.003,
            'USDCAD': 0.0003,
            'AUDUSD': 0.0004,
            'NZDUSD': 0.0005,
            'USDCHF': 0.0003,
            
            # Precious metals (medium volatility)
            'XAUUSD': 0.002,
            'XAGUSD': 0.005,
            
            # Cryptocurrencies (high volatility)
            'BTCUSD': 0.01,
            'ETHUSD': 0.015,
            'LTCUSD': 0.02,
            'XRPUSD': 0.03
        }
        
        return volatilities.get(symbol, 0.001)
    
    def get_historical_data(self, symbol, timeframe='M1', count=100):
        """Generate historical data for technical analysis"""
        try:
            current_price = self.get_current_price(symbol)
            volatility = self._get_symbol_volatility(symbol)
            
            # Generate historical prices using random walk
            prices = []
            timestamps = []
            
            # Start from count periods ago
            start_time = datetime.now() - timedelta(minutes=count)
            
            price = current_price * (1 + np.random.normal(0, volatility * 10))  # Start price
            
            for i in range(count):
                # Generate price movement
                change = np.random.normal(0, volatility) * price
                price += change
                
                # Ensure positive price
                price = max(price, current_price * 0.5)
                
                prices.append(price)
                timestamps.append(start_time + timedelta(minutes=i))
            
            # Adjust last price to current price
            prices[-1] = current_price
            
            # Create OHLC data
            ohlc_data = []
            for i, price in enumerate(prices):
                # Generate realistic OHLC from price
                volatility_range = price * volatility * 2
                
                open_price = price + np.random.uniform(-volatility_range, volatility_range)
                close_price = price
                high_price = max(open_price, close_price) + abs(np.random.uniform(0, volatility_range))
                low_price = min(open_price, close_price) - abs(np.random.uniform(0, volatility_range))
                
                ohlc_data.append({
                    'time': timestamps[i],
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': np.random.uniform(1000, 10000)  # Simulated volume
                })
            
            # Convert to numpy array format for compatibility
            data = {
                'time': np.array([d['time'] for d in ohlc_data]),
                'open': np.array([d['open'] for d in ohlc_data]),
                'high': np.array([d['high'] for d in ohlc_data]),
                'low': np.array([d['low'] for d in ohlc_data]),
                'close': np.array([d['close'] for d in ohlc_data]),
                'volume': np.array([d['volume'] for d in ohlc_data])
            }
            
            return data
            
        except Exception as e:
            print(f"Error generating historical data: {e}")
            return None
    
    def get_symbol_info(self, symbol):
        """Get symbol information for position sizing"""
        symbol_specs = {
            'XAUUSD': {
                'point': 0.01,
                'digits': 2,
                'trade_tick_value': 0.01,
                'volume_min': 0.01,
                'volume_max': 100.0,
                'volume_step': 0.01
            },
            'EURUSD': {
                'point': 0.00001,
                'digits': 5,
                'trade_tick_value': 0.0001,
                'volume_min': 0.01,
                'volume_max': 100.0,
                'volume_step': 0.01
            },
            'BTCUSD': {
                'point': 0.01,
                'digits': 2,
                'trade_tick_value': 0.01,
                'volume_min': 0.001,
                'volume_max': 10.0,
                'volume_step': 0.001
            }
        }
        
        # Default specs for unknown symbols
        default_spec = {
            'point': 0.00001,
            'digits': 5,
            'trade_tick_value': 0.0001,
            'volume_min': 0.01,
            'volume_max': 100.0,
            'volume_step': 0.01
        }
        
        return symbol_specs.get(symbol, default_spec)
    
    def test_connection(self):
        """Test API connectivity"""
        try:
            price = self.get_current_price("XAUUSD")
            return price is not None and price > 0
        except:
            return False