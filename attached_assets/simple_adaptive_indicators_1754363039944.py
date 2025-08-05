"""
Simplified Adaptive Indicators - Compatible Version
Provides basic adaptive technical indicators without type errors
"""

import numpy as np
from config import config

class AdaptiveIndicators:
    def __init__(self, market_data_api=None):
        self.market_api = market_data_api
        self.price_history = {}
        self.adaptive_periods = {}
        print("✅ Enhanced Adaptive Indicators initialized - ACTIVE")
    
    def add_price_data(self, symbol: str, price: float):
        """Add price data for adaptive analysis"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(price)
        
        # Keep only last 100 data points
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
    
    def calculate_adaptive_period(self, symbol: str, base_period: int = 14) -> int:
        """Calculate adaptive period based on volatility"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < 20:
            return base_period
        
        prices = self.price_history[symbol][-20:]
        price_changes = [abs(prices[i] - prices[i-1])/prices[i-1] for i in range(1, len(prices))]
        volatility = np.std(price_changes) if price_changes else 0
        
        # Adapt period based on volatility
        if volatility > 0.02:  # High volatility
            adaptive_period = max(5, base_period - 5)
        elif volatility < 0.005:  # Low volatility
            adaptive_period = min(30, base_period + 5)
        else:
            adaptive_period = base_period
        
        self.adaptive_periods[symbol] = adaptive_period
        return adaptive_period
    
    def adaptive_ma(self, prices, symbol="UNKNOWN", mode=None):
        """Enhanced adaptive moving average"""
        try:
            if len(prices) < 5:
                return None
            
            # Add current price to history
            if len(prices) > 0:
                self.add_price_data(symbol, prices[-1])
            
            # Calculate adaptive period
            period = self.calculate_adaptive_period(symbol, 14)
            period = min(period, len(prices))
            
            return float(np.mean(prices[-period:]))
        except:
            return None
    
    def adaptive_rsi(self, prices, symbol="UNKNOWN", mode=None):
        """Enhanced adaptive RSI"""
        try:
            if len(prices) < 10:
                return 50.0
            
            # Add current price to history
            if len(prices) > 0:
                self.add_price_data(symbol, prices[-1])
            
            # Calculate adaptive period
            period = self.calculate_adaptive_period(symbol, 14)
            period = min(period, len(prices) - 1)
            period = max(5, period)
            
            deltas = np.diff(prices[-period-1:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
        except:
            return 50.0
    
    def adaptive_bollinger_bands(self, prices, symbol="UNKNOWN", mode=None):
        """Enhanced adaptive Bollinger Bands"""
        try:
            if len(prices) < 10:
                return None
            
            # Add current price to history
            if len(prices) > 0:
                self.add_price_data(symbol, prices[-1])
            
            # Calculate adaptive period
            period = self.calculate_adaptive_period(symbol, 20)
            period = min(period, len(prices))
            period = max(10, period)
            
            recent_prices = prices[-period:]
            ma = float(np.mean(recent_prices))
            std = float(np.std(recent_prices))
            
            # Adaptive standard deviation multiplier
            volatility = std / ma if ma > 0 else 0.02
            multiplier = 2.0 + (volatility * 10)  # Adaptive multiplier
            multiplier = min(3.0, max(1.5, multiplier))
            
            upper = ma + (multiplier * std)
            lower = ma - (multiplier * std)
            
            return (upper, ma, lower)
        except:
            return None
    
    def get_adaptive_status(self) -> dict:
        """Get adaptive indicators status"""
        return {
            'status': 'ACTIVE',
            'symbols_tracked': len(self.price_history),
            'adaptive_periods': self.adaptive_periods,
            'enabled': True
        }