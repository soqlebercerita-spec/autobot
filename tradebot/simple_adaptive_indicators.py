"""
Simplified Adaptive Indicators - Compatible Version
Provides basic adaptive technical indicators without type errors
"""

import numpy as np
from config import config

class AdaptiveIndicators:
    def __init__(self, market_data_api=None):
        self.market_api = market_data_api
        print("✅ Simplified Adaptive Indicators initialized")
    
    def adaptive_ma(self, prices, symbol="UNKNOWN", mode=None):
        """Simple adaptive moving average"""
        try:
            if len(prices) < 10:
                return None
            period = min(20, len(prices))
            return float(np.mean(prices[-period:]))
        except:
            return None
    
    def adaptive_rsi(self, prices, symbol="UNKNOWN", mode=None):
        """Simple adaptive RSI"""
        try:
            if len(prices) < 15:
                return 50.0
            
            period = 14
            deltas = np.diff(prices[-period-1:])
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi)
        except:
            return 50.0
    
    def adaptive_bollinger_bands(self, prices, symbol="UNKNOWN", mode=None):
        """Simple adaptive Bollinger Bands"""
        try:
            if len(prices) < 20:
                return None
            
            period = 20
            recent_prices = prices[-period:]
            ma = float(np.mean(recent_prices))
            std = float(np.std(recent_prices))
            
            upper = ma + (2.0 * std)
            lower = ma - (2.0 * std)
            
            return (upper, ma, lower)
        except:
            return None