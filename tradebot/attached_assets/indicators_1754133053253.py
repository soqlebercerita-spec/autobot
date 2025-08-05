"""
Technical Indicators for Trading Bot
"""

import numpy as np
import MetaTrader5 as mt5

class TechnicalIndicators:
    def __init__(self):
        pass
    
    def calculate_ma(self, data, period=10):
        """Calculate Simple Moving Average"""
        try:
            if len(data['close']) < period:
                return None
            return np.mean(data['close'][-period:])
        except Exception as e:
            print(f"Error calculating MA: {e}")
            return None
    
    def calculate_ema(self, data, period=9):
        """Calculate Exponential Moving Average"""
        try:
            prices = data['close']
            if len(prices) < period:
                return None
            
            # Calculate EMA using numpy
            weights = np.exp(np.linspace(-1., 0., period))
            weights /= weights.sum()
            
            # Pad prices to handle edge cases
            padded_prices = np.concatenate([np.repeat(prices[0], period-1), prices])
            ema_values = np.convolve(padded_prices, weights, mode='valid')
            
            return ema_values[-1] if len(ema_values) > 0 else None
        except Exception as e:
            print(f"Error calculating EMA: {e}")
            return None
    
    def calculate_wma(self, data, period=5):
        """Calculate Weighted Moving Average"""
        try:
            if len(data['close']) < period:
                return None
            
            weights = np.arange(1, period + 1)
            prices = data['close'][-period:]
            return np.average(prices, weights=weights)
        except Exception as e:
            print(f"Error calculating WMA: {e}")
            return None
    
    def calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        try:
            prices = data['close']
            if len(prices) < period + 1:
                return 50  # Neutral RSI if not enough data
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calculate average gains and losses
            avg_gains = np.mean(gains[-period:])
            avg_losses = np.mean(losses[-period:])
            
            if avg_losses == 0:
                return 100  # Avoid division by zero
            
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return 50
    
    def get_bollinger_bands(self, data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        try:
            if len(data['close']) < period:
                current_price = data['close'][-1]
                return current_price, current_price, current_price
            
            closes = data['close'][-period:]
            sma = np.mean(closes)
            std = np.std(closes)
            
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            
            return upper_band, lower_band, sma
        except Exception as e:
            print(f"Error calculating Bollinger Bands: {e}")
            current_price = data['close'][-1] if len(data['close']) > 0 else 0
            return current_price, current_price, current_price
    
    def calculate_macd(self, data, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(data['close']) < slow_period:
                return 0, 0, 0
            
            # Calculate EMAs
            fast_ema = self._calculate_ema_series(data['close'], fast_period)
            slow_ema = self._calculate_ema_series(data['close'], slow_period)
            
            # Calculate MACD line
            macd_line = fast_ema[-1] - slow_ema[-1]
            
            # Calculate signal line (EMA of MACD)
            macd_values = fast_ema[slow_period-fast_period:] - slow_ema
            signal_line = self._calculate_ema_series(macd_values, signal_period)[-1]
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        except Exception as e:
            print(f"Error calculating MACD: {e}")
            return 0, 0, 0
    
    def _calculate_ema_series(self, prices, period):
        """Calculate EMA series for internal use"""
        try:
            ema_values = []
            multiplier = 2 / (period + 1)
            
            # Start with SMA for first value
            ema = np.mean(prices[:period])
            ema_values.append(ema)
            
            # Calculate EMA for remaining values
            for i in range(period, len(prices)):
                ema = (prices[i] * multiplier) + (ema * (1 - multiplier))
                ema_values.append(ema)
            
            return np.array(ema_values)
        except:
            return np.array([prices[-1]] * len(prices))
    
    def calculate_stochastic(self, data, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        try:
            if len(data) < k_period:
                return 50, 50
            
            highs = data['high'][-k_period:]
            lows = data['low'][-k_period:]
            closes = data['close'][-k_period:]
            
            highest_high = np.max(highs)
            lowest_low = np.min(lows)
            current_close = closes[-1]
            
            if highest_high == lowest_low:
                k_percent = 50
            else:
                k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            
            # For simplicity, use current %K as %D (normally it's SMA of %K)
            d_percent = k_percent
            
            return k_percent, d_percent
        except Exception as e:
            print(f"Error calculating Stochastic: {e}")
            return 50, 50
    
    def calculate_atr(self, data, period=14):
        """Calculate Average True Range"""
        try:
            if len(data) < period + 1:
                return 0
            
            high = data['high'][-period-1:]
            low = data['low'][-period-1:]
            close = data['close'][-period-1:]
            
            tr_list = []
            for i in range(1, len(high)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr = max(tr1, tr2, tr3)
                tr_list.append(tr)
            
            atr = np.mean(tr_list)
            return atr
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            return 0
    
    def detect_price_spike(self, data, threshold_percent=1.0):
        """Detect unusual price spikes"""
        try:
            if len(data['close']) < 2:
                return False
            
            current_price = data['close'][-1]
            previous_price = data['close'][-2]
            
            price_change_percent = abs((current_price - previous_price) / previous_price) * 100
            
            return price_change_percent > threshold_percent
        except Exception as e:
            print(f"Error detecting price spike: {e}")
            return False
    
    def calculate_support_resistance(self, data, window=20):
        """Calculate basic support and resistance levels"""
        try:
            if len(data) < window:
                current_price = data['close'][-1]
                return current_price, current_price
            
            highs = data['high'][-window:]
            lows = data['low'][-window:]
            
            resistance = np.max(highs)
            support = np.min(lows)
            
            return support, resistance
        except Exception as e:
            print(f"Error calculating support/resistance: {e}")
            current_price = data['close'][-1] if len(data['close']) > 0 else 0
            return current_price, current_price
    
    def get_trend_direction(self, data, short_period=20, long_period=50):
        """Determine overall trend direction"""
        try:
            if len(data['close']) < long_period:
                return "SIDEWAYS"
            
            short_ma = self.calculate_ma(data, short_period)
            long_ma = self.calculate_ma(data, long_period)
            
            if short_ma is None or long_ma is None:
                return "SIDEWAYS"
            
            if short_ma > long_ma:
                return "UPTREND"
            elif short_ma < long_ma:
                return "DOWNTREND"
            else:
                return "SIDEWAYS"
        except Exception as e:
            print(f"Error determining trend: {e}")
            return "SIDEWAYS"
    
    def calculate_volatility(self, data, period=20):
        """Calculate price volatility"""
        try:
            if len(data['close']) < period:
                return 0
            
            closes = data['close'][-period:]
            returns = np.diff(np.log(closes))
            volatility = np.std(returns) * np.sqrt(period)
            
            return volatility
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return 0
