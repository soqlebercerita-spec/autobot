"""
Technical Analysis Module for AuraTrade Bot
Custom implementation of technical indicators without TA-Lib dependency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

from utils.logger import Logger

class TechnicalAnalysis:
    """Custom Technical Analysis Implementation"""
    
    def __init__(self):
        self.logger = Logger()
        
    def calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < period:
                return np.full(len(prices), np.nan)
            
            sma = np.full(len(prices), np.nan)
            for i in range(period - 1, len(prices)):
                sma[i] = np.mean(prices[i - period + 1:i + 1])
            
            return sma
            
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {str(e)}")
            return np.full(len(prices), np.nan)
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return np.full(len(prices), np.nan)
            
            ema = np.full(len(prices), np.nan)
            multiplier = 2.0 / (period + 1)
            
            # Initialize with SMA
            ema[period - 1] = np.mean(prices[:period])
            
            # Calculate EMA
            for i in range(period, len(prices)):
                ema[i] = (prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
            
            return ema
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {str(e)}")
            return np.full(len(prices), np.nan)
    
    def calculate_wma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Weighted Moving Average"""
        try:
            if len(prices) < period:
                return np.full(len(prices), np.nan)
            
            wma = np.full(len(prices), np.nan)
            weights = np.arange(1, period + 1)
            weight_sum = np.sum(weights)
            
            for i in range(period - 1, len(prices)):
                window = prices[i - period + 1:i + 1]
                wma[i] = np.sum(window * weights) / weight_sum
            
            return wma
            
        except Exception as e:
            self.logger.error(f"Error calculating WMA: {str(e)}")
            return np.full(len(prices), np.nan)
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return np.full(len(prices), 50.0)
            
            rsi = np.full(len(prices), 50.0)
            
            # Calculate price changes
            deltas = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calculate initial average gain and loss
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            
            # Calculate RSI for the first valid point
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi[period] = 100 - (100 / (1 + rs))
            
            # Calculate subsequent RSI values using smoothed averages
            for i in range(period + 1, len(prices)):
                gain = gains[i - 1] if i - 1 < len(gains) else 0
                loss = losses[i - 1] if i - 1 < len(losses) else 0
                
                # Smoothed averages
                avg_gain = (avg_gain * (period - 1) + gain) / period
                avg_loss = (avg_loss * (period - 1) + loss) / period
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
                else:
                    rsi[i] = 100
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            return np.full(len(prices), 50.0)
    
    def calculate_macd(self, prices: np.ndarray, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD, Signal line, and Histogram"""
        try:
            # Calculate EMAs
            ema_fast = self.calculate_ema(prices, fast_period)
            ema_slow = self.calculate_ema(prices, slow_period)
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line (EMA of MACD)
            signal_line = self.calculate_ema(macd_line[~np.isnan(macd_line)], signal_period)
            
            # Align signal line with MACD line
            aligned_signal = np.full(len(macd_line), np.nan)
            signal_start = len(macd_line) - len(signal_line)
            aligned_signal[signal_start:] = signal_line
            
            # Calculate histogram
            histogram = macd_line - aligned_signal
            
            return macd_line, aligned_signal, histogram
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {str(e)}")
            return (np.full(len(prices), np.nan), 
                   np.full(len(prices), np.nan), 
                   np.full(len(prices), np.nan))
    
    def calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                nan_array = np.full(len(prices), np.nan)
                return nan_array, nan_array, nan_array
            
            # Calculate middle band (SMA)
            middle_band = self.calculate_sma(prices, period)
            
            # Calculate standard deviation
            std_values = np.full(len(prices), np.nan)
            for i in range(period - 1, len(prices)):
                window = prices[i - period + 1:i + 1]
                std_values[i] = np.std(window)
            
            # Calculate upper and lower bands
            upper_band = middle_band + (std_values * std_dev)
            lower_band = middle_band - (std_values * std_dev)
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            nan_array = np.full(len(prices), np.nan)
            return nan_array, nan_array, nan_array
    
    def calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                           k_period: int = 14, d_period: int = 3) -> np.ndarray:
        """Calculate Stochastic %K"""
        try:
            if len(high) < k_period or len(low) < k_period or len(close) < k_period:
                return np.full(len(close), 50.0)
            
            stoch_k = np.full(len(close), 50.0)
            
            for i in range(k_period - 1, len(close)):
                # Get the period window
                high_window = high[i - k_period + 1:i + 1]
                low_window = low[i - k_period + 1:i + 1]
                current_close = close[i]
                
                # Calculate %K
                highest_high = np.max(high_window)
                lowest_low = np.min(low_window)
                
                if highest_high != lowest_low:
                    stoch_k[i] = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
                else:
                    stoch_k[i] = 50.0
            
            return stoch_k
            
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {str(e)}")
            return np.full(len(close), 50.0)
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        try:
            if len(high) < 2 or len(low) < 2 or len(close) < 2:
                return np.full(len(close), 0.001)
            
            # Calculate True Range
            tr = np.full(len(close), 0.0)
            
            for i in range(1, len(close)):
                tr1 = high[i] - low[i]  # Current high - current low
                tr2 = abs(high[i] - close[i - 1])  # Current high - previous close
                tr3 = abs(low[i] - close[i - 1])   # Current low - previous close
                
                tr[i] = max(tr1, tr2, tr3)
            
            # Calculate ATR using smoothed average
            atr = np.full(len(close), 0.001)
            
            if len(tr) >= period:
                # Initial ATR
                atr[period - 1] = np.mean(tr[1:period])
                
                # Subsequent ATR values (smoothed)
                for i in range(period, len(close)):
                    atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {str(e)}")
            return np.full(len(close), 0.001)
    
    def calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                           period: int = 14) -> np.ndarray:
        """Calculate Williams %R"""
        try:
            if len(high) < period or len(low) < period or len(close) < period:
                return np.full(len(close), -50.0)
            
            williams_r = np.full(len(close), -50.0)
            
            for i in range(period - 1, len(close)):
                high_window = high[i - period + 1:i + 1]
                low_window = low[i - period + 1:i + 1]
                current_close = close[i]
                
                highest_high = np.max(high_window)
                lowest_low = np.min(low_window)
                
                if highest_high != lowest_low:
                    williams_r[i] = ((highest_high - current_close) / (highest_high - lowest_low)) * -100
                else:
                    williams_r[i] = -50.0
            
            return williams_r
            
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {str(e)}")
            return np.full(len(close), -50.0)
    
    def calculate_cci(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                     period: int = 20) -> np.ndarray:
        """Calculate Commodity Channel Index"""
        try:
            if len(high) < period or len(low) < period or len(close) < period:
                return np.full(len(close), 0.0)
            
            # Calculate Typical Price
            typical_price = (high + low + close) / 3
            
            # Calculate SMA of Typical Price
            sma_tp = self.calculate_sma(typical_price, period)
            
            # Calculate Mean Deviation
            mean_deviation = np.full(len(close), 0.0)
            
            for i in range(period - 1, len(close)):
                window = typical_price[i - period + 1:i + 1]
                sma_value = sma_tp[i]
                
                if not np.isnan(sma_value):
                    deviations = np.abs(window - sma_value)
                    mean_deviation[i] = np.mean(deviations)
            
            # Calculate CCI
            cci = np.full(len(close), 0.0)
            
            for i in range(period - 1, len(close)):
                if mean_deviation[i] != 0 and not np.isnan(sma_tp[i]):
                    cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mean_deviation[i])
            
            return cci
            
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {str(e)}")
            return np.full(len(close), 0.0)
    
    def calculate_momentum(self, prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Calculate Price Momentum"""
        try:
            momentum = np.full(len(prices), 0.0)
            
            for i in range(period, len(prices)):
                momentum[i] = prices[i] - prices[i - period]
            
            return momentum
            
        except Exception as e:
            self.logger.error(f"Error calculating Momentum: {str(e)}")
            return np.full(len(prices), 0.0)
    
    def calculate_roc(self, prices: np.ndarray, period: int = 10) -> np.ndarray:
        """Calculate Rate of Change"""
        try:
            roc = np.full(len(prices), 0.0)
            
            for i in range(period, len(prices)):
                if prices[i - period] != 0:
                    roc[i] = ((prices[i] - prices[i - period]) / prices[i - period]) * 100
            
            return roc
            
        except Exception as e:
            self.logger.error(f"Error calculating ROC: {str(e)}")
            return np.full(len(prices), 0.0)
    
    def detect_support_resistance(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                                 window: int = 20, min_touches: int = 2) -> Dict[str, List[float]]:
        """Detect Support and Resistance Levels"""
        try:
            if len(high) < window * 2:
                return {'support': [], 'resistance': []}
            
            support_levels = []
            resistance_levels = []
            
            # Find local minima for support
            for i in range(window, len(low) - window):
                if low[i] == np.min(low[i - window:i + window + 1]):
                    # Check for multiple touches
                    touches = 0
                    level = low[i]
                    tolerance = (np.max(high) - np.min(low)) * 0.002  # 0.2% tolerance
                    
                    for j in range(max(0, i - window * 2), min(len(low), i + window * 2)):
                        if abs(low[j] - level) <= tolerance:
                            touches += 1
                    
                    if touches >= min_touches:
                        support_levels.append(level)
            
            # Find local maxima for resistance
            for i in range(window, len(high) - window):
                if high[i] == np.max(high[i - window:i + window + 1]):
                    # Check for multiple touches
                    touches = 0
                    level = high[i]
                    tolerance = (np.max(high) - np.min(low)) * 0.002  # 0.2% tolerance
                    
                    for j in range(max(0, i - window * 2), min(len(high), i + window * 2)):
                        if abs(high[j] - level) <= tolerance:
                            touches += 1
                    
                    if touches >= min_touches:
                        resistance_levels.append(level)
            
            # Remove duplicates and sort
            support_levels = sorted(list(set(support_levels)))
            resistance_levels = sorted(list(set(resistance_levels)))
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting support/resistance: {str(e)}")
            return {'support': [], 'resistance': []}
    
    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Calculate Pivot Points"""
        try:
            pivot = (high + low + close) / 3
            
            # Resistance levels
            r1 = (2 * pivot) - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            # Support levels
            s1 = (2 * pivot) - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            return {
                'pivot': pivot,
                'r1': r1, 'r2': r2, 'r3': r3,
                's1': s1, 's2': s2, 's3': s3
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating pivot points: {str(e)}")
            return {}
    
    def calculate_fibonacci_retracements(self, high: float, low: float) -> Dict[str, float]:
        """Calculate Fibonacci Retracement Levels"""
        try:
            diff = high - low
            
            levels = {
                '0.0': high,
                '23.6': high - (diff * 0.236),
                '38.2': high - (diff * 0.382),
                '50.0': high - (diff * 0.500),
                '61.8': high - (diff * 0.618),
                '78.6': high - (diff * 0.786),
                '100.0': low
            }
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Error calculating Fibonacci retracements: {str(e)}")
            return {}
    
    def analyze_trend(self, prices: np.ndarray, period: int = 20) -> Dict[str, Any]:
        """Analyze price trend"""
        try:
            if len(prices) < period:
                return {'trend': 'sideways', 'strength': 0.0, 'angle': 0.0}
            
            # Use linear regression to determine trend
            x = np.arange(len(prices))
            coefficients = np.polyfit(x[-period:], prices[-period:], 1)
            slope = coefficients[0]
            
            # Calculate trend strength (R-squared)
            y_pred = np.polyval(coefficients, x[-period:])
            ss_res = np.sum((prices[-period:] - y_pred) ** 2)
            ss_tot = np.sum((prices[-period:] - np.mean(prices[-period:])) ** 2)
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Determine trend direction
            if slope > 0.0001:
                trend = 'uptrend'
            elif slope < -0.0001:
                trend = 'downtrend'
            else:
                trend = 'sideways'
            
            # Calculate angle in degrees
            angle = np.degrees(np.arctan(slope))
            
            return {
                'trend': trend,
                'strength': abs(r_squared),
                'angle': angle,
                'slope': slope
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend: {str(e)}")
            return {'trend': 'sideways', 'strength': 0.0, 'angle': 0.0}
    
    def calculate_volume_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Volume-based indicators"""
        try:
            if len(prices) != len(volumes) or len(prices) < 10:
                default_array = np.full(len(prices), 0.0)
                return {
                    'obv': default_array,
                    'volume_sma': default_array,
                    'volume_ratio': np.ones(len(prices))
                }
            
            # On Balance Volume (OBV)
            obv = np.zeros(len(prices))
            for i in range(1, len(prices)):
                if prices[i] > prices[i - 1]:
                    obv[i] = obv[i - 1] + volumes[i]
                elif prices[i] < prices[i - 1]:
                    obv[i] = obv[i - 1] - volumes[i]
                else:
                    obv[i] = obv[i - 1]
            
            # Volume SMA
            volume_sma = self.calculate_sma(volumes, 20)
            
            # Volume Ratio
            volume_ratio = np.ones(len(volumes))
            for i in range(len(volumes)):
                if not np.isnan(volume_sma[i]) and volume_sma[i] > 0:
                    volume_ratio[i] = volumes[i] / volume_sma[i]
            
            return {
                'obv': obv,
                'volume_sma': volume_sma,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {str(e)}")
            default_array = np.full(len(prices), 0.0)
            return {
                'obv': default_array,
                'volume_sma': default_array,
                'volume_ratio': np.ones(len(prices))
            }
    
    def get_comprehensive_analysis(self, high: np.ndarray, low: np.ndarray, 
                                 close: np.ndarray, volumes: np.ndarray = None) -> Dict[str, Any]:
        """Get comprehensive technical analysis"""
        try:
            if volumes is None:
                volumes = np.ones(len(close))
            
            # Basic indicators
            sma_20 = self.calculate_sma(close, 20)
            ema_12 = self.calculate_ema(close, 12)
            rsi = self.calculate_rsi(close, 14)
            
            # MACD
            macd_line, signal_line, histogram = self.calculate_macd(close)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(close)
            
            # Stochastic
            stoch_k = self.calculate_stochastic(high, low, close)
            
            # ATR
            atr = self.calculate_atr(high, low, close)
            
            # Support/Resistance
            sr_levels = self.detect_support_resistance(high, low, close)
            
            # Trend Analysis
            trend_info = self.analyze_trend(close)
            
            # Volume indicators
            volume_indicators = self.calculate_volume_indicators(close, volumes)
            
            # Current values (last available)
            current_values = {
                'price': close[-1] if len(close) > 0 else 0,
                'sma_20': sma_20[-1] if not np.isnan(sma_20[-1]) else 0,
                'ema_12': ema_12[-1] if not np.isnan(ema_12[-1]) else 0,
                'rsi': rsi[-1],
                'macd': macd_line[-1] if not np.isnan(macd_line[-1]) else 0,
                'signal': signal_line[-1] if not np.isnan(signal_line[-1]) else 0,
                'bb_upper': bb_upper[-1] if not np.isnan(bb_upper[-1]) else 0,
                'bb_lower': bb_lower[-1] if not np.isnan(bb_lower[-1]) else 0,
                'stoch_k': stoch_k[-1],
                'atr': atr[-1]
            }
            
            return {
                'indicators': {
                    'sma_20': sma_20,
                    'ema_12': ema_12,
                    'rsi': rsi,
                    'macd_line': macd_line,
                    'signal_line': signal_line,
                    'histogram': histogram,
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'stoch_k': stoch_k,
                    'atr': atr
                },
                'current_values': current_values,
                'support_resistance': sr_levels,
                'trend_analysis': trend_info,
                'volume_analysis': volume_indicators,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            return {'error': str(e)}
