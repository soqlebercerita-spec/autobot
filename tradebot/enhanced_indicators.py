
"""
Enhanced Technical Indicators for AuraTrade Bot
Advanced signal generation with multiple confirmations
"""

import math
import time
from typing import List, Dict, Any, Optional, Tuple

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Mock numpy for basic calculations
    class MockNumpy:
        @staticmethod
        def array(data): return data
        @staticmethod
        def mean(data): return sum(data) / len(data) if data else 0
        @staticmethod
        def std(data):
            if not data: return 0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return math.sqrt(variance)
        @staticmethod
        def max(data): return max(data) if data else 0
        @staticmethod
        def min(data): return min(data) if data else 0
        @staticmethod
        def corrcoef(x, y): return [[1.0, 0.5], [0.5, 1.0]]
        @staticmethod
        def polyfit(x, y, deg): return [0.1, 0.2, 0.3][:deg+1]
        @staticmethod
        def percentile(data, pct):
            if not data: return 0
            sorted_data = sorted(data)
            k = int((len(sorted_data) - 1) * pct / 100)
            return sorted_data[k]
    
    np = MockNumpy()

class EnhancedIndicators:
    """Enhanced technical indicators with advanced signal analysis"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 30  # 30 seconds cache
        
        print("📈 Enhanced Technical Indicators initialized")
    
    def _cache_key(self, indicator_name: str, params: tuple) -> str:
        """Generate cache key"""
        return f"{indicator_name}_{hash(params)}"
    
    def _get_cached(self, key: str):
        """Get cached result if valid"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_timeout:
                return value
        return None
    
    def _set_cache(self, key: str, value):
        """Set cache value"""
        self.cache[key] = (value, time.time())
    
    def calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Simple Moving Average"""
        if len(prices) < period:
            return None
        
        cache_key = self._cache_key("sma", (tuple(prices[-period:]), period))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        result = np.mean(prices[-period:])
        self._set_cache(cache_key, result)
        return result
    
    def calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Exponential Moving Average"""
        if len(prices) < period:
            return None
        
        cache_key = self._cache_key("ema", (tuple(prices), period))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        multiplier = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        self._set_cache(cache_key, ema)
        return ema
    
    def calculate_wma(self, prices: List[float], period: int) -> Optional[float]:
        """Weighted Moving Average"""
        if len(prices) < period:
            return None
        
        cache_key = self._cache_key("wma", (tuple(prices[-period:]), period))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        weights = list(range(1, period + 1))
        weighted_prices = [prices[-period + i] * weights[i] for i in range(period)]
        result = sum(weighted_prices) / sum(weights)
        
        self._set_cache(cache_key, result)
        return result
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        cache_key = self._cache_key("rsi", (tuple(prices), period))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return None
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            result = 100
        else:
            rs = avg_gain / avg_loss
            result = 100 - (100 / (1 + rs))
        
        self._set_cache(cache_key, result)
        return result
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """MACD Indicator"""
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        cache_key = self._cache_key("macd", (tuple(prices), fast, slow, signal))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        
        if not ema_fast or not ema_slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        macd_line = ema_fast - ema_slow
        
        # Simplified signal line calculation
        signal_line = macd_line * 0.9  # Approximation
        histogram = macd_line - signal_line
        
        result = {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
        
        self._set_cache(cache_key, result)
        return result
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Bollinger Bands"""
        if len(prices) < period:
            return None, None, None
        
        cache_key = self._cache_key("bb", (tuple(prices[-period:]), period, std_dev))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        middle = self.calculate_sma(prices, period)
        if not middle:
            return None, None, None
        
        std = np.std(prices[-period:])
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        result = (upper, lower, middle)
        self._set_cache(cache_key, result)
        return result
    
    def calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Stochastic Oscillator"""
        if len(closes) < k_period or len(highs) < k_period or len(lows) < k_period:
            return {'k': 50, 'd': 50}
        
        cache_key = self._cache_key("stoch", (tuple(highs[-k_period:]), tuple(lows[-k_period:]), 
                                            tuple(closes[-k_period:]), k_period, d_period))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        highest_high = np.max(highs[-k_period:])
        lowest_low = np.min(lows[-k_period:])
        current_close = closes[-1]
        
        if highest_high == lowest_low:
            k_percent = 50
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Simplified D% calculation
        d_percent = k_percent * 0.9  # Approximation
        
        result = {'k': k_percent, 'd': d_percent}
        self._set_cache(cache_key, result)
        return result
    
    def calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """Average True Range"""
        if len(closes) < period + 1 or len(highs) < period + 1 or len(lows) < period + 1:
            return None
        
        cache_key = self._cache_key("atr", (tuple(highs), tuple(lows), tuple(closes), period))
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        true_ranges = []
        
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i-1])
            low_close_prev = abs(lows[i] - closes[i-1])
            
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)
        
        if len(true_ranges) < period:
            return None
        
        result = np.mean(true_ranges[-period:])
        self._set_cache(cache_key, result)
        return result
    
    def calculate_adx(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict[str, float]:
        """Average Directional Index"""
        if len(closes) < period + 1:
            return {'adx': 25, 'di_plus': 25, 'di_minus': 25}
        
        # Simplified ADX calculation
        atr = self.calculate_atr(highs, lows, closes, period)
        if not atr:
            return {'adx': 25, 'di_plus': 25, 'di_minus': 25}
        
        # Basic trend strength calculation
        price_change = closes[-1] - closes[-period]
        volatility = np.std(closes[-period:])
        
        if volatility > 0:
            trend_strength = min(abs(price_change / volatility) * 10, 100)
        else:
            trend_strength = 25
        
        if price_change > 0:
            di_plus = trend_strength
            di_minus = 100 - trend_strength
        else:
            di_plus = 100 - trend_strength
            di_minus = trend_strength
        
        adx = (abs(di_plus - di_minus) / (di_plus + di_minus)) * 100 if (di_plus + di_minus) > 0 else 25
        
        return {
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus
        }
    
    def calculate_fibonacci_levels(self, prices: List[float], period: int = 50) -> Dict[str, float]:
        """Fibonacci Retracement Levels"""
        if len(prices) < period:
            return {}
        
        recent_prices = prices[-period:]
        high = np.max(recent_prices)
        low = np.min(recent_prices)
        
        diff = high - low
        
        return {
            'level_0': high,
            'level_23.6': high - 0.236 * diff,
            'level_38.2': high - 0.382 * diff,
            'level_50': high - 0.5 * diff,
            'level_61.8': high - 0.618 * diff,
            'level_100': low
        }
    
    def calculate_pivot_points(self, high: float, low: float, close: float) -> Dict[str, float]:
        """Pivot Points"""
        pivot = (high + low + close) / 3
        
        return {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low),
            's1': 2 * pivot - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot)
        }
    
    def detect_patterns(self, prices: List[float], period: int = 20) -> Dict[str, Any]:
        """Pattern Recognition"""
        if len(prices) < period:
            return {'patterns': [], 'strength': 0}
        
        patterns = []
        recent_prices = prices[-period:]
        
        # Double Top/Bottom
        if self._detect_double_top(recent_prices):
            patterns.append('Double Top')
        
        if self._detect_double_bottom(recent_prices):
            patterns.append('Double Bottom')
        
        # Head and Shoulders
        if self._detect_head_shoulders(recent_prices):
            patterns.append('Head and Shoulders')
        
        # Triangle
        if self._detect_triangle(recent_prices):
            patterns.append('Triangle')
        
        # Flag/Pennant
        if self._detect_flag(recent_prices):
            patterns.append('Flag')
        
        strength = len(patterns) * 20  # Each pattern adds 20% strength
        
        return {
            'patterns': patterns,
            'strength': min(strength, 100)
        }
    
    def _detect_double_top(self, prices: List[float]) -> bool:
        """Detect Double Top pattern"""
        if len(prices) < 10:
            return False
        
        # Find peaks
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        
        if len(peaks) < 2:
            return False
        
        # Check if last two peaks are similar height
        last_two_peaks = peaks[-2:]
        height_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1])
        avg_height = (last_two_peaks[0][1] + last_two_peaks[1][1]) / 2
        
        return height_diff / avg_height < 0.02  # Within 2%
    
    def _detect_double_bottom(self, prices: List[float]) -> bool:
        """Detect Double Bottom pattern"""
        if len(prices) < 10:
            return False
        
        # Find troughs
        troughs = []
        for i in range(1, len(prices) - 1):
            if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append((i, prices[i]))
        
        if len(troughs) < 2:
            return False
        
        # Check if last two troughs are similar depth
        last_two_troughs = troughs[-2:]
        depth_diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1])
        avg_depth = (last_two_troughs[0][1] + last_two_troughs[1][1]) / 2
        
        return depth_diff / avg_depth < 0.02  # Within 2%
    
    def _detect_head_shoulders(self, prices: List[float]) -> bool:
        """Detect Head and Shoulders pattern"""
        if len(prices) < 15:
            return False
        
        # Find peaks
        peaks = []
        for i in range(2, len(prices) - 2):
            if (prices[i] > prices[i-1] and prices[i] > prices[i+1] and
                prices[i] > prices[i-2] and prices[i] > prices[i+2]):
                peaks.append((i, prices[i]))
        
        if len(peaks) < 3:
            return False
        
        # Check for head and shoulders pattern
        last_three = peaks[-3:]
        left_shoulder = last_three[0][1]
        head = last_three[1][1]
        right_shoulder = last_three[2][1]
        
        # Head should be higher than shoulders
        if head > left_shoulder and head > right_shoulder:
            # Shoulders should be roughly equal
            shoulder_diff = abs(left_shoulder - right_shoulder)
            shoulder_avg = (left_shoulder + right_shoulder) / 2
            return shoulder_diff / shoulder_avg < 0.05  # Within 5%
        
        return False
    
    def _detect_triangle(self, prices: List[float]) -> bool:
        """Detect Triangle pattern"""
        if len(prices) < 15:
            return False
        
        # Calculate trend lines
        x = list(range(len(prices)))
        upper_trend = np.polyfit(x, prices, 1)[0]
        
        # Check if price is consolidating (decreasing volatility)
        first_half_vol = np.std(prices[:len(prices)//2])
        second_half_vol = np.std(prices[len(prices)//2:])
        
        return second_half_vol < first_half_vol * 0.8  # Volatility decreased by 20%
    
    def _detect_flag(self, prices: List[float]) -> bool:
        """Detect Flag pattern"""
        if len(prices) < 10:
            return False
        
        # Look for strong move followed by consolidation
        first_half = prices[:len(prices)//2]
        second_half = prices[len(prices)//2:]
        
        first_trend = (first_half[-1] - first_half[0]) / len(first_half)
        second_volatility = np.std(second_half)
        
        # Strong initial move with low volatility consolidation
        return abs(first_trend) > second_volatility * 2
    
    def enhanced_signal_analysis(self, prices: List[float], highs: List[float] = None, 
                                lows: List[float] = None, symbol: str = "EURUSD") -> Dict[str, Any]:
        """Enhanced signal analysis with multiple confirmations"""
        
        if len(prices) < 50:
            return {
                'signal': 'HOLD',
                'confidence': 0.5,
                'strength': 0.5,
                'indicators': {},
                'scores': {'buy_score': 0, 'sell_score': 0},
                'reasons': ['Insufficient data'],
                'risk_level': 'HIGH'
            }
        
        # Use prices for highs/lows if not provided
        if not highs:
            highs = prices.copy()
        if not lows:
            lows = prices.copy()
        
        # Calculate all indicators
        indicators = {}
        
        # Moving Averages
        indicators['sma_10'] = self.calculate_sma(prices, 10)
        indicators['sma_20'] = self.calculate_sma(prices, 20)
        indicators['sma_50'] = self.calculate_sma(prices, 50)
        indicators['ema_12'] = self.calculate_ema(prices, 12)
        indicators['ema_26'] = self.calculate_ema(prices, 26)
        indicators['wma_10'] = self.calculate_wma(prices, 10)
        
        # Oscillators
        indicators['rsi'] = self.calculate_rsi(prices, 14)
        indicators['macd'] = self.calculate_macd(prices)
        indicators['stoch'] = self.calculate_stochastic(highs, lows, prices)
        
        # Volatility and Trend
        indicators['bb_upper'], indicators['bb_lower'], indicators['bb_middle'] = self.calculate_bollinger_bands(prices)
        indicators['atr'] = self.calculate_atr(highs, lows, prices)
        indicators['adx'] = self.calculate_adx(highs, lows, prices)
        
        # Pattern Recognition
        indicators['patterns'] = self.detect_patterns(prices)
        
        # Fibonacci and Support/Resistance
        indicators['fibonacci'] = self.calculate_fibonacci_levels(prices)
        if len(prices) >= 3:
            indicators['pivot'] = self.calculate_pivot_points(max(prices[-3:]), min(prices[-3:]), prices[-1])
        
        # Current price
        current_price = prices[-1]
        
        # Signal Scoring System
        buy_score = 0
        sell_score = 0
        reasons = []
        
        # 1. Trend Analysis (Weight: 25%)
        if indicators['sma_20'] and indicators['sma_50']:
            if indicators['sma_20'] > indicators['sma_50']:
                buy_score += 2.5
                reasons.append("Bullish trend (SMA20 > SMA50)")
            else:
                sell_score += 2.5
                reasons.append("Bearish trend (SMA20 < SMA50)")
        
        # 2. Moving Average Crossovers (Weight: 20%)
        if indicators['ema_12'] and indicators['ema_26']:
            if indicators['ema_12'] > indicators['ema_26']:
                buy_score += 2
                reasons.append("EMA bullish crossover")
            else:
                sell_score += 2
                reasons.append("EMA bearish crossover")
        
        # 3. RSI Analysis (Weight: 15%)
        if indicators['rsi']:
            if indicators['rsi'] < 30:
                buy_score += 2
                reasons.append(f"RSI oversold ({indicators['rsi']:.1f})")
            elif indicators['rsi'] > 70:
                sell_score += 2
                reasons.append(f"RSI overbought ({indicators['rsi']:.1f})")
            elif 45 < indicators['rsi'] < 55:
                buy_score += 0.5
                sell_score += 0.5
        
        # 4. MACD Analysis (Weight: 15%)
        if indicators['macd']['histogram'] > 0:
            buy_score += 1.5
            reasons.append("MACD bullish momentum")
        elif indicators['macd']['histogram'] < 0:
            sell_score += 1.5
            reasons.append("MACD bearish momentum")
        
        # 5. Bollinger Bands (Weight: 10%)
        if indicators['bb_upper'] and indicators['bb_lower']:
            if current_price <= indicators['bb_lower']:
                buy_score += 1.5
                reasons.append("Price at lower Bollinger Band")
            elif current_price >= indicators['bb_upper']:
                sell_score += 1.5
                reasons.append("Price at upper Bollinger Band")
        
        # 6. Stochastic (Weight: 10%)
        if indicators['stoch']['k'] < 20:
            buy_score += 1
            reasons.append("Stochastic oversold")
        elif indicators['stoch']['k'] > 80:
            sell_score += 1
            reasons.append("Stochastic overbought")
        
        # 7. ADX Trend Strength (Weight: 5%)
        if indicators['adx']['adx'] > 25:
            strength_multiplier = 1.2
            if indicators['adx']['di_plus'] > indicators['adx']['di_minus']:
                buy_score *= strength_multiplier
                reasons.append("Strong bullish trend (ADX)")
            else:
                sell_score *= strength_multiplier
                reasons.append("Strong bearish trend (ADX)")
        
        # 8. Pattern Recognition Bonus
        if indicators['patterns']['patterns']:
            pattern_bonus = indicators['patterns']['strength'] / 100
            if any(pattern in ['Double Bottom', 'Head and Shoulders', 'Flag'] 
                   for pattern in indicators['patterns']['patterns']):
                buy_score += pattern_bonus
                reasons.append(f"Bullish pattern: {', '.join(indicators['patterns']['patterns'])}")
            elif any(pattern in ['Double Top'] for pattern in indicators['patterns']['patterns']):
                sell_score += pattern_bonus
                reasons.append(f"Bearish pattern: {', '.join(indicators['patterns']['patterns'])}")
        
        # 9. Volatility Analysis
        if indicators['atr']:
            volatility = indicators['atr'] / current_price
            if volatility > 0.02:  # High volatility
                buy_score *= 1.1
                sell_score *= 1.1
                reasons.append("High volatility - enhanced signals")
            elif volatility < 0.005:  # Low volatility
                buy_score *= 0.8
                sell_score *= 0.8
        
        # 10. Market Session Bonus
        current_hour = time.localtime().tm_hour
        if 8 <= current_hour <= 17:  # Active trading hours
            buy_score *= 1.1
            sell_score *= 1.1
        
        # Determine final signal
        total_score = buy_score + sell_score
        if total_score > 0:
            buy_confidence = buy_score / total_score
            sell_confidence = sell_score / total_score
        else:
            buy_confidence = sell_confidence = 0.5
        
        # Signal determination with enhanced thresholds
        if buy_score > sell_score and buy_confidence > 0.6:
            signal = 'BUY'
            confidence = buy_confidence
            strength = min(buy_score / 10, 1.0)
        elif sell_score > buy_score and sell_confidence > 0.6:
            signal = 'SELL'
            confidence = sell_confidence
            strength = min(sell_score / 10, 1.0)
        else:
            signal = 'HOLD'
            confidence = 0.5
            strength = 0.5
        
        # Risk Assessment
        risk_factors = 0
        if indicators['atr'] and indicators['atr'] / current_price > 0.03:
            risk_factors += 1
        if indicators['rsi'] and (indicators['rsi'] > 80 or indicators['rsi'] < 20):
            risk_factors += 1
        if not indicators['patterns']['patterns']:
            risk_factors += 1
        
        if risk_factors >= 2:
            risk_level = 'HIGH'
        elif risk_factors == 1:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'signal': signal,
            'confidence': confidence,
            'strength': strength,
            'indicators': indicators,
            'scores': {
                'buy_score': buy_score,
                'sell_score': sell_score,
                'total_score': total_score
            },
            'reasons': reasons[:5],  # Top 5 reasons
            'risk_level': risk_level,
            'market_conditions': {
                'trend': 'BULLISH' if buy_score > sell_score else 'BEARISH' if sell_score > buy_score else 'SIDEWAYS',
                'volatility': 'HIGH' if indicators['atr'] and indicators['atr'] / current_price > 0.02 else 'NORMAL',
                'momentum': 'STRONG' if strength > 0.7 else 'WEAK' if strength < 0.3 else 'MODERATE'
            }
        }

# Test function
def test_indicators():
    """Test enhanced indicators"""
    import random
    
    # Generate test data
    prices = [1.0850 + (i * 0.0001) + random.uniform(-0.001, 0.001) for i in range(100)]
    highs = [p + random.uniform(0, 0.0005) for p in prices]
    lows = [p - random.uniform(0, 0.0005) for p in prices]
    
    indicators = EnhancedIndicators()
    
    print("🧪 Testing Enhanced Indicators")
    print("=" * 50)
    
    # Test signal analysis
    result = indicators.enhanced_signal_analysis(prices, highs, lows, "EURUSD")
    
    print(f"Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Strength: {result['strength']:.2f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Reasons: {', '.join(result['reasons'])}")
    print(f"Market Conditions: {result['market_conditions']}")
    
    print("=" * 50)

if __name__ == "__main__":
    test_indicators()
