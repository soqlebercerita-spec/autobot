"""
Adaptive Technical Indicators
Self-adjusting technical analysis based on market conditions
"""

import numpy as np
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from config import config

class AdaptiveMode(Enum):
    STATIC = "static"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    TREND_ADJUSTED = "trend_adjusted"
    VOLUME_ADJUSTED = "volume_adjusted"
    REGIME_ADJUSTED = "regime_adjusted"
    FULL_ADAPTIVE = "full_adaptive"

@dataclass
class AdaptiveParams:
    base_period: int
    adjusted_period: int
    sensitivity: float
    volatility_factor: float
    trend_factor: float
    volume_factor: float
    confidence: float
    last_update: datetime

class AdaptiveIndicators:
    def __init__(self, market_data_api):
        self.market_api = market_data_api
        
        # Adaptive parameters for each indicator
        self.adaptive_params = {
            'ma': AdaptiveParams(20, 20, 1.0, 1.0, 1.0, 1.0, 0.8, datetime.now()),
            'ema': AdaptiveParams(12, 12, 1.0, 1.0, 1.0, 1.0, 0.8, datetime.now()),
            'rsi': AdaptiveParams(14, 14, 1.0, 1.0, 1.0, 1.0, 0.8, datetime.now()),
            'macd': AdaptiveParams(26, 26, 1.0, 1.0, 1.0, 1.0, 0.8, datetime.now()),
            'bollinger': AdaptiveParams(20, 20, 2.0, 1.0, 1.0, 1.0, 0.8, datetime.now()),
            'stochastic': AdaptiveParams(14, 14, 1.0, 1.0, 1.0, 1.0, 0.8, datetime.now())
        }
        
        # Market condition cache
        self.market_conditions = {}
        self.adaptation_history = {}
        
        # Adaptation settings
        self.adaptation_config = {
            'update_frequency': 300,  # 5 minutes
            'volatility_threshold': 0.02,
            'trend_threshold': 0.001,
            'volume_threshold': 1.2,
            'max_period_adjustment': 0.5,  # ±50% of base period
            'min_confidence_for_adaptation': 0.6,
            'adaptation_smoothing': 0.3  # EMA smoothing for adaptations
        }
        
        print("🔄 Adaptive Indicators initialized")
        print(f"   • Volatility Adaptation: ✅")
        print(f"   • Trend Adaptation: ✅")
        print(f"   • Volume Adaptation: ✅")
        print(f"   • Regime Adaptation: ✅")
        print(f"   • Update Frequency: {self.adaptation_config['update_frequency']}s")
    
    def adaptive_ma(self, prices: List[float], symbol: str = "UNKNOWN", mode: AdaptiveMode = AdaptiveMode.FULL_ADAPTIVE) -> Optional[float]:
        """Adaptive Moving Average"""
        try:
            if len(prices) < 10:
                return None
            
            # Update adaptive parameters
            self._update_adaptive_params('ma', prices, symbol, mode)
            
            # Get adjusted period
            period = int(self.adaptive_params['ma'].adjusted_period)
            period = max(5, min(period, len(prices)))
            
            # Calculate adaptive MA
            if mode == AdaptiveMode.VOLATILITY_ADJUSTED:
                return self._volatility_adjusted_ma(prices, period)
            elif mode == AdaptiveMode.TREND_ADJUSTED:
                return self._trend_adjusted_ma(prices, period)
            elif mode == AdaptiveMode.VOLUME_ADJUSTED:
                return self._volume_adjusted_ma(prices, period, symbol)
            elif mode == AdaptiveMode.FULL_ADAPTIVE:
                return self._full_adaptive_ma(prices, period, symbol)
            else:
                return np.mean(prices[-period:])
        
        except Exception as e:
            print(f"❌ Adaptive MA error: {e}")
            return np.mean(prices[-20:]) if len(prices) >= 20 else None
    
    def adaptive_ema(self, prices: List[float], symbol: str = "UNKNOWN", mode: AdaptiveMode = AdaptiveMode.FULL_ADAPTIVE) -> Optional[float]:
        """Adaptive Exponential Moving Average"""
        try:
            if len(prices) < 10:
                return None
            
            # Update adaptive parameters
            self._update_adaptive_params('ema', prices, symbol, mode)
            
            # Get adjusted parameters
            params = self.adaptive_params['ema']
            base_alpha = 2.0 / (params.base_period + 1)
            
            # Adjust alpha based on market conditions
            if mode == AdaptiveMode.VOLATILITY_ADJUSTED:
                volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
                alpha = base_alpha * (1 + volatility * params.volatility_factor)
            elif mode == AdaptiveMode.TREND_ADJUSTED:
                trend = self._calculate_trend_strength(prices)
                alpha = base_alpha * (1 + abs(trend) * params.trend_factor)
            else:
                # Full adaptive
                volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
                trend = self._calculate_trend_strength(prices)
                alpha = base_alpha * (1 + volatility * 0.5 + abs(trend) * 0.5)
            
            # Ensure alpha is within bounds
            alpha = max(0.01, min(alpha, 0.9))
            
            # Calculate EMA
            ema = prices[0]
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
        
        except Exception as e:
            print(f"❌ Adaptive EMA error: {e}")
            return self._calculate_ema_simple(prices, 12)
    
    def adaptive_rsi(self, prices: List[float], symbol: str = "UNKNOWN", mode: AdaptiveMode = AdaptiveMode.FULL_ADAPTIVE) -> Optional[float]:
        """Adaptive RSI with dynamic overbought/oversold levels"""
        try:
            if len(prices) < 20:
                return None
            
            # Update adaptive parameters
            self._update_adaptive_params('rsi', prices, symbol, mode)
            
            # Get adjusted period
            period = int(self.adaptive_params['rsi'].adjusted_period)
            period = max(5, min(period, len(prices) - 1))
            
            # Calculate price changes
            changes = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            # Calculate adaptive averages
            if mode == AdaptiveMode.VOLATILITY_ADJUSTED:
                volatility = np.std(changes) / np.mean(prices)
                smoothing = max(0.1, min(0.9, 1 / (1 + volatility * 10)))
            else:
                smoothing = 1.0 / period
            
            # Calculate average gain and loss
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Adaptive overbought/oversold levels
            volatility = np.std(prices[-50:]) / np.mean(prices[-50:]) if len(prices) >= 50 else 0.02
            
            # Adjust RSI interpretation based on volatility
            if volatility > 0.03:  # High volatility
                # Wider RSI bands
                self.adaptive_params['rsi'].sensitivity = 0.8
            elif volatility < 0.01:  # Low volatility
                # Tighter RSI bands
                self.adaptive_params['rsi'].sensitivity = 1.2
            
            return rsi
        
        except Exception as e:
            print(f"❌ Adaptive RSI error: {e}")
            return self._calculate_rsi_simple(prices, 14)
    
    def adaptive_macd(self, prices: List[float], symbol: str = "UNKNOWN", mode: AdaptiveMode = AdaptiveMode.FULL_ADAPTIVE) -> Optional[Dict]:
        """Adaptive MACD with dynamic periods"""
        try:
            if len(prices) < 50:
                return None
            
            # Update adaptive parameters
            self._update_adaptive_params('macd', prices, symbol, mode)
            
            # Base periods
            fast_base = 12
            slow_base = 26
            signal_base = 9
            
            # Adjust periods based on market conditions
            if mode == AdaptiveMode.TREND_ADJUSTED:
                trend_strength = abs(self._calculate_trend_strength(prices))
                if trend_strength > 0.001:  # Strong trend
                    fast_period = int(fast_base * 0.8)  # Faster
                    slow_period = int(slow_base * 0.8)
                else:  # Weak trend
                    fast_period = int(fast_base * 1.2)  # Slower
                    slow_period = int(slow_base * 1.2)
            elif mode == AdaptiveMode.VOLATILITY_ADJUSTED:
                volatility = np.std(prices[-30:]) / np.mean(prices[-30:])
                if volatility > 0.02:  # High volatility
                    fast_period = int(fast_base * 0.7)  # Much faster
                    slow_period = int(slow_base * 0.7)
                else:  # Low volatility
                    fast_period = int(fast_base * 1.3)  # Slower
                    slow_period = int(slow_base * 1.3)
            else:
                # Full adaptive
                trend_strength = abs(self._calculate_trend_strength(prices))
                volatility = np.std(prices[-30:]) / np.mean(prices[-30:])
                
                adaptation_factor = 1.0 - (trend_strength * 0.3 + volatility * 20 * 0.3)
                adaptation_factor = max(0.6, min(1.4, adaptation_factor))
                
                fast_period = int(fast_base * adaptation_factor)
                slow_period = int(slow_base * adaptation_factor)
            
            # Calculate EMAs
            fast_ema = self._calculate_ema_simple(prices, fast_period)
            slow_ema = self._calculate_ema_simple(prices, slow_period)
            
            if not fast_ema or not slow_ema:
                return None
            
            # MACD line
            macd_line = fast_ema - slow_ema
            
            # Signal line (EMA of MACD line)
            # For signal line, we need MACD history - simplified for demo
            signal_line = macd_line * 0.9  # Simplified
            
            # Histogram
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram,
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_base
            }
        
        except Exception as e:
            print(f"❌ Adaptive MACD error: {e}")
            return None
    
    def adaptive_bollinger_bands(self, prices: List[float], symbol: str = "UNKNOWN", mode: AdaptiveMode = AdaptiveMode.FULL_ADAPTIVE) -> Optional[Tuple[float, float, float]]:
        """Adaptive Bollinger Bands with dynamic deviation"""
        try:
            if len(prices) < 20:
                return None
            
            # Update adaptive parameters
            self._update_adaptive_params('bollinger', prices, symbol, mode)
            
            # Get adaptive parameters
            params = self.adaptive_params['bollinger']
            period = int(params.adjusted_period)
            base_deviation = params.sensitivity  # Base is 2.0
            
            # Calculate middle band (SMA)
            middle_band = np.mean(prices[-period:])
            
            # Calculate standard deviation
            std_dev = np.std(prices[-period:])
            
            # Adaptive deviation based on market conditions
            if mode == AdaptiveMode.VOLATILITY_ADJUSTED:
                volatility = std_dev / middle_band
                if volatility > 0.02:  # High volatility
                    deviation = base_deviation * 1.3  # Wider bands
                elif volatility < 0.005:  # Low volatility
                    deviation = base_deviation * 0.7  # Tighter bands
                else:
                    deviation = base_deviation
            elif mode == AdaptiveMode.TREND_ADJUSTED:
                trend_strength = abs(self._calculate_trend_strength(prices))
                if trend_strength > 0.001:  # Strong trend
                    deviation = base_deviation * 1.2  # Wider bands
                else:
                    deviation = base_deviation * 0.8  # Tighter bands
            else:
                # Full adaptive
                volatility = std_dev / middle_band
                trend_strength = abs(self._calculate_trend_strength(prices))
                
                vol_adjustment = 1.0 + (volatility - 0.01) * 15  # Scale volatility
                trend_adjustment = 1.0 + trend_strength * 200    # Scale trend
                
                deviation = base_deviation * vol_adjustment * trend_adjustment
                deviation = max(0.5, min(4.0, deviation))  # Reasonable bounds
            
            # Calculate bands
            upper_band = middle_band + (deviation * std_dev)
            lower_band = middle_band - (deviation * std_dev)
            
            return upper_band, lower_band, middle_band
        
        except Exception as e:
            print(f"❌ Adaptive Bollinger Bands error: {e}")
            return None
    
    def adaptive_stochastic(self, highs: List[float], lows: List[float], closes: List[float], 
                          symbol: str = "UNKNOWN", mode: AdaptiveMode = AdaptiveMode.FULL_ADAPTIVE) -> Optional[Dict]:
        """Adaptive Stochastic Oscillator"""
        try:
            if len(closes) < 20:
                return None
            
            # Update adaptive parameters
            self._update_adaptive_params('stochastic', closes, symbol, mode)
            
            # Get adaptive period
            period = int(self.adaptive_params['stochastic'].adjusted_period)
            period = max(5, min(period, len(closes)))
            
            # Use closes as highs/lows if not provided
            if not highs:
                highs = closes
            if not lows:
                lows = closes
            
            # Calculate %K
            highest_high = max(highs[-period:])
            lowest_low = min(lows[-period:])
            current_close = closes[-1]
            
            if highest_high == lowest_low:
                k_percent = 50.0
            else:
                k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
            
            # Adaptive %D smoothing
            if mode == AdaptiveMode.VOLATILITY_ADJUSTED:
                volatility = np.std(closes[-20:]) / np.mean(closes[-20:])
                if volatility > 0.02:  # High volatility
                    d_period = 2  # Faster %D
                else:
                    d_period = 5  # Slower %D
            else:
                d_period = 3  # Default
            
            # Calculate %D (simplified - would need K history for proper calculation)
            d_percent = k_percent * 0.9  # Simplified
            
            # Adaptive overbought/oversold levels
            volatility = np.std(closes[-30:]) / np.mean(closes[-30:]) if len(closes) >= 30 else 0.02
            
            if volatility > 0.03:  # High volatility
                overbought_level = 85
                oversold_level = 15
            elif volatility < 0.01:  # Low volatility
                overbought_level = 75
                oversold_level = 25
            else:
                overbought_level = 80
                oversold_level = 20
            
            return {
                'k_percent': k_percent,
                'd_percent': d_percent,
                'overbought_level': overbought_level,
                'oversold_level': oversold_level,
                'period': period,
                'volatility': volatility
            }
        
        except Exception as e:
            print(f"❌ Adaptive Stochastic error: {e}")
            return None
    
    def _update_adaptive_params(self, indicator: str, prices: List[float], symbol: str, mode: AdaptiveMode):
        """Update adaptive parameters for an indicator"""
        try:
            current_time = datetime.now()
            params = self.adaptive_params[indicator]
            
            # Check if update is needed
            if (current_time - params.last_update).seconds < self.adaptation_config['update_frequency']:
                return
            
            # Calculate market conditions
            volatility = np.std(prices[-30:]) / np.mean(prices[-30:]) if len(prices) >= 30 else 0.02
            trend_strength = self._calculate_trend_strength(prices)
            volume_factor = self._estimate_volume_factor(symbol)
            
            # Calculate adaptation factors
            vol_factor = 1.0
            trend_factor = 1.0
            
            if mode in [AdaptiveMode.VOLATILITY_ADJUSTED, AdaptiveMode.FULL_ADAPTIVE]:
                if volatility > self.adaptation_config['volatility_threshold']:
                    vol_factor = 1.0 - min(0.3, (volatility - 0.01) * 10)  # Shorten periods in high vol
                elif volatility < 0.005:
                    vol_factor = 1.0 + 0.2  # Lengthen periods in low vol
            
            if mode in [AdaptiveMode.TREND_ADJUSTED, AdaptiveMode.FULL_ADAPTIVE]:
                if abs(trend_strength) > self.adaptation_config['trend_threshold']:
                    trend_factor = 0.9  # Shorten periods in strong trends
                else:
                    trend_factor = 1.1  # Lengthen periods in weak trends
            
            # Combine factors
            combined_factor = vol_factor * trend_factor
            combined_factor = max(1 - self.adaptation_config['max_period_adjustment'], 
                                min(1 + self.adaptation_config['max_period_adjustment'], combined_factor))
            
            # Smooth the adaptation
            old_adjustment = params.adjusted_period / params.base_period
            new_adjustment = (old_adjustment * (1 - self.adaptation_config['adaptation_smoothing']) + 
                            combined_factor * self.adaptation_config['adaptation_smoothing'])
            
            # Update parameters
            params.adjusted_period = int(params.base_period * new_adjustment)
            params.volatility_factor = volatility
            params.trend_factor = abs(trend_strength)
            params.volume_factor = volume_factor
            params.last_update = current_time
            
            # Update confidence based on how much we're adapting
            adaptation_magnitude = abs(new_adjustment - 1.0)
            params.confidence = max(0.5, 1.0 - adaptation_magnitude * 2)
            
            # Log significant adaptations
            if abs(new_adjustment - 1.0) > 0.1:
                print(f"🔄 {indicator.upper()} adapted: {params.base_period} → {params.adjusted_period} "
                      f"(Vol: {volatility:.3f}, Trend: {trend_strength:.3f})")
        
        except Exception as e:
            print(f"❌ Parameter adaptation error for {indicator}: {e}")
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength"""
        try:
            if len(prices) < 20:
                return 0.0
            
            # Linear regression slope
            x = np.arange(len(prices))
            slope, _ = np.polyfit(x, prices, 1)
            
            # Normalize by average price
            trend_strength = slope / np.mean(prices)
            
            return trend_strength
        except:
            return 0.0
    
    def _estimate_volume_factor(self, symbol: str) -> float:
        """Estimate volume factor (simplified)"""
        # In production, this would use real volume data
        # For now, return a mock factor based on symbol type
        if 'USD' in symbol:
            return 1.2  # High volume currency pairs
        elif 'XAU' in symbol:
            return 0.8  # Lower volume for gold
        else:
            return 1.0
    
    def _volatility_adjusted_ma(self, prices: List[float], period: int) -> float:
        """Volatility-adjusted moving average"""
        volatility = np.std(prices[-period:]) / np.mean(prices[-period:])
        weights = np.exp(-np.arange(period) * volatility * 10)  # Exponential weighting
        weights = weights[::-1] / np.sum(weights)  # Reverse and normalize
        return np.sum(prices[-period:] * weights)
    
    def _trend_adjusted_ma(self, prices: List[float], period: int) -> float:
        """Trend-adjusted moving average"""
        trend = self._calculate_trend_strength(prices[-period:])
        if abs(trend) > 0.001:  # Strong trend
            # Give more weight to recent prices
            weights = np.exp(np.arange(period) * abs(trend) * 1000)
        else:
            # Equal weights
            weights = np.ones(period)
        
        weights = weights / np.sum(weights)
        return np.sum(prices[-period:] * weights)
    
    def _volume_adjusted_ma(self, prices: List[float], period: int, symbol: str) -> float:
        """Volume-adjusted moving average"""
        volume_factor = self._estimate_volume_factor(symbol)
        # Simulate volume-weighted average
        weights = np.random.uniform(0.5, 1.5, period) * volume_factor
        weights = weights / np.sum(weights)
        return np.sum(prices[-period:] * weights)
    
    def _full_adaptive_ma(self, prices: List[float], period: int, symbol: str) -> float:
        """Fully adaptive moving average combining all factors"""
        volatility = np.std(prices[-period:]) / np.mean(prices[-period:])
        trend = self._calculate_trend_strength(prices[-period:])
        volume_factor = self._estimate_volume_factor(symbol)
        
        # Create adaptive weights
        base_weights = np.ones(period)
        
        # Volatility adjustment
        vol_weights = np.exp(-np.arange(period) * volatility * 5)[::-1]
        
        # Trend adjustment
        if abs(trend) > 0.001:
            trend_weights = np.exp(np.arange(period) * abs(trend) * 500)
        else:
            trend_weights = np.ones(period)
        
        # Volume adjustment
        volume_weights = np.random.uniform(0.8, 1.2, period) * volume_factor
        
        # Combine weights
        combined_weights = base_weights * vol_weights * trend_weights * volume_weights
        combined_weights = combined_weights / np.sum(combined_weights)
        
        return np.sum(prices[-period:] * combined_weights)
    
    def _calculate_ema_simple(self, prices: List[float], period: int) -> Optional[float]:
        """Simple EMA calculation"""
        try:
            if len(prices) < period:
                return None
            
            alpha = 2.0 / (period + 1)
            ema = prices[0]
            
            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema
            
            return ema
        except:
            return None
    
    def _calculate_rsi_simple(self, prices: List[float], period: int) -> Optional[float]:
        """Simple RSI calculation"""
        try:
            if len(prices) < period + 1:
                return None
            
            changes = np.diff(prices)
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            return None
    
    def get_adaptation_summary(self) -> Dict:
        """Get summary of all adaptive parameters"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'indicators': {},
                'market_conditions': self.market_conditions,
                'adaptation_config': self.adaptation_config
            }
            
            for indicator, params in self.adaptive_params.items():
                summary['indicators'][indicator] = {
                    'base_period': params.base_period,
                    'adjusted_period': params.adjusted_period,
                    'adaptation_factor': params.adjusted_period / params.base_period,
                    'sensitivity': params.sensitivity,
                    'volatility_factor': params.volatility_factor,
                    'trend_factor': params.trend_factor,
                    'volume_factor': params.volume_factor,
                    'confidence': params.confidence,
                    'last_update': params.last_update.isoformat()
                }
            
            return summary
        
        except Exception as e:
            print(f"❌ Adaptation summary error: {e}")
            return {}

# Global instance
adaptive_indicators = None  # Will be initialized with dependencies