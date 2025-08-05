"""
Enhanced Indicators Module
Provides advanced technical analysis without TA-Lib dependency
"""

import numpy as np
from datetime import datetime
import random

class EnhancedIndicators:
    """Enhanced technical indicators without TA-Lib dependency"""

    def __init__(self):
        self.indicators_cache = {}

    def enhanced_signal_analysis(self, close_prices, high_prices=None, low_prices=None):
        """Enhanced signal analysis combining multiple indicators"""
        try:
            if len(close_prices) < 20:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.3,
                    'strength': 0.0,
                    'indicators': {}
                }

            # Calculate indicators
            rsi = self.calculate_rsi(close_prices)
            ma_short = self.calculate_sma(close_prices, 10)
            ma_long = self.calculate_sma(close_prices, 20)
            ema_fast = self.calculate_ema(close_prices, 12)
            ema_slow = self.calculate_ema(close_prices, 26)

            # Signal logic
            signal = 'HOLD'
            confidence = 0.5
            strength = 0.0

            # RSI-based signals
            if rsi < 30:
                signal = 'BUY'
                confidence += 0.2
                strength += 0.3
            elif rsi > 70:
                signal = 'SELL'
                confidence += 0.2
                strength += 0.3

            # Moving average crossover
            if ma_short > ma_long and close_prices[-1] > ma_short:
                if signal == 'HOLD':
                    signal = 'BUY'
                elif signal == 'BUY':
                    confidence += 0.1
                    strength += 0.2
            elif ma_short < ma_long and close_prices[-1] < ma_short:
                if signal == 'HOLD':
                    signal = 'SELL'
                elif signal == 'SELL':
                    confidence += 0.1
                    strength += 0.2

            # EMA trend
            if ema_fast > ema_slow:
                if signal == 'BUY':
                    confidence += 0.1
                    strength += 0.1
            else:
                if signal == 'SELL':
                    confidence += 0.1
                    strength += 0.1

            # Price momentum
            if len(close_prices) >= 5:
                momentum = (close_prices[-1] - close_prices[-5]) / close_prices[-5] * 100
                if momentum > 0.5:
                    if signal == 'BUY':
                        strength += 0.2
                elif momentum < -0.5:
                    if signal == 'SELL':
                        strength += 0.2

            # Cap values
            confidence = min(1.0, max(0.0, confidence))
            strength = min(1.0, max(0.0, strength))

            return {
                'signal': signal,
                'confidence': confidence,
                'strength': strength,
                'indicators': {
                    'rsi': rsi,
                    'ma_short': ma_short,
                    'ma_long': ma_long,
                    'ema_fast': ema_fast,
                    'ema_slow': ema_slow,
                    'ma_signal': 'BUY' if ma_short > ma_long else 'SELL',
                    'ema_signal': 'BUY' if ema_fast > ema_slow else 'SELL'
                }
            }

        except Exception as e:
            print(f"Signal analysis error: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.3,
                'strength': 0.0,
                'indicators': {}
            }

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI without TA-Lib"""
        try:
            if len(prices) < period + 1:
                return 50.0

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

        except Exception as e:
            print(f"RSI calculation error: {e}")
            return 50.0

    def calculate_sma(self, prices, period):
        """Calculate Simple Moving Average"""
        try:
            if len(prices) < period:
                return float(np.mean(prices))
            return float(np.mean(prices[-period:]))
        except Exception as e:
            print(f"SMA calculation error: {e}")
            return float(prices[-1]) if len(prices) > 0 else 0.0

    def calculate_ema(self, prices, period):
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return float(np.mean(prices))

            multiplier = 2 / (period + 1)
            ema = prices[0]

            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))

            return float(ema)

        except Exception as e:
            print(f"EMA calculation error: {e}")
            return float(prices[-1]) if len(prices) > 0 else 0.0