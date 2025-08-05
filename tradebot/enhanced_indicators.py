"""
Enhanced Technical Indicators with Advanced Signal Generation
Optimized for high success rate trading
"""

import numpy as np
from typing import Dict, List, Optional

class EnhancedIndicators:
    """Enhanced technical indicators with advanced signal generation"""

    def __init__(self):
        self.cache = {}
        self.signal_history = []

    def calculate_sma(self, prices: List[float], period: int = 20) -> Optional[float]:
        """Simple Moving Average"""
        try:
            if len(prices) < period:
                return None
            return float(np.mean(prices[-period:]))
        except Exception as e:
            print(f"SMA calculation error: {e}")
            return None

    def calculate_ema(self, prices: List[float], period: int = 20) -> Optional[float]:
        """Exponential Moving Average"""
        try:
            if len(prices) < period:
                return None

            multiplier = 2 / (period + 1)
            ema = prices[0]

            for price in prices[1:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))

            return float(ema)
        except Exception as e:
            print(f"EMA calculation error: {e}")
            return None

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """RSI with enhanced sensitivity"""
        try:
            if len(prices) < period + 1:
                return 50.0

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return float(rsi)
        except Exception as e:
            print(f"RSI calculation error: {e}")
            return 50.0

    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands calculation"""
        try:
            if len(prices) < period:
                return None, None, None

            recent_prices = prices[-period:]
            sma = np.mean(recent_prices)
            std = np.std(recent_prices)

            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)

            return float(upper), float(sma), float(lower)
        except Exception as e:
            print(f"Bollinger Bands calculation error: {e}")
            return None, None, None

    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD calculation"""
        try:
            if len(prices) < slow:
                return {'macd': 0, 'signal': 0, 'histogram': 0}

            ema_fast = self.calculate_ema(prices, fast) or 0
            ema_slow = self.calculate_ema(prices, slow) or 0

            macd_line = ema_fast - ema_slow

            # Simplified signal line calculation
            signal_line = macd_line * 0.9
            histogram = macd_line - signal_line

            return {
                'macd': float(macd_line),
                'signal': float(signal_line),
                'histogram': float(histogram)
            }
        except Exception as e:
            print(f"MACD calculation error: {e}")
            return {'macd': 0, 'signal': 0, 'histogram': 0}

    def enhanced_signal_analysis(self, prices: List[float], high_prices: Optional[List[float]] = None, 
                                low_prices: Optional[List[float]] = None) -> Dict:
        """Enhanced signal analysis with multiple confirmations"""
        try:
            if len(prices) < 20:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'strength': 0.0,
                    'indicators': {},
                    'reasons': ['Insufficient data']
                }

            current_price = prices[-1]

            # Calculate all indicators
            sma_short = self.calculate_sma(prices, 10)
            sma_long = self.calculate_sma(prices, 20)
            ema_fast = self.calculate_ema(prices, 9)
            ema_slow = self.calculate_ema(prices, 21)
            rsi = self.calculate_rsi(prices)
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(prices)
            macd = self.calculate_macd(prices)

            # Signal scoring system
            buy_score = 0
            sell_score = 0
            reasons = []

            # Trend Analysis (Weight: 3)
            if sma_short and sma_long:
                if sma_short > sma_long and current_price > sma_short:
                    buy_score += 3
                    reasons.append("Bullish trend confirmed")
                elif sma_short < sma_long and current_price < sma_short:
                    sell_score += 3
                    reasons.append("Bearish trend confirmed")

            # EMA Momentum (Weight: 2)
            if ema_fast and ema_slow:
                if ema_fast > ema_slow and current_price > ema_fast:
                    buy_score += 2
                    reasons.append("Bullish momentum")
                elif ema_fast < ema_slow and current_price < ema_fast:
                    sell_score += 2
                    reasons.append("Bearish momentum")

            # RSI Analysis (Weight: 2)
            if rsi <= 30:
                buy_score += 3
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi >= 70:
                sell_score += 3
                reasons.append(f"RSI overbought ({rsi:.1f})")
            elif 30 < rsi < 45:
                buy_score += 1
                reasons.append("RSI bullish zone")
            elif 55 < rsi < 70:
                sell_score += 1
                reasons.append("RSI bearish zone")

            # Bollinger Bands (Weight: 2)
            if bb_upper and bb_lower and bb_middle:
                if current_price <= bb_lower:
                    buy_score += 2
                    reasons.append("Price at lower Bollinger Band")
                elif current_price >= bb_upper:
                    sell_score += 2
                    reasons.append("Price at upper Bollinger Band")
                elif bb_middle and current_price > bb_middle:
                    buy_score += 1
                elif bb_middle and current_price < bb_middle:
                    sell_score += 1

            # MACD Analysis (Weight: 1)
            if macd['histogram'] > 0:
                buy_score += 1
                reasons.append("MACD bullish")
            elif macd['histogram'] < 0:
                sell_score += 1
                reasons.append("MACD bearish")

            # Price momentum (Weight: 1)
            if len(prices) >= 5:
                price_change = (current_price - prices[-5]) / prices[-5] * 100
                if price_change > 0.1:
                    buy_score += 1
                    reasons.append("Strong upward momentum")
                elif price_change < -0.1:
                    sell_score += 1
                    reasons.append("Strong downward momentum")

            # Determine signal
            total_possible_score = 12

            if buy_score > sell_score and buy_score >= 4:
                signal = 'BUY'
                confidence = min(buy_score / total_possible_score, 0.95)
                strength = min(buy_score / 8, 1.0)
            elif sell_score > buy_score and sell_score >= 4:
                signal = 'SELL'
                confidence = min(sell_score / total_possible_score, 0.95)
                strength = min(sell_score / 8, 1.0)
            else:
                signal = 'HOLD'
                confidence = 0.3
                strength = 0.3

            return {
                'signal': signal,
                'confidence': confidence,
                'strength': strength,
                'indicators': {
                    'sma_short': sma_short,
                    'sma_long': sma_long,
                    'ema_fast': ema_fast,
                    'ema_slow': ema_slow,
                    'rsi': rsi,
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'macd': macd,
                    'current_price': current_price
                },
                'scores': {
                    'buy_score': buy_score,
                    'sell_score': sell_score
                },
                'reasons': reasons[:3]  # Top 3 reasons
            }

        except Exception as e:
            print(f"Enhanced signal analysis error: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'strength': 0.0,
                'indicators': {},
                'reasons': [f'Analysis error: {str(e)[:50]}']
            }

    def get_signal(self, market_data: List[float], symbol: str = "UNKNOWN") -> Dict:
        """Compatible get_signal method for the bot"""
        try:
            if not market_data or len(market_data) == 0:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'strength': 0.0,
                    'indicators': {}
                }

            # Use enhanced signal analysis
            result = self.enhanced_signal_analysis(market_data)

            # Convert to expected format
            action = result['signal']
            if action not in ['BUY', 'SELL']:
                action = 'HOLD'

            return {
                'action': action,
                'confidence': result['confidence'],
                'strength': result['strength'],
                'indicators': result.get('indicators', {}),
                'scores': result.get('scores', {}),
                'reasons': result.get('reasons', [])
            }

        except Exception as e:
            print(f"Get signal error: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'strength': 0.0,
                'indicators': {}
            }