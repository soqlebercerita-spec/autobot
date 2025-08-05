"""
Pattern Recognition Module for AuraTrade Bot
Advanced price pattern detection and analysis
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

from utils.logger import Logger

class PatternType(Enum):
    """Supported pattern types"""
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    FLAG_BULLISH = "flag_bullish"
    FLAG_BEARISH = "flag_bearish"
    PENNANT = "pennant"
    RECTANGLE = "rectangle"
    CUP_HANDLE = "cup_handle"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    HAMMER = "hammer"
    DOJI = "doji"
    SHOOTING_STAR = "shooting_star"

class PatternRecognition:
    """Advanced Pattern Recognition System"""
    
    def __init__(self):
        self.logger = Logger()
        
        # Pattern detection parameters
        self.min_pattern_length = 10
        self.max_pattern_length = 100
        self.tolerance = 0.02  # 2% tolerance for pattern matching
        
        # Candlestick pattern parameters
        self.body_size_threshold = 0.6  # 60% of total range
        self.shadow_ratio_threshold = 2.0
        self.doji_threshold = 0.1  # 10% body size for doji
        
        # Chart pattern parameters
        self.support_resistance_touches = 2
        self.breakout_threshold = 0.001  # 0.1% for breakout confirmation
        
        self.logger.info("Pattern Recognition initialized")
    
    def detect_all_patterns(self, ohlc_data: Dict[str, np.ndarray], 
                           volume_data: np.ndarray = None) -> Dict[str, Any]:
        """Detect all supported patterns in OHLC data"""
        try:
            patterns = {
                'chart_patterns': [],
                'candlestick_patterns': [],
                'volume_patterns': [],
                'pattern_summary': {},
                'timestamp': datetime.now()
            }
            
            # Extract OHLC arrays
            open_prices = ohlc_data.get('open', np.array([]))
            high_prices = ohlc_data.get('high', np.array([]))
            low_prices = ohlc_data.get('low', np.array([]))
            close_prices = ohlc_data.get('close', np.array([]))
            
            if len(close_prices) < self.min_pattern_length:
                return patterns
            
            # Detect chart patterns
            patterns['chart_patterns'] = self._detect_chart_patterns(
                open_prices, high_prices, low_prices, close_prices
            )
            
            # Detect candlestick patterns
            patterns['candlestick_patterns'] = self._detect_candlestick_patterns(
                open_prices, high_prices, low_prices, close_prices
            )
            
            # Detect volume patterns if volume data available
            if volume_data is not None and len(volume_data) == len(close_prices):
                patterns['volume_patterns'] = self._detect_volume_patterns(
                    close_prices, volume_data
                )
            
            # Create pattern summary
            patterns['pattern_summary'] = self._create_pattern_summary(patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return {'error': str(e)}
    
    def _detect_chart_patterns(self, open_arr: np.ndarray, high_arr: np.ndarray, 
                              low_arr: np.ndarray, close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect chart patterns"""
        patterns = []
        
        try:
            # Double Top/Bottom
            patterns.extend(self._detect_double_patterns(high_arr, low_arr, close_arr))
            
            # Head and Shoulders
            patterns.extend(self._detect_head_shoulders(high_arr, low_arr, close_arr))
            
            # Triangles
            patterns.extend(self._detect_triangles(high_arr, low_arr, close_arr))
            
            # Wedges
            patterns.extend(self._detect_wedges(high_arr, low_arr, close_arr))
            
            # Flags and Pennants
            patterns.extend(self._detect_flags_pennants(high_arr, low_arr, close_arr))
            
            # Rectangle/Channel
            patterns.extend(self._detect_rectangles(high_arr, low_arr, close_arr))
            
            # Cup and Handle
            patterns.extend(self._detect_cup_handle(high_arr, low_arr, close_arr))
            
        except Exception as e:
            self.logger.error(f"Error detecting chart patterns: {str(e)}")
        
        return patterns
    
    def _detect_candlestick_patterns(self, open_arr: np.ndarray, high_arr: np.ndarray,
                                   low_arr: np.ndarray, close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect candlestick patterns"""
        patterns = []
        
        try:
            min_length = min(len(open_arr), len(high_arr), len(low_arr), len(close_arr))
            
            for i in range(1, min_length):
                # Single candlestick patterns
                patterns.extend(self._check_single_candle_patterns(i, open_arr, high_arr, low_arr, close_arr))
                
                # Two candlestick patterns
                if i >= 2:
                    patterns.extend(self._check_two_candle_patterns(i, open_arr, high_arr, low_arr, close_arr))
                
                # Three candlestick patterns
                if i >= 3:
                    patterns.extend(self._check_three_candle_patterns(i, open_arr, high_arr, low_arr, close_arr))
            
        except Exception as e:
            self.logger.error(f"Error detecting candlestick patterns: {str(e)}")
        
        return patterns
    
    def _detect_volume_patterns(self, close_prices: np.ndarray, volume_data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect volume-based patterns"""
        patterns = []
        
        try:
            if len(volume_data) < 10:
                return patterns
            
            # Volume spike detection
            volume_sma = self._calculate_sma(volume_data, 20)
            
            for i in range(20, len(volume_data)):
                if volume_sma[i] > 0:
                    volume_ratio = volume_data[i] / volume_sma[i]
                    
                    # Volume spike (>2x average)
                    if volume_ratio > 2.0:
                        patterns.append({
                            'type': 'volume_spike',
                            'position': i,
                            'ratio': volume_ratio,
                            'confidence': min(volume_ratio / 2.0, 1.0),
                            'description': f'Volume spike: {volume_ratio:.1f}x average'
                        })
                    
                    # Volume climax (>3x average with price movement)
                    if volume_ratio > 3.0 and i > 0:
                        price_change = abs(close_prices[i] - close_prices[i-1]) / close_prices[i-1]
                        if price_change > 0.01:  # 1% price movement
                            patterns.append({
                                'type': 'volume_climax',
                                'position': i,
                                'ratio': volume_ratio,
                                'price_change': price_change,
                                'confidence': min(volume_ratio / 3.0, 1.0),
                                'description': f'Volume climax: {volume_ratio:.1f}x volume, {price_change:.1%} price move'
                            })
            
            # On Balance Volume divergence
            patterns.extend(self._detect_obv_divergence(close_prices, volume_data))
            
        except Exception as e:
            self.logger.error(f"Error detecting volume patterns: {str(e)}")
        
        return patterns
    
    def _detect_double_patterns(self, high_arr: np.ndarray, low_arr: np.ndarray, 
                               close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect double top and double bottom patterns"""
        patterns = []
        
        try:
            # Double Top
            peaks = self._find_peaks(high_arr, distance=5)
            if len(peaks) >= 2:
                for i in range(len(peaks) - 1):
                    peak1_idx, peak2_idx = peaks[i], peaks[i + 1]
                    
                    # Check if peaks are similar height
                    peak1_price = high_arr[peak1_idx]
                    peak2_price = high_arr[peak2_idx]
                    
                    if abs(peak1_price - peak2_price) / peak1_price < self.tolerance:
                        # Find valley between peaks
                        valley_idx = np.argmin(low_arr[peak1_idx:peak2_idx]) + peak1_idx
                        valley_price = low_arr[valley_idx]
                        
                        # Check valley depth
                        if (peak1_price - valley_price) / peak1_price > 0.02:  # 2% minimum
                            patterns.append({
                                'type': PatternType.DOUBLE_TOP.value,
                                'start': peak1_idx,
                                'end': peak2_idx,
                                'peaks': [peak1_idx, peak2_idx],
                                'valley': valley_idx,
                                'resistance_level': (peak1_price + peak2_price) / 2,
                                'support_level': valley_price,
                                'confidence': self._calculate_pattern_confidence(
                                    'double_top', peak1_price, peak2_price, valley_price
                                ),
                                'description': f'Double Top: Resistance at {(peak1_price + peak2_price) / 2:.5f}'
                            })
            
            # Double Bottom
            troughs = self._find_troughs(low_arr, distance=5)
            if len(troughs) >= 2:
                for i in range(len(troughs) - 1):
                    trough1_idx, trough2_idx = troughs[i], troughs[i + 1]
                    
                    # Check if troughs are similar depth
                    trough1_price = low_arr[trough1_idx]
                    trough2_price = low_arr[trough2_idx]
                    
                    if abs(trough1_price - trough2_price) / trough1_price < self.tolerance:
                        # Find peak between troughs
                        peak_idx = np.argmax(high_arr[trough1_idx:trough2_idx]) + trough1_idx
                        peak_price = high_arr[peak_idx]
                        
                        # Check peak height
                        if (peak_price - trough1_price) / trough1_price > 0.02:  # 2% minimum
                            patterns.append({
                                'type': PatternType.DOUBLE_BOTTOM.value,
                                'start': trough1_idx,
                                'end': trough2_idx,
                                'troughs': [trough1_idx, trough2_idx],
                                'peak': peak_idx,
                                'support_level': (trough1_price + trough2_price) / 2,
                                'resistance_level': peak_price,
                                'confidence': self._calculate_pattern_confidence(
                                    'double_bottom', trough1_price, trough2_price, peak_price
                                ),
                                'description': f'Double Bottom: Support at {(trough1_price + trough2_price) / 2:.5f}'
                            })
        
        except Exception as e:
            self.logger.error(f"Error detecting double patterns: {str(e)}")
        
        return patterns
    
    def _detect_head_shoulders(self, high_arr: np.ndarray, low_arr: np.ndarray, 
                              close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect head and shoulders patterns"""
        patterns = []
        
        try:
            peaks = self._find_peaks(high_arr, distance=3)
            
            if len(peaks) >= 3:
                for i in range(len(peaks) - 2):
                    left_shoulder = peaks[i]
                    head = peaks[i + 1]
                    right_shoulder = peaks[i + 2]
                    
                    left_price = high_arr[left_shoulder]
                    head_price = high_arr[head]
                    right_price = high_arr[right_shoulder]
                    
                    # Check head and shoulders formation
                    if (head_price > left_price and head_price > right_price and
                        abs(left_price - right_price) / left_price < self.tolerance):
                        
                        # Find neckline (valleys between shoulders and head)
                        left_valley = np.argmin(low_arr[left_shoulder:head]) + left_shoulder
                        right_valley = np.argmin(low_arr[head:right_shoulder]) + head
                        
                        neckline_level = (low_arr[left_valley] + low_arr[right_valley]) / 2
                        
                        patterns.append({
                            'type': PatternType.HEAD_SHOULDERS.value,
                            'start': left_shoulder,
                            'end': right_shoulder,
                            'left_shoulder': left_shoulder,
                            'head': head,
                            'right_shoulder': right_shoulder,
                            'neckline_level': neckline_level,
                            'target': neckline_level - (head_price - neckline_level),
                            'confidence': self._calculate_hs_confidence(left_price, head_price, right_price),
                            'description': f'Head & Shoulders: Target {neckline_level - (head_price - neckline_level):.5f}'
                        })
            
            # Inverse Head and Shoulders
            troughs = self._find_troughs(low_arr, distance=3)
            
            if len(troughs) >= 3:
                for i in range(len(troughs) - 2):
                    left_shoulder = troughs[i]
                    head = troughs[i + 1]
                    right_shoulder = troughs[i + 2]
                    
                    left_price = low_arr[left_shoulder]
                    head_price = low_arr[head]
                    right_price = low_arr[right_shoulder]
                    
                    # Check inverse head and shoulders formation
                    if (head_price < left_price and head_price < right_price and
                        abs(left_price - right_price) / left_price < self.tolerance):
                        
                        # Find neckline (peaks between shoulders and head)
                        left_peak = np.argmax(high_arr[left_shoulder:head]) + left_shoulder
                        right_peak = np.argmax(high_arr[head:right_shoulder]) + head
                        
                        neckline_level = (high_arr[left_peak] + high_arr[right_peak]) / 2
                        
                        patterns.append({
                            'type': PatternType.INVERSE_HEAD_SHOULDERS.value,
                            'start': left_shoulder,
                            'end': right_shoulder,
                            'left_shoulder': left_shoulder,
                            'head': head,
                            'right_shoulder': right_shoulder,
                            'neckline_level': neckline_level,
                            'target': neckline_level + (neckline_level - head_price),
                            'confidence': self._calculate_hs_confidence(left_price, head_price, right_price),
                            'description': f'Inverse H&S: Target {neckline_level + (neckline_level - head_price):.5f}'
                        })
        
        except Exception as e:
            self.logger.error(f"Error detecting head and shoulders: {str(e)}")
        
        return patterns
    
    def _detect_triangles(self, high_arr: np.ndarray, low_arr: np.ndarray, 
                         close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect triangle patterns"""
        patterns = []
        
        try:
            if len(high_arr) < 20:
                return patterns
            
            # Look for triangle patterns in recent data
            window_size = min(50, len(high_arr))
            recent_highs = high_arr[-window_size:]
            recent_lows = low_arr[-window_size:]
            
            # Find trending support and resistance lines
            high_peaks = self._find_peaks(recent_highs, distance=3)
            low_troughs = self._find_troughs(recent_lows, distance=3)
            
            if len(high_peaks) >= 2 and len(low_troughs) >= 2:
                # Calculate trend lines
                high_slope = self._calculate_trend_slope([high_peaks[-2], high_peaks[-1]], 
                                                       [recent_highs[high_peaks[-2]], recent_highs[high_peaks[-1]]])
                low_slope = self._calculate_trend_slope([low_troughs[-2], low_troughs[-1]], 
                                                      [recent_lows[low_troughs[-2]], recent_lows[low_troughs[-1]]])
                
                # Classify triangle type
                if high_slope < -0.0001 and low_slope > 0.0001:
                    # Symmetrical triangle
                    patterns.append({
                        'type': PatternType.TRIANGLE_SYMMETRICAL.value,
                        'start': len(high_arr) - window_size,
                        'end': len(high_arr) - 1,
                        'upper_slope': high_slope,
                        'lower_slope': low_slope,
                        'confidence': 0.7,
                        'description': 'Symmetrical Triangle: Consolidation pattern'
                    })
                
                elif abs(high_slope) < 0.0001 and low_slope > 0.0001:
                    # Ascending triangle
                    patterns.append({
                        'type': PatternType.TRIANGLE_ASCENDING.value,
                        'start': len(high_arr) - window_size,
                        'end': len(high_arr) - 1,
                        'resistance_level': np.mean([recent_highs[p] for p in high_peaks[-2:]]),
                        'support_slope': low_slope,
                        'confidence': 0.8,
                        'description': 'Ascending Triangle: Bullish breakout expected'
                    })
                
                elif high_slope < -0.0001 and abs(low_slope) < 0.0001:
                    # Descending triangle
                    patterns.append({
                        'type': PatternType.TRIANGLE_DESCENDING.value,
                        'start': len(high_arr) - window_size,
                        'end': len(high_arr) - 1,
                        'support_level': np.mean([recent_lows[t] for t in low_troughs[-2:]]),
                        'resistance_slope': high_slope,
                        'confidence': 0.8,
                        'description': 'Descending Triangle: Bearish breakdown expected'
                    })
        
        except Exception as e:
            self.logger.error(f"Error detecting triangles: {str(e)}")
        
        return patterns
    
    def _detect_wedges(self, high_arr: np.ndarray, low_arr: np.ndarray, 
                      close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect wedge patterns"""
        patterns = []
        
        try:
            if len(high_arr) < 20:
                return patterns
            
            window_size = min(40, len(high_arr))
            recent_highs = high_arr[-window_size:]
            recent_lows = low_arr[-window_size:]
            
            high_peaks = self._find_peaks(recent_highs, distance=3)
            low_troughs = self._find_troughs(recent_lows, distance=3)
            
            if len(high_peaks) >= 2 and len(low_troughs) >= 2:
                high_slope = self._calculate_trend_slope([high_peaks[-2], high_peaks[-1]], 
                                                       [recent_highs[high_peaks[-2]], recent_highs[high_peaks[-1]]])
                low_slope = self._calculate_trend_slope([low_troughs[-2], low_troughs[-1]], 
                                                      [recent_lows[low_troughs[-2]], recent_lows[low_troughs[-1]]])
                
                # Rising wedge (both slopes positive, but upper slope less steep)
                if high_slope > 0 and low_slope > 0 and high_slope < low_slope:
                    patterns.append({
                        'type': PatternType.WEDGE_RISING.value,
                        'start': len(high_arr) - window_size,
                        'end': len(high_arr) - 1,
                        'upper_slope': high_slope,
                        'lower_slope': low_slope,
                        'confidence': 0.7,
                        'description': 'Rising Wedge: Bearish reversal pattern'
                    })
                
                # Falling wedge (both slopes negative, but lower slope less steep)
                elif high_slope < 0 and low_slope < 0 and abs(high_slope) > abs(low_slope):
                    patterns.append({
                        'type': PatternType.WEDGE_FALLING.value,
                        'start': len(high_arr) - window_size,
                        'end': len(high_arr) - 1,
                        'upper_slope': high_slope,
                        'lower_slope': low_slope,
                        'confidence': 0.7,
                        'description': 'Falling Wedge: Bullish reversal pattern'
                    })
        
        except Exception as e:
            self.logger.error(f"Error detecting wedges: {str(e)}")
        
        return patterns
    
    def _detect_flags_pennants(self, high_arr: np.ndarray, low_arr: np.ndarray, 
                              close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect flag and pennant patterns"""
        patterns = []
        
        try:
            if len(close_arr) < 30:
                return patterns
            
            # Look for strong moves followed by consolidation
            for i in range(20, len(close_arr) - 10):
                # Check for strong move (flagpole)
                flagpole_start = max(0, i - 15)
                flagpole_move = (close_arr[i] - close_arr[flagpole_start]) / close_arr[flagpole_start]
                
                if abs(flagpole_move) > 0.03:  # 3% minimum move
                    # Check for consolidation period
                    consolidation_end = min(len(close_arr), i + 10)
                    consolidation_range = np.max(high_arr[i:consolidation_end]) - np.min(low_arr[i:consolidation_end])
                    consolidation_ratio = consolidation_range / close_arr[i]
                    
                    if consolidation_ratio < 0.02:  # Less than 2% range
                        pattern_type = PatternType.FLAG_BULLISH.value if flagpole_move > 0 else PatternType.FLAG_BEARISH.value
                        
                        patterns.append({
                            'type': pattern_type,
                            'flagpole_start': flagpole_start,
                            'flag_start': i,
                            'flag_end': consolidation_end - 1,
                            'flagpole_move': flagpole_move,
                            'consolidation_range': consolidation_ratio,
                            'confidence': 0.8,
                            'description': f'{pattern_type.replace("_", " ").title()}: Continuation pattern'
                        })
        
        except Exception as e:
            self.logger.error(f"Error detecting flags and pennants: {str(e)}")
        
        return patterns
    
    def _detect_rectangles(self, high_arr: np.ndarray, low_arr: np.ndarray, 
                          close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect rectangle/channel patterns"""
        patterns = []
        
        try:
            if len(high_arr) < 30:
                return patterns
            
            window_size = min(50, len(high_arr))
            recent_highs = high_arr[-window_size:]
            recent_lows = low_arr[-window_size:]
            
            # Find horizontal support and resistance levels
            resistance_level = np.percentile(recent_highs, 95)
            support_level = np.percentile(recent_lows, 5)
            
            # Check if price has respected these levels multiple times
            resistance_touches = np.sum(recent_highs > resistance_level * 0.998)
            support_touches = np.sum(recent_lows < support_level * 1.002)
            
            if resistance_touches >= 2 and support_touches >= 2:
                range_size = (resistance_level - support_level) / support_level
                
                if 0.02 < range_size < 0.1:  # 2-10% range
                    patterns.append({
                        'type': PatternType.RECTANGLE.value,
                        'start': len(high_arr) - window_size,
                        'end': len(high_arr) - 1,
                        'resistance_level': resistance_level,
                        'support_level': support_level,
                        'range_size': range_size,
                        'resistance_touches': resistance_touches,
                        'support_touches': support_touches,
                        'confidence': min(0.9, (resistance_touches + support_touches) / 8),
                        'description': f'Rectangle: Range {range_size:.1%}'
                    })
        
        except Exception as e:
            self.logger.error(f"Error detecting rectangles: {str(e)}")
        
        return patterns
    
    def _detect_cup_handle(self, high_arr: np.ndarray, low_arr: np.ndarray, 
                          close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect cup and handle patterns"""
        patterns = []
        
        try:
            if len(close_arr) < 50:
                return patterns
            
            # Look for cup pattern in recent data
            window_size = min(100, len(close_arr))
            recent_prices = close_arr[-window_size:]
            
            # Find the cup (U-shaped pattern)
            start_price = recent_prices[0]
            end_price = recent_prices[-1]
            min_price = np.min(recent_prices)
            min_idx = np.argmin(recent_prices)
            
            # Check cup formation criteria
            if (abs(start_price - end_price) / start_price < 0.05 and  # Similar levels
                (start_price - min_price) / start_price > 0.15 and    # Significant depth
                10 < min_idx < window_size - 10):                     # Bottom not at edges
                
                # Check for handle (small consolidation at the right)
                handle_start = int(window_size * 0.7)
                handle_prices = recent_prices[handle_start:]
                handle_range = (np.max(handle_prices) - np.min(handle_prices)) / np.max(handle_prices)
                
                if handle_range < 0.1:  # Handle range < 10%
                    patterns.append({
                        'type': PatternType.CUP_HANDLE.value,
                        'start': len(close_arr) - window_size,
                        'end': len(close_arr) - 1,
                        'cup_bottom': len(close_arr) - window_size + min_idx,
                        'handle_start': len(close_arr) - window_size + handle_start,
                        'rim_level': max(start_price, end_price),
                        'depth': (start_price - min_price) / start_price,
                        'confidence': 0.8,
                        'description': f'Cup & Handle: Bullish pattern, depth {(start_price - min_price) / start_price:.1%}'
                    })
        
        except Exception as e:
            self.logger.error(f"Error detecting cup and handle: {str(e)}")
        
        return patterns
    
    def _check_single_candle_patterns(self, idx: int, open_arr: np.ndarray, high_arr: np.ndarray,
                                    low_arr: np.ndarray, close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Check single candlestick patterns"""
        patterns = []
        
        try:
            open_price = open_arr[idx]
            high_price = high_arr[idx]
            low_price = low_arr[idx]
            close_price = close_arr[idx]
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            
            if total_range == 0:
                return patterns
            
            # Doji
            if body_size / total_range < self.doji_threshold:
                patterns.append({
                    'type': PatternType.DOJI.value,
                    'position': idx,
                    'confidence': 0.8,
                    'description': 'Doji: Indecision candle'
                })
            
            # Hammer (bullish reversal)
            elif (lower_shadow > body_size * 2 and 
                  upper_shadow < body_size * 0.5 and
                  idx > 5):  # Need some history
                
                # Check if in downtrend
                recent_trend = (close_arr[idx] - close_arr[max(0, idx-5)]) / close_arr[max(0, idx-5)]
                if recent_trend < -0.02:  # 2% down
                    patterns.append({
                        'type': PatternType.HAMMER.value,
                        'position': idx,
                        'confidence': 0.9,
                        'description': 'Hammer: Bullish reversal signal'
                    })
            
            # Shooting Star (bearish reversal)
            elif (upper_shadow > body_size * 2 and 
                  lower_shadow < body_size * 0.5 and
                  idx > 5):
                
                # Check if in uptrend
                recent_trend = (close_arr[idx] - close_arr[max(0, idx-5)]) / close_arr[max(0, idx-5)]
                if recent_trend > 0.02:  # 2% up
                    patterns.append({
                        'type': PatternType.SHOOTING_STAR.value,
                        'position': idx,
                        'confidence': 0.9,
                        'description': 'Shooting Star: Bearish reversal signal'
                    })
        
        except Exception as e:
            self.logger.error(f"Error checking single candle patterns: {str(e)}")
        
        return patterns
    
    def _check_two_candle_patterns(self, idx: int, open_arr: np.ndarray, high_arr: np.ndarray,
                                 low_arr: np.ndarray, close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Check two-candlestick patterns"""
        patterns = []
        
        try:
            # Current candle
            open1, high1, low1, close1 = open_arr[idx], high_arr[idx], low_arr[idx], close_arr[idx]
            # Previous candle
            open0, high0, low0, close0 = open_arr[idx-1], high_arr[idx-1], low_arr[idx-1], close_arr[idx-1]
            
            # Bullish Engulfing
            if (close0 < open0 and  # Previous candle is bearish
                close1 > open1 and  # Current candle is bullish
                open1 < close0 and  # Current open below previous close
                close1 > open0):    # Current close above previous open
                
                patterns.append({
                    'type': PatternType.ENGULFING_BULLISH.value,
                    'position': idx,
                    'confidence': 0.9,
                    'description': 'Bullish Engulfing: Strong reversal signal'
                })
            
            # Bearish Engulfing
            elif (close0 > open0 and  # Previous candle is bullish
                  close1 < open1 and  # Current candle is bearish
                  open1 > close0 and  # Current open above previous close
                  close1 < open0):    # Current close below previous open
                
                patterns.append({
                    'type': PatternType.ENGULFING_BEARISH.value,
                    'position': idx,
                    'confidence': 0.9,
                    'description': 'Bearish Engulfing: Strong reversal signal'
                })
        
        except Exception as e:
            self.logger.error(f"Error checking two candle patterns: {str(e)}")
        
        return patterns
    
    def _check_three_candle_patterns(self, idx: int, open_arr: np.ndarray, high_arr: np.ndarray,
                                   low_arr: np.ndarray, close_arr: np.ndarray) -> List[Dict[str, Any]]:
        """Check three-candlestick patterns"""
        patterns = []
        
        try:
            # Implementation for three-candle patterns like morning star, evening star, etc.
            # This is a simplified version
            
            close2 = close_arr[idx]    # Current
            close1 = close_arr[idx-1]  # Previous
            close0 = close_arr[idx-2]  # Two candles ago
            
            # Simple three-candle trend pattern
            if close2 > close1 > close0:
                trend_strength = (close2 - close0) / close0
                if trend_strength > 0.02:  # 2% move
                    patterns.append({
                        'type': 'three_white_soldiers',
                        'position': idx,
                        'confidence': 0.7,
                        'description': 'Three White Soldiers: Bullish continuation'
                    })
            
            elif close2 < close1 < close0:
                trend_strength = (close0 - close2) / close0
                if trend_strength > 0.02:  # 2% move
                    patterns.append({
                        'type': 'three_black_crows',
                        'position': idx,
                        'confidence': 0.7,
                        'description': 'Three Black Crows: Bearish continuation'
                    })
        
        except Exception as e:
            self.logger.error(f"Error checking three candle patterns: {str(e)}")
        
        return patterns
    
    def _detect_obv_divergence(self, close_prices: np.ndarray, volume_data: np.ndarray) -> List[Dict[str, Any]]:
        """Detect On Balance Volume divergence"""
        patterns = []
        
        try:
            if len(close_prices) < 20 or len(volume_data) < 20:
                return patterns
            
            # Calculate OBV
            obv = np.zeros(len(close_prices))
            for i in range(1, len(close_prices)):
                if close_prices[i] > close_prices[i-1]:
                    obv[i] = obv[i-1] + volume_data[i]
                elif close_prices[i] < close_prices[i-1]:
                    obv[i] = obv[i-1] - volume_data[i]
                else:
                    obv[i] = obv[i-1]
            
            # Look for divergence in recent data
            recent_length = min(20, len(close_prices) - 1)
            recent_prices = close_prices[-recent_length:]
            recent_obv = obv[-recent_length:]
            
            # Calculate trends
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            obv_trend = (recent_obv[-1] - recent_obv[0]) / abs(recent_obv[0]) if recent_obv[0] != 0 else 0
            
            # Check for divergence
            if price_trend > 0.01 and obv_trend < -0.01:  # Price up, OBV down
                patterns.append({
                    'type': 'obv_bearish_divergence',
                    'position': len(close_prices) - 1,
                    'price_trend': price_trend,
                    'obv_trend': obv_trend,
                    'confidence': 0.7,
                    'description': 'OBV Bearish Divergence: Price rising but volume declining'
                })
            
            elif price_trend < -0.01 and obv_trend > 0.01:  # Price down, OBV up
                patterns.append({
                    'type': 'obv_bullish_divergence',
                    'position': len(close_prices) - 1,
                    'price_trend': price_trend,
                    'obv_trend': obv_trend,
                    'confidence': 0.7,
                    'description': 'OBV Bullish Divergence: Price falling but volume rising'
                })
        
        except Exception as e:
            self.logger.error(f"Error detecting OBV divergence: {str(e)}")
        
        return patterns
    
    # Helper methods
    def _find_peaks(self, data: np.ndarray, distance: int = 1) -> List[int]:
        """Find peaks in data"""
        peaks = []
        for i in range(distance, len(data) - distance):
            if all(data[i] >= data[i-j] for j in range(1, distance+1)) and \
               all(data[i] >= data[i+j] for j in range(1, distance+1)):
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, data: np.ndarray, distance: int = 1) -> List[int]:
        """Find troughs in data"""
        troughs = []
        for i in range(distance, len(data) - distance):
            if all(data[i] <= data[i-j] for j in range(1, distance+1)) and \
               all(data[i] <= data[i+j] for j in range(1, distance+1)):
                troughs.append(i)
        return troughs
    
    def _calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        sma = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1:i + 1])
        return sma
    
    def _calculate_trend_slope(self, x_points: List[int], y_points: List[float]) -> float:
        """Calculate slope of trend line"""
        if len(x_points) != len(y_points) or len(x_points) < 2:
            return 0.0
        
        x_mean = np.mean(x_points)
        y_mean = np.mean(y_points)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_points, y_points))
        denominator = sum((x - x_mean) ** 2 for x in x_points)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _calculate_pattern_confidence(self, pattern_type: str, *args) -> float:
        """Calculate confidence for pattern"""
        # Simplified confidence calculation
        if pattern_type in ['double_top', 'double_bottom']:
            val1, val2, val3 = args
            similarity = 1 - abs(val1 - val2) / val1
            return min(similarity * 1.2, 1.0)
        
        return 0.7  # Default confidence
    
    def _calculate_hs_confidence(self, left: float, head: float, right: float) -> float:
        """Calculate head and shoulders pattern confidence"""
        shoulder_similarity = 1 - abs(left - right) / left
        head_prominence = (head - max(left, right)) / head
        return min((shoulder_similarity + head_prominence) / 2, 1.0)
    
    def _create_pattern_summary(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of detected patterns"""
        try:
            chart_patterns = patterns.get('chart_patterns', [])
            candlestick_patterns = patterns.get('candlestick_patterns', [])
            volume_patterns = patterns.get('volume_patterns', [])
            
            # Count patterns by type
            pattern_counts = {}
            for pattern in chart_patterns + candlestick_patterns + volume_patterns:
                pattern_type = pattern.get('type', 'unknown')
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            # Get most recent significant patterns
            recent_patterns = []
            all_patterns = chart_patterns + candlestick_patterns + volume_patterns
            
            # Sort by confidence and recency
            for pattern in sorted(all_patterns, key=lambda x: x.get('confidence', 0), reverse=True)[:5]:
                recent_patterns.append({
                    'type': pattern.get('type'),
                    'confidence': pattern.get('confidence', 0),
                    'description': pattern.get('description', '')
                })
            
            # Overall market sentiment based on patterns
            bullish_patterns = ['double_bottom', 'inverse_head_shoulders', 'triangle_ascending', 
                              'hammer', 'engulfing_bullish', 'cup_handle']
            bearish_patterns = ['double_top', 'head_shoulders', 'triangle_descending', 
                              'shooting_star', 'engulfing_bearish', 'wedge_rising']
            
            bullish_count = sum(1 for p in all_patterns if p.get('type') in bullish_patterns)
            bearish_count = sum(1 for p in all_patterns if p.get('type') in bearish_patterns)
            
            if bullish_count > bearish_count:
                sentiment = 'bullish'
            elif bearish_count > bullish_count:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            return {
                'total_patterns': len(all_patterns),
                'chart_patterns_count': len(chart_patterns),
                'candlestick_patterns_count': len(candlestick_patterns),
                'volume_patterns_count': len(volume_patterns),
                'pattern_counts': pattern_counts,
                'recent_significant_patterns': recent_patterns,
                'market_sentiment': sentiment,
                'bullish_signals': bullish_count,
                'bearish_signals': bearish_count
            }
            
        except Exception as e:
            self.logger.error(f"Error creating pattern summary: {str(e)}")
            return {'error': str(e)}
