"""
Market Conditions Analysis for AuraTrade Bot
Real-time market condition detection and classification
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import statistics

from core.mt5_connector import MT5Connector
from analysis.technical_analysis import TechnicalAnalysis
from utils.logger import Logger

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"

class TradingSession(Enum):
    """Trading session classifications"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    ASIAN_QUIET = "asian_quiet"
    WEEKEND = "weekend"

class MarketConditions:
    """Comprehensive Market Conditions Analysis"""
    
    def __init__(self, mt5_connector: Optional[MT5Connector] = None):
        self.mt5_connector = mt5_connector
        self.logger = Logger()
        self.technical_analysis = TechnicalAnalysis()
        
        # Analysis parameters
        self.volatility_lookback = 20
        self.trend_lookback = 50
        self.volume_lookback = 30
        self.correlation_lookback = 100
        
        # Thresholds
        self.high_volatility_threshold = 0.008  # 0.8% daily volatility
        self.low_volatility_threshold = 0.002   # 0.2% daily volatility
        self.trend_strength_threshold = 0.7     # 70% trend confidence
        self.volume_spike_threshold = 2.0       # 2x average volume
        
        # Market data cache
        self.market_data_cache = {}
        self.condition_history = []
        self.last_update = None
        
        # Economic calendar events (simplified)
        self.high_impact_events = [
            "NFP", "FOMC", "CPI", "GDP", "ECB", "BOE", "BOJ",
            "Interest Rate Decision", "Employment", "Inflation"
        ]
        
        self.logger.info("Market Conditions analyzer initialized")
    
    def analyze_current_conditions(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Analyze current market conditions across multiple timeframes"""
        try:
            if symbols is None:
                symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
            
            conditions = {
                'timestamp': datetime.now(),
                'overall_regime': MarketRegime.RANGING,
                'trading_session': self.get_trading_session(),
                'symbol_conditions': {},
                'cross_market_analysis': {},
                'risk_environment': {},
                'volatility_analysis': {},
                'correlation_analysis': {},
                'session_characteristics': {},
                'news_impact_assessment': {}
            }
            
            # Analyze each symbol
            for symbol in symbols:
                symbol_conditions = self._analyze_symbol_conditions(symbol)
                conditions['symbol_conditions'][symbol] = symbol_conditions
            
            # Cross-market analysis
            conditions['cross_market_analysis'] = self._analyze_cross_market_conditions(symbols)
            
            # Overall risk environment
            conditions['risk_environment'] = self._assess_risk_environment(conditions['symbol_conditions'])
            
            # Volatility regime analysis
            conditions['volatility_analysis'] = self._analyze_volatility_regime(symbols)
            
            # Correlation analysis
            conditions['correlation_analysis'] = self._analyze_market_correlations(symbols)
            
            # Session characteristics
            conditions['session_characteristics'] = self._analyze_session_characteristics()
            
            # News impact assessment
            conditions['news_impact_assessment'] = self._assess_news_impact()
            
            # Determine overall market regime
            conditions['overall_regime'] = self._determine_overall_regime(conditions)
            
            # Cache results
            self.condition_history.append(conditions)
            if len(self.condition_history) > 100:
                self.condition_history = self.condition_history[-100:]
            
            self.last_update = datetime.now()
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing market conditions: {str(e)}")
            return self._get_default_conditions()
    
    def _analyze_symbol_conditions(self, symbol: str) -> Dict[str, Any]:
        """Analyze conditions for a specific symbol"""
        try:
            # Get market data
            rates_h1 = self._get_rates_data(symbol, 'H1', 100)
            rates_m15 = self._get_rates_data(symbol, 'M15', 200)
            rates_m5 = self._get_rates_data(symbol, 'M5', 300)
            
            if rates_h1 is None:
                return self._get_default_symbol_conditions()
            
            close_h1 = rates_h1['close'].values
            high_h1 = rates_h1['high'].values
            low_h1 = rates_h1['low'].values
            volume_h1 = rates_h1.get('tick_volume', rates_h1.get('real_volume', np.ones(len(close_h1)))).values
            
            conditions = {
                'symbol': symbol,
                'current_price': close_h1[-1],
                'trend_analysis': {},
                'volatility_metrics': {},
                'support_resistance': {},
                'momentum_indicators': {},
                'volume_analysis': {},
                'timeframe_alignment': {},
                'market_structure': {},
                'breakout_potential': {}
            }
            
            # Trend analysis
            conditions['trend_analysis'] = self._analyze_trend_conditions(close_h1, high_h1, low_h1)
            
            # Volatility metrics
            conditions['volatility_metrics'] = self._calculate_volatility_metrics(close_h1, high_h1, low_h1)
            
            # Support and resistance
            conditions['support_resistance'] = self._identify_key_levels(high_h1, low_h1, close_h1)
            
            # Momentum indicators
            conditions['momentum_indicators'] = self._analyze_momentum_conditions(close_h1, volume_h1)
            
            # Volume analysis
            conditions['volume_analysis'] = self._analyze_volume_conditions(close_h1, volume_h1)
            
            # Multi-timeframe alignment
            if rates_m15 is not None and rates_m5 is not None:
                conditions['timeframe_alignment'] = self._analyze_timeframe_alignment(
                    rates_h1, rates_m15, rates_m5
                )
            
            # Market structure
            conditions['market_structure'] = self._analyze_market_structure(high_h1, low_h1, close_h1)
            
            # Breakout potential
            conditions['breakout_potential'] = self._assess_breakout_potential(
                high_h1, low_h1, close_h1, volume_h1
            )
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Error analyzing conditions for {symbol}: {str(e)}")
            return self._get_default_symbol_conditions()
    
    def _analyze_trend_conditions(self, close_prices: np.ndarray, high_prices: np.ndarray, 
                                 low_prices: np.ndarray) -> Dict[str, Any]:
        """Analyze trend conditions"""
        try:
            # Multiple trend analysis methods
            trend_analysis = {}
            
            # Linear regression trend
            x = np.arange(len(close_prices))
            coefficients = np.polyfit(x[-self.trend_lookback:], close_prices[-self.trend_lookback:], 1)
            slope = coefficients[0]
            
            # R-squared for trend strength
            y_pred = np.polyval(coefficients, x[-self.trend_lookback:])
            ss_res = np.sum((close_prices[-self.trend_lookback:] - y_pred) ** 2)
            ss_tot = np.sum((close_prices[-self.trend_lookback:] - np.mean(close_prices[-self.trend_lookback:])) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Moving averages trend
            sma_20 = self.technical_analysis.calculate_sma(close_prices, 20)
            sma_50 = self.technical_analysis.calculate_sma(close_prices, 50)
            ema_12 = self.technical_analysis.calculate_ema(close_prices, 12)
            ema_26 = self.technical_analysis.calculate_ema(close_prices, 26)
            
            # ADX for trend strength
            atr = self.technical_analysis.calculate_atr(high_prices, low_prices, close_prices, 14)
            
            current_price = close_prices[-1]
            
            # Determine trend direction
            if slope > 0.0001 and current_price > sma_20[-1] and sma_20[-1] > sma_50[-1]:
                trend_direction = 'bullish'
            elif slope < -0.0001 and current_price < sma_20[-1] and sma_20[-1] < sma_50[-1]:
                trend_direction = 'bearish'
            else:
                trend_direction = 'sideways'
            
            # Trend strength (0-1)
            trend_strength = min(abs(r_squared), 1.0)
            
            # Price position relative to moving averages
            ma_position = 'above' if current_price > sma_20[-1] else 'below'
            
            trend_analysis = {
                'direction': trend_direction,
                'strength': trend_strength,
                'slope': slope,
                'r_squared': r_squared,
                'ma_alignment': ema_12[-1] > ema_26[-1],
                'price_vs_ma': ma_position,
                'atr': atr[-1],
                'trend_quality': 'strong' if trend_strength > 0.7 else 'weak' if trend_strength < 0.3 else 'moderate'
            }
            
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trend conditions: {str(e)}")
            return {'direction': 'sideways', 'strength': 0.0}
    
    def _calculate_volatility_metrics(self, close_prices: np.ndarray, high_prices: np.ndarray, 
                                    low_prices: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive volatility metrics"""
        try:
            # Historical volatility (annualized)
            returns = np.diff(np.log(close_prices))
            historical_vol = np.std(returns[-self.volatility_lookback:]) * np.sqrt(252) if len(returns) > 0 else 0
            
            # Average True Range
            atr = self.technical_analysis.calculate_atr(high_prices, low_prices, close_prices, 14)
            current_atr = atr[-1]
            atr_sma = np.mean(atr[-20:]) if len(atr) >= 20 else current_atr
            
            # Intraday range volatility
            daily_ranges = (high_prices - low_prices) / close_prices
            avg_daily_range = np.mean(daily_ranges[-self.volatility_lookback:])
            
            # Volatility regime classification
            if historical_vol > self.high_volatility_threshold:
                vol_regime = 'high'
            elif historical_vol < self.low_volatility_threshold:
                vol_regime = 'low'
            else:
                vol_regime = 'normal'
            
            # Volatility trend (increasing/decreasing)
            if len(atr) >= 10:
                recent_atr = np.mean(atr[-5:])
                previous_atr = np.mean(atr[-10:-5])
                vol_trend = 'increasing' if recent_atr > previous_atr * 1.1 else 'decreasing' if recent_atr < previous_atr * 0.9 else 'stable'
            else:
                vol_trend = 'stable'
            
            return {
                'historical_volatility': historical_vol,
                'atr': current_atr,
                'atr_normalized': current_atr / atr_sma if atr_sma > 0 else 1.0,
                'avg_daily_range': avg_daily_range,
                'volatility_regime': vol_regime,
                'volatility_trend': vol_trend,
                'volatility_percentile': self._calculate_volatility_percentile(historical_vol)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility metrics: {str(e)}")
            return {'historical_volatility': 0.0, 'volatility_regime': 'normal'}
    
    def _identify_key_levels(self, high_prices: np.ndarray, low_prices: np.ndarray, 
                           close_prices: np.ndarray) -> Dict[str, Any]:
        """Identify key support and resistance levels"""
        try:
            # Support and resistance detection
            sr_levels = self.technical_analysis.detect_support_resistance(
                high_prices, low_prices, close_prices, window=20, min_touches=2
            )
            
            current_price = close_prices[-1]
            
            # Find nearest levels
            support_levels = sr_levels.get('support', [])
            resistance_levels = sr_levels.get('resistance', [])
            
            # Nearest support and resistance
            nearest_support = max([s for s in support_levels if s < current_price], default=None)
            nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
            
            # Calculate distances
            support_distance = ((current_price - nearest_support) / current_price) if nearest_support else None
            resistance_distance = ((nearest_resistance - current_price) / current_price) if nearest_resistance else None
            
            # Pivot points
            if len(high_prices) >= 1 and len(low_prices) >= 1:
                yesterday_high = high_prices[-2] if len(high_prices) > 1 else high_prices[-1]
                yesterday_low = low_prices[-2] if len(low_prices) > 1 else low_prices[-1]
                yesterday_close = close_prices[-2] if len(close_prices) > 1 else close_prices[-1]
                
                pivot_points = self.technical_analysis.calculate_pivot_points(
                    yesterday_high, yesterday_low, yesterday_close
                )
            else:
                pivot_points = {}
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'support_distance_pct': support_distance,
                'resistance_distance_pct': resistance_distance,
                'pivot_points': pivot_points,
                'level_density': len(support_levels) + len(resistance_levels)
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying key levels: {str(e)}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def _analyze_momentum_conditions(self, close_prices: np.ndarray, volume_data: np.ndarray) -> Dict[str, Any]:
        """Analyze momentum conditions"""
        try:
            # RSI
            rsi = self.technical_analysis.calculate_rsi(close_prices, 14)
            current_rsi = rsi[-1]
            
            # MACD
            macd_line, signal_line, histogram = self.technical_analysis.calculate_macd(close_prices)
            
            # Price momentum
            momentum_10 = self.technical_analysis.calculate_momentum(close_prices, 10)
            momentum_20 = self.technical_analysis.calculate_momentum(close_prices, 20)
            
            # Rate of Change
            roc = self.technical_analysis.calculate_roc(close_prices, 10)
            
            # Momentum classification
            if current_rsi > 70 and momentum_10[-1] > 0:
                momentum_state = 'overbought_strong'
            elif current_rsi < 30 and momentum_10[-1] < 0:
                momentum_state = 'oversold_weak'
            elif 40 <= current_rsi <= 60:
                momentum_state = 'neutral'
            elif current_rsi > 60:
                momentum_state = 'bullish'
            else:
                momentum_state = 'bearish'
            
            # MACD signal
            macd_signal = 'bullish' if (not np.isnan(macd_line[-1]) and not np.isnan(signal_line[-1]) and 
                                      macd_line[-1] > signal_line[-1]) else 'bearish'
            
            return {
                'rsi': current_rsi,
                'rsi_state': 'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral',
                'macd_line': macd_line[-1] if not np.isnan(macd_line[-1]) else 0,
                'macd_signal': macd_signal,
                'momentum_10': momentum_10[-1],
                'momentum_20': momentum_20[-1],
                'roc': roc[-1],
                'momentum_state': momentum_state,
                'momentum_divergence': self._check_momentum_divergence(close_prices, rsi)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum conditions: {str(e)}")
            return {'rsi': 50, 'momentum_state': 'neutral'}
    
    def _analyze_volume_conditions(self, close_prices: np.ndarray, volume_data: np.ndarray) -> Dict[str, Any]:
        """Analyze volume conditions"""
        try:
            if len(volume_data) < 10:
                return {'volume_state': 'normal', 'volume_trend': 'stable'}
            
            # Volume moving average
            volume_sma = self.technical_analysis.calculate_sma(volume_data, 20)
            current_volume = volume_data[-1]
            avg_volume = volume_sma[-1] if not np.isnan(volume_sma[-1]) else current_volume
            
            # Volume ratio
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Volume state
            if volume_ratio > self.volume_spike_threshold:
                volume_state = 'high'
            elif volume_ratio < 0.5:
                volume_state = 'low'
            else:
                volume_state = 'normal'
            
            # Volume trend
            recent_volumes = volume_data[-5:]
            previous_volumes = volume_data[-10:-5] if len(volume_data) >= 10 else volume_data[-5:]
            
            if len(recent_volumes) > 0 and len(previous_volumes) > 0:
                recent_avg = np.mean(recent_volumes)
                previous_avg = np.mean(previous_volumes)
                
                if recent_avg > previous_avg * 1.2:
                    volume_trend = 'increasing'
                elif recent_avg < previous_avg * 0.8:
                    volume_trend = 'decreasing'
                else:
                    volume_trend = 'stable'
            else:
                volume_trend = 'stable'
            
            # On Balance Volume
            obv = self._calculate_obv(close_prices, volume_data)
            obv_trend = self._analyze_obv_trend(obv)
            
            return {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'volume_state': volume_state,
                'volume_trend': volume_trend,
                'obv_trend': obv_trend,
                'volume_price_relationship': self._analyze_volume_price_relationship(close_prices, volume_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume conditions: {str(e)}")
            return {'volume_state': 'normal', 'volume_trend': 'stable'}
    
    def _analyze_cross_market_conditions(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze conditions across multiple markets"""
        try:
            cross_analysis = {
                'market_correlation': {},
                'relative_strength': {},
                'sector_rotation': {},
                'currency_strength': {},
                'safe_haven_flows': {}
            }
            
            # Get price data for all symbols
            symbol_data = {}
            for symbol in symbols:
                rates = self._get_rates_data(symbol, 'H1', 50)
                if rates is not None:
                    symbol_data[symbol] = rates['close'].values
            
            if len(symbol_data) >= 2:
                # Calculate correlations
                correlations = {}
                for i, symbol1 in enumerate(symbols):
                    for j, symbol2 in enumerate(symbols[i+1:], i+1):
                        if symbol1 in symbol_data and symbol2 in symbol_data:
                            corr = np.corrcoef(symbol_data[symbol1][-30:], symbol_data[symbol2][-30:])[0, 1]
                            correlations[f"{symbol1}_{symbol2}"] = corr
                
                cross_analysis['market_correlation'] = correlations
                
                # Relative strength analysis
                for symbol in symbols:
                    if symbol in symbol_data and len(symbol_data[symbol]) >= 20:
                        recent_change = (symbol_data[symbol][-1] - symbol_data[symbol][-20]) / symbol_data[symbol][-20]
                        cross_analysis['relative_strength'][symbol] = recent_change
                
                # Currency strength (simplified)
                if any('USD' in s for s in symbols):
                    cross_analysis['currency_strength'] = self._calculate_currency_strength(symbol_data)
                
                # Safe haven analysis
                if 'XAUUSD' in symbol_data or 'USDJPY' in symbol_data:
                    cross_analysis['safe_haven_flows'] = self._analyze_safe_haven_flows(symbol_data)
            
            return cross_analysis
            
        except Exception as e:
            self.logger.error(f"Error in cross-market analysis: {str(e)}")
            return {}
    
    def _assess_risk_environment(self, symbol_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk environment"""
        try:
            risk_metrics = {
                'overall_risk_level': 'moderate',
                'volatility_risk': 'normal',
                'correlation_risk': 'low',
                'liquidity_risk': 'low',
                'news_risk': 'low',
                'risk_score': 0.5  # 0-1 scale
            }
            
            # Aggregate volatility across symbols
            volatilities = []
            for symbol, conditions in symbol_conditions.items():
                vol_metrics = conditions.get('volatility_metrics', {})
                if 'historical_volatility' in vol_metrics:
                    volatilities.append(vol_metrics['historical_volatility'])
            
            if volatilities:
                avg_volatility = np.mean(volatilities)
                max_volatility = np.max(volatilities)
                
                if max_volatility > self.high_volatility_threshold:
                    risk_metrics['volatility_risk'] = 'high'
                    risk_metrics['risk_score'] += 0.2
                elif avg_volatility < self.low_volatility_threshold:
                    risk_metrics['volatility_risk'] = 'low'
                    risk_metrics['risk_score'] -= 0.1
            
            # Trading session risk
            current_session = self.get_trading_session()
            if current_session in [TradingSession.ASIAN_QUIET, TradingSession.WEEKEND]:
                risk_metrics['liquidity_risk'] = 'moderate'
                risk_metrics['risk_score'] += 0.1
            elif current_session == TradingSession.OVERLAP_LONDON_NY:
                risk_metrics['liquidity_risk'] = 'low'
                risk_metrics['risk_score'] -= 0.1
            
            # Overall risk level
            if risk_metrics['risk_score'] > 0.7:
                risk_metrics['overall_risk_level'] = 'high'
            elif risk_metrics['risk_score'] < 0.3:
                risk_metrics['overall_risk_level'] = 'low'
            else:
                risk_metrics['overall_risk_level'] = 'moderate'
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error assessing risk environment: {str(e)}")
            return {'overall_risk_level': 'moderate', 'risk_score': 0.5}
    
    def get_trading_session(self) -> TradingSession:
        """Determine current trading session"""
        try:
            utc_now = datetime.utcnow()
            utc_hour = utc_now.hour
            weekday = utc_now.weekday()
            
            # Weekend check
            if weekday >= 5:  # Saturday = 5, Sunday = 6
                return TradingSession.WEEKEND
            
            # Session times (UTC)
            # Asian: 22:00 - 08:00 UTC
            # London: 07:00 - 16:00 UTC  
            # New York: 12:00 - 21:00 UTC
            
            if 22 <= utc_hour or utc_hour < 7:
                return TradingSession.ASIAN
            elif 7 <= utc_hour < 12:
                return TradingSession.LONDON
            elif 12 <= utc_hour < 16:
                return TradingSession.OVERLAP_LONDON_NY
            elif 16 <= utc_hour < 21:
                return TradingSession.NEW_YORK
            else:
                return TradingSession.ASIAN_QUIET
                
        except Exception as e:
            self.logger.error(f"Error determining trading session: {str(e)}")
            return TradingSession.ASIAN_QUIET
    
    def _analyze_volatility_regime(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze overall volatility regime"""
        try:
            volatility_data = {}
            
            for symbol in symbols:
                rates = self._get_rates_data(symbol, 'H1', 100)
                if rates is not None:
                    close_prices = rates['close'].values
                    returns = np.diff(np.log(close_prices))
                    volatility = np.std(returns[-self.volatility_lookback:]) * np.sqrt(252)
                    volatility_data[symbol] = volatility
            
            if not volatility_data:
                return {'regime': 'normal', 'level': 0.5}
            
            avg_volatility = np.mean(list(volatility_data.values()))
            max_volatility = np.max(list(volatility_data.values()))
            
            # Determine volatility regime
            if avg_volatility > self.high_volatility_threshold:
                regime = 'high'
            elif avg_volatility < self.low_volatility_threshold:
                regime = 'low'
            else:
                regime = 'normal'
            
            return {
                'regime': regime,
                'average_volatility': avg_volatility,
                'maximum_volatility': max_volatility,
                'symbol_volatilities': volatility_data,
                'volatility_dispersion': np.std(list(volatility_data.values())) if len(volatility_data) > 1 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility regime: {str(e)}")
            return {'regime': 'normal', 'level': 0.5}
    
    def _analyze_market_correlations(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze market correlations"""
        try:
            correlation_matrix = {}
            symbol_returns = {}
            
            # Get returns for each symbol
            for symbol in symbols:
                rates = self._get_rates_data(symbol, 'H1', self.correlation_lookback)
                if rates is not None:
                    close_prices = rates['close'].values
                    returns = np.diff(np.log(close_prices))
                    symbol_returns[symbol] = returns
            
            # Calculate correlation matrix
            if len(symbol_returns) >= 2:
                for symbol1 in symbol_returns:
                    correlation_matrix[symbol1] = {}
                    for symbol2 in symbol_returns:
                        if len(symbol_returns[symbol1]) == len(symbol_returns[symbol2]):
                            corr = np.corrcoef(symbol_returns[symbol1], symbol_returns[symbol2])[0, 1]
                            correlation_matrix[symbol1][symbol2] = corr if not np.isnan(corr) else 0.0
                        else:
                            correlation_matrix[symbol1][symbol2] = 0.0
            
            # Calculate average correlation
            correlations = []
            for symbol1 in correlation_matrix:
                for symbol2 in correlation_matrix[symbol1]:
                    if symbol1 != symbol2:
                        correlations.append(abs(correlation_matrix[symbol1][symbol2]))
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
            
            # Correlation regime
            if avg_correlation > 0.8:
                correlation_regime = 'high'
            elif avg_correlation < 0.3:
                correlation_regime = 'low'
            else:
                correlation_regime = 'moderate'
            
            return {
                'correlation_matrix': correlation_matrix,
                'average_correlation': avg_correlation,
                'correlation_regime': correlation_regime,
                'risk_diversification': 1 - avg_correlation  # Higher is better for diversification
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {str(e)}")
            return {'correlation_regime': 'moderate', 'average_correlation': 0.5}
    
    def _analyze_session_characteristics(self) -> Dict[str, Any]:
        """Analyze current session characteristics"""
        try:
            current_session = self.get_trading_session()
            
            # Session characteristics database
            session_data = {
                TradingSession.ASIAN: {
                    'liquidity': 'low',
                    'volatility': 'low',
                    'spread_environment': 'wide',
                    'major_pairs': ['USDJPY', 'AUDUSD', 'NZDUSD'],
                    'risk_level': 'moderate'
                },
                TradingSession.LONDON: {
                    'liquidity': 'high',
                    'volatility': 'moderate',
                    'spread_environment': 'tight',
                    'major_pairs': ['EURUSD', 'GBPUSD', 'EURGBP'],
                    'risk_level': 'low'
                },
                TradingSession.NEW_YORK: {
                    'liquidity': 'high',
                    'volatility': 'high',
                    'spread_environment': 'tight',
                    'major_pairs': ['EURUSD', 'GBPUSD', 'USDCAD'],
                    'risk_level': 'moderate'
                },
                TradingSession.OVERLAP_LONDON_NY: {
                    'liquidity': 'very_high',
                    'volatility': 'high',
                    'spread_environment': 'very_tight',
                    'major_pairs': ['EURUSD', 'GBPUSD', 'USDCHF'],
                    'risk_level': 'low'
                },
                TradingSession.ASIAN_QUIET: {
                    'liquidity': 'very_low',
                    'volatility': 'very_low',
                    'spread_environment': 'very_wide',
                    'major_pairs': [],
                    'risk_level': 'high'
                },
                TradingSession.WEEKEND: {
                    'liquidity': 'none',
                    'volatility': 'none',
                    'spread_environment': 'none',
                    'major_pairs': [],
                    'risk_level': 'very_high'
                }
            }
            
            characteristics = session_data.get(current_session, session_data[TradingSession.ASIAN_QUIET])
            characteristics['current_session'] = current_session.value
            characteristics['session_start_time'] = self._get_session_start_time(current_session)
            characteristics['session_end_time'] = self._get_session_end_time(current_session)
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error analyzing session characteristics: {str(e)}")
            return {'current_session': 'unknown', 'liquidity': 'unknown'}
    
    def _assess_news_impact(self) -> Dict[str, Any]:
        """Assess potential news impact on markets"""
        try:
            # This is a simplified implementation
            # In production, this would integrate with economic calendar APIs
            
            current_time = datetime.now()
            
            # Check if it's a typical high-impact news time
            news_times = [
                (8, 30),   # 8:30 UTC - US economic data
                (12, 30),  # 12:30 UTC - ECB events
                (13, 30),  # 13:30 UTC - More US data
                (14, 0),   # 14:00 UTC - Fed events
            ]
            
            current_hour_minute = (current_time.hour, current_time.minute)
            
            # Check if within 30 minutes of news time
            news_risk = 'low'
            upcoming_events = []
            
            for news_hour, news_minute in news_times:
                news_time = current_time.replace(hour=news_hour, minute=news_minute, second=0, microsecond=0)
                time_diff = abs((current_time - news_time).total_seconds())
                
                if time_diff <= 1800:  # Within 30 minutes
                    news_risk = 'high'
                    upcoming_events.append(f"{news_hour:02d}:{news_minute:02d}")
            
            # Check if it's Friday (weekly close effects)
            if current_time.weekday() == 4:  # Friday
                news_risk = 'moderate' if news_risk == 'low' else news_risk
                upcoming_events.append("Weekly close effects")
            
            return {
                'news_risk_level': news_risk,
                'upcoming_events': upcoming_events,
                'current_day': current_time.strftime('%A'),
                'market_open_time': current_time.hour in range(7, 22),  # London to NY close
                'recommendation': self._get_news_trading_recommendation(news_risk)
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing news impact: {str(e)}")
            return {'news_risk_level': 'unknown', 'upcoming_events': []}
    
    def _determine_overall_regime(self, conditions: Dict[str, Any]) -> MarketRegime:
        """Determine overall market regime"""
        try:
            # Analyze symbol conditions to determine overall regime
            symbol_conditions = conditions.get('symbol_conditions', {})
            volatility_analysis = conditions.get('volatility_analysis', {})
            risk_environment = conditions.get('risk_environment', {})
            
            # Count regime indicators
            trending_count = 0
            ranging_count = 0
            volatile_count = 0
            
            for symbol, symbol_data in symbol_conditions.items():
                trend_analysis = symbol_data.get('trend_analysis', {})
                vol_metrics = symbol_data.get('volatility_metrics', {})
                
                trend_strength = trend_analysis.get('strength', 0)
                vol_regime = vol_metrics.get('volatility_regime', 'normal')
                
                if trend_strength > 0.7:
                    trending_count += 1
                elif vol_regime == 'high':
                    volatile_count += 1
                else:
                    ranging_count += 1
            
            total_symbols = len(symbol_conditions)
            
            if total_symbols == 0:
                return MarketRegime.RANGING
            
            # Determine regime based on majority
            trending_ratio = trending_count / total_symbols
            volatile_ratio = volatile_count / total_symbols
            
            # Overall volatility consideration
            overall_vol_regime = volatility_analysis.get('regime', 'normal')
            
            if overall_vol_regime == 'high':
                return MarketRegime.VOLATILE
            elif trending_ratio > 0.6:
                # Determine if bullish or bearish trending
                bullish_count = 0
                for symbol_data in symbol_conditions.values():
                    trend_direction = symbol_data.get('trend_analysis', {}).get('direction', 'sideways')
                    if trend_direction == 'bullish':
                        bullish_count += 1
                
                if bullish_count > len(symbol_conditions) / 2:
                    return MarketRegime.TRENDING_BULLISH
                else:
                    return MarketRegime.TRENDING_BEARISH
            elif volatile_ratio > 0.5:
                return MarketRegime.VOLATILE
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            self.logger.error(f"Error determining overall regime: {str(e)}")
            return MarketRegime.RANGING
    
    # Helper methods
    def _get_rates_data(self, symbol: str, timeframe: str, count: int):
        """Get rates data from MT5 or cache"""
        try:
            if self.mt5_connector:
                return self.mt5_connector.get_rates(symbol, timeframe, count)
            else:
                # Return None if no connector available
                return None
        except Exception:
            return None
    
    def _get_default_conditions(self) -> Dict[str, Any]:
        """Get default conditions when analysis fails"""
        return {
            'timestamp': datetime.now(),
            'overall_regime': MarketRegime.RANGING,
            'trading_session': TradingSession.ASIAN_QUIET,
            'symbol_conditions': {},
            'error': 'Failed to analyze market conditions'
        }
    
    def _get_default_symbol_conditions(self) -> Dict[str, Any]:
        """Get default symbol conditions"""
        return {
            'symbol': 'UNKNOWN',
            'current_price': 0.0,
            'trend_analysis': {'direction': 'sideways', 'strength': 0.0},
            'volatility_metrics': {'volatility_regime': 'normal'},
            'momentum_indicators': {'rsi': 50, 'momentum_state': 'neutral'}
        }
    
    def _calculate_volatility_percentile(self, current_vol: float) -> float:
        """Calculate volatility percentile based on historical data"""
        # Simplified implementation
        if current_vol > 0.01:
            return 90.0
        elif current_vol > 0.005:
            return 70.0
        elif current_vol < 0.001:
            return 10.0
        else:
            return 50.0
    
    def _check_momentum_divergence(self, prices: np.ndarray, rsi: np.ndarray) -> bool:
        """Check for momentum divergence"""
        try:
            if len(prices) < 10 or len(rsi) < 10:
                return False
            
            # Simple divergence check
            recent_price_trend = prices[-5:] - prices[-10:-5]
            recent_rsi_trend = rsi[-5:] - rsi[-10:-5]
            
            price_direction = np.mean(recent_price_trend)
            rsi_direction = np.mean(recent_rsi_trend)
            
            # Divergence if price and RSI move in opposite directions
            return (price_direction > 0 > rsi_direction) or (price_direction < 0 < rsi_direction)
            
        except Exception:
            return False
    
    def _calculate_obv(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate On Balance Volume"""
        obv = np.zeros(len(prices))
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                obv[i] = obv[i-1] + volumes[i]
            elif prices[i] < prices[i-1]:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        return obv
    
    def _analyze_obv_trend(self, obv: np.ndarray) -> str:
        """Analyze OBV trend"""
        if len(obv) < 10:
            return 'neutral'
        
        recent_obv = np.mean(obv[-5:])
        previous_obv = np.mean(obv[-10:-5])
        
        if recent_obv > previous_obv * 1.05:
            return 'increasing'
        elif recent_obv < previous_obv * 0.95:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_volume_price_relationship(self, prices: np.ndarray, volumes: np.ndarray) -> str:
        """Analyze volume-price relationship"""
        try:
            if len(prices) < 5 or len(volumes) < 5:
                return 'neutral'
            
            price_changes = np.diff(prices[-5:])
            volume_changes = np.diff(volumes[-5:])
            
            # Correlation between price and volume changes
            if len(price_changes) > 2 and len(volume_changes) > 2:
                correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                
                if correlation > 0.5:
                    return 'confirming'  # Volume confirms price movement
                elif correlation < -0.5:
                    return 'diverging'   # Volume diverges from price
                else:
                    return 'neutral'
            
            return 'neutral'
            
        except Exception:
            return 'neutral'
    
    def _calculate_currency_strength(self, symbol_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate relative currency strength"""
        try:
            currency_strength = {}
            currencies = set()
            
            # Extract currencies from symbols
            for symbol in symbol_data.keys():
                if len(symbol) == 6:  # Standard forex pair
                    base = symbol[:3]
                    quote = symbol[3:]
                    currencies.add(base)
                    currencies.add(quote)
            
            # Calculate strength for each currency
            for currency in currencies:
                strength_sum = 0
                pair_count = 0
                
                for symbol, prices in symbol_data.items():
                    if len(symbol) == 6 and len(prices) >= 20:
                        base = symbol[:3]
                        quote = symbol[3:]
                        
                        if currency in [base, quote]:
                            # Calculate price change
                            price_change = (prices[-1] - prices[-20]) / prices[-20]
                            
                            if currency == base:
                                strength_sum += price_change
                            else:  # currency == quote
                                strength_sum -= price_change
                            
                            pair_count += 1
                
                if pair_count > 0:
                    currency_strength[currency] = strength_sum / pair_count
            
            return currency_strength
            
        except Exception as e:
            self.logger.error(f"Error calculating currency strength: {str(e)}")
            return {}
    
    def _analyze_safe_haven_flows(self, symbol_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze safe haven flows"""
        try:
            safe_haven_analysis = {}
            
            # Check gold (XAUUSD) performance
            if 'XAUUSD' in symbol_data and len(symbol_data['XAUUSD']) >= 10:
                gold_prices = symbol_data['XAUUSD']
                gold_change = (gold_prices[-1] - gold_prices[-10]) / gold_prices[-10]
                safe_haven_analysis['gold_performance'] = gold_change
                
                if gold_change > 0.01:  # 1% gain
                    safe_haven_analysis['gold_signal'] = 'risk_off'
                elif gold_change < -0.01:
                    safe_haven_analysis['gold_signal'] = 'risk_on'
                else:
                    safe_haven_analysis['gold_signal'] = 'neutral'
            
            # Check JPY strength (safe haven currency)
            jpy_pairs = [s for s in symbol_data.keys() if 'JPY' in s]
            if jpy_pairs:
                jpy_strength = 0
                for pair in jpy_pairs:
                    if len(symbol_data[pair]) >= 10:
                        change = (symbol_data[pair][-1] - symbol_data[pair][-10]) / symbol_data[pair][-10]
                        if pair.endswith('JPY'):
                            jpy_strength -= change  # JPY strengthening means pair goes down
                        else:
                            jpy_strength += change  # JPY strengthening means pair goes up
                
                jpy_strength /= len(jpy_pairs)
                safe_haven_analysis['jpy_strength'] = jpy_strength
                
                if jpy_strength > 0.005:
                    safe_haven_analysis['jpy_signal'] = 'risk_off'
                elif jpy_strength < -0.005:
                    safe_haven_analysis['jpy_signal'] = 'risk_on'
                else:
                    safe_haven_analysis['jpy_signal'] = 'neutral'
            
            # Overall safe haven assessment
            gold_signal = safe_haven_analysis.get('gold_signal', 'neutral')
            jpy_signal = safe_haven_analysis.get('jpy_signal', 'neutral')
            
            if gold_signal == 'risk_off' and jpy_signal == 'risk_off':
                safe_haven_analysis['overall_sentiment'] = 'strong_risk_off'
            elif gold_signal == 'risk_on' and jpy_signal == 'risk_on':
                safe_haven_analysis['overall_sentiment'] = 'strong_risk_on'
            elif gold_signal == 'risk_off' or jpy_signal == 'risk_off':
                safe_haven_analysis['overall_sentiment'] = 'mild_risk_off'
            elif gold_signal == 'risk_on' or jpy_signal == 'risk_on':
                safe_haven_analysis['overall_sentiment'] = 'mild_risk_on'
            else:
                safe_haven_analysis['overall_sentiment'] = 'neutral'
            
            return safe_haven_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing safe haven flows: {str(e)}")
            return {'overall_sentiment': 'neutral'}
    
    def _analyze_timeframe_alignment(self, h1_data, m15_data, m5_data) -> Dict[str, Any]:
        """Analyze alignment across timeframes"""
        try:
            alignment = {}
            
            # Get close prices for each timeframe
            h1_close = h1_data['close'].values
            m15_close = m15_data['close'].values
            m5_close = m5_data['close'].values
            
            # Calculate trends for each timeframe
            timeframes = {
                'H1': h1_close,
                'M15': m15_close,
                'M5': m5_close
            }
            
            trends = {}
            for tf, prices in timeframes.items():
                if len(prices) >= 20:
                    trend_analysis = self._analyze_trend_conditions(prices, prices, prices)
                    trends[tf] = trend_analysis.get('direction', 'sideways')
            
            # Check alignment
            bullish_count = sum(1 for trend in trends.values() if trend == 'bullish')
            bearish_count = sum(1 for trend in trends.values() if trend == 'bearish')
            
            if bullish_count >= 2:
                alignment['overall_direction'] = 'bullish'
                alignment['strength'] = bullish_count / len(trends)
            elif bearish_count >= 2:
                alignment['overall_direction'] = 'bearish'
                alignment['strength'] = bearish_count / len(trends)
            else:
                alignment['overall_direction'] = 'mixed'
                alignment['strength'] = 0.5
            
            alignment['timeframe_trends'] = trends
            alignment['alignment_quality'] = 'strong' if alignment['strength'] > 0.8 else 'weak' if alignment['strength'] < 0.4 else 'moderate'
            
            return alignment
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe alignment: {str(e)}")
            return {'overall_direction': 'mixed', 'strength': 0.5}
    
    def _analyze_market_structure(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict[str, Any]:
        """Analyze market structure (higher highs, lower lows, etc.)"""
        try:
            if len(highs) < 10:
                return {'structure': 'unknown', 'quality': 'poor'}
            
            # Find recent swing highs and lows
            recent_highs = []
            recent_lows = []
            
            # Simple peak/trough detection
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    recent_highs.append((i, highs[i]))
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    recent_lows.append((i, lows[i]))
            
            # Take last few swings
            recent_highs = recent_highs[-3:] if len(recent_highs) >= 3 else recent_highs
            recent_lows = recent_lows[-3:] if len(recent_lows) >= 3 else recent_lows
            
            structure_analysis = {
                'structure': 'unknown',
                'quality': 'poor',
                'recent_highs': len(recent_highs),
                'recent_lows': len(recent_lows)
            }
            
            # Analyze structure
            if len(recent_highs) >= 2:
                higher_highs = all(recent_highs[i][1] > recent_highs[i-1][1] for i in range(1, len(recent_highs)))
                if higher_highs and len(recent_lows) >= 2:
                    higher_lows = all(recent_lows[i][1] > recent_lows[i-1][1] for i in range(1, len(recent_lows)))
                    if higher_lows:
                        structure_analysis['structure'] = 'uptrend'
                        structure_analysis['quality'] = 'good'
            
            if len(recent_lows) >= 2:
                lower_lows = all(recent_lows[i][1] < recent_lows[i-1][1] for i in range(1, len(recent_lows)))
                if lower_lows and len(recent_highs) >= 2:
                    lower_highs = all(recent_highs[i][1] < recent_highs[i-1][1] for i in range(1, len(recent_highs)))
                    if lower_highs:
                        structure_analysis['structure'] = 'downtrend'
                        structure_analysis['quality'] = 'good'
            
            if structure_analysis['structure'] == 'unknown':
                structure_analysis['structure'] = 'sideways'
                structure_analysis['quality'] = 'moderate'
            
            return structure_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {str(e)}")
            return {'structure': 'unknown', 'quality': 'poor'}
    
    def _assess_breakout_potential(self, highs: np.ndarray, lows: np.ndarray, 
                                 closes: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """Assess breakout potential"""
        try:
            if len(closes) < 20:
                return {'potential': 'low', 'direction': 'unknown'}
            
            # Calculate recent range
            recent_period = 20
            recent_high = np.max(highs[-recent_period:])
            recent_low = np.min(lows[-recent_period:])
            current_price = closes[-1]
            
            # Range analysis
            range_size = (recent_high - recent_low) / recent_low
            price_position = (current_price - recent_low) / (recent_high - recent_low)
            
            # Volume analysis
            if len(volumes) >= 20:
                recent_volume = np.mean(volumes[-5:])
                avg_volume = np.mean(volumes[-20:])
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            # Volatility squeeze detection
            volatility = np.std(closes[-recent_period:]) / np.mean(closes[-recent_period:])
            avg_volatility = np.std(closes[-50:-recent_period]) / np.mean(closes[-50:-recent_period]) if len(closes) >= 50 else volatility
            
            volatility_ratio = volatility / avg_volatility if avg_volatility > 0 else 1.0
            
            breakout_assessment = {
                'potential': 'low',
                'direction': 'unknown',
                'range_size': range_size,
                'price_position': price_position,
                'volume_ratio': volume_ratio,
                'volatility_ratio': volatility_ratio
            }
            
            # Assess breakout potential
            if range_size < 0.02 and volatility_ratio < 0.7:  # Tight range + low volatility
                breakout_assessment['potential'] = 'high'
                
                # Determine likely direction
                if price_position > 0.8:
                    breakout_assessment['direction'] = 'upward'
                elif price_position < 0.2:
                    breakout_assessment['direction'] = 'downward'
                else:
                    breakout_assessment['direction'] = 'unclear'
            
            elif range_size < 0.05 and volume_ratio > 1.5:  # Moderate range + high volume
                breakout_assessment['potential'] = 'moderate'
                breakout_assessment['direction'] = 'upward' if price_position > 0.6 else 'downward' if price_position < 0.4 else 'unclear'
            
            return breakout_assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing breakout potential: {str(e)}")
            return {'potential': 'low', 'direction': 'unknown'}
    
    def _get_session_start_time(self, session: TradingSession) -> str:
        """Get session start time"""
        session_times = {
            TradingSession.ASIAN: "22:00 UTC",
            TradingSession.LONDON: "07:00 UTC",
            TradingSession.NEW_YORK: "12:00 UTC",
            TradingSession.OVERLAP_LONDON_NY: "12:00 UTC",
            TradingSession.ASIAN_QUIET: "00:00 UTC",
            TradingSession.WEEKEND: "N/A"
        }
        return session_times.get(session, "Unknown")
    
    def _get_session_end_time(self, session: TradingSession) -> str:
        """Get session end time"""
        session_times = {
            TradingSession.ASIAN: "08:00 UTC",
            TradingSession.LONDON: "16:00 UTC",
            TradingSession.NEW_YORK: "21:00 UTC",
            TradingSession.OVERLAP_LONDON_NY: "16:00 UTC",
            TradingSession.ASIAN_QUIET: "07:00 UTC",
            TradingSession.WEEKEND: "N/A"
        }
        return session_times.get(session, "Unknown")
    
    def _get_news_trading_recommendation(self, news_risk: str) -> str:
        """Get trading recommendation based on news risk"""
        recommendations = {
            'low': 'Normal trading conditions. Standard risk management applies.',
            'moderate': 'Elevated caution advised. Consider reduced position sizes.',
            'high': 'High-impact news expected. Consider avoiding new positions or using tight stops.',
            'unknown': 'Unable to assess news risk. Use conservative approach.'
        }
        return recommendations.get(news_risk, recommendations['unknown'])
    
    def get_condition_summary(self) -> str:
        """Get a brief summary of current market conditions"""
        try:
            if not self.condition_history:
                return "No market condition data available."
            
            latest = self.condition_history[-1]
            regime = latest.get('overall_regime', MarketRegime.RANGING)
            session = latest.get('trading_session', TradingSession.ASIAN_QUIET)
            
            risk_level = latest.get('risk_environment', {}).get('overall_risk_level', 'moderate')
            
            return f"Market Regime: {regime.value.replace('_', ' ').title()}, " \
                   f"Session: {session.value.replace('_', ' ').title()}, " \
                   f"Risk Level: {risk_level.title()}"
                   
        except Exception as e:
            self.logger.error(f"Error creating condition summary: {str(e)}")
            return "Error generating market condition summary."
