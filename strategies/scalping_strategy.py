"""
Scalping Strategy for AuraTrade Bot
Short-term trading strategy for quick profits
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque

from config.config import config
from config.settings import settings
from core.mt5_connector import MT5Connector
from core.order_manager import OrderManager
from core.risk_manager import RiskManager
from analysis.technical_analysis import TechnicalAnalysis
from analysis.market_conditions import MarketConditions
from utils.logger import Logger

class ScalpingStrategy:
    """Scalping Strategy Implementation"""
    
    def __init__(self, mt5_connector: MT5Connector, order_manager: OrderManager, risk_manager: RiskManager):
        self.mt5_connector = mt5_connector
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.logger = Logger()
        self.technical_analysis = TechnicalAnalysis()
        self.market_conditions = MarketConditions()
        
        # Strategy parameters
        self.enabled = True
        self.timeframes = ['M1', 'M5']  # Focus on short timeframes
        self.max_hold_time_minutes = 15  # Maximum position hold time
        self.min_profit_pips = 3  # Minimum profit target
        self.max_loss_pips = 5   # Maximum loss tolerance
        self.volume_threshold = 50  # Minimum volume for signal
        
        # Technical indicator parameters
        self.fast_ema_period = 5
        self.slow_ema_period = 13
        self.rsi_period = 9  # Shorter RSI for scalping
        self.bb_period = 12
        self.bb_deviation = 1.5
        self.stoch_k_period = 5
        self.stoch_d_period = 3
        
        # Market condition filters
        self.min_volatility = 0.0005  # Minimum volatility for trading
        self.max_volatility = 0.0030  # Maximum volatility threshold
        self.min_spread_pips = 0.5
        self.max_spread_pips = 2.5
        
        # Position tracking
        self.active_scalp_positions = {}
        self.signal_history = deque(maxlen=1000)
        self.performance_tracking = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pips': 0.0,
            'avg_hold_time_minutes': 0.0,
            'best_trade_pips': 0.0,
            'worst_trade_pips': 0.0
        }
        
        # Signal filtering
        self.last_signal_time = {}  # symbol -> timestamp
        self.min_signal_interval_seconds = 30  # Minimum time between signals
        
        self.logger.info("Scalping Strategy initialized")
    
    def is_enabled(self) -> bool:
        """Check if scalping strategy is enabled"""
        return self.enabled and config.STRATEGY_ENABLED.get('scalping', True)
    
    def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate scalping trading signals"""
        if not self.is_enabled():
            return []
        
        signals = []
        
        for symbol in settings.trading.enabled_symbols:
            try:
                # Check market conditions first
                if not self._check_market_conditions(symbol):
                    continue
                
                # Check signal frequency
                if not self._can_generate_signal(symbol):
                    continue
                
                # Analyze each timeframe
                for timeframe in self.timeframes:
                    scalp_signals = self._analyze_scalping_opportunity(symbol, timeframe)
                    signals.extend(scalp_signals)
                
            except Exception as e:
                self.logger.error(f"Error generating scalping signals for {symbol}: {str(e)}")
        
        return signals
    
    def _check_market_conditions(self, symbol: str) -> bool:
        """Check if market conditions are suitable for scalping"""
        try:
            # Get current market data
            tick = self.mt5_connector.get_tick(symbol)
            if not tick:
                return False
            
            # Check spread
            spread = tick.ask - tick.bid
            pip_size = self._get_pip_size(symbol)
            spread_pips = spread / pip_size
            
            if spread_pips < self.min_spread_pips or spread_pips > self.max_spread_pips:
                return False
            
            # Check volatility
            volatility = self._get_current_volatility(symbol)
            if volatility < self.min_volatility or volatility > self.max_volatility:
                return False
            
            # Check trading session
            market_session = self.market_conditions.get_trading_session()
            if market_session in ['asian_quiet', 'weekend']:
                return False  # Avoid quiet sessions
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking market conditions: {str(e)}")
            return False
    
    def _can_generate_signal(self, symbol: str) -> bool:
        """Check if enough time has passed since last signal"""
        try:
            if symbol not in self.last_signal_time:
                return True
            
            time_since_last = datetime.now() - self.last_signal_time[symbol]
            return time_since_last.total_seconds() >= self.min_signal_interval_seconds
            
        except Exception as e:
            self.logger.error(f"Error checking signal frequency: {str(e)}")
            return True
    
    def _analyze_scalping_opportunity(self, symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """Analyze scalping opportunities for symbol and timeframe"""
        signals = []
        
        try:
            # Get OHLC data
            rates = self.mt5_connector.get_rates(symbol, timeframe, 100)
            if rates is None or len(rates) < 50:
                return signals
            
            # Calculate technical indicators
            indicators = self._calculate_scalping_indicators(rates)
            
            # Get current market data
            tick = self.mt5_connector.get_tick(symbol)
            if not tick:
                return signals
            
            # Analyze different scalping patterns
            signals.extend(self._check_ema_crossover_scalp(symbol, timeframe, rates, indicators, tick))
            signals.extend(self._check_rsi_divergence_scalp(symbol, timeframe, rates, indicators, tick))
            signals.extend(self._check_bollinger_bounce_scalp(symbol, timeframe, rates, indicators, tick))
            signals.extend(self._check_stochastic_scalp(symbol, timeframe, rates, indicators, tick))
            signals.extend(self._check_momentum_scalp(symbol, timeframe, rates, indicators, tick))
            
        except Exception as e:
            self.logger.error(f"Error analyzing scalping opportunity: {str(e)}")
        
        return signals
    
    def _calculate_scalping_indicators(self, rates) -> Dict[str, np.ndarray]:
        """Calculate technical indicators for scalping"""
        try:
            close_prices = rates['close'].values
            high_prices = rates['high'].values
            low_prices = rates['low'].values
            volumes = rates.get('tick_volume', rates.get('real_volume', np.ones(len(close_prices)))).values
            
            indicators = {
                'ema_fast': self.technical_analysis.calculate_ema(close_prices, self.fast_ema_period),
                'ema_slow': self.technical_analysis.calculate_ema(close_prices, self.slow_ema_period),
                'rsi': self.technical_analysis.calculate_rsi(close_prices, self.rsi_period),
                'stoch_k': self.technical_analysis.calculate_stochastic(high_prices, low_prices, close_prices, self.stoch_k_period),
                'atr': self.technical_analysis.calculate_atr(high_prices, low_prices, close_prices, 14),
                'volume_sma': self.technical_analysis.calculate_sma(volumes, 20)
            }
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.technical_analysis.calculate_bollinger_bands(
                close_prices, self.bb_period, self.bb_deviation
            )
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            
            # Stochastic %D
            indicators['stoch_d'] = self.technical_analysis.calculate_sma(indicators['stoch_k'], self.stoch_d_period)
            
            # Price momentum
            indicators['momentum'] = self._calculate_momentum(close_prices, 3)
            
            # Volume ratio
            if len(volumes) > 1:
                indicators['volume_ratio'] = volumes / indicators['volume_sma']
            else:
                indicators['volume_ratio'] = np.ones_like(volumes)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating scalping indicators: {str(e)}")
            return {}
    
    def _check_ema_crossover_scalp(self, symbol: str, timeframe: str, rates, indicators: Dict, tick) -> List[Dict[str, Any]]:
        """Check for EMA crossover scalping opportunities"""
        signals = []
        
        try:
            if 'ema_fast' not in indicators or 'ema_slow' not in indicators:
                return signals
            
            ema_fast = indicators['ema_fast']
            ema_slow = indicators['ema_slow']
            
            if len(ema_fast) < 3 or len(ema_slow) < 3:
                return signals
            
            # Check for crossover
            current_fast = ema_fast[-1]
            current_slow = ema_slow[-1]
            prev_fast = ema_fast[-2]
            prev_slow = ema_slow[-2]
            
            # Golden cross (bullish)
            if prev_fast <= prev_slow and current_fast > current_slow:
                # Additional confirmation
                if self._confirm_bullish_scalp(indicators):
                    signal = self._create_scalp_signal(
                        symbol, 'BUY', 'ema_crossover', 
                        f'EMA golden cross on {timeframe}', 
                        tick, indicators
                    )
                    if signal:
                        signals.append(signal)
            
            # Death cross (bearish)
            elif prev_fast >= prev_slow and current_fast < current_slow:
                # Additional confirmation
                if self._confirm_bearish_scalp(indicators):
                    signal = self._create_scalp_signal(
                        symbol, 'SELL', 'ema_crossover',
                        f'EMA death cross on {timeframe}',
                        tick, indicators
                    )
                    if signal:
                        signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error checking EMA crossover scalp: {str(e)}")
        
        return signals
    
    def _check_rsi_divergence_scalp(self, symbol: str, timeframe: str, rates, indicators: Dict, tick) -> List[Dict[str, Any]]:
        """Check for RSI divergence scalping opportunities"""
        signals = []
        
        try:
            if 'rsi' not in indicators:
                return signals
            
            rsi = indicators['rsi']
            close_prices = rates['close'].values
            
            if len(rsi) < 10 or len(close_prices) < 10:
                return signals
            
            # Check for oversold/overbought with momentum
            current_rsi = rsi[-1]
            
            # RSI oversold bounce
            if current_rsi < 25 and rsi[-2] < current_rsi:  # RSI turning up from oversold
                if close_prices[-1] > close_prices[-2]:  # Price confirming
                    signal = self._create_scalp_signal(
                        symbol, 'BUY', 'rsi_bounce',
                        f'RSI oversold bounce: {current_rsi:.1f}',
                        tick, indicators
                    )
                    if signal:
                        signals.append(signal)
            
            # RSI overbought drop
            elif current_rsi > 75 and rsi[-2] > current_rsi:  # RSI turning down from overbought
                if close_prices[-1] < close_prices[-2]:  # Price confirming
                    signal = self._create_scalp_signal(
                        symbol, 'SELL', 'rsi_bounce',
                        f'RSI overbought drop: {current_rsi:.1f}',
                        tick, indicators
                    )
                    if signal:
                        signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error checking RSI divergence scalp: {str(e)}")
        
        return signals
    
    def _check_bollinger_bounce_scalp(self, symbol: str, timeframe: str, rates, indicators: Dict, tick) -> List[Dict[str, Any]]:
        """Check for Bollinger Band bounce scalping opportunities"""
        signals = []
        
        try:
            if not all(key in indicators for key in ['bb_upper', 'bb_lower', 'bb_middle']):
                return signals
            
            bb_upper = indicators['bb_upper']
            bb_lower = indicators['bb_lower']
            bb_middle = indicators['bb_middle']
            close_prices = rates['close'].values
            
            if len(bb_upper) < 3:
                return signals
            
            current_price = close_prices[-1]
            prev_price = close_prices[-2]
            
            # Bounce from lower band (bullish)
            if (prev_price <= bb_lower[-2] and 
                current_price > bb_lower[-1] and
                current_price < bb_middle[-1]):  # Still below middle
                
                signal = self._create_scalp_signal(
                    symbol, 'BUY', 'bb_bounce',
                    f'Bollinger lower band bounce',
                    tick, indicators
                )
                if signal:
                    signals.append(signal)
            
            # Bounce from upper band (bearish)
            elif (prev_price >= bb_upper[-2] and 
                  current_price < bb_upper[-1] and
                  current_price > bb_middle[-1]):  # Still above middle
                
                signal = self._create_scalp_signal(
                    symbol, 'SELL', 'bb_bounce',
                    f'Bollinger upper band bounce',
                    tick, indicators
                )
                if signal:
                    signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error checking Bollinger bounce scalp: {str(e)}")
        
        return signals
    
    def _check_stochastic_scalp(self, symbol: str, timeframe: str, rates, indicators: Dict, tick) -> List[Dict[str, Any]]:
        """Check for Stochastic scalping opportunities"""
        signals = []
        
        try:
            if not all(key in indicators for key in ['stoch_k', 'stoch_d']):
                return signals
            
            stoch_k = indicators['stoch_k']
            stoch_d = indicators['stoch_d']
            
            if len(stoch_k) < 3 or len(stoch_d) < 3:
                return signals
            
            current_k = stoch_k[-1]
            current_d = stoch_d[-1]
            prev_k = stoch_k[-2]
            prev_d = stoch_d[-2]
            
            # Stochastic bullish crossover in oversold region
            if (current_k < 25 and current_d < 25 and  # Oversold
                prev_k <= prev_d and current_k > current_d):  # Bullish crossover
                
                signal = self._create_scalp_signal(
                    symbol, 'BUY', 'stoch_crossover',
                    f'Stochastic bullish crossover in oversold',
                    tick, indicators
                )
                if signal:
                    signals.append(signal)
            
            # Stochastic bearish crossover in overbought region
            elif (current_k > 75 and current_d > 75 and  # Overbought
                  prev_k >= prev_d and current_k < current_d):  # Bearish crossover
                
                signal = self._create_scalp_signal(
                    symbol, 'SELL', 'stoch_crossover',
                    f'Stochastic bearish crossover in overbought',
                    tick, indicators
                )
                if signal:
                    signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error checking stochastic scalp: {str(e)}")
        
        return signals
    
    def _check_momentum_scalp(self, symbol: str, timeframe: str, rates, indicators: Dict, tick) -> List[Dict[str, Any]]:
        """Check for momentum scalping opportunities"""
        signals = []
        
        try:
            if 'momentum' not in indicators or 'volume_ratio' not in indicators:
                return signals
            
            momentum = indicators['momentum']
            volume_ratio = indicators['volume_ratio']
            
            if len(momentum) < 3 or len(volume_ratio) < 3:
                return signals
            
            current_momentum = momentum[-1]
            current_volume_ratio = volume_ratio[-1]
            
            # Strong bullish momentum with volume
            if (current_momentum > 0.0002 and  # Positive momentum
                current_volume_ratio > 1.2):   # Above average volume
                
                signal = self._create_scalp_signal(
                    symbol, 'BUY', 'momentum',
                    f'Strong bullish momentum with volume',
                    tick, indicators
                )
                if signal:
                    signals.append(signal)
            
            # Strong bearish momentum with volume
            elif (current_momentum < -0.0002 and  # Negative momentum
                  current_volume_ratio > 1.2):    # Above average volume
                
                signal = self._create_scalp_signal(
                    symbol, 'SELL', 'momentum',
                    f'Strong bearish momentum with volume',
                    tick, indicators
                )
                if signal:
                    signals.append(signal)
            
        except Exception as e:
            self.logger.error(f"Error checking momentum scalp: {str(e)}")
        
        return signals
    
    def _confirm_bullish_scalp(self, indicators: Dict) -> bool:
        """Additional confirmation for bullish scalp signals"""
        try:
            # RSI not overbought
            if 'rsi' in indicators and len(indicators['rsi']) > 0:
                if indicators['rsi'][-1] > 70:
                    return False
            
            # Stochastic not overbought
            if 'stoch_k' in indicators and len(indicators['stoch_k']) > 0:
                if indicators['stoch_k'][-1] > 80:
                    return False
            
            # Volume confirmation
            if 'volume_ratio' in indicators and len(indicators['volume_ratio']) > 0:
                if indicators['volume_ratio'][-1] < 0.8:  # Below average volume
                    return False
            
            return True
            
        except Exception:
            return True  # Default to allow signal if confirmation fails
    
    def _confirm_bearish_scalp(self, indicators: Dict) -> bool:
        """Additional confirmation for bearish scalp signals"""
        try:
            # RSI not oversold
            if 'rsi' in indicators and len(indicators['rsi']) > 0:
                if indicators['rsi'][-1] < 30:
                    return False
            
            # Stochastic not oversold
            if 'stoch_k' in indicators and len(indicators['stoch_k']) > 0:
                if indicators['stoch_k'][-1] < 20:
                    return False
            
            # Volume confirmation
            if 'volume_ratio' in indicators and len(indicators['volume_ratio']) > 0:
                if indicators['volume_ratio'][-1] < 0.8:  # Below average volume
                    return False
            
            return True
            
        except Exception:
            return True  # Default to allow signal if confirmation fails
    
    def _create_scalp_signal(self, symbol: str, action: str, pattern: str, 
                           reason: str, tick, indicators: Dict) -> Optional[Dict[str, Any]]:
        """Create a scalping signal with appropriate parameters"""
        try:
            pip_size = self._get_pip_size(symbol)
            
            # Calculate entry price
            if action == 'BUY':
                entry_price = tick.ask
            else:
                entry_price = tick.bid
            
            # Calculate stop loss and take profit
            atr = indicators.get('atr', np.array([0.0001]))[-1] if 'atr' in indicators else 0.0001
            
            # Dynamic stop loss based on ATR and volatility
            stop_distance = max(atr * 1.5, pip_size * self.max_loss_pips)
            profit_distance = max(atr * 2.0, pip_size * self.min_profit_pips)
            
            if action == 'BUY':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + profit_distance
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - profit_distance
            
            # Calculate confidence based on multiple factors
            confidence = self._calculate_signal_confidence(pattern, indicators)
            
            # Calculate lot size
            lot_size = self._calculate_scalping_lot_size(symbol, confidence, stop_distance)
            
            signal = {
                'strategy': 'scalping',
                'pattern': pattern,
                'symbol': symbol,
                'action': action,
                'lot_size': lot_size,
                'price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'max_hold_time_minutes': self.max_hold_time_minutes,
                'reason': reason,
                'timestamp': datetime.now(),
                'indicators': {
                    'rsi': indicators.get('rsi', [0])[-1] if 'rsi' in indicators and len(indicators['rsi']) > 0 else 0,
                    'atr': atr,
                    'volume_ratio': indicators.get('volume_ratio', [1])[-1] if 'volume_ratio' in indicators and len(indicators['volume_ratio']) > 0 else 1
                }
            }
            
            # Update last signal time
            self.last_signal_time[symbol] = datetime.now()
            
            # Add to signal history
            self.signal_history.append(signal.copy())
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating scalp signal: {str(e)}")
            return None
    
    def _calculate_signal_confidence(self, pattern: str, indicators: Dict) -> float:
        """Calculate confidence score for the signal"""
        try:
            confidence = 0.5  # Base confidence
            
            # Pattern-specific confidence
            pattern_weights = {
                'ema_crossover': 0.7,
                'rsi_bounce': 0.6,
                'bb_bounce': 0.8,
                'stoch_crossover': 0.6,
                'momentum': 0.5
            }
            confidence = pattern_weights.get(pattern, 0.5)
            
            # Volume confirmation
            if 'volume_ratio' in indicators and len(indicators['volume_ratio']) > 0:
                volume_ratio = indicators['volume_ratio'][-1]
                if volume_ratio > 1.5:
                    confidence += 0.1
                elif volume_ratio < 0.8:
                    confidence -= 0.1
            
            # RSI confirmation
            if 'rsi' in indicators and len(indicators['rsi']) > 0:
                rsi = indicators['rsi'][-1]
                if 30 <= rsi <= 70:  # Not extreme
                    confidence += 0.05
            
            # Multiple indicator alignment
            aligned_indicators = 0
            if 'ema_fast' in indicators and 'ema_slow' in indicators:
                if len(indicators['ema_fast']) > 0 and len(indicators['ema_slow']) > 0:
                    if indicators['ema_fast'][-1] > indicators['ema_slow'][-1]:
                        aligned_indicators += 1
            
            if aligned_indicators > 0:
                confidence += 0.05
            
            return min(max(confidence, 0.1), 0.95)  # Clamp between 10% and 95%
            
        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {str(e)}")
            return 0.5
    
    def _calculate_scalping_lot_size(self, symbol: str, confidence: float, stop_distance: float) -> float:
        """Calculate appropriate lot size for scalping"""
        try:
            base_lot = settings.trading.default_lot_size
            
            # Adjust based on confidence
            confidence_multiplier = 0.5 + (confidence * 1.0)  # 0.5x to 1.5x based on confidence
            
            # Adjust based on risk (stop distance)
            pip_size = self._get_pip_size(symbol)
            stop_pips = stop_distance / pip_size
            
            # Smaller lots for wider stops
            if stop_pips > 8:
                risk_multiplier = 0.6
            elif stop_pips > 5:
                risk_multiplier = 0.8
            else:
                risk_multiplier = 1.0
            
            lot_size = base_lot * confidence_multiplier * risk_multiplier
            
            # Apply limits
            min_lot = 0.01
            max_lot = base_lot * 2.0  # Maximum 2x base lot for scalping
            
            return round(max(min_lot, min(lot_size, max_lot)), 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating scalping lot size: {str(e)}")
            return settings.trading.default_lot_size
    
    def _calculate_momentum(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate price momentum"""
        momentum = np.full(len(prices), 0.0)
        for i in range(period, len(prices)):
            momentum[i] = prices[i] - prices[i - period]
        return momentum
    
    def _get_current_volatility(self, symbol: str) -> float:
        """Get current market volatility"""
        try:
            # Get recent price data
            rates = self.mt5_connector.get_rates(symbol, 'M1', 50)
            if rates is None or len(rates) < 20:
                return 0.001  # Default volatility
            
            close_prices = rates['close'].values
            returns = np.diff(np.log(close_prices))
            volatility = np.std(returns)
            
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error getting current volatility: {str(e)}")
            return 0.001
    
    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol"""
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001
    
    def manage_scalp_positions(self):
        """Manage existing scalping positions"""
        try:
            current_time = datetime.now()
            positions_to_close = []
            
            # Get current positions
            all_positions = self.mt5_connector.get_positions()
            
            for position in all_positions:
                # Check if this is a scalping position
                if position.comment and 'scalping' in position.comment.lower():
                    # Check hold time
                    position_time = datetime.fromtimestamp(position.time)
                    hold_time_minutes = (current_time - position_time).total_seconds() / 60
                    
                    # Close if maximum hold time exceeded
                    if hold_time_minutes > self.max_hold_time_minutes:
                        positions_to_close.append((position, 'max_hold_time'))
                    
                    # Check for quick profit opportunities
                    elif position.profit > 0:
                        profit_pips = self._calculate_profit_pips(position)
                        if profit_pips >= self.min_profit_pips * 0.7:  # 70% of target
                            positions_to_close.append((position, 'partial_profit'))
            
            # Close positions
            for position, reason in positions_to_close:
                self._close_scalp_position(position, reason)
            
        except Exception as e:
            self.logger.error(f"Error managing scalp positions: {str(e)}")
    
    def _close_scalp_position(self, position, reason: str):
        """Close a scalping position"""
        try:
            result = self.order_manager.close_position(position.ticket)
            
            if result and result.retcode == 10009:
                # Update performance tracking
                profit_pips = self._calculate_profit_pips(position)
                self.performance_tracking['total_trades'] += 1
                self.performance_tracking['total_pips'] += profit_pips
                
                if profit_pips > 0:
                    self.performance_tracking['winning_trades'] += 1
                    self.performance_tracking['best_trade_pips'] = max(
                        self.performance_tracking['best_trade_pips'], 
                        profit_pips
                    )
                else:
                    self.performance_tracking['worst_trade_pips'] = min(
                        self.performance_tracking['worst_trade_pips'],
                        profit_pips
                    )
                
                # Calculate hold time
                position_time = datetime.fromtimestamp(position.time)
                hold_time_minutes = (datetime.now() - position_time).total_seconds() / 60
                
                # Update average hold time
                total_trades = self.performance_tracking['total_trades']
                current_avg = self.performance_tracking['avg_hold_time_minutes']
                new_avg = ((current_avg * (total_trades - 1)) + hold_time_minutes) / total_trades
                self.performance_tracking['avg_hold_time_minutes'] = new_avg
                
                self.logger.info(f"Scalping position closed: {position.symbol} "
                               f"Profit: {profit_pips:.1f} pips, Reason: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error closing scalp position: {str(e)}")
    
    def _calculate_profit_pips(self, position) -> float:
        """Calculate profit in pips for a position"""
        try:
            pip_size = self._get_pip_size(position.symbol)
            
            if position.type == 0:  # Buy position
                profit_points = position.price_current - position.price_open
            else:  # Sell position
                profit_points = position.price_open - position.price_current
            
            return profit_points / pip_size
            
        except Exception as e:
            self.logger.error(f"Error calculating profit pips: {str(e)}")
            return 0.0
    
    def get_scalping_performance(self) -> Dict[str, Any]:
        """Get scalping strategy performance metrics"""
        try:
            total_trades = self.performance_tracking['total_trades']
            winning_trades = self.performance_tracking['winning_trades']
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_profit_per_trade = (self.performance_tracking['total_pips'] / total_trades) if total_trades > 0 else 0
            
            return {
                'strategy': 'scalping',
                'enabled': self.is_enabled(),
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'total_pips': self.performance_tracking['total_pips'],
                'avg_profit_per_trade_pips': avg_profit_per_trade,
                'best_trade_pips': self.performance_tracking['best_trade_pips'],
                'worst_trade_pips': self.performance_tracking['worst_trade_pips'],
                'avg_hold_time_minutes': self.performance_tracking['avg_hold_time_minutes'],
                'signals_generated_today': len([s for s in self.signal_history 
                                              if s['timestamp'].date() == datetime.now().date()]),
                'last_signal_time': max(self.last_signal_time.values()) if self.last_signal_time else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting scalping performance: {str(e)}")
            return {'strategy': 'scalping', 'error': str(e)}
    
    def update_state(self):
        """Update strategy state"""
        try:
            # Manage existing positions
            self.manage_scalp_positions()
            
            # Clean old signal history
            cutoff_time = datetime.now() - timedelta(hours=24)
            while self.signal_history and self.signal_history[0]['timestamp'] < cutoff_time:
                self.signal_history.popleft()
            
        except Exception as e:
            self.logger.error(f"Error updating scalping state: {str(e)}")
