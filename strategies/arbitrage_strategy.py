"""
Arbitrage Strategy for AuraTrade Bot
Statistical and Cross-Market Arbitrage Implementation
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import statistics

from config.config import config
from config.settings import settings
from core.mt5_connector import MT5Connector
from core.order_manager import OrderManager
from core.risk_manager import RiskManager
from analysis.technical_analysis import TechnicalAnalysis
from utils.logger import Logger

class ArbitrageStrategy:
    """Arbitrage Strategy Implementation"""
    
    def __init__(self, mt5_connector: MT5Connector, order_manager: OrderManager, risk_manager: RiskManager):
        self.mt5_connector = mt5_connector
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.logger = Logger()
        self.technical_analysis = TechnicalAnalysis()
        
        # Strategy parameters
        self.enabled = config.STRATEGY_ENABLED.get('arbitrage', False)
        self.min_profit_threshold = 0.0002  # Minimum profit threshold (20 pips for 4-digit)
        self.max_execution_time = 5.0  # Maximum execution time in seconds
        self.correlation_threshold = 0.8  # Minimum correlation for pair arbitrage
        self.mean_reversion_period = 50  # Period for mean reversion calculation
        
        # Statistical arbitrage parameters
        self.z_score_entry = 2.0  # Z-score threshold for entry
        self.z_score_exit = 0.5   # Z-score threshold for exit
        self.lookback_period = 100  # Historical data lookback
        self.cointegration_threshold = 0.05  # P-value threshold for cointegration
        
        # Cross-pair arbitrage
        self.currency_triangles = [
            ['EURUSD', 'GBPUSD', 'EURGBP'],
            ['USDJPY', 'EURJPY', 'EURUSD'],
            ['AUDUSD', 'NZDUSD', 'AUDNZD'],
            ['USDCAD', 'USDCHF', 'CADCHF']
        ]
        
        # Data storage
        self.price_history = {}  # symbol -> deque of prices
        self.spread_history = {}  # pair -> deque of spreads
        self.correlation_matrix = {}
        self.cointegration_results = {}
        
        # Position tracking
        self.arbitrage_positions = {}  # pair -> position info
        self.pending_arbitrage = {}   # pending arbitrage opportunities
        
        # Performance tracking
        self.performance_metrics = {
            'total_arbitrage_trades': 0,
            'successful_arbitrages': 0,
            'total_arbitrage_profit': 0.0,
            'avg_arbitrage_profit': 0.0,
            'max_arbitrage_profit': 0.0,
            'arbitrage_opportunities_detected': 0,
            'arbitrage_opportunities_executed': 0
        }
        
        self.logger.info("Arbitrage Strategy initialized")
    
    def is_enabled(self) -> bool:
        """Check if arbitrage strategy is enabled"""
        return self.enabled and settings.hft.arbitrage_enabled
    
    def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate arbitrage trading signals"""
        if not self.is_enabled():
            return []
        
        signals = []
        
        try:
            # Update price data
            self._update_price_data()
            
            # Check statistical arbitrage opportunities
            signals.extend(self._check_statistical_arbitrage())
            
            # Check triangular arbitrage opportunities
            signals.extend(self._check_triangular_arbitrage())
            
            # Check currency cross arbitrage
            signals.extend(self._check_cross_currency_arbitrage())
            
            # Check mean reversion arbitrage
            signals.extend(self._check_mean_reversion_arbitrage())
            
        except Exception as e:
            self.logger.error(f"Error generating arbitrage signals: {str(e)}")
        
        return signals
    
    def _update_price_data(self):
        """Update price data for all symbols"""
        try:
            for symbol in settings.trading.enabled_symbols:
                # Get current tick
                tick = self.mt5_connector.get_tick(symbol)
                if not tick:
                    continue
                
                mid_price = (tick.bid + tick.ask) / 2.0
                timestamp = datetime.fromtimestamp(tick.time)
                
                # Initialize price history if needed
                if symbol not in self.price_history:
                    self.price_history[symbol] = deque(maxlen=self.lookback_period)
                
                # Add price data
                self.price_history[symbol].append({
                    'price': mid_price,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'timestamp': timestamp
                })
                
        except Exception as e:
            self.logger.error(f"Error updating price data: {str(e)}")
    
    def _check_statistical_arbitrage(self) -> List[Dict[str, Any]]:
        """Check for statistical arbitrage opportunities between correlated pairs"""
        signals = []
        
        try:
            symbols = list(self.price_history.keys())
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    
                    # Skip if insufficient data
                    if (len(self.price_history[symbol1]) < self.mean_reversion_period or
                        len(self.price_history[symbol2]) < self.mean_reversion_period):
                        continue
                    
                    # Calculate correlation
                    correlation = self._calculate_correlation(symbol1, symbol2)
                    if abs(correlation) < self.correlation_threshold:
                        continue
                    
                    # Check for cointegration
                    if not self._check_cointegration(symbol1, symbol2):
                        continue
                    
                    # Calculate spread and z-score
                    spread, z_score = self._calculate_spread_zscore(symbol1, symbol2)
                    
                    # Check for arbitrage opportunity
                    if abs(z_score) > self.z_score_entry:
                        pair_key = f"{symbol1}_{symbol2}"
                        
                        # Determine direction
                        if z_score > 0:  # symbol1 overvalued relative to symbol2
                            # Sell symbol1, Buy symbol2
                            signal1 = self._create_arbitrage_signal(symbol1, 'SELL', 'stat_arb', z_score, pair_key)
                            signal2 = self._create_arbitrage_signal(symbol2, 'BUY', 'stat_arb', z_score, pair_key)
                        else:  # symbol1 undervalued relative to symbol2
                            # Buy symbol1, Sell symbol2
                            signal1 = self._create_arbitrage_signal(symbol1, 'BUY', 'stat_arb', z_score, pair_key)
                            signal2 = self._create_arbitrage_signal(symbol2, 'SELL', 'stat_arb', z_score, pair_key)
                        
                        if signal1 and signal2:
                            signals.extend([signal1, signal2])
                            self.performance_metrics['arbitrage_opportunities_detected'] += 1
                            
                            self.logger.info(f"Statistical arbitrage opportunity: {symbol1}-{symbol2}, "
                                           f"Z-score: {z_score:.2f}, Correlation: {correlation:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error checking statistical arbitrage: {str(e)}")
        
        return signals
    
    def _check_triangular_arbitrage(self) -> List[Dict[str, Any]]:
        """Check for triangular arbitrage opportunities"""
        signals = []
        
        try:
            for triangle in self.currency_triangles:
                if len(triangle) != 3:
                    continue
                
                symbol1, symbol2, symbol3 = triangle
                
                # Check if all symbols have data
                if not all(symbol in self.price_history for symbol in triangle):
                    continue
                
                if not all(len(self.price_history[symbol]) > 0 for symbol in triangle):
                    continue
                
                # Get current prices
                prices = {}
                for symbol in triangle:
                    latest_data = self.price_history[symbol][-1]
                    prices[symbol] = {
                        'bid': latest_data['bid'],
                        'ask': latest_data['ask'],
                        'mid': latest_data['price']
                    }
                
                # Calculate triangular arbitrage opportunity
                arbitrage_profit = self._calculate_triangular_arbitrage(prices, triangle)
                
                if arbitrage_profit > self.min_profit_threshold:
                    # Create triangular arbitrage signals
                    triangle_signals = self._create_triangular_signals(triangle, prices, arbitrage_profit)
                    signals.extend(triangle_signals)
                    
                    self.performance_metrics['arbitrage_opportunities_detected'] += 1
                    
                    self.logger.info(f"Triangular arbitrage opportunity: {triangle}, "
                                   f"Profit: {arbitrage_profit:.6f}")
            
        except Exception as e:
            self.logger.error(f"Error checking triangular arbitrage: {str(e)}")
        
        return signals
    
    def _check_cross_currency_arbitrage(self) -> List[Dict[str, Any]]:
        """Check for cross-currency arbitrage opportunities"""
        signals = []
        
        try:
            # Look for price discrepancies in cross currency pairs
            usd_pairs = [s for s in self.price_history.keys() if 'USD' in s]
            
            for base_pair in usd_pairs:
                if len(self.price_history[base_pair]) == 0:
                    continue
                
                base_price = self.price_history[base_pair][-1]['price']
                
                # Find related cross pairs
                base_currency = base_pair.replace('USD', '')
                
                for quote_pair in usd_pairs:
                    if quote_pair == base_pair or len(self.price_history[quote_pair]) == 0:
                        continue
                    
                    quote_currency = quote_pair.replace('USD', '')
                    cross_symbol = f"{base_currency}{quote_currency}"
                    
                    if cross_symbol in self.price_history and len(self.price_history[cross_symbol]) > 0:
                        quote_price = self.price_history[quote_pair][-1]['price']
                        cross_price = self.price_history[cross_symbol][-1]['price']
                        
                        # Calculate implied cross rate
                        if 'USD' == quote_pair[:3]:  # USD is base currency
                            implied_cross = base_price / quote_price
                        else:  # USD is quote currency
                            implied_cross = base_price * quote_price
                        
                        # Check for arbitrage opportunity
                        price_discrepancy = abs(cross_price - implied_cross) / implied_cross
                        
                        if price_discrepancy > self.min_profit_threshold:
                            # Create cross currency arbitrage signals
                            cross_signals = self._create_cross_currency_signals(
                                base_pair, quote_pair, cross_symbol, 
                                implied_cross, cross_price, price_discrepancy
                            )
                            signals.extend(cross_signals)
                            
                            self.logger.info(f"Cross currency arbitrage: {cross_symbol}, "
                                           f"Discrepancy: {price_discrepancy:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error checking cross currency arbitrage: {str(e)}")
        
        return signals
    
    def _check_mean_reversion_arbitrage(self) -> List[Dict[str, Any]]:
        """Check for mean reversion arbitrage opportunities"""
        signals = []
        
        try:
            for symbol in self.price_history:
                if len(self.price_history[symbol]) < self.mean_reversion_period:
                    continue
                
                prices = [data['price'] for data in list(self.price_history[symbol])[-self.mean_reversion_period:]]
                current_price = prices[-1]
                
                # Calculate mean and standard deviation
                mean_price = statistics.mean(prices)
                std_price = statistics.stdev(prices)
                
                if std_price == 0:
                    continue
                
                # Calculate z-score
                z_score = (current_price - mean_price) / std_price
                
                # Check for mean reversion opportunity
                if abs(z_score) > self.z_score_entry:
                    # Create mean reversion signal
                    action = 'SELL' if z_score > 0 else 'BUY'  # Bet on reversion
                    
                    signal = self._create_arbitrage_signal(
                        symbol, action, 'mean_reversion', z_score, f"mean_rev_{symbol}"
                    )
                    
                    if signal:
                        signals.append(signal)
                        
                        self.logger.info(f"Mean reversion opportunity: {symbol}, "
                                       f"Z-score: {z_score:.2f}, Action: {action}")
            
        except Exception as e:
            self.logger.error(f"Error checking mean reversion arbitrage: {str(e)}")
        
        return signals
    
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        try:
            prices1 = [data['price'] for data in list(self.price_history[symbol1])[-self.mean_reversion_period:]]
            prices2 = [data['price'] for data in list(self.price_history[symbol2])[-self.mean_reversion_period:]]
            
            if len(prices1) != len(prices2) or len(prices1) < 10:
                return 0.0
            
            return np.corrcoef(prices1, prices2)[0, 1]
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0
    
    def _check_cointegration(self, symbol1: str, symbol2: str) -> bool:
        """Check if two price series are cointegrated (simplified)"""
        try:
            # Simplified cointegration test using correlation of price differences
            prices1 = [data['price'] for data in list(self.price_history[symbol1])[-self.mean_reversion_period:]]
            prices2 = [data['price'] for data in list(self.price_history[symbol2])[-self.mean_reversion_period:]]
            
            if len(prices1) != len(prices2) or len(prices1) < 20:
                return False
            
            # Calculate price differences (returns)
            returns1 = np.diff(np.log(prices1))
            returns2 = np.diff(np.log(prices2))
            
            # Simple cointegration proxy: correlation of returns should be high
            returns_correlation = np.corrcoef(returns1, returns2)[0, 1]
            
            return abs(returns_correlation) > 0.3  # Simplified threshold
            
        except Exception as e:
            self.logger.error(f"Error checking cointegration: {str(e)}")
            return False
    
    def _calculate_spread_zscore(self, symbol1: str, symbol2: str) -> Tuple[float, float]:
        """Calculate spread and z-score between two symbols"""
        try:
            prices1 = [data['price'] for data in list(self.price_history[symbol1])[-self.mean_reversion_period:]]
            prices2 = [data['price'] for data in list(self.price_history[symbol2])[-self.mean_reversion_period:]]
            
            # Calculate spread (price ratio)
            spreads = [p1 / p2 for p1, p2 in zip(prices1, prices2)]
            
            current_spread = spreads[-1]
            mean_spread = statistics.mean(spreads)
            std_spread = statistics.stdev(spreads)
            
            if std_spread == 0:
                return current_spread, 0.0
            
            z_score = (current_spread - mean_spread) / std_spread
            
            return current_spread, z_score
            
        except Exception as e:
            self.logger.error(f"Error calculating spread z-score: {str(e)}")
            return 0.0, 0.0
    
    def _calculate_triangular_arbitrage(self, prices: Dict, triangle: List[str]) -> float:
        """Calculate triangular arbitrage profit potential"""
        try:
            symbol1, symbol2, symbol3 = triangle
            
            # Calculate cross rates
            # Example: EUR/USD * GBP/USD * EUR/GBP should equal 1
            
            # Path 1: Buy symbol1, Sell symbol2, Buy symbol3
            path1_rate = (prices[symbol1]['ask'] * 
                         (1 / prices[symbol2]['bid']) * 
                         prices[symbol3]['bid'])
            
            # Path 2: Sell symbol1, Buy symbol2, Sell symbol3
            path2_rate = ((1 / prices[symbol1]['bid']) * 
                         prices[symbol2]['ask'] * 
                         (1 / prices[symbol3]['ask']))
            
            # Calculate arbitrage opportunity
            arbitrage1 = abs(path1_rate - 1.0)
            arbitrage2 = abs(path2_rate - 1.0)
            
            return max(arbitrage1, arbitrage2)
            
        except Exception as e:
            self.logger.error(f"Error calculating triangular arbitrage: {str(e)}")
            return 0.0
    
    def _create_arbitrage_signal(self, symbol: str, action: str, strategy_type: str, 
                               z_score: float, pair_key: str) -> Optional[Dict[str, Any]]:
        """Create an arbitrage signal"""
        try:
            # Get current tick
            tick = self.mt5_connector.get_tick(symbol)
            if not tick:
                return None
            
            # Calculate entry price
            entry_price = tick.ask if action == 'BUY' else tick.bid
            
            # Calculate lot size (smaller for arbitrage)
            lot_size = settings.trading.default_lot_size * 0.5
            
            # Calculate stop loss and take profit based on strategy
            stop_loss, take_profit = self._calculate_arbitrage_levels(
                symbol, action, entry_price, strategy_type, abs(z_score)
            )
            
            signal = {
                'strategy': 'arbitrage',
                'sub_strategy': strategy_type,
                'symbol': symbol,
                'action': action,
                'lot_size': lot_size,
                'price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'z_score': z_score,
                'pair_key': pair_key,
                'urgency': 'immediate',  # Arbitrage requires fast execution
                'max_execution_time': self.max_execution_time,
                'reason': f'{strategy_type} arbitrage: Z-score {z_score:.2f}',
                'timestamp': datetime.now()
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating arbitrage signal: {str(e)}")
            return None
    
    def _create_triangular_signals(self, triangle: List[str], prices: Dict, 
                                 arbitrage_profit: float) -> List[Dict[str, Any]]:
        """Create signals for triangular arbitrage"""
        signals = []
        
        try:
            symbol1, symbol2, symbol3 = triangle
            
            # Create coordinated signals for triangular arbitrage
            # This is a simplified implementation
            
            # Signal 1: Buy symbol1
            signal1 = {
                'strategy': 'arbitrage',
                'sub_strategy': 'triangular',
                'symbol': symbol1,
                'action': 'BUY',
                'lot_size': settings.trading.default_lot_size * 0.3,
                'price': prices[symbol1]['ask'],
                'arbitrage_group': f"triangle_{symbol1}_{symbol2}_{symbol3}",
                'expected_profit': arbitrage_profit,
                'urgency': 'immediate',
                'reason': f'Triangular arbitrage profit: {arbitrage_profit:.6f}'
            }
            
            # Signal 2: Sell symbol2
            signal2 = {
                'strategy': 'arbitrage',
                'sub_strategy': 'triangular',
                'symbol': symbol2,
                'action': 'SELL',
                'lot_size': settings.trading.default_lot_size * 0.3,
                'price': prices[symbol2]['bid'],
                'arbitrage_group': f"triangle_{symbol1}_{symbol2}_{symbol3}",
                'expected_profit': arbitrage_profit,
                'urgency': 'immediate',
                'reason': f'Triangular arbitrage profit: {arbitrage_profit:.6f}'
            }
            
            signals.extend([signal1, signal2])
            
        except Exception as e:
            self.logger.error(f"Error creating triangular signals: {str(e)}")
        
        return signals
    
    def _create_cross_currency_signals(self, base_pair: str, quote_pair: str, cross_symbol: str,
                                     implied_cross: float, actual_cross: float, 
                                     discrepancy: float) -> List[Dict[str, Any]]:
        """Create signals for cross currency arbitrage"""
        signals = []
        
        try:
            # Determine direction based on price discrepancy
            if actual_cross > implied_cross:
                # Cross pair is overvalued
                # Sell cross pair, buy components
                cross_action = 'SELL'
                base_action = 'BUY'
                quote_action = 'SELL'
            else:
                # Cross pair is undervalued
                # Buy cross pair, sell components
                cross_action = 'BUY'
                base_action = 'SELL'
                quote_action = 'BUY'
            
            lot_size = settings.trading.default_lot_size * 0.4
            
            # Cross pair signal
            signal1 = {
                'strategy': 'arbitrage',
                'sub_strategy': 'cross_currency',
                'symbol': cross_symbol,
                'action': cross_action,
                'lot_size': lot_size,
                'discrepancy': discrepancy,
                'arbitrage_group': f"cross_{base_pair}_{quote_pair}_{cross_symbol}",
                'urgency': 'immediate',
                'reason': f'Cross currency arbitrage: {discrepancy:.4f} discrepancy'
            }
            
            signals.append(signal1)
            
        except Exception as e:
            self.logger.error(f"Error creating cross currency signals: {str(e)}")
        
        return signals
    
    def _calculate_arbitrage_levels(self, symbol: str, action: str, entry_price: float,
                                  strategy_type: str, confidence: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit for arbitrage trades"""
        try:
            pip_size = self._get_pip_size(symbol)
            
            # Tighter levels for arbitrage
            if strategy_type == 'statistical':
                stop_pips = 5.0 / confidence  # Inverse relationship with confidence
                profit_pips = 3.0 * confidence
            elif strategy_type == 'triangular':
                stop_pips = 3.0
                profit_pips = 2.0
            elif strategy_type == 'mean_reversion':
                stop_pips = 4.0 / confidence
                profit_pips = 2.5 * confidence
            else:
                stop_pips = 5.0
                profit_pips = 3.0
            
            stop_distance = pip_size * stop_pips
            profit_distance = pip_size * profit_pips
            
            if action == 'BUY':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + profit_distance
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - profit_distance
            
            return stop_loss, take_profit
            
        except Exception as e:
            self.logger.error(f"Error calculating arbitrage levels: {str(e)}")
            return None, None
    
    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol"""
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001
    
    def manage_arbitrage_positions(self):
        """Manage existing arbitrage positions"""
        try:
            current_time = datetime.now()
            
            # Check for exit conditions
            for pair_key, position_info in list(self.arbitrage_positions.items()):
                # Check if positions should be closed based on z-score
                if position_info['sub_strategy'] == 'stat_arb':
                    self._check_statistical_arbitrage_exit(pair_key, position_info)
                elif position_info['sub_strategy'] == 'mean_reversion':
                    self._check_mean_reversion_exit(pair_key, position_info)
                
                # Check maximum hold time
                entry_time = position_info.get('entry_time', current_time)
                hold_time = (current_time - entry_time).total_seconds()
                
                if hold_time > 300:  # 5 minutes maximum for arbitrage
                    self._close_arbitrage_position(pair_key, 'max_hold_time')
            
        except Exception as e:
            self.logger.error(f"Error managing arbitrage positions: {str(e)}")
    
    def _check_statistical_arbitrage_exit(self, pair_key: str, position_info: Dict):
        """Check exit conditions for statistical arbitrage"""
        try:
            symbols = pair_key.split('_')
            if len(symbols) != 2:
                return
            
            symbol1, symbol2 = symbols
            
            # Calculate current z-score
            _, current_z_score = self._calculate_spread_zscore(symbol1, symbol2)
            
            # Check for exit condition
            if abs(current_z_score) < self.z_score_exit:
                self._close_arbitrage_position(pair_key, 'z_score_exit')
            
        except Exception as e:
            self.logger.error(f"Error checking statistical arbitrage exit: {str(e)}")
    
    def _check_mean_reversion_exit(self, pair_key: str, position_info: Dict):
        """Check exit conditions for mean reversion arbitrage"""
        try:
            symbol = pair_key.replace('mean_rev_', '')
            
            if symbol not in self.price_history or len(self.price_history[symbol]) < self.mean_reversion_period:
                return
            
            prices = [data['price'] for data in list(self.price_history[symbol])[-self.mean_reversion_period:]]
            current_price = prices[-1]
            
            mean_price = statistics.mean(prices)
            std_price = statistics.stdev(prices)
            
            if std_price == 0:
                return
            
            z_score = (current_price - mean_price) / std_price
            
            # Exit when z-score reverses sufficiently
            if abs(z_score) < self.z_score_exit:
                self._close_arbitrage_position(pair_key, 'mean_reversion_exit')
            
        except Exception as e:
            self.logger.error(f"Error checking mean reversion exit: {str(e)}")
    
    def _close_arbitrage_position(self, pair_key: str, reason: str):
        """Close arbitrage position"""
        try:
            if pair_key not in self.arbitrage_positions:
                return
            
            position_info = self.arbitrage_positions[pair_key]
            tickets = position_info.get('tickets', [])
            
            total_profit = 0.0
            closed_count = 0
            
            for ticket in tickets:
                result = self.order_manager.close_position(ticket)
                if result and result.retcode == 10009:
                    closed_count += 1
                    # Calculate profit (simplified)
                    # In real implementation, get actual profit from position
                    total_profit += 10.0  # Placeholder
            
            # Update performance metrics
            if closed_count > 0:
                self.performance_metrics['total_arbitrage_trades'] += 1
                self.performance_metrics['total_arbitrage_profit'] += total_profit
                
                if total_profit > 0:
                    self.performance_metrics['successful_arbitrages'] += 1
                    self.performance_metrics['max_arbitrage_profit'] = max(
                        self.performance_metrics['max_arbitrage_profit'], total_profit
                    )
                
                # Update average
                total_trades = self.performance_metrics['total_arbitrage_trades']
                self.performance_metrics['avg_arbitrage_profit'] = (
                    self.performance_metrics['total_arbitrage_profit'] / total_trades
                )
            
            # Remove from tracking
            del self.arbitrage_positions[pair_key]
            
            self.logger.info(f"Arbitrage position closed: {pair_key}, Reason: {reason}, "
                           f"Profit: {total_profit:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error closing arbitrage position: {str(e)}")
    
    def get_arbitrage_performance(self) -> Dict[str, Any]:
        """Get arbitrage strategy performance metrics"""
        try:
            total_trades = self.performance_metrics['total_arbitrage_trades']
            successful_trades = self.performance_metrics['successful_arbitrages']
            
            success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            opportunities_detected = self.performance_metrics['arbitrage_opportunities_detected']
            opportunities_executed = self.performance_metrics['arbitrage_opportunities_executed']
            
            execution_rate = (opportunities_executed / opportunities_detected * 100) if opportunities_detected > 0 else 0
            
            return {
                'strategy': 'arbitrage',
                'enabled': self.is_enabled(),
                'total_arbitrage_trades': total_trades,
                'successful_arbitrages': successful_trades,
                'success_rate': success_rate,
                'total_arbitrage_profit': self.performance_metrics['total_arbitrage_profit'],
                'avg_arbitrage_profit': self.performance_metrics['avg_arbitrage_profit'],
                'max_arbitrage_profit': self.performance_metrics['max_arbitrage_profit'],
                'opportunities_detected': opportunities_detected,
                'opportunities_executed': opportunities_executed,
                'execution_rate': execution_rate,
                'active_arbitrage_positions': len(self.arbitrage_positions),
                'tracked_symbols': len(self.price_history),
                'currency_triangles': len(self.currency_triangles)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting arbitrage performance: {str(e)}")
            return {'strategy': 'arbitrage', 'error': str(e)}
    
    def update_state(self):
        """Update strategy state"""
        try:
            # Manage existing positions
            self.manage_arbitrage_positions()
            
            # Clean old price data
            cutoff_time = datetime.now() - timedelta(hours=2)
            for symbol in self.price_history:
                while (self.price_history[symbol] and 
                       self.price_history[symbol][0]['timestamp'] < cutoff_time):
                    self.price_history[symbol].popleft()
            
        except Exception as e:
            self.logger.error(f"Error updating arbitrage state: {str(e)}")
