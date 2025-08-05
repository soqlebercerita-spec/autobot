"""
Position Sizing System for AuraTrade Bot
Calculates optimal position sizes based on risk and market conditions
"""

import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

from config.config import config
from config.settings import settings
from utils.logger import Logger

class PositionSizing:
    """Advanced position sizing algorithms"""
    
    def __init__(self):
        self.logger = Logger()
        
        # Position sizing parameters
        self.default_risk_per_trade = settings.risk.max_risk_per_trade
        self.max_position_size = settings.trading.default_lot_size * 10
        self.min_position_size = 0.01
        
        # Volatility tracking
        self.volatility_history = {}
        self.correlation_matrix = {}
        
        self.logger.info("Position Sizing system initialized")
    
    def calculate_lot_size(self, signal: Dict[str, Any]) -> float:
        """Calculate optimal lot size for signal"""
        try:
            method = settings.risk.position_sizing_method
            
            if method == "fixed_lot":
                return self._fixed_lot_sizing(signal)
            elif method == "fixed_percentage":
                return self._fixed_percentage_sizing(signal)
            elif method == "volatility_adjusted":
                return self._volatility_adjusted_sizing(signal)
            elif method == "kelly_criterion":
                return self._kelly_criterion_sizing(signal)
            elif method == "risk_parity":
                return self._risk_parity_sizing(signal)
            else:
                # Default to fixed percentage
                return self._fixed_percentage_sizing(signal)
                
        except Exception as e:
            self.logger.error(f"Error calculating lot size: {str(e)}")
            return settings.trading.default_lot_size
    
    def _fixed_lot_sizing(self, signal: Dict[str, Any]) -> float:
        """Fixed lot size method"""
        return settings.trading.default_lot_size
    
    def _fixed_percentage_sizing(self, signal: Dict[str, Any]) -> float:
        """Fixed percentage of account risk"""
        try:
            # Get account balance
            account_balance = self._get_account_balance()
            
            # Calculate risk amount
            risk_amount = account_balance * self.default_risk_per_trade
            
            # Get stop loss distance
            entry_price = signal.get('price', self._get_current_price(signal['symbol'], signal['action']))
            stop_loss = signal.get('stop_loss')
            
            if not stop_loss:
                stop_loss = self._estimate_stop_loss(signal['symbol'], signal['action'], entry_price)
            
            if not stop_loss:
                return settings.trading.default_lot_size
            
            # Calculate pip distance
            pip_distance = abs(entry_price - stop_loss) / self._get_pip_size(signal['symbol'])
            
            if pip_distance <= 0:
                return settings.trading.default_lot_size
            
            # Calculate pip value
            pip_value = self._get_pip_value(signal['symbol'], 1.0)  # For 1 lot
            
            # Calculate lot size
            lot_size = risk_amount / (pip_distance * pip_value)
            
            # Apply limits
            lot_size = max(self.min_position_size, min(lot_size, self.max_position_size))
            
            return round(lot_size, 2)
            
        except Exception as e:
            self.logger.error(f"Error in fixed percentage sizing: {str(e)}")
            return settings.trading.default_lot_size
    
    def _volatility_adjusted_sizing(self, signal: Dict[str, Any]) -> float:
        """Volatility-adjusted position sizing"""
        try:
            symbol = signal['symbol']
            
            # Get base lot size from fixed percentage method
            base_lot_size = self._fixed_percentage_sizing(signal)
            
            # Get current volatility
            current_volatility = self._get_current_volatility(symbol)
            average_volatility = self._get_average_volatility(symbol)
            
            if current_volatility <= 0 or average_volatility <= 0:
                return base_lot_size
            
            # Volatility adjustment factor
            volatility_ratio = average_volatility / current_volatility
            
            # Limit adjustment to prevent extreme position sizes
            volatility_ratio = max(0.5, min(volatility_ratio, 2.0))
            
            # Apply volatility adjustment
            adjusted_lot_size = base_lot_size * volatility_ratio
            
            # Apply limits
            adjusted_lot_size = max(self.min_position_size, min(adjusted_lot_size, self.max_position_size))
            
            return round(adjusted_lot_size, 2)
            
        except Exception as e:
            self.logger.error(f"Error in volatility adjusted sizing: {str(e)}")
            return self._fixed_percentage_sizing(signal)
    
    def _kelly_criterion_sizing(self, signal: Dict[str, Any]) -> float:
        """Kelly Criterion optimal position sizing"""
        try:
            # Get historical win rate and average win/loss
            win_rate, avg_win, avg_loss = self._get_strategy_statistics(signal.get('strategy', 'default'))
            
            if win_rate <= 0 or avg_win <= 0 or avg_loss <= 0:
                return self._fixed_percentage_sizing(signal)
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Conservative Kelly (use fraction of optimal)
            conservative_kelly = kelly_fraction * 0.25  # Use 25% of Kelly
            
            # Limit Kelly fraction
            conservative_kelly = max(0.01, min(conservative_kelly, 0.1))  # 1% to 10%
            
            # Apply Kelly fraction to account
            account_balance = self._get_account_balance()
            risk_amount = account_balance * conservative_kelly
            
            # Convert to lot size
            entry_price = signal.get('price', self._get_current_price(signal['symbol'], signal['action']))
            stop_loss = signal.get('stop_loss')
            
            if not stop_loss:
                stop_loss = self._estimate_stop_loss(signal['symbol'], signal['action'], entry_price)
            
            if not stop_loss:
                return settings.trading.default_lot_size
            
            pip_distance = abs(entry_price - stop_loss) / self._get_pip_size(signal['symbol'])
            pip_value = self._get_pip_value(signal['symbol'], 1.0)
            
            lot_size = risk_amount / (pip_distance * pip_value)
            lot_size = max(self.min_position_size, min(lot_size, self.max_position_size))
            
            return round(lot_size, 2)
            
        except Exception as e:
            self.logger.error(f"Error in Kelly criterion sizing: {str(e)}")
            return self._fixed_percentage_sizing(signal)
    
    def _risk_parity_sizing(self, signal: Dict[str, Any]) -> float:
        """Risk parity position sizing"""
        try:
            symbol = signal['symbol']
            
            # Get current portfolio positions
            current_positions = self._get_current_positions()
            
            if not current_positions:
                return self._fixed_percentage_sizing(signal)
            
            # Calculate target risk allocation per position
            total_positions = len(current_positions) + 1  # Including new position
            target_risk_per_position = self.default_risk_per_trade / total_positions
            
            # Calculate lot size for target risk
            account_balance = self._get_account_balance()
            risk_amount = account_balance * target_risk_per_position
            
            entry_price = signal.get('price', self._get_current_price(symbol, signal['action']))
            stop_loss = signal.get('stop_loss')
            
            if not stop_loss:
                stop_loss = self._estimate_stop_loss(symbol, signal['action'], entry_price)
            
            if not stop_loss:
                return settings.trading.default_lot_size
            
            pip_distance = abs(entry_price - stop_loss) / self._get_pip_size(symbol)
            pip_value = self._get_pip_value(symbol, 1.0)
            
            lot_size = risk_amount / (pip_distance * pip_value)
            lot_size = max(self.min_position_size, min(lot_size, self.max_position_size))
            
            return round(lot_size, 2)
            
        except Exception as e:
            self.logger.error(f"Error in risk parity sizing: {str(e)}")
            return self._fixed_percentage_sizing(signal)
    
    def calculate_dynamic_sizing(self, signal: Dict[str, Any], market_conditions: Dict[str, Any]) -> float:
        """Calculate position size based on market conditions"""
        try:
            base_lot_size = self.calculate_lot_size(signal)
            
            # Market condition adjustments
            volatility_regime = market_conditions.get('volatility_regime', 'normal')
            trend_strength = market_conditions.get('trend_strength', 0.5)
            market_hours = market_conditions.get('market_hours', 'london')
            
            adjustment_factor = 1.0
            
            # Volatility regime adjustment
            if volatility_regime == 'high':
                adjustment_factor *= 0.7  # Reduce size in high volatility
            elif volatility_regime == 'low':
                adjustment_factor *= 1.2  # Increase size in low volatility
            
            # Trend strength adjustment
            if trend_strength > 0.8:
                adjustment_factor *= 1.1  # Increase size in strong trends
            elif trend_strength < 0.3:
                adjustment_factor *= 0.9  # Reduce size in weak trends
            
            # Market hours adjustment
            if market_hours in ['london', 'new_york']:
                adjustment_factor *= 1.0  # Full size during major sessions
            elif market_hours == 'asian':
                adjustment_factor *= 0.8  # Reduced size during Asian session
            else:
                adjustment_factor *= 0.6  # Further reduced during off hours
            
            # Apply adjustment
            adjusted_lot_size = base_lot_size * adjustment_factor
            adjusted_lot_size = max(self.min_position_size, min(adjusted_lot_size, self.max_position_size))
            
            return round(adjusted_lot_size, 2)
            
        except Exception as e:
            self.logger.error(f"Error in dynamic sizing: {str(e)}")
            return self.calculate_lot_size(signal)
    
    def _get_current_volatility(self, symbol: str) -> float:
        """Get current volatility for symbol"""
        try:
            # This would calculate actual volatility from recent price data
            # Simplified implementation
            return 0.001  # Default volatility
        except Exception as e:
            self.logger.error(f"Error getting current volatility: {str(e)}")
            return 0.001
    
    def _get_average_volatility(self, symbol: str, period_days: int = 30) -> float:
        """Get average volatility over period"""
        try:
            # This would calculate average volatility from historical data
            # Simplified implementation
            return 0.0012  # Default average volatility
        except Exception as e:
            self.logger.error(f"Error getting average volatility: {str(e)}")
            return 0.0012
    
    def _get_strategy_statistics(self, strategy: str) -> tuple:
        """Get strategy performance statistics"""
        try:
            # This would get actual strategy statistics
            # Simplified implementation
            win_rate = 0.6  # 60% win rate
            avg_win = 0.02  # 2% average win
            avg_loss = 0.01  # 1% average loss
            
            return win_rate, avg_win, avg_loss
            
        except Exception as e:
            self.logger.error(f"Error getting strategy statistics: {str(e)}")
            return 0.5, 0.01, 0.01
    
    def _get_current_positions(self) -> list:
        """Get current open positions"""
        try:
            # This would get actual current positions
            # Simplified implementation
            return []
        except Exception as e:
            self.logger.error(f"Error getting current positions: {str(e)}")
            return []
    
    def _get_account_balance(self) -> float:
        """Get account balance"""
        try:
            # This would get actual account balance
            return 10000.0  # Default $10,000
        except Exception as e:
            self.logger.error(f"Error getting account balance: {str(e)}")
            return 10000.0
    
    def _get_current_price(self, symbol: str, action: str) -> float:
        """Get current market price"""
        try:
            # This would get actual current price
            if action.upper() == 'BUY':
                return 1.1000  # Default ask
            else:
                return 1.0999  # Default bid
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            return 1.1000
    
    def _estimate_stop_loss(self, symbol: str, action: str, entry_price: float) -> float:
        """Estimate stop loss level"""
        try:
            pip_size = self._get_pip_size(symbol)
            stop_pips = config.STOP_LOSS_PIPS
            
            if action.upper() == 'BUY':
                return entry_price - (stop_pips * pip_size)
            else:
                return entry_price + (stop_pips * pip_size)
                
        except Exception as e:
            self.logger.error(f"Error estimating stop loss: {str(e)}")
            return entry_price * 0.99 if action.upper() == 'BUY' else entry_price * 1.01
    
    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol"""
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001
    
    def _get_pip_value(self, symbol: str, lot_size: float) -> float:
        """Calculate pip value"""
        if 'JPY' in symbol:
            return lot_size * 100 * 0.01
        else:
            return lot_size * 100000 * 0.0001
    
    def update_volatility_history(self, symbol: str, volatility: float):
        """Update volatility history for symbol"""
        try:
            if symbol not in self.volatility_history:
                self.volatility_history[symbol] = []
            
            self.volatility_history[symbol].append({
                'timestamp': datetime.now(),
                'volatility': volatility
            })
            
            # Keep only last 100 readings
            if len(self.volatility_history[symbol]) > 100:
                self.volatility_history[symbol] = self.volatility_history[symbol][-100:]
                
        except Exception as e:
            self.logger.error(f"Error updating volatility history: {str(e)}")
    
    def get_position_sizing_report(self) -> Dict[str, Any]:
        """Generate position sizing report"""
        try:
            report = {
                'sizing_method': settings.risk.position_sizing_method,
                'default_risk_per_trade': self.default_risk_per_trade,
                'min_position_size': self.min_position_size,
                'max_position_size': self.max_position_size,
                'volatility_data': {
                    symbol: len(history) for symbol, history in self.volatility_history.items()
                },
                'recent_sizes': []  # Would include recent position sizes
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating position sizing report: {str(e)}")
            return {}
    
    def optimize_portfolio_allocation(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize position sizes across multiple signals"""
        try:
            if not signals:
                return signals
            
            # Calculate individual risks
            total_risk = 0
            for signal in signals:
                risk = self._calculate_signal_risk(signal)
                signal['individual_risk'] = risk
                total_risk += risk
            
            # If total risk exceeds limit, scale down proportionally
            max_total_risk = self.default_risk_per_trade * len(signals)
            
            if total_risk > max_total_risk:
                scale_factor = max_total_risk / total_risk
                
                for signal in signals:
                    original_lot = signal.get('lot_size', settings.trading.default_lot_size)
                    scaled_lot = original_lot * scale_factor
                    signal['lot_size'] = max(self.min_position_size, round(scaled_lot, 2))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio allocation: {str(e)}")
            return signals
    
    def _calculate_signal_risk(self, signal: Dict[str, Any]) -> float:
        """Calculate risk for individual signal"""
        try:
            lot_size = signal.get('lot_size', settings.trading.default_lot_size)
            entry_price = signal.get('price', self._get_current_price(signal['symbol'], signal['action']))
            stop_loss = signal.get('stop_loss')
            
            if not stop_loss:
                stop_loss = self._estimate_stop_loss(signal['symbol'], signal['action'], entry_price)
            
            pip_distance = abs(entry_price - stop_loss) / self._get_pip_size(signal['symbol'])
            pip_value = self._get_pip_value(signal['symbol'], lot_size)
            
            risk_amount = pip_distance * pip_value
            account_balance = self._get_account_balance()
            
            return risk_amount / account_balance if account_balance > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating signal risk: {str(e)}")
            return 0.01  # Default 1% risk
