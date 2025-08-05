"""
Risk Management System for AuraTrade Bot
Handles position sizing, risk limits, and portfolio protection
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from config.config import config
from config.settings import settings
from utils.logger import Logger

@dataclass
class RiskMetrics:
    """Risk metrics data class"""
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    current_exposure: float
    portfolio_beta: float

class RiskManager:
    """Comprehensive risk management system"""
    
    def __init__(self):
        self.logger = Logger()
        
        # Risk tracking
        self.daily_pnl = []
        self.drawdown_history = []
        self.position_history = []
        self.correlation_matrix = {}
        
        # Risk limits
        self.max_position_size = settings.risk.exposure_limit
        self.max_portfolio_risk = settings.risk.max_drawdown
        self.max_correlation = settings.risk.correlation_limit
        
        # Performance tracking
        self.risk_events = []
        self.breach_count = 0
        
        self.logger.info("Risk Manager initialized")
    
    def calculate_position_risk(self, signal: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics for a position"""
        try:
            symbol = signal['symbol']
            lot_size = signal['lot_size']
            action = signal['action']
            entry_price = signal.get('price', 0)
            stop_loss = signal.get('stop_loss')
            
            # Get symbol info for calculations
            pip_value = self._get_pip_value(symbol, lot_size)
            
            # Calculate potential loss
            if stop_loss and entry_price:
                if action.upper() == 'BUY':
                    risk_pips = abs(entry_price - stop_loss) / self._get_pip_size(symbol)
                else:
                    risk_pips = abs(stop_loss - entry_price) / self._get_pip_size(symbol)
                
                risk_amount = risk_pips * pip_value
            else:
                # Default risk calculation based on ATR
                risk_amount = self._estimate_default_risk(symbol, lot_size)
            
            # Calculate risk as percentage of account
            account_balance = self._get_account_balance()
            risk_percentage = risk_amount / account_balance if account_balance > 0 else 0
            
            return {
                'risk_amount': risk_amount,
                'risk_percentage': risk_percentage,
                'risk_pips': risk_pips if 'risk_pips' in locals() else 0,
                'pip_value': pip_value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk: {str(e)}")
            return {'risk_amount': 0, 'risk_percentage': 0, 'risk_pips': 0, 'pip_value': 0}
    
    def validate_risk_limits(self, signal: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate if position meets risk limits"""
        try:
            # Calculate position risk
            risk_metrics = self.calculate_position_risk(signal)
            
            # Check individual position risk
            if risk_metrics['risk_percentage'] > settings.risk.max_risk_per_trade:
                return False, f"Position risk {risk_metrics['risk_percentage']:.2%} exceeds limit {settings.risk.max_risk_per_trade:.2%}"
            
            # Check portfolio exposure
            current_exposure = self._calculate_current_exposure()
            new_exposure = current_exposure + risk_metrics['risk_percentage']
            
            if new_exposure > self.max_position_size:
                return False, f"Portfolio exposure {new_exposure:.2%} would exceed limit {self.max_position_size:.2%}"
            
            # Check correlation limits
            if not self._check_correlation_limits(signal['symbol']):
                return False, f"Adding {signal['symbol']} would exceed correlation limits"
            
            # Check drawdown limits
            current_drawdown = self._calculate_current_drawdown()
            if current_drawdown > self.max_portfolio_risk:
                return False, f"Current drawdown {current_drawdown:.2%} exceeds limit {self.max_portfolio_risk:.2%}"
            
            return True, "Risk limits validated"
            
        except Exception as e:
            self.logger.error(f"Error validating risk limits: {str(e)}")
            return False, f"Risk validation error: {str(e)}"
    
    def add_risk_parameters(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Add stop loss and take profit to signal"""
        try:
            symbol = signal['symbol']
            action = signal['action']
            entry_price = signal.get('price')
            
            if not entry_price:
                # Get current market price
                entry_price = self._get_current_price(symbol, action)
                signal['price'] = entry_price
            
            # Calculate stop loss based on method
            stop_loss = self._calculate_stop_loss(symbol, action, entry_price)
            if stop_loss:
                signal['stop_loss'] = stop_loss
            
            # Calculate take profit based on risk/reward ratio
            take_profit = self._calculate_take_profit(symbol, action, entry_price, stop_loss)
            if take_profit:
                signal['take_profit'] = take_profit
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error adding risk parameters: {str(e)}")
            return signal
    
    def _calculate_stop_loss(self, symbol: str, action: str, entry_price: float) -> Optional[float]:
        """Calculate stop loss based on configured method"""
        try:
            method = settings.risk.stop_loss_method
            
            if method == "fixed_pips":
                pip_size = self._get_pip_size(symbol)
                stop_pips = config.STOP_LOSS_PIPS
                
                if action.upper() == 'BUY':
                    return entry_price - (stop_pips * pip_size)
                else:
                    return entry_price + (stop_pips * pip_size)
            
            elif method == "atr_based":
                atr = self._get_atr(symbol)
                if atr:
                    multiplier = 2.0  # 2x ATR
                    if action.upper() == 'BUY':
                        return entry_price - (atr * multiplier)
                    else:
                        return entry_price + (atr * multiplier)
            
            elif method == "percentage":
                percentage = 0.02  # 2% stop loss
                if action.upper() == 'BUY':
                    return entry_price * (1 - percentage)
                else:
                    return entry_price * (1 + percentage)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return None
    
    def _calculate_take_profit(self, symbol: str, action: str, entry_price: float, 
                              stop_loss: Optional[float]) -> Optional[float]:
        """Calculate take profit based on risk/reward ratio"""
        try:
            if not stop_loss:
                return None
            
            method = settings.risk.take_profit_method
            
            if method == "risk_reward":
                risk_reward_ratio = 2.0  # 1:2 risk/reward
                
                if action.upper() == 'BUY':
                    risk = entry_price - stop_loss
                    return entry_price + (risk * risk_reward_ratio)
                else:
                    risk = stop_loss - entry_price
                    return entry_price - (risk * risk_reward_ratio)
            
            elif method == "fixed_pips":
                pip_size = self._get_pip_size(symbol)
                tp_pips = config.TAKE_PROFIT_PIPS
                
                if action.upper() == 'BUY':
                    return entry_price + (tp_pips * pip_size)
                else:
                    return entry_price - (tp_pips * pip_size)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return None
    
    def calculate_risk(self, signal: Dict[str, Any]) -> float:
        """Calculate total risk amount for position"""
        risk_metrics = self.calculate_position_risk(signal)
        return risk_metrics['risk_amount']
    
    def update_portfolio_metrics(self, positions: List[Any], account_info: Any):
        """Update portfolio risk metrics"""
        try:
            if not account_info:
                return
            
            # Calculate current portfolio metrics
            total_equity = account_info.equity
            total_balance = account_info.balance
            
            # Update drawdown history
            if total_balance > 0:
                current_drawdown = (total_balance - total_equity) / total_balance
                self.drawdown_history.append({
                    'timestamp': datetime.now(),
                    'drawdown': current_drawdown,
                    'equity': total_equity,
                    'balance': total_balance
                })
            
            # Update daily PnL
            today = datetime.now().date()
            daily_pnl = sum(pos.profit for pos in positions)
            
            # Store daily PnL (keep only last 30 days)
            self.daily_pnl.append({
                'date': today,
                'pnl': daily_pnl,
                'equity': total_equity
            })
            
            if len(self.daily_pnl) > 30:
                self.daily_pnl = self.daily_pnl[-30:]
            
            # Calculate updated risk metrics
            self._calculate_portfolio_risk_metrics()
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {str(e)}")
    
    def _calculate_portfolio_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            if len(self.daily_pnl) < 2:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 1.0)
            
            # Extract PnL series
            pnl_series = np.array([d['pnl'] for d in self.daily_pnl])
            
            # Calculate VaR (Value at Risk)
            var_95 = np.percentile(pnl_series, 5) if len(pnl_series) > 0 else 0
            var_99 = np.percentile(pnl_series, 1) if len(pnl_series) > 0 else 0
            
            # Calculate maximum drawdown
            equity_series = np.array([d['equity'] for d in self.daily_pnl])
            running_max = np.maximum.accumulate(equity_series)
            drawdown_series = (running_max - equity_series) / running_max
            max_drawdown = np.max(drawdown_series) if len(drawdown_series) > 0 else 0
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            mean_return = np.mean(pnl_series) if len(pnl_series) > 0 else 0
            std_return = np.std(pnl_series) if len(pnl_series) > 0 else 1
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
            
            # Calculate Sortino ratio (downside deviation)
            negative_returns = pnl_series[pnl_series < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1
            sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
            
            # Calculate Calmar ratio
            calmar_ratio = mean_return / max_drawdown if max_drawdown > 0 else 0
            
            # Calculate current exposure
            current_exposure = self._calculate_current_exposure()
            
            metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                current_exposure=current_exposure,
                portfolio_beta=1.0  # Simplified
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio risk metrics: {str(e)}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 1.0)
    
    def _calculate_current_exposure(self) -> float:
        """Calculate current portfolio exposure"""
        try:
            # This would get actual positions and calculate exposure
            # Simplified implementation
            return 0.05  # 5% default exposure
        except Exception as e:
            self.logger.error(f"Error calculating current exposure: {str(e)}")
            return 0.0
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown"""
        try:
            if self.drawdown_history:
                return self.drawdown_history[-1]['drawdown']
            return 0.0
        except Exception as e:
            self.logger.error(f"Error calculating current drawdown: {str(e)}")
            return 0.0
    
    def _check_correlation_limits(self, symbol: str) -> bool:
        """Check if adding symbol would exceed correlation limits"""
        try:
            # Simplified correlation check
            # In production, this would calculate actual correlations
            current_symbols = self._get_current_symbols()
            
            # For major pairs, check correlation
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
            if symbol in major_pairs:
                major_positions = sum(1 for s in current_symbols if s in major_pairs)
                return major_positions < 3  # Max 3 major pairs
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking correlation limits: {str(e)}")
            return True
    
    def _get_current_symbols(self) -> List[str]:
        """Get symbols of current positions"""
        try:
            # This would get actual current positions
            # Simplified implementation
            return []
        except Exception as e:
            self.logger.error(f"Error getting current symbols: {str(e)}")
            return []
    
    def _get_pip_value(self, symbol: str, lot_size: float) -> float:
        """Calculate pip value for position"""
        try:
            # Simplified pip value calculation
            # This would use actual symbol info and account currency
            if 'JPY' in symbol:
                return lot_size * 100 * 0.01  # JPY pairs
            else:
                return lot_size * 100000 * 0.0001  # USD pairs
        except Exception as e:
            self.logger.error(f"Error calculating pip value: {str(e)}")
            return 10.0  # Default value
    
    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol"""
        try:
            if 'JPY' in symbol:
                return 0.01
            else:
                return 0.0001
        except Exception as e:
            self.logger.error(f"Error getting pip size: {str(e)}")
            return 0.0001
    
    def _get_account_balance(self) -> float:
        """Get account balance"""
        try:
            # This would get actual account balance
            # Simplified implementation
            return 10000.0  # Default $10,000
        except Exception as e:
            self.logger.error(f"Error getting account balance: {str(e)}")
            return 10000.0
    
    def _get_current_price(self, symbol: str, action: str) -> float:
        """Get current market price"""
        try:
            # This would get actual current price from MT5
            # Simplified implementation
            if action.upper() == 'BUY':
                return 1.1000  # Default ask price
            else:
                return 1.0999  # Default bid price
        except Exception as e:
            self.logger.error(f"Error getting current price: {str(e)}")
            return 1.1000
    
    def _get_atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """Get Average True Range for symbol"""
        try:
            # This would calculate actual ATR from price data
            # Simplified implementation
            return 0.0020  # Default ATR value
        except Exception as e:
            self.logger.error(f"Error getting ATR: {str(e)}")
            return None
    
    def _estimate_default_risk(self, symbol: str, lot_size: float) -> float:
        """Estimate default risk amount"""
        try:
            # Default risk estimation based on volatility
            atr = self._get_atr(symbol)
            if atr:
                return atr * self._get_pip_value(symbol, lot_size) * 2
            else:
                return lot_size * 100  # Default risk
        except Exception as e:
            self.logger.error(f"Error estimating default risk: {str(e)}")
            return 100.0
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            metrics = self._calculate_portfolio_risk_metrics()
            
            report = {
                'risk_metrics': {
                    'var_95': metrics.var_95,
                    'var_99': metrics.var_99,
                    'max_drawdown': metrics.max_drawdown,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'sortino_ratio': metrics.sortino_ratio,
                    'calmar_ratio': metrics.calmar_ratio,
                    'current_exposure': metrics.current_exposure
                },
                'risk_limits': {
                    'max_position_risk': settings.risk.max_risk_per_trade,
                    'max_portfolio_risk': settings.risk.max_drawdown,
                    'max_correlation': settings.risk.correlation_limit,
                    'max_exposure': self.max_position_size
                },
                'current_status': {
                    'total_positions': len(self._get_current_symbols()),
                    'current_drawdown': self._calculate_current_drawdown(),
                    'risk_events_today': len([e for e in self.risk_events 
                                           if e['timestamp'].date() == datetime.now().date()]),
                    'breach_count': self.breach_count
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {str(e)}")
            return {}
    
    def log_risk_event(self, event_type: str, description: str, severity: str = "WARNING"):
        """Log risk management event"""
        try:
            event = {
                'timestamp': datetime.now(),
                'type': event_type,
                'description': description,
                'severity': severity
            }
            
            self.risk_events.append(event)
            
            # Keep only last 1000 events
            if len(self.risk_events) > 1000:
                self.risk_events = self.risk_events[-1000:]
            
            if severity in ["ERROR", "CRITICAL"]:
                self.breach_count += 1
            
            self.logger.warning(f"Risk Event [{severity}]: {description}")
            
        except Exception as e:
            self.logger.error(f"Error logging risk event: {str(e)}")
