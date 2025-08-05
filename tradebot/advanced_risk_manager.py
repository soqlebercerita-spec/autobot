"""
Advanced Risk Management System
VAR calculations, Kelly Criterion, Emergency Stops, Correlation Limits
"""

import numpy as np
import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math
from config import config

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class EmergencyTrigger(Enum):
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    CORRELATION_BREACH = "correlation_breach"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    SYSTEM_ERROR = "system_error"

@dataclass
class RiskMetrics:
    var_1day: float  # 1-day Value at Risk
    var_5day: float  # 5-day Value at Risk
    expected_shortfall: float  # Expected Shortfall (CVaR)
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float  # Market beta
    correlation_risk: float
    liquidity_risk: float
    concentration_risk: float
    total_risk_score: float
    risk_level: RiskLevel
    timestamp: datetime

@dataclass
class PositionRisk:
    symbol: str
    position_size: float
    market_value: float
    var_contribution: float
    correlation_exposure: Dict[str, float]
    liquidity_score: float
    concentration_pct: float
    risk_score: float
    recommended_action: str  # HOLD, REDUCE, CLOSE
    timestamp: datetime

class AdvancedRiskManager:
    def __init__(self, market_data_api, trade_logger):
        self.market_api = market_data_api
        self.trade_logger = trade_logger
        
        # Risk parameters
        self.risk_config = {
            'max_daily_loss_pct': 5.0,  # Maximum 5% daily loss
            'max_drawdown_pct': 15.0,   # Maximum 15% drawdown
            'var_confidence': 0.95,     # 95% VaR confidence
            'max_position_size_pct': 10.0,  # Max 10% in single position
            'max_correlation': 0.7,     # Maximum position correlation
            'min_liquidity_score': 0.3, # Minimum liquidity requirement
            'kelly_fraction': 0.25,     # Kelly Criterion fraction
            'volatility_limit': 0.05,   # 5% volatility limit
            'concentration_limit': 0.3,  # 30% max in single asset class
            'emergency_stop_enabled': True,
            'circuit_breaker_enabled': True
        }
        
        # Emergency stops
        self.emergency_stops = {
            EmergencyTrigger.DAILY_LOSS_LIMIT: {'triggered': False, 'threshold': -5.0},
            EmergencyTrigger.DRAWDOWN_LIMIT: {'triggered': False, 'threshold': -15.0},
            EmergencyTrigger.CORRELATION_BREACH: {'triggered': False, 'threshold': 0.8},
            EmergencyTrigger.VOLATILITY_SPIKE: {'triggered': False, 'threshold': 0.1},
            EmergencyTrigger.LIQUIDITY_CRISIS: {'triggered': False, 'threshold': 0.1},
            EmergencyTrigger.SYSTEM_ERROR: {'triggered': False, 'threshold': None}
        }
        
        # Portfolio tracking
        self.portfolio = {}
        self.historical_returns = []
        self.correlation_matrix = {}
        self.risk_metrics_history = []
        
        # Performance tracking
        self.account_balance = 10000.0  # Starting balance
        self.peak_balance = 10000.0
        self.daily_pnl = 0.0
        self.start_of_day_balance = 10000.0
        
        # Risk monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        self.last_risk_check = datetime.now()
        
        print("🛡️ Advanced Risk Manager initialized")
        print(f"   • Max Daily Loss: {self.risk_config['max_daily_loss_pct']}%")
        print(f"   • Max Drawdown: {self.risk_config['max_drawdown_pct']}%")
        print(f"   • VaR Confidence: {self.risk_config['var_confidence']*100}%")
        print(f"   • Emergency Stops: {'✅' if self.risk_config['emergency_stop_enabled'] else '❌'}")
        print(f"   • Circuit Breakers: {'✅' if self.risk_config['circuit_breaker_enabled'] else '❌'}")
    
    def start_risk_monitoring(self):
        """Start continuous risk monitoring"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._risk_monitoring_loop, daemon=True)
            self.monitor_thread.start()
            print("🔍 Risk monitoring started")
    
    def stop_risk_monitoring(self):
        """Stop risk monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("🔍 Risk monitoring stopped")
    
    def _risk_monitoring_loop(self):
        """Continuous risk monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_emergency_conditions()
                self._update_risk_metrics()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                print(f"❌ Risk monitoring error: {e}")
                time.sleep(30)
    
    def calculate_position_size_kelly(self, symbol: str, win_rate: float, avg_win: float, avg_loss: float, balance: float) -> float:
        """Calculate optimal position size using Kelly Criterion"""
        try:
            if win_rate <= 0 or win_rate >= 1 or avg_loss <= 0:
                return 0.01  # Minimum position size
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Apply safety margin (use only fraction of Kelly)
            safe_kelly = kelly_fraction * self.risk_config['kelly_fraction']
            
            # Ensure reasonable bounds
            safe_kelly = max(0.01, min(safe_kelly, 0.1))  # Between 1% and 10%
            
            # Calculate actual position size
            risk_amount = balance * safe_kelly
            
            # Get current price and calculate lot size
            current_price = self.market_api.get_price(symbol)
            if not current_price:
                return 0.01
            
            # Simplified lot calculation (adjust for actual broker requirements)
            lot_size = risk_amount / (current_price * 100)  # Assuming $100 per lot
            lot_size = max(0.01, min(lot_size, 1.0))  # Reasonable bounds
            
            print(f"📊 Kelly Position Size for {symbol}: {lot_size:.3f} lots "
                  f"(Kelly: {kelly_fraction:.3f}, Safe: {safe_kelly:.3f})")
            
            return lot_size
        
        except Exception as e:
            print(f"❌ Kelly calculation error: {e}")
            return 0.01
    
    def calculate_var(self, returns: List[float], confidence: float = 0.95, horizon: int = 1) -> Tuple[float, float]:
        """Calculate Value at Risk and Expected Shortfall"""
        try:
            if len(returns) < 30:
                return 0.0, 0.0
            
            returns_array = np.array(returns)
            
            # Scale for horizon (square root of time rule)
            if horizon > 1:
                returns_array = returns_array * math.sqrt(horizon)
            
            # Sort returns
            sorted_returns = np.sort(returns_array)
            
            # Calculate VaR
            var_index = int((1 - confidence) * len(sorted_returns))
            var = abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else 0
            
            # Calculate Expected Shortfall (CVaR)
            tail_returns = sorted_returns[:var_index]
            expected_shortfall = abs(np.mean(tail_returns)) if len(tail_returns) > 0 else 0
            
            return var, expected_shortfall
        
        except Exception as e:
            print(f"❌ VaR calculation error: {e}")
            return 0.0, 0.0
    
    def calculate_portfolio_risk(self) -> RiskMetrics:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Get recent returns
            if len(self.historical_returns) < 30:
                # Generate mock returns for demo
                self.historical_returns = list(np.random.normal(0.001, 0.02, 100))
            
            returns = self.historical_returns[-100:]  # Last 100 periods
            
            # Calculate VaR
            var_1day, es_1day = self.calculate_var(returns, self.risk_config['var_confidence'], 1)
            var_5day, es_5day = self.calculate_var(returns, self.risk_config['var_confidence'], 5)
            
            # Calculate other metrics
            volatility = np.std(returns) * math.sqrt(252)  # Annualized
            mean_return = np.mean(returns) * 252  # Annualized
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # Calculate drawdown
            cumulative_returns = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns))
            
            # Portfolio specific metrics
            correlation_risk = self._calculate_correlation_risk()
            liquidity_risk = self._calculate_liquidity_risk()
            concentration_risk = self._calculate_concentration_risk()
            
            # Beta calculation (simplified, assuming market correlation)
            market_returns = np.random.normal(0.0008, 0.015, len(returns))  # Mock market
            beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns)
            
            # Total risk score
            risk_components = [
                var_1day * 5,  # VaR weight
                volatility * 2,  # Volatility weight
                max_drawdown * 3,  # Drawdown weight
                correlation_risk,
                liquidity_risk,
                concentration_risk
            ]
            
            total_risk_score = np.mean(risk_components)
            
            # Determine risk level
            if total_risk_score < 0.02:
                risk_level = RiskLevel.LOW
            elif total_risk_score < 0.05:
                risk_level = RiskLevel.MEDIUM
            elif total_risk_score < 0.1:
                risk_level = RiskLevel.HIGH
            elif total_risk_score < 0.2:
                risk_level = RiskLevel.CRITICAL
            else:
                risk_level = RiskLevel.EMERGENCY
            
            risk_metrics = RiskMetrics(
                var_1day=var_1day,
                var_5day=var_5day,
                expected_shortfall=es_1day,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                total_risk_score=total_risk_score,
                risk_level=risk_level,
                timestamp=datetime.now()
            )
            
            self.risk_metrics_history.append(risk_metrics)
            
            return risk_metrics
        
        except Exception as e:
            print(f"❌ Portfolio risk calculation error: {e}")
            return RiskMetrics(
                var_1day=0.01, var_5day=0.02, expected_shortfall=0.015,
                sharpe_ratio=0.0, max_drawdown=0.05, volatility=0.02,
                beta=1.0, correlation_risk=0.1, liquidity_risk=0.1,
                concentration_risk=0.1, total_risk_score=0.05,
                risk_level=RiskLevel.MEDIUM, timestamp=datetime.now()
            )
    
    def _calculate_correlation_risk(self) -> float:
        """Calculate portfolio correlation risk"""
        try:
            if len(self.portfolio) < 2:
                return 0.0
            
            symbols = list(self.portfolio.keys())
            correlations = []
            
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    # Get correlation (mock for demo)
                    corr = abs(np.random.uniform(-0.5, 0.8))  # Mock correlation
                    correlations.append(corr)
                    
                    if corr > self.risk_config['max_correlation']:
                        print(f"⚠️ High correlation detected: {symbol1}-{symbol2}: {corr:.2f}")
            
            return np.mean(correlations) if correlations else 0.0
        
        except Exception as e:
            print(f"❌ Correlation risk calculation error: {e}")
            return 0.1
    
    def _calculate_liquidity_risk(self) -> float:
        """Calculate portfolio liquidity risk"""
        try:
            if not self.portfolio:
                return 0.0
            
            liquidity_scores = []
            for symbol, position in self.portfolio.items():
                # Mock liquidity score (in production, use real market data)
                if 'USD' in symbol:
                    liquidity_score = 0.9  # High liquidity for USD pairs
                elif 'XAU' in symbol:
                    liquidity_score = 0.7  # Medium liquidity for gold
                else:
                    liquidity_score = 0.5  # Default
                
                liquidity_scores.append(liquidity_score)
            
            avg_liquidity = np.mean(liquidity_scores)
            liquidity_risk = 1 - avg_liquidity  # Inverse of liquidity
            
            return liquidity_risk
        
        except Exception as e:
            print(f"❌ Liquidity risk calculation error: {e}")
            return 0.2
    
    def _calculate_concentration_risk(self) -> float:
        """Calculate portfolio concentration risk"""
        try:
            if not self.portfolio:
                return 0.0
            
            total_exposure = sum(abs(pos.get('market_value', 0)) for pos in self.portfolio.values())
            if total_exposure == 0:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index
            concentrations = []
            for position in self.portfolio.values():
                weight = abs(position.get('market_value', 0)) / total_exposure
                concentrations.append(weight * weight)
            
            hhi = sum(concentrations)
            
            # Normalize to 0-1 scale (1 = maximum concentration)
            max_hhi = 1.0  # Single position
            min_hhi = 1.0 / len(self.portfolio) if self.portfolio else 1.0
            
            normalized_concentration = (hhi - min_hhi) / (max_hhi - min_hhi) if max_hhi != min_hhi else 0
            
            return max(0, min(1, normalized_concentration))
        
        except Exception as e:
            print(f"❌ Concentration risk calculation error: {e}")
            return 0.1
    
    def _check_emergency_conditions(self):
        """Check for emergency stop conditions"""
        try:
            current_balance = self.get_current_balance()
            
            # Daily loss limit
            daily_loss_pct = (current_balance - self.start_of_day_balance) / self.start_of_day_balance * 100
            if daily_loss_pct <= -self.risk_config['max_daily_loss_pct']:
                self._trigger_emergency_stop(EmergencyTrigger.DAILY_LOSS_LIMIT, 
                                           f"Daily loss: {daily_loss_pct:.2f}%")
            
            # Drawdown limit
            drawdown_pct = (current_balance - self.peak_balance) / self.peak_balance * 100
            if drawdown_pct <= -self.risk_config['max_drawdown_pct']:
                self._trigger_emergency_stop(EmergencyTrigger.DRAWDOWN_LIMIT,
                                           f"Drawdown: {drawdown_pct:.2f}%")
            
            # Volatility spike
            if len(self.historical_returns) >= 10:
                recent_vol = np.std(self.historical_returns[-10:])
                if recent_vol > self.risk_config['volatility_limit']:
                    self._trigger_emergency_stop(EmergencyTrigger.VOLATILITY_SPIKE,
                                               f"Volatility spike: {recent_vol:.3f}")
            
            # Update peak balance
            if current_balance > self.peak_balance:
                self.peak_balance = current_balance
        
        except Exception as e:
            print(f"❌ Emergency check error: {e}")
            self._trigger_emergency_stop(EmergencyTrigger.SYSTEM_ERROR, str(e))
    
    def _trigger_emergency_stop(self, trigger: EmergencyTrigger, message: str):
        """Trigger emergency stop"""
        if not self.emergency_stops[trigger]['triggered']:
            self.emergency_stops[trigger]['triggered'] = True
            
            print(f"🚨 EMERGENCY STOP TRIGGERED: {trigger.value}")
            print(f"   Reason: {message}")
            print(f"   Time: {datetime.now()}")
            
            # Log emergency stop
            self.trade_logger.log_error(f"Emergency stop: {trigger.value} - {message}", "EMERGENCY")
            
            # In production, this would:
            # 1. Close all positions immediately
            # 2. Cancel all pending orders
            # 3. Send urgent notifications
            # 4. Disable trading
            
            print("🛑 All trading operations halted")
            print("🔧 Manual intervention required to resume")
    
    def reset_emergency_stop(self, trigger: EmergencyTrigger, password: str = "RESET123"):
        """Reset emergency stop (requires manual intervention)"""
        if password != "RESET123":
            print("❌ Invalid reset password")
            return False
        
        self.emergency_stops[trigger]['triggered'] = False
        print(f"✅ Emergency stop reset: {trigger.value}")
        self.trade_logger.log_info(f"Emergency stop reset: {trigger.value}")
        return True
    
    def can_place_trade(self, symbol: str, volume: float, action: str) -> Tuple[bool, str]:
        """Check if trade can be placed based on risk limits"""
        try:
            # Check emergency stops
            for trigger, status in self.emergency_stops.items():
                if status['triggered']:
                    return False, f"Emergency stop active: {trigger.value}"
            
            # Check daily loss limit
            daily_pnl_pct = self.daily_pnl / self.start_of_day_balance * 100
            if daily_pnl_pct <= -self.risk_config['max_daily_loss_pct'] * 0.8:  # 80% of limit
                return False, f"Approaching daily loss limit: {daily_pnl_pct:.2f}%"
            
            # Check position size limits
            current_price = self.market_api.get_price(symbol)
            if current_price:
                position_value = volume * current_price * 100  # Simplified
                position_pct = position_value / self.get_current_balance() * 100
                
                if position_pct > self.risk_config['max_position_size_pct']:
                    return False, f"Position too large: {position_pct:.2f}% > {self.risk_config['max_position_size_pct']}%"
            
            # Check correlation limits
            correlation_risk = self._calculate_correlation_risk()
            if correlation_risk > self.risk_config['max_correlation']:
                return False, f"Correlation risk too high: {correlation_risk:.2f}"
            
            # Check concentration limits
            concentration_risk = self._calculate_concentration_risk()
            if concentration_risk > self.risk_config['concentration_limit']:
                return False, f"Concentration risk too high: {concentration_risk:.2f}"
            
            return True, "Trade approved"
        
        except Exception as e:
            return False, f"Risk check error: {e}"
    
    def update_portfolio(self, symbol: str, volume: float, price: float, action: str):
        """Update portfolio tracking"""
        try:
            if symbol not in self.portfolio:
                self.portfolio[symbol] = {
                    'volume': 0.0,
                    'avg_price': 0.0,
                    'market_value': 0.0,
                    'unrealized_pnl': 0.0
                }
            
            position = self.portfolio[symbol]
            
            if action.upper() == "BUY":
                # Add to position
                total_cost = (position['volume'] * position['avg_price']) + (volume * price)
                total_volume = position['volume'] + volume
                position['avg_price'] = total_cost / total_volume if total_volume > 0 else price
                position['volume'] = total_volume
            elif action.upper() == "SELL":
                # Reduce position
                position['volume'] -= volume
                if position['volume'] <= 0:
                    del self.portfolio[symbol]
                    return
            
            # Update market value
            current_price = self.market_api.get_price(symbol) or price
            position['market_value'] = position['volume'] * current_price
            position['unrealized_pnl'] = (current_price - position['avg_price']) * position['volume']
            
        except Exception as e:
            print(f"❌ Portfolio update error: {e}")
    
    def get_current_balance(self) -> float:
        """Get current account balance"""
        # In production, this would query the actual broker balance
        return self.account_balance
    
    def _update_risk_metrics(self):
        """Update risk metrics periodically"""
        try:
            current_time = datetime.now()
            if (current_time - self.last_risk_check).seconds >= 60:  # Update every minute
                risk_metrics = self.calculate_portfolio_risk()
                
                if risk_metrics.risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
                    print(f"⚠️ HIGH RISK DETECTED: {risk_metrics.risk_level.value}")
                    print(f"   Total Risk Score: {risk_metrics.total_risk_score:.3f}")
                    print(f"   VaR (1-day): {risk_metrics.var_1day:.3f}")
                    print(f"   Max Drawdown: {risk_metrics.max_drawdown:.3f}")
                
                self.last_risk_check = current_time
        
        except Exception as e:
            print(f"❌ Risk metrics update error: {e}")
    
    def get_risk_summary(self) -> Dict:
        """Get comprehensive risk summary"""
        try:
            risk_metrics = self.calculate_portfolio_risk()
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'account_balance': self.get_current_balance(),
                'daily_pnl': self.daily_pnl,
                'daily_pnl_pct': self.daily_pnl / self.start_of_day_balance * 100,
                'peak_balance': self.peak_balance,
                'current_drawdown_pct': (self.get_current_balance() - self.peak_balance) / self.peak_balance * 100,
                'risk_metrics': {
                    'var_1day': risk_metrics.var_1day,
                    'var_5day': risk_metrics.var_5day,
                    'expected_shortfall': risk_metrics.expected_shortfall,
                    'sharpe_ratio': risk_metrics.sharpe_ratio,
                    'volatility': risk_metrics.volatility,
                    'max_drawdown': risk_metrics.max_drawdown,
                    'correlation_risk': risk_metrics.correlation_risk,
                    'liquidity_risk': risk_metrics.liquidity_risk,
                    'concentration_risk': risk_metrics.concentration_risk,
                    'total_risk_score': risk_metrics.total_risk_score,
                    'risk_level': risk_metrics.risk_level.value
                },
                'emergency_stops': {
                    trigger.value: status['triggered'] 
                    for trigger, status in self.emergency_stops.items()
                },
                'portfolio': {
                    symbol: {
                        'volume': pos['volume'],
                        'market_value': pos['market_value'],
                        'unrealized_pnl': pos['unrealized_pnl']
                    } for symbol, pos in self.portfolio.items()
                },
                'risk_limits': self.risk_config
            }
            
            return summary
        
        except Exception as e:
            print(f"❌ Risk summary error: {e}")
            return {}

# Global instance
advanced_risk_manager = None  # Will be initialized with dependencies