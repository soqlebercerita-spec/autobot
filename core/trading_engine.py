"""
Main Trading Engine for AuraTrade Bot
Coordinates all trading activities and strategies
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np

from config.config import config
from config.settings import settings
from core.mt5_connector import MT5Connector
from core.order_manager import OrderManager
from core.risk_manager import RiskManager
from core.position_sizing import PositionSizing
from strategies.hft_strategy import HFTStrategy
from strategies.scalping_strategy import ScalpingStrategy
from analysis.technical_analysis import TechnicalAnalysis
from analysis.market_conditions import MarketConditions
from data.data_manager import DataManager
from utils.logger import Logger

class TradingEngine:
    """Main trading engine that coordinates all trading activities"""
    
    def __init__(self, mt5_connector: MT5Connector):
        self.mt5_connector = mt5_connector
        self.logger = Logger()
        
        # Initialize core components
        self.order_manager = OrderManager(mt5_connector)
        self.risk_manager = RiskManager()
        self.position_sizing = PositionSizing()
        self.data_manager = DataManager(mt5_connector)
        self.technical_analysis = TechnicalAnalysis()
        self.market_conditions = MarketConditions()
        
        # Initialize strategies
        self.strategies = {}
        self._initialize_strategies()
        
        # Engine state
        self.running = False
        self.last_cycle_time = 0
        self.cycle_count = 0
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
        
        # Risk monitoring
        self.daily_trades = 0
        self.daily_trade_reset = datetime.now().date()
        self.emergency_stop = False
        
        self.logger.info("Trading Engine initialized")
    
    def _initialize_strategies(self):
        """Initialize trading strategies"""
        try:
            if settings.hft.enabled and config.STRATEGY_ENABLED['hft']:
                self.strategies['hft'] = HFTStrategy(
                    self.mt5_connector, 
                    self.order_manager,
                    self.risk_manager
                )
            
            if config.STRATEGY_ENABLED['scalping']:
                self.strategies['scalping'] = ScalpingStrategy(
                    self.mt5_connector,
                    self.order_manager,
                    self.risk_manager
                )
            
            self.logger.info(f"Initialized {len(self.strategies)} strategies")
            
        except Exception as e:
            self.logger.error(f"Error initializing strategies: {str(e)}")
    
    def start(self):
        """Start the trading engine"""
        if not self.mt5_connector.is_connected():
            self.logger.error("MT5 not connected. Cannot start trading engine.")
            return False
        
        if not self._pre_trading_checks():
            return False
        
        self.running = True
        self.logger.info("Trading Engine started")
        return True
    
    def stop(self):
        """Stop the trading engine"""
        self.running = False
        
        # Close all open positions if emergency stop
        if self.emergency_stop:
            self._emergency_close_all_positions()
        
        self.logger.info("Trading Engine stopped")
    
    def run_cycle(self):
        """Run one trading cycle"""
        if not self.running:
            return
        
        cycle_start = time.time()
        
        try:
            # Check if we need to reset daily counters
            self._check_daily_reset()
            
            # Pre-cycle checks
            if not self._pre_cycle_checks():
                return
            
            # Update market data
            self._update_market_data()
            
            # Run risk management checks
            if not self._risk_management_checks():
                return
            
            # Execute strategies
            self._execute_strategies()
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Post-cycle cleanup
            self._post_cycle_cleanup()
            
            # Calculate cycle performance
            cycle_time = (time.time() - cycle_start) * 1000  # Convert to milliseconds
            self.last_cycle_time = cycle_time
            self.cycle_count += 1
            
            # Log performance for HFT monitoring
            if self.cycle_count % 1000 == 0:
                self.logger.info(f"Cycle {self.cycle_count}: {cycle_time:.2f}ms avg")
            
        except Exception as e:
            self.logger.error(f"Trading cycle error: {str(e)}")
    
    def _pre_trading_checks(self) -> bool:
        """Perform pre-trading checks"""
        # Check trading hours
        if not settings.is_trading_time():
            self.logger.info("Outside trading hours")
            return False
        
        # Check weekend trading
        if not settings.is_weekend_trading_allowed():
            self.logger.info("Weekend trading not allowed")
            return False
        
        # Check account balance
        account_info = self.mt5_connector.get_account_info()
        if account_info and account_info.balance < 100:  # Minimum balance check
            self.logger.error("Insufficient account balance")
            return False
        
        return True
    
    def _pre_cycle_checks(self) -> bool:
        """Perform pre-cycle checks"""
        # Check connection
        if not self.mt5_connector.is_connected():
            self.logger.error("MT5 connection lost")
            return False
        
        # Check emergency stop
        if self.emergency_stop:
            self.logger.warning("Emergency stop activated")
            return False
        
        # Check daily trade limit
        if self.daily_trades >= config.MAX_DAILY_TRADES:
            self.logger.warning("Daily trade limit reached")
            return False
        
        return True
    
    def _update_market_data(self):
        """Update market data for all symbols"""
        try:
            for symbol in settings.trading.enabled_symbols:
                # Update tick data
                self.data_manager.update_tick_data(symbol)
                
                # Update OHLC data for multiple timeframes
                for timeframe in config.TA_TIMEFRAMES:
                    self.data_manager.update_ohlc_data(symbol, timeframe)
            
        except Exception as e:
            self.logger.error(f"Error updating market data: {str(e)}")
    
    def _risk_management_checks(self) -> bool:
        """Perform risk management checks"""
        try:
            # Check maximum drawdown
            current_equity = self.mt5_connector.get_account_info().equity
            account_balance = self.mt5_connector.get_account_info().balance
            
            if account_balance > 0:
                drawdown = (account_balance - current_equity) / account_balance
                
                if drawdown > config.MAX_DRAWDOWN:
                    self.logger.error(f"Maximum drawdown exceeded: {drawdown:.2%}")
                    self.emergency_stop = True
                    return False
                
                # Update max drawdown metric
                self.performance_metrics['max_drawdown'] = max(
                    self.performance_metrics['max_drawdown'], 
                    drawdown
                )
            
            # Check position limits
            total_positions = len(self.mt5_connector.get_positions())
            if total_positions >= config.MAX_POSITIONS:
                self.logger.warning("Maximum positions limit reached")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Risk management check failed: {str(e)}")
            return False
    
    def _execute_strategies(self):
        """Execute all active strategies"""
        for strategy_name, strategy in self.strategies.items():
            try:
                if strategy.is_enabled():
                    signals = strategy.generate_signals()
                    
                    for signal in signals:
                        if self._validate_signal(signal):
                            self._execute_signal(signal)
                
            except Exception as e:
                self.logger.error(f"Error executing strategy {strategy_name}: {str(e)}")
    
    def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate trading signal"""
        try:
            # Check required fields
            required_fields = ['symbol', 'action', 'lot_size', 'price']
            if not all(field in signal for field in required_fields):
                return False
            
            # Check symbol is enabled
            if signal['symbol'] not in settings.trading.enabled_symbols:
                return False
            
            # Check lot size limits
            symbol_config = config.get_symbol_config(signal['symbol'])
            if signal['lot_size'] < symbol_config['min_lot'] or signal['lot_size'] > symbol_config['max_lot']:
                return False
            
            # Check risk limits
            risk_amount = self.risk_manager.calculate_risk(signal)
            if risk_amount > config.MAX_RISK_PER_TRADE:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {str(e)}")
            return False
    
    def _execute_signal(self, signal: Dict[str, Any]):
        """Execute trading signal"""
        try:
            # Calculate position size
            lot_size = self.position_sizing.calculate_lot_size(signal)
            signal['lot_size'] = lot_size
            
            # Add stop loss and take profit
            signal = self.risk_manager.add_risk_parameters(signal)
            
            # Execute order
            result = self.order_manager.place_order(signal)
            
            if result and result.retcode == 10009:  # TRADE_RETCODE_DONE
                self.daily_trades += 1
                self.performance_metrics['total_trades'] += 1
                
                self.logger.info(f"Order executed: {signal['symbol']} {signal['action']} {lot_size}")
            else:
                error_msg = f"Order failed: {result.comment if result else 'Unknown error'}"
                self.logger.error(error_msg)
            
        except Exception as e:
            self.logger.error(f"Signal execution error: {str(e)}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            # Get all closed positions for today
            closed_positions = self.order_manager.get_closed_positions_today()
            
            if closed_positions:
                winning_trades = sum(1 for pos in closed_positions if pos.profit > 0)
                total_trades = len(closed_positions)
                total_profit = sum(pos.profit for pos in closed_positions)
                
                self.performance_metrics.update({
                    'winning_trades': winning_trades,
                    'losing_trades': total_trades - winning_trades,
                    'total_profit': total_profit,
                    'win_rate': winning_trades / total_trades if total_trades > 0 else 0
                })
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {str(e)}")
    
    def _post_cycle_cleanup(self):
        """Post-cycle cleanup tasks"""
        try:
            # Clean up old data
            self.data_manager.cleanup_old_data()
            
            # Update strategy states
            for strategy in self.strategies.values():
                strategy.update_state()
            
        except Exception as e:
            self.logger.error(f"Post-cycle cleanup error: {str(e)}")
    
    def _check_daily_reset(self):
        """Check if we need to reset daily counters"""
        current_date = datetime.now().date()
        if current_date > self.daily_trade_reset:
            self.daily_trades = 0
            self.daily_trade_reset = current_date
            self.logger.info("Daily counters reset")
    
    def _emergency_close_all_positions(self):
        """Emergency close all open positions"""
        try:
            positions = self.mt5_connector.get_positions()
            for position in positions:
                self.order_manager.close_position(position.ticket)
            
            self.logger.warning(f"Emergency closed {len(positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Emergency close error: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status"""
        return {
            'running': self.running,
            'cycle_count': self.cycle_count,
            'last_cycle_time_ms': self.last_cycle_time,
            'daily_trades': self.daily_trades,
            'emergency_stop': self.emergency_stop,
            'performance_metrics': self.performance_metrics.copy(),
            'active_strategies': list(self.strategies.keys()),
            'connected_symbols': settings.trading.enabled_symbols
        }
    
    def get_real_time_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time data for a symbol"""
        try:
            return self.data_manager.get_real_time_data(symbol)
        except Exception as e:
            self.logger.error(f"Error getting real-time data for {symbol}: {str(e)}")
            return None
    
    def force_emergency_stop(self, reason: str = "Manual emergency stop"):
        """Force emergency stop"""
        self.logger.warning(f"Emergency stop triggered: {reason}")
        self.emergency_stop = True
        self.stop()
