"""
Advanced Trading Engine - Integration Hub
Menghubungkan semua sistem advanced: Strategies, ML/AI, Risk Management, Adaptive Indicators
"""

import threading
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Import simplified modules (compatible versions)
from simple_trading_strategies import TradingStrategies, StrategyType, TradeSignal
from simple_ml_engine import MLEngine, ModelType, MLPrediction, MarketCondition, MarketRegime
from simple_risk_manager import AdvancedRiskManager, RiskLevel
from simple_adaptive_indicators import AdaptiveIndicators
from telegram_notifier import telegram_notifier
from trade_logger import trade_logger
from performance_optimizer import performance_optimizer
from config import config

class TradingMode(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    HFT_MODE = "hft_mode"
    AI_OPTIMIZED = "ai_optimized"

@dataclass
class TradingDecision:
    symbol: str
    action: str  # BUY/SELL/HOLD
    volume: float
    confidence: float
    strategy_source: str
    ml_confirmation: bool
    risk_approved: bool
    price_target: float
    stop_loss: float
    take_profit: float
    urgency: int
    metadata: Dict
    timestamp: datetime

class AdvancedTradingEngine:
    def __init__(self, market_data_api, simulation_trading=None):
        self.market_api = market_data_api
        self.mt5 = simulation_trading
        
        # Initialize all advanced components
        print("🚀 Initializing Advanced Trading Engine...")
        
        # Core engines
        self.trading_strategies = TradingStrategies(market_data_api, None, None)
        self.ml_engine = MLEngine(market_data_api, None)
        self.advanced_risk = AdvancedRiskManager(market_data_api, trade_logger)
        self.adaptive_indicators = AdaptiveIndicators(market_data_api)
        
        # Update references (optional, simplified modules handle None gracefully)
        if hasattr(self.trading_strategies, 'indicators'):
            self.trading_strategies.indicators = self.adaptive_indicators
        if hasattr(self.trading_strategies, 'risk_manager'):
            self.trading_strategies.risk_manager = self.advanced_risk
        if hasattr(self.ml_engine, 'indicators'):
            self.ml_engine.indicators = self.adaptive_indicators
        
        # Trading configuration
        self.trading_config = {
            'mode': TradingMode.BALANCED,
            'max_concurrent_trades': 10,
            'min_confidence_threshold': 0.6,
            'ml_confirmation_required': True,
            'risk_override_allowed': False,
            'emergency_stop_enabled': True,
            'adaptive_indicators_enabled': True,
            'hft_enabled': True,
            'scalping_target': 300,  # trades per day
            'correlation_monitoring': True
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_profit': 0.0,
            'best_strategy': None,
            'best_ml_model': None,
            'avg_execution_time': 0.0,
            'emergency_stops_triggered': 0,
            'adaptation_count': 0
        }
        
        # Real-time decision engine
        self.decision_queue = []
        self.active_trades = {}
        self.market_regime_cache = {}
        
        # Threading for real-time operations
        self.engine_running = False
        self.decision_thread = None
        self.monitoring_thread = None
        
        print("✅ Advanced Trading Engine initialized")
        print(f"   • Trading Strategies: 5 strategies active")
        print(f"   • ML Engine: 3 models + ensemble")
        print(f"   • Advanced Risk Manager: VAR + Kelly + Emergency stops")
        print(f"   • Adaptive Indicators: Full adaptation enabled")
        print(f"   • Mode: {self.trading_config['mode'].value}")
    
    def start_engine(self):
        """Start the advanced trading engine"""
        if self.engine_running:
            print("⚠️ Engine already running")
            return
        
        self.engine_running = True
        
        # Start all monitoring systems
        self.advanced_risk.start_risk_monitoring()
        performance_optimizer.start_monitoring()
        
        # Start decision engine
        self.decision_thread = threading.Thread(target=self._decision_engine_loop, daemon=True)
        self.decision_thread.start()
        
        # Start market monitoring
        self.monitoring_thread = threading.Thread(target=self._market_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print("🚀 Advanced Trading Engine started")
        telegram_notifier.notify_bot_status("STARTED", "Advanced engine with AI, HFT, and ML active")
    
    def stop_engine(self):
        """Stop the advanced trading engine"""
        self.engine_running = False
        
        # Stop monitoring systems
        self.advanced_risk.stop_risk_monitoring()
        performance_optimizer.stop_monitoring()
        
        # Wait for threads
        if self.decision_thread:
            self.decision_thread.join(timeout=2)
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
        
        print("🛑 Advanced Trading Engine stopped")
        telegram_notifier.notify_bot_status("STOPPED", "All advanced systems deactivated")
    
    def _decision_engine_loop(self):
        """Main decision engine loop"""
        while self.engine_running:
            try:
                # Process symbols
                symbols = ['XAUUSDm', 'EURUSD', 'GBPUSD', 'BTCUSD']
                
                for symbol in symbols:
                    if not self.engine_running:
                        break
                    
                    # Generate comprehensive trading decision
                    decision = self._make_trading_decision(symbol)
                    
                    if decision and decision.risk_approved:
                        self._execute_decision(decision)
                
                # Adjust sleep based on trading mode
                if self.trading_config['mode'] == TradingMode.HFT_MODE:
                    time.sleep(0.1)  # 100ms for HFT
                elif self.trading_config['mode'] == TradingMode.AGGRESSIVE:
                    time.sleep(1)    # 1 second
                else:
                    time.sleep(5)    # 5 seconds for balanced/conservative
                
            except Exception as e:
                print(f"❌ Decision engine error: {e}")
                time.sleep(10)
    
    def _market_monitoring_loop(self):
        """Market condition monitoring loop"""
        while self.engine_running:
            try:
                # Monitor market regimes for all symbols
                symbols = ['XAUUSDm', 'EURUSD', 'GBPUSD', 'BTCUSD']
                
                for symbol in symbols:
                    market_condition = self.ml_engine.detect_market_regime(symbol)
                    self.market_regime_cache[symbol] = market_condition
                    
                    # Update adaptive parameters based on regime
                    self.ml_engine.update_adaptive_parameters(market_condition)
                
                # Check for regime changes
                self._handle_regime_changes()
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                print(f"❌ Market monitoring error: {e}")
                time.sleep(60)
    
    def _make_trading_decision(self, symbol: str) -> Optional[TradingDecision]:
        """Make comprehensive trading decision combining all systems"""
        try:
            start_time = time.time()
            
            # 1. Get strategy signals
            strategy_signals = self.trading_strategies.generate_signals(symbol)
            if not strategy_signals:
                return None
            
            # 2. Get ML predictions
            ml_predictions = self.ml_engine.generate_ml_predictions(symbol)
            
            # 3. Get market regime
            market_condition = self.market_regime_cache.get(
                symbol, 
                self.ml_engine.detect_market_regime(symbol)
            )
            
            # 4. Analyze with adaptive indicators
            prices = self.market_api.get_recent_prices(symbol, count=100)
            if not prices or len(prices) < 50:
                return None
            
            # Get adaptive signals
            adaptive_ma = self.adaptive_indicators.adaptive_ma(prices, symbol, AdaptiveMode.FULL_ADAPTIVE)
            adaptive_rsi = self.adaptive_indicators.adaptive_rsi(prices, symbol, AdaptiveMode.FULL_ADAPTIVE)
            adaptive_bb = self.adaptive_indicators.adaptive_bollinger_bands(prices, symbol, AdaptiveMode.FULL_ADAPTIVE)
            
            # 5. Combine all signals
            combined_decision = self._combine_signals(
                strategy_signals, ml_predictions, market_condition,
                adaptive_ma, adaptive_rsi, adaptive_bb, prices[-1]
            )
            
            if not combined_decision:
                return None
            
            # 6. Risk assessment
            position_size = self._calculate_optimal_position_size(
                symbol, combined_decision['confidence'], combined_decision['action']
            )
            
            risk_approved, risk_message = self.advanced_risk.can_place_trade(
                symbol, position_size, combined_decision['action']
            )
            
            # 7. Create final decision
            decision = TradingDecision(
                symbol=symbol,
                action=combined_decision['action'],
                volume=position_size,
                confidence=combined_decision['confidence'],
                strategy_source=combined_decision['primary_strategy'],
                ml_confirmation=combined_decision['ml_confirmed'],
                risk_approved=risk_approved,
                price_target=combined_decision['price_target'],
                stop_loss=combined_decision['stop_loss'],
                take_profit=combined_decision['take_profit'],
                urgency=combined_decision['urgency'],
                metadata={
                    'strategy_signals': len(strategy_signals),
                    'ml_predictions': len(ml_predictions),
                    'market_regime': market_condition.regime.value,
                    'execution_time_ms': (time.time() - start_time) * 1000,
                    'risk_message': risk_message,
                    'adaptive_ma': adaptive_ma,
                    'adaptive_rsi': adaptive_rsi,
                    'bollinger_bands': adaptive_bb
                },
                timestamp=datetime.now()
            )
            
            return decision
        
        except Exception as e:
            print(f"❌ Decision making error for {symbol}: {e}")
            return None
    
    def _combine_signals(self, strategy_signals: List[TradeSignal], 
                        ml_predictions: List[MLPrediction],
                        market_condition: MarketCondition,
                        adaptive_ma: float, adaptive_rsi: float, 
                        adaptive_bb: Tuple, current_price: float) -> Optional[Dict]:
        """Combine all signals into final decision"""
        try:
            if not strategy_signals:
                return None
            
            # Weight different signal sources
            signal_weights = {
                StrategyType.HFT: 0.3,
                StrategyType.SCALPING: 0.25,
                StrategyType.ARBITRAGE: 0.2,
                StrategyType.INTRADAY: 0.15,
                StrategyType.SWING: 0.1
            }
            
            ml_weights = {
                ModelType.LSTM: 0.3,
                ModelType.CNN: 0.25,
                ModelType.TRANSFORMER: 0.25,
                ModelType.ENSEMBLE: 0.4
            }
            
            # Calculate weighted strategy score
            strategy_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            total_strategy_weight = 0
            
            for signal in strategy_signals:
                weight = signal_weights.get(signal.strategy, 0.1) * signal.confidence
                strategy_scores[signal.action] += weight
                total_strategy_weight += weight
            
            # Calculate weighted ML score
            ml_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            total_ml_weight = 0
            
            for prediction in ml_predictions:
                weight = ml_weights.get(prediction.model_type, 0.1) * prediction.confidence
                ml_scores[prediction.prediction] += weight
                total_ml_weight += weight
            
            # Adaptive indicators confirmation
            adaptive_score = 0
            if adaptive_ma and adaptive_rsi and adaptive_bb:
                upper_bb, lower_bb, middle_bb = adaptive_bb
                
                # MA trend
                if current_price > adaptive_ma:
                    adaptive_score += 1
                elif current_price < adaptive_ma:
                    adaptive_score -= 1
                
                # RSI momentum
                if adaptive_rsi < 35:  # Oversold
                    adaptive_score += 1
                elif adaptive_rsi > 65:  # Overbought
                    adaptive_score -= 1
                
                # Bollinger Bands
                if current_price < lower_bb:  # Below lower band
                    adaptive_score += 1
                elif current_price > upper_bb:  # Above upper band
                    adaptive_score -= 1
            
            # Market regime adjustment
            regime_multiplier = 1.0
            if market_condition.regime == MarketRegime.TRENDING:
                regime_multiplier = 1.2  # Amplify signals in trending market
            elif market_condition.regime == MarketRegime.VOLATILE:
                regime_multiplier = 0.8  # Reduce signals in volatile market
            elif market_condition.regime == MarketRegime.CRISIS:
                regime_multiplier = 0.3  # Very conservative in crisis
            
            # Combine scores
            final_scores = {}
            for action in ['BUY', 'SELL', 'HOLD']:
                strategy_component = strategy_scores[action] / max(total_strategy_weight, 0.1)
                ml_component = ml_scores[action] / max(total_ml_weight, 0.1) if total_ml_weight > 0 else 0
                adaptive_component = adaptive_score / 3.0 if action == 'BUY' and adaptive_score > 0 else (
                    -adaptive_score / 3.0 if action == 'SELL' and adaptive_score < 0 else 0
                )
                
                # Weight the components
                final_scores[action] = (
                    strategy_component * 0.4 +
                    ml_component * 0.4 +
                    adaptive_component * 0.2
                ) * regime_multiplier
            
            # Get final decision
            final_action = max(final_scores, key=final_scores.get)
            final_confidence = final_scores[final_action]
            
            # Check minimum confidence threshold
            if final_confidence < self.trading_config['min_confidence_threshold']:
                return None
            
            # ML confirmation check
            ml_confirmed = False
            if ml_predictions:
                ml_consensus = max(ml_scores, key=ml_scores.get)
                ml_confirmed = (ml_consensus == final_action and ml_scores[ml_consensus] > 0.3)
            
            if self.trading_config['ml_confirmation_required'] and not ml_confirmed:
                return None
            
            # Calculate targets
            volatility = market_condition.volatility
            
            if final_action == 'BUY':
                price_target = current_price * (1 + volatility * 2)
                stop_loss = current_price * (1 - volatility * 1.5)
                take_profit = current_price * (1 + volatility * 3)
            elif final_action == 'SELL':
                price_target = current_price * (1 - volatility * 2)
                stop_loss = current_price * (1 + volatility * 1.5)
                take_profit = current_price * (1 - volatility * 3)
            else:
                return None  # No HOLD actions
            
            # Determine primary strategy
            best_signal = max(strategy_signals, key=lambda x: x.confidence)
            primary_strategy = best_signal.strategy.value
            
            # Determine urgency
            urgency = 1
            if any(s.strategy == StrategyType.HFT for s in strategy_signals):
                urgency = 5
            elif any(s.strategy == StrategyType.SCALPING for s in strategy_signals):
                urgency = 4
            elif any(s.strategy == StrategyType.ARBITRAGE for s in strategy_signals):
                urgency = 3
            
            return {
                'action': final_action,
                'confidence': final_confidence,
                'ml_confirmed': ml_confirmed,
                'primary_strategy': primary_strategy,
                'price_target': price_target,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'urgency': urgency,
                'regime_multiplier': regime_multiplier,
                'adaptive_score': adaptive_score
            }
        
        except Exception as e:
            print(f"❌ Signal combination error: {e}")
            return None
    
    def _calculate_optimal_position_size(self, symbol: str, confidence: float, action: str) -> float:
        """Calculate optimal position size using multiple methods"""
        try:
            # Get historical performance for Kelly calculation
            performance = trade_logger.get_performance_summary()
            win_rate = performance.get('win_rate', 0.5)
            avg_profit = performance.get('avg_profit', 0.01)
            avg_loss = performance.get('avg_loss', 0.01)
            
            # Kelly Criterion
            balance = self.advanced_risk.get_current_balance()
            kelly_size = self.advanced_risk.calculate_position_size_kelly(
                symbol, win_rate, avg_profit, avg_loss, balance
            )
            
            # Confidence adjustment
            confidence_adjusted_size = kelly_size * confidence
            
            # Volatility adjustment
            market_condition = self.market_regime_cache.get(symbol)
            if market_condition:
                if market_condition.regime == MarketRegime.VOLATILE:
                    confidence_adjusted_size *= 0.7  # Reduce in volatile conditions
                elif market_condition.regime == MarketRegime.CRISIS:
                    confidence_adjusted_size *= 0.3  # Very small positions in crisis
            
            # Apply limits
            max_position = balance * 0.1  # Max 10% of balance
            min_position = 0.01
            
            final_size = max(min_position, min(confidence_adjusted_size, max_position))
            
            return round(final_size, 3)
        
        except Exception as e:
            print(f"❌ Position sizing error: {e}")
            return 0.01
    
    def _execute_decision(self, decision: TradingDecision):
        """Execute trading decision"""
        try:
            if decision.urgency >= 4:  # HFT/Scalping
                execution_start = time.time()
            
            # Log decision
            trade_logger.log_info(
                f"Executing {decision.action} {decision.symbol} "
                f"Vol: {decision.volume} Confidence: {decision.confidence:.2%} "
                f"Strategy: {decision.strategy_source}"
            )
            
            # Execute trade (simplified for demo)
            if self.mt5:
                # In production, this would place actual trades
                success = True  # Mock success
                ticket = f"ADV_{int(time.time())}"
                
                if success:
                    # Update portfolio tracking
                    self.advanced_risk.update_portfolio(
                        decision.symbol, decision.volume, 
                        decision.metadata.get('current_price', 0), 
                        decision.action
                    )
                    
                    # Log successful execution
                    trade_logger.log_trade_entry({
                        'symbol': decision.symbol,
                        'action': decision.action,
                        'volume': decision.volume,
                        'price': decision.metadata.get('current_price', 0),
                        'tp': decision.take_profit,
                        'sl': decision.stop_loss,
                        'confidence': decision.confidence,
                        'ticket': ticket,
                        'strategy': decision.strategy_source,
                        'ml_confirmed': decision.ml_confirmation
                    })
                    
                    # Send notification
                    telegram_notifier.notify_trade_opened(
                        decision.symbol, decision.action,
                        decision.metadata.get('current_price', 0),
                        decision.volume, decision.take_profit, decision.stop_loss
                    )
                    
                    # Update performance
                    self.performance_metrics['total_trades'] += 1
                    
                    # Track execution time for HFT
                    if decision.urgency >= 4:
                        execution_time = (time.time() - execution_start) * 1000
                        self.performance_metrics['avg_execution_time'] = (
                            (self.performance_metrics['avg_execution_time'] * 
                             (self.performance_metrics['total_trades'] - 1) + execution_time) /
                            self.performance_metrics['total_trades']
                        )
                        
                        print(f"⚡ HFT Execution: {execution_time:.2f}ms")
                    
                    print(f"✅ Trade executed: {decision.action} {decision.symbol}")
                else:
                    print(f"❌ Trade execution failed: {decision.symbol}")
            
        except Exception as e:
            print(f"❌ Execution error: {e}")
            trade_logger.log_error(f"Execution failed: {e}", "EXECUTION")
    
    def _handle_regime_changes(self):
        """Handle market regime changes"""
        try:
            for symbol, condition in self.market_regime_cache.items():
                # Check for significant regime changes
                if condition.confidence > 0.8:
                    # Adjust trading parameters based on regime
                    if condition.regime == MarketRegime.CRISIS:
                        print(f"🚨 Crisis mode detected for {symbol}")
                        # Reduce position sizes, increase stops
                        self.trading_config['min_confidence_threshold'] = 0.8
                    elif condition.regime == MarketRegime.TRENDING:
                        print(f"📈 Strong trend detected for {symbol}")
                        # Enable trend-following strategies
                        self.trading_config['min_confidence_threshold'] = 0.5
                    elif condition.regime == MarketRegime.VOLATILE:
                        print(f"🌊 High volatility detected for {symbol}")
                        # Enable scalping, reduce swing trading
                        self.trading_config['min_confidence_threshold'] = 0.65
        
        except Exception as e:
            print(f"❌ Regime handling error: {e}")
    
    def get_engine_status(self) -> Dict:
        """Get comprehensive engine status"""
        try:
            risk_summary = self.advanced_risk.get_risk_summary()
            ml_summary = self.ml_engine.get_ml_summary()
            strategy_summary = self.trading_strategies.get_strategy_summary()
            adaptation_summary = self.adaptive_indicators.get_adaptation_summary()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'engine_running': self.engine_running,
                'trading_mode': self.trading_config['mode'].value,
                'performance_metrics': self.performance_metrics,
                'active_trades': len(self.active_trades),
                'market_regimes': {
                    symbol: condition.regime.value 
                    for symbol, condition in self.market_regime_cache.items()
                },
                'risk_status': risk_summary,
                'ml_status': ml_summary,
                'strategy_status': strategy_summary,
                'adaptation_status': adaptation_summary,
                'config': self.trading_config
            }
        
        except Exception as e:
            print(f"❌ Status report error: {e}")
            return {'error': str(e)}

# Global instance
advanced_trading_engine = None  # Will be initialized with dependencies