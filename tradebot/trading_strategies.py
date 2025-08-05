"""
Advanced Trading Strategies Engine
HFT, Scalping, Arbitrage, Intraday, dan Swing Trading
"""

import time
import numpy as np
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from config import config

class StrategyType(Enum):
    HFT = "hft"
    SCALPING = "scalping"
    ARBITRAGE = "arbitrage"
    INTRADAY = "intraday"
    SWING = "swing"

@dataclass
class TradeSignal:
    strategy: StrategyType
    symbol: str
    action: str  # BUY/SELL
    confidence: float
    price: float
    volume: float
    tp: float
    sl: float
    urgency: int  # 1-5, 5 = ultra urgent (HFT)
    timestamp: datetime
    metadata: Dict

class TradingStrategies:
    def __init__(self, market_data_api, indicators, risk_manager):
        self.market_api = market_data_api
        self.indicators = indicators
        self.risk_manager = risk_manager
        
        # Strategy configurations
        self.strategies_config = {
            StrategyType.HFT: {
                'enabled': True,
                'max_execution_time_ms': 1,
                'min_profit_pips': 0.5,
                'max_trades_per_second': 10,
                'symbols': ['EURUSD', 'GBPUSD', 'USDJPY'],
                'max_position_size': 0.01
            },
            StrategyType.SCALPING: {
                'enabled': True,
                'target_trades_per_day': 300,
                'min_profit_percent': 0.001,  # 0.1%
                'max_profit_percent': 0.003,  # 0.3%
                'max_holding_time_minutes': 5,
                'symbols': ['XAUUSDm', 'EURUSD', 'GBPUSD']
            },
            StrategyType.ARBITRAGE: {
                'enabled': True,
                'min_spread_percent': 0.0002,  # 0.02%
                'max_execution_time_ms': 100,
                'symbols': ['BTCUSD', 'ETHUSD', 'XAUUSDm']
            },
            StrategyType.INTRADAY: {
                'enabled': True,
                'max_holding_time_hours': 8,
                'target_profit_percent': 0.01,  # 1%
                'max_positions': 3
            },
            StrategyType.SWING: {
                'enabled': True,
                'max_holding_time_days': 5,
                'target_profit_percent': 0.03,  # 3%
                'max_positions': 2
            }
        }
        
        # Strategy performance tracking
        self.strategy_performance = {
            strategy: {
                'total_trades': 0,
                'winning_trades': 0,
                'total_profit': 0.0,
                'avg_execution_time_ms': 0.0,
                'last_trade_time': None
            } for strategy in StrategyType
        }
        
        # Real-time execution tracking
        self.execution_times = []
        self.active_trades = {}
        self.trade_counter = 0
        
        print("🚀 Advanced Trading Strategies Engine initialized")
        print(f"   • HFT: {'✅' if self.strategies_config[StrategyType.HFT]['enabled'] else '❌'}")
        print(f"   • Scalping: {'✅' if self.strategies_config[StrategyType.SCALPING]['enabled'] else '❌'}")
        print(f"   • Arbitrage: {'✅' if self.strategies_config[StrategyType.ARBITRAGE]['enabled'] else '❌'}")
        print(f"   • Intraday: {'✅' if self.strategies_config[StrategyType.INTRADAY]['enabled'] else '❌'}")
        print(f"   • Swing: {'✅' if self.strategies_config[StrategyType.SWING]['enabled'] else '❌'}")
    
    def generate_signals(self, symbol: str) -> List[TradeSignal]:
        """Generate trading signals from all enabled strategies"""
        signals = []
        
        try:
            # Get market data
            price = self.market_api.get_price(symbol)
            if not price:
                return signals
            
            # HFT Strategy
            if self.strategies_config[StrategyType.HFT]['enabled']:
                hft_signal = self._hft_strategy(symbol, price)
                if hft_signal:
                    signals.append(hft_signal)
            
            # Scalping Strategy
            if self.strategies_config[StrategyType.SCALPING]['enabled']:
                scalping_signal = self._scalping_strategy(symbol, price)
                if scalping_signal:
                    signals.append(scalping_signal)
            
            # Arbitrage Strategy
            if self.strategies_config[StrategyType.ARBITRAGE]['enabled']:
                arbitrage_signal = self._arbitrage_strategy(symbol, price)
                if arbitrage_signal:
                    signals.append(arbitrage_signal)
            
            # Intraday Strategy
            if self.strategies_config[StrategyType.INTRADAY]['enabled']:
                intraday_signal = self._intraday_strategy(symbol, price)
                if intraday_signal:
                    signals.append(intraday_signal)
            
            # Swing Strategy
            if self.strategies_config[StrategyType.SWING]['enabled']:
                swing_signal = self._swing_strategy(symbol, price)
                if swing_signal:
                    signals.append(swing_signal)
        
        except Exception as e:
            print(f"❌ Strategy signal generation error: {e}")
        
        return signals
    
    def _hft_strategy(self, symbol: str, price: float) -> Optional[TradeSignal]:
        """High-Frequency Trading Strategy - Ultra-fast execution"""
        try:
            start_time = time.time()
            
            # Check if symbol is suitable for HFT
            if symbol not in self.strategies_config[StrategyType.HFT]['symbols']:
                return None
            
            # Get ultra-short timeframe data (1-second bars simulated)
            recent_prices = self.market_api.get_recent_prices(symbol, count=10)
            if len(recent_prices) < 5:
                return None
            
            # Ultra-fast momentum detection
            momentum = (recent_prices[-1] - recent_prices[-3]) / recent_prices[-3]
            volatility = np.std(recent_prices[-5:]) / np.mean(recent_prices[-5:])
            
            # Micro-trend analysis
            micro_trend = np.polyfit(range(5), recent_prices[-5:], 1)[0]
            
            # HFT signal conditions
            signal_strength = 0
            action = None
            
            # Momentum breakout
            if momentum > 0.0001 and micro_trend > 0:  # 0.01% momentum
                signal_strength += 3
                action = "BUY"
            elif momentum < -0.0001 and micro_trend < 0:
                signal_strength += 3
                action = "SELL"
            
            # Volatility filter
            if 0.0005 < volatility < 0.002:  # Optimal volatility range
                signal_strength += 2
            
            # Execution time check (must be <1ms)
            execution_time_ms = (time.time() - start_time) * 1000
            if execution_time_ms > self.strategies_config[StrategyType.HFT]['max_execution_time_ms']:
                return None
            
            if signal_strength >= 4 and action:
                # Ultra-tight SL and TP for HFT
                pip_value = 0.0001 if 'JPY' not in symbol else 0.01
                tp = price + (0.5 * pip_value) if action == "BUY" else price - (0.5 * pip_value)
                sl = price - (1.0 * pip_value) if action == "BUY" else price + (1.0 * pip_value)
                
                return TradeSignal(
                    strategy=StrategyType.HFT,
                    symbol=symbol,
                    action=action,
                    confidence=signal_strength / 5.0,
                    price=price,
                    volume=self.strategies_config[StrategyType.HFT]['max_position_size'],
                    tp=tp,
                    sl=sl,
                    urgency=5,  # Ultra urgent
                    timestamp=datetime.now(),
                    metadata={
                        'momentum': momentum,
                        'volatility': volatility,
                        'execution_time_ms': execution_time_ms,
                        'micro_trend': micro_trend
                    }
                )
        
        except Exception as e:
            print(f"❌ HFT strategy error: {e}")
        
        return None
    
    def _scalping_strategy(self, symbol: str, price: float) -> Optional[TradeSignal]:
        """Scalping Strategy - 300+ trades/day, 0.1-0.3% profit"""
        try:
            # Check daily trade count
            today_trades = self._count_today_trades(StrategyType.SCALPING)
            if today_trades >= self.strategies_config[StrategyType.SCALPING]['target_trades_per_day']:
                return None
            
            # Get short-term data
            prices = self.market_api.get_recent_prices(symbol, count=20)
            if len(prices) < 15:
                return None
            
            # Fast EMA crossover
            ema5 = self.indicators.calculate_ema(prices, 5)
            ema10 = self.indicators.calculate_ema(prices, 10)
            
            if not ema5 or not ema10:
                return None
            
            # RSI for overbought/oversold
            rsi = self.indicators.calculate_rsi(prices, 7)  # Fast RSI
            
            # Bollinger Bands squeeze detection
            bb_upper, bb_lower, bb_middle = self.indicators.calculate_bollinger_bands(prices, 10, 1.5)
            bb_squeeze = (bb_upper - bb_lower) / bb_middle < 0.02  # Tight bands
            
            # Scalping conditions
            signal_strength = 0
            action = None
            
            # EMA crossover
            prev_ema5 = self.indicators.calculate_ema(prices[:-1], 5)
            prev_ema10 = self.indicators.calculate_ema(prices[:-1], 10)
            
            if prev_ema5 and prev_ema10:
                if prev_ema5 <= prev_ema10 and ema5 > ema10:  # Bullish crossover
                    signal_strength += 3
                    action = "BUY"
                elif prev_ema5 >= prev_ema10 and ema5 < ema10:  # Bearish crossover
                    signal_strength += 3
                    action = "SELL"
            
            # RSI confirmation
            if action == "BUY" and rsi < 45:
                signal_strength += 2
            elif action == "SELL" and rsi > 55:
                signal_strength += 2
            
            # Volatility requirement
            volatility = np.std(prices[-10:]) / np.mean(prices[-10:])
            if volatility > 0.001:  # Minimum volatility for scalping
                signal_strength += 1
            
            if signal_strength >= 5 and action:
                # Scalping-specific TP/SL (0.1-0.3% profit)
                profit_percent = np.random.uniform(0.001, 0.003)  # 0.1-0.3%
                
                if action == "BUY":
                    tp = price * (1 + profit_percent)
                    sl = price * (1 - profit_percent * 2)  # 2:1 risk ratio
                else:
                    tp = price * (1 - profit_percent)
                    sl = price * (1 + profit_percent * 2)
                
                return TradeSignal(
                    strategy=StrategyType.SCALPING,
                    symbol=symbol,
                    action=action,
                    confidence=signal_strength / 6.0,
                    price=price,
                    volume=0.02,  # Small size for scalping
                    tp=tp,
                    sl=sl,
                    urgency=4,  # High urgency
                    timestamp=datetime.now(),
                    metadata={
                        'rsi': rsi,
                        'volatility': volatility,
                        'bb_squeeze': bb_squeeze,
                        'today_trades': today_trades,
                        'target_profit_pct': profit_percent
                    }
                )
        
        except Exception as e:
            print(f"❌ Scalping strategy error: {e}")
        
        return None
    
    def _arbitrage_strategy(self, symbol: str, price: float) -> Optional[TradeSignal]:
        """Arbitrage Strategy - Statistical dan cross-asset arbitrage"""
        try:
            # Statistical arbitrage - mean reversion
            prices = self.market_api.get_recent_prices(symbol, count=50)
            if len(prices) < 30:
                return None
            
            # Calculate statistical measures
            mean_price = np.mean(prices)
            std_price = np.std(prices)
            z_score = (price - mean_price) / std_price
            
            # Cross-asset correlation (example: Gold vs USD pairs)
            correlation_signal = 0
            if symbol == 'XAUUSDm':
                # Check inverse correlation with USD pairs
                eur_usd_price = self.market_api.get_price('EURUSD')
                if eur_usd_price:
                    eur_prices = self.market_api.get_recent_prices('EURUSD', count=20)
                    if len(eur_prices) >= 10:
                        # Simple correlation check
                        gold_changes = np.diff(prices[-10:])
                        eur_changes = np.diff(eur_prices[-10:])
                        if len(gold_changes) == len(eur_changes):
                            corr = np.corrcoef(gold_changes, eur_changes)[0, 1]
                            if corr < -0.5:  # Strong negative correlation
                                correlation_signal = 2
            
            # Arbitrage conditions
            signal_strength = 0
            action = None
            
            # Mean reversion signal
            if z_score > 2.0:  # Price too high
                signal_strength += 3
                action = "SELL"
            elif z_score < -2.0:  # Price too low
                signal_strength += 3
                action = "BUY"
            
            # Add correlation signal
            signal_strength += correlation_signal
            
            # Volume analysis for arbitrage opportunities
            recent_volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
            if recent_volatility > 0.002:  # High volatility = arbitrage opportunity
                signal_strength += 1
            
            # Minimum spread requirement
            bid_ask_spread = self.market_api.get_spread(symbol)
            min_spread = self.strategies_config[StrategyType.ARBITRAGE]['min_spread_percent']
            
            if bid_ask_spread and bid_ask_spread < min_spread:
                signal_strength += 2
            
            if signal_strength >= 4 and action and abs(z_score) > 1.5:
                # Arbitrage-specific TP/SL
                reversion_target = mean_price + (0.5 * std_price * (-1 if action == "BUY" else 1))
                
                if action == "BUY":
                    tp = reversion_target
                    sl = price * 0.995  # 0.5% stop loss
                else:
                    tp = reversion_target
                    sl = price * 1.005  # 0.5% stop loss
                
                return TradeSignal(
                    strategy=StrategyType.ARBITRAGE,
                    symbol=symbol,
                    action=action,
                    confidence=min(signal_strength / 6.0, 0.95),
                    price=price,
                    volume=0.05,  # Medium size for arbitrage
                    tp=tp,
                    sl=sl,
                    urgency=3,  # Medium urgency
                    timestamp=datetime.now(),
                    metadata={
                        'z_score': z_score,
                        'mean_price': mean_price,
                        'std_price': std_price,
                        'correlation_signal': correlation_signal,
                        'bid_ask_spread': bid_ask_spread,
                        'volatility': recent_volatility
                    }
                )
        
        except Exception as e:
            print(f"❌ Arbitrage strategy error: {e}")
        
        return None
    
    def _intraday_strategy(self, symbol: str, price: float) -> Optional[TradeSignal]:
        """Intraday Strategy - Medium-term positions"""
        try:
            # Check current positions for this strategy
            active_positions = self._count_active_positions(StrategyType.INTRADAY)
            if active_positions >= self.strategies_config[StrategyType.INTRADAY]['max_positions']:
                return None
            
            # Get hourly-equivalent data
            prices = self.market_api.get_recent_prices(symbol, count=100)
            if len(prices) < 50:
                return None
            
            # Medium-term indicators
            sma20 = self.indicators.calculate_ma(prices, 20)
            sma50 = self.indicators.calculate_ma(prices, 50)
            rsi = self.indicators.calculate_rsi(prices, 14)
            macd = self.indicators.calculate_macd(prices)
            
            if not all([sma20, sma50, rsi, macd]):
                return None
            
            # Trend analysis
            trend_strength = (sma20 - sma50) / sma50
            momentum = macd.get('histogram', 0)
            
            signal_strength = 0
            action = None
            
            # Trend following
            if trend_strength > 0.005 and momentum > 0:  # Strong uptrend
                signal_strength += 3
                action = "BUY"
            elif trend_strength < -0.005 and momentum < 0:  # Strong downtrend
                signal_strength += 3
                action = "SELL"
            
            # RSI confirmation
            if action == "BUY" and 30 < rsi < 60:
                signal_strength += 2
            elif action == "SELL" and 40 < rsi < 70:
                signal_strength += 2
            
            # Volume confirmation (simulated)
            volume_trend = np.mean(prices[-5:]) / np.mean(prices[-20:])
            if 1.01 < volume_trend < 1.05:  # Healthy growth
                signal_strength += 1
            
            if signal_strength >= 5 and action:
                # Intraday-specific TP/SL (1% target)
                target_percent = self.strategies_config[StrategyType.INTRADAY]['target_profit_percent']
                
                if action == "BUY":
                    tp = price * (1 + target_percent)
                    sl = price * (1 - target_percent * 0.5)  # 0.5% SL
                else:
                    tp = price * (1 - target_percent)
                    sl = price * (1 + target_percent * 0.5)
                
                return TradeSignal(
                    strategy=StrategyType.INTRADAY,
                    symbol=symbol,
                    action=action,
                    confidence=signal_strength / 6.0,
                    price=price,
                    volume=0.1,  # Larger size for intraday
                    tp=tp,
                    sl=sl,
                    urgency=2,  # Medium-low urgency
                    timestamp=datetime.now(),
                    metadata={
                        'trend_strength': trend_strength,
                        'rsi': rsi,
                        'macd': macd,
                        'volume_trend': volume_trend,
                        'active_positions': active_positions
                    }
                )
        
        except Exception as e:
            print(f"❌ Intraday strategy error: {e}")
        
        return None
    
    def _swing_strategy(self, symbol: str, price: float) -> Optional[TradeSignal]:
        """Swing Strategy - Longer-term positions"""
        try:
            # Check current positions
            active_positions = self._count_active_positions(StrategyType.SWING)
            if active_positions >= self.strategies_config[StrategyType.SWING]['max_positions']:
                return None
            
            # Get longer-term data
            prices = self.market_api.get_recent_prices(symbol, count=200)
            if len(prices) < 100:
                return None
            
            # Long-term indicators
            sma50 = self.indicators.calculate_ma(prices, 50)
            sma100 = self.indicators.calculate_ma(prices, 100)
            rsi = self.indicators.calculate_rsi(prices, 21)
            
            # Support/Resistance levels
            highs = [max(prices[i:i+10]) for i in range(0, len(prices)-10, 10)]
            lows = [min(prices[i:i+10]) for i in range(0, len(prices)-10, 10)]
            
            resistance = np.percentile(highs, 90)
            support = np.percentile(lows, 10)
            
            signal_strength = 0
            action = None
            
            # Major trend identification
            long_term_trend = (sma50 - sma100) / sma100
            
            # Swing conditions
            if long_term_trend > 0.01 and price > sma50:  # Strong uptrend
                if price < resistance * 0.98:  # Not at resistance
                    signal_strength += 4
                    action = "BUY"
            elif long_term_trend < -0.01 and price < sma50:  # Strong downtrend
                if price > support * 1.02:  # Not at support
                    signal_strength += 4
                    action = "SELL"
            
            # RSI confirmation for swing
            if action == "BUY" and rsi < 50:
                signal_strength += 2
            elif action == "SELL" and rsi > 50:
                signal_strength += 2
            
            if signal_strength >= 5 and action:
                # Swing-specific TP/SL (3% target)
                target_percent = self.strategies_config[StrategyType.SWING]['target_profit_percent']
                
                if action == "BUY":
                    tp = min(price * (1 + target_percent), resistance)
                    sl = max(price * (1 - target_percent * 0.7), support)
                else:
                    tp = max(price * (1 - target_percent), support)
                    sl = min(price * (1 + target_percent * 0.7), resistance)
                
                return TradeSignal(
                    strategy=StrategyType.SWING,
                    symbol=symbol,
                    action=action,
                    confidence=signal_strength / 6.0,
                    price=price,
                    volume=0.15,  # Larger size for swing
                    tp=tp,
                    sl=sl,
                    urgency=1,  # Low urgency
                    timestamp=datetime.now(),
                    metadata={
                        'long_term_trend': long_term_trend,
                        'rsi': rsi,
                        'resistance': resistance,
                        'support': support,
                        'active_positions': active_positions
                    }
                )
        
        except Exception as e:
            print(f"❌ Swing strategy error: {e}")
        
        return None
    
    def _count_today_trades(self, strategy: StrategyType) -> int:
        """Count trades made today for specific strategy"""
        # In real implementation, this would query trade history
        return self.strategy_performance[strategy]['total_trades']
    
    def _count_active_positions(self, strategy: StrategyType) -> int:
        """Count active positions for specific strategy"""
        count = 0
        for trade_id, trade_info in self.active_trades.items():
            if trade_info.get('strategy') == strategy:
                count += 1
        return count
    
    def update_strategy_performance(self, strategy: StrategyType, profit: float, execution_time_ms: float):
        """Update strategy performance metrics"""
        try:
            perf = self.strategy_performance[strategy]
            perf['total_trades'] += 1
            perf['total_profit'] += profit
            
            if profit > 0:
                perf['winning_trades'] += 1
            
            # Update average execution time
            perf['avg_execution_time_ms'] = (
                (perf['avg_execution_time_ms'] * (perf['total_trades'] - 1) + execution_time_ms) 
                / perf['total_trades']
            )
            
            perf['last_trade_time'] = datetime.now()
            
            # Calculate win rate
            win_rate = perf['winning_trades'] / perf['total_trades'] if perf['total_trades'] > 0 else 0
            
            print(f"📊 {strategy.value.upper()} Performance: "
                  f"Trades: {perf['total_trades']}, "
                  f"Win Rate: {win_rate:.1%}, "
                  f"Total P&L: ${perf['total_profit']:.2f}, "
                  f"Avg Exec: {perf['avg_execution_time_ms']:.2f}ms")
            
        except Exception as e:
            print(f"❌ Performance update error: {e}")
    
    def get_strategy_summary(self) -> Dict:
        """Get comprehensive strategy performance summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'strategies': {}
        }
        
        for strategy, perf in self.strategy_performance.items():
            win_rate = perf['winning_trades'] / perf['total_trades'] if perf['total_trades'] > 0 else 0
            avg_profit = perf['total_profit'] / perf['total_trades'] if perf['total_trades'] > 0 else 0
            
            summary['strategies'][strategy.value] = {
                'enabled': self.strategies_config[strategy]['enabled'],
                'total_trades': perf['total_trades'],
                'winning_trades': perf['winning_trades'],
                'win_rate': win_rate,
                'total_profit': perf['total_profit'],
                'avg_profit_per_trade': avg_profit,
                'avg_execution_time_ms': perf['avg_execution_time_ms'],
                'last_trade': perf['last_trade_time'].isoformat() if perf['last_trade_time'] else None
            }
        
        return summary

# Global instance
trading_strategies = None  # Will be initialized with dependencies