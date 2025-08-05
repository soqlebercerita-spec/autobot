"""
High-Frequency Trading Strategy for AuraTrade Bot
Ultra-fast execution strategy for scalping and market making
"""

import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import threading

from config.config import config
from config.settings import settings
from core.mt5_connector import MT5Connector
from core.order_manager import OrderManager
from core.risk_manager import RiskManager
from analysis.technical_analysis import TechnicalAnalysis
from utils.logger import Logger

class TickData:
    """Tick data structure for HFT"""
    def __init__(self, symbol: str, bid: float, ask: float, volume: float, timestamp: datetime):
        self.symbol = symbol
        self.bid = bid
        self.ask = ask
        self.volume = volume
        self.timestamp = timestamp
        self.spread = ask - bid
        self.mid_price = (bid + ask) / 2.0

class OrderBookLevel:
    """Order book level data"""
    def __init__(self, price: float, volume: float, orders: int = 1):
        self.price = price
        self.volume = volume
        self.orders = orders

class HFTStrategy:
    """High-Frequency Trading Strategy Implementation"""
    
    def __init__(self, mt5_connector: MT5Connector, order_manager: OrderManager, risk_manager: RiskManager):
        self.mt5_connector = mt5_connector
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        self.logger = Logger()
        self.technical_analysis = TechnicalAnalysis()
        
        # HFT Configuration
        self.enabled = settings.hft.enabled
        self.max_execution_time = settings.hft.max_execution_time_ms / 1000.0  # Convert to seconds
        self.order_book_depth = settings.hft.order_book_depth
        self.latency_threshold = settings.hft.latency_threshold_ms / 1000.0
        
        # Strategy parameters
        self.min_spread_threshold = 0.5  # Minimum spread in pips
        self.max_spread_threshold = 3.0  # Maximum spread in pips
        self.volume_threshold = 100  # Minimum volume for execution
        self.profit_target_pips = 0.5  # Minimum profit target
        self.max_position_hold_time = 30  # Maximum seconds to hold position
        
        # Market microstructure data
        self.tick_buffer = {}  # symbol -> deque of TickData
        self.buffer_size = 1000
        self.order_book_data = {}  # symbol -> order book
        self.market_impact_data = {}
        
        # Strategy state
        self.active_positions = {}
        self.pending_orders = {}
        self.execution_times = deque(maxlen=1000)
        self.pnl_tracking = deque(maxlen=10000)
        
        # Performance metrics
        self.trades_today = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.avg_execution_time = 0.0
        self.max_drawdown = 0.0
        
        # Threading for real-time processing
        self.processing_thread = None
        self.stop_processing = False
        
        self.logger.info("HFT Strategy initialized")
    
    def is_enabled(self) -> bool:
        """Check if HFT strategy is enabled"""
        return self.enabled and settings.hft.enabled
    
    def start(self):
        """Start HFT processing thread"""
        if not self.is_enabled():
            return
        
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self._hft_processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("HFT Strategy started")
    
    def stop(self):
        """Stop HFT processing"""
        self.stop_processing = True
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        self.logger.info("HFT Strategy stopped")
    
    def generate_signals(self) -> List[Dict[str, Any]]:
        """Generate HFT trading signals"""
        if not self.is_enabled():
            return []
        
        signals = []
        
        for symbol in settings.trading.enabled_symbols:
            try:
                # Get latest market data
                tick = self.mt5_connector.get_tick(symbol)
                if not tick:
                    continue
                
                # Update tick buffer
                self._update_tick_buffer(symbol, tick)
                
                # Check for HFT opportunities
                hft_signals = self._analyze_hft_opportunities(symbol)
                signals.extend(hft_signals)
                
            except Exception as e:
                self.logger.error(f"Error generating HFT signals for {symbol}: {str(e)}")
        
        return signals
    
    def _hft_processing_loop(self):
        """Main HFT processing loop for ultra-fast execution"""
        while not self.stop_processing:
            start_time = time.time()
            
            try:
                # Process each enabled symbol
                for symbol in settings.trading.enabled_symbols:
                    self._process_symbol_hft(symbol)
                
                # Manage existing positions
                self._manage_hft_positions()
                
                # Clean up old data
                self._cleanup_old_data()
                
            except Exception as e:
                self.logger.error(f"HFT processing error: {str(e)}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Sleep for remaining time to maintain frequency
            target_cycle_time = 0.001  # 1ms target cycle time
            if processing_time < target_cycle_time:
                time.sleep(target_cycle_time - processing_time)
    
    def _process_symbol_hft(self, symbol: str):
        """Process HFT opportunities for a single symbol"""
        try:
            # Get latest tick
            tick = self.mt5_connector.get_tick(symbol)
            if not tick:
                return
            
            # Update market data
            self._update_tick_buffer(symbol, tick)
            self._update_order_book(symbol, tick)
            
            # Check for scalping opportunities
            if settings.hft.tick_scalping_enabled:
                self._check_tick_scalping(symbol, tick)
            
            # Check for market making opportunities
            if settings.hft.market_making_enabled:
                self._check_market_making(symbol, tick)
            
            # Check for arbitrage opportunities
            if settings.hft.arbitrage_enabled:
                self._check_arbitrage(symbol, tick)
            
        except Exception as e:
            self.logger.error(f"Error processing HFT for {symbol}: {str(e)}")
    
    def _update_tick_buffer(self, symbol: str, tick: Any):
        """Update tick data buffer"""
        if symbol not in self.tick_buffer:
            self.tick_buffer[symbol] = deque(maxlen=self.buffer_size)
        
        tick_data = TickData(
            symbol=symbol,
            bid=tick.bid,
            ask=tick.ask,
            volume=getattr(tick, 'volume', 0),
            timestamp=datetime.fromtimestamp(tick.time)
        )
        
        self.tick_buffer[symbol].append(tick_data)
    
    def _update_order_book(self, symbol: str, tick: Any):
        """Update order book data (simplified)"""
        if symbol not in self.order_book_data:
            self.order_book_data[symbol] = {
                'bids': [],
                'asks': [],
                'last_update': datetime.now()
            }
        
        # Simplified order book update
        # In real implementation, this would connect to Level II data
        spread = tick.ask - tick.bid
        
        self.order_book_data[symbol] = {
            'bids': [OrderBookLevel(tick.bid, 100)],  # Simplified
            'asks': [OrderBookLevel(tick.ask, 100)],  # Simplified
            'spread': spread,
            'mid_price': (tick.bid + tick.ask) / 2.0,
            'last_update': datetime.now()
        }
    
    def _check_tick_scalping(self, symbol: str, tick: Any):
        """Check for tick scalping opportunities"""
        try:
            if symbol not in self.tick_buffer or len(self.tick_buffer[symbol]) < 10:
                return
            
            ticks = list(self.tick_buffer[symbol])[-10:]  # Last 10 ticks
            
            # Calculate price momentum
            price_changes = []
            for i in range(1, len(ticks)):
                price_change = ticks[i].mid_price - ticks[i-1].mid_price
                price_changes.append(price_change)
            
            if not price_changes:
                return
            
            recent_momentum = sum(price_changes[-3:]) if len(price_changes) >= 3 else 0
            pip_size = self._get_pip_size(symbol)
            momentum_pips = recent_momentum / pip_size
            
            # Check for scalping opportunity
            current_spread = tick.ask - tick.bid
            spread_pips = current_spread / pip_size
            
            # Scalping conditions
            if (abs(momentum_pips) > 0.3 and  # Minimum momentum
                spread_pips < self.max_spread_threshold and  # Acceptable spread
                spread_pips > self.min_spread_threshold):  # Minimum spread
                
                # Determine direction
                action = 'BUY' if momentum_pips > 0 else 'SELL'
                
                # Calculate lot size
                lot_size = self._calculate_hft_lot_size(symbol, spread_pips)
                
                # Create scalping signal
                signal = {
                    'strategy': 'hft_scalping',
                    'symbol': symbol,
                    'action': action,
                    'lot_size': lot_size,
                    'price': tick.ask if action == 'BUY' else tick.bid,
                    'stop_loss': self._calculate_scalping_stop_loss(symbol, tick, action),
                    'take_profit': self._calculate_scalping_take_profit(symbol, tick, action),
                    'urgency': 'immediate',  # HFT urgency
                    'max_execution_time': self.max_execution_time,
                    'reason': f'Tick scalping: {momentum_pips:.1f} pip momentum'
                }
                
                # Execute immediately if conditions are met
                if self._validate_hft_signal(signal):
                    self._execute_hft_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error in tick scalping: {str(e)}")
    
    def _check_market_making(self, symbol: str, tick: Any):
        """Check for market making opportunities"""
        try:
            if symbol not in self.order_book_data:
                return
            
            order_book = self.order_book_data[symbol]
            spread = order_book['spread']
            pip_size = self._get_pip_size(symbol)
            spread_pips = spread / pip_size
            
            # Market making conditions
            if (spread_pips > 1.0 and  # Sufficient spread for market making
                spread_pips < 5.0):    # Not too wide
                
                # Calculate optimal bid/ask prices
                mid_price = order_book['mid_price']
                optimal_spread = max(spread * 0.3, pip_size * 0.5)  # Tighter spread
                
                bid_price = mid_price - optimal_spread / 2
                ask_price = mid_price + optimal_spread / 2
                
                lot_size = self._calculate_market_making_lot_size(symbol, spread_pips)
                
                # Create market making orders (both sides)
                buy_signal = {
                    'strategy': 'hft_market_making',
                    'symbol': symbol,
                    'action': 'BUY',
                    'lot_size': lot_size,
                    'price': bid_price,
                    'order_type': 'limit',
                    'take_profit': bid_price + (pip_size * self.profit_target_pips),
                    'urgency': 'normal',
                    'reason': f'Market making bid at {bid_price}'
                }
                
                sell_signal = {
                    'strategy': 'hft_market_making',
                    'symbol': symbol,
                    'action': 'SELL',
                    'lot_size': lot_size,
                    'price': ask_price,
                    'order_type': 'limit',
                    'take_profit': ask_price - (pip_size * self.profit_target_pips),
                    'urgency': 'normal',
                    'reason': f'Market making ask at {ask_price}'
                }
                
                # Execute market making orders
                if self._validate_hft_signal(buy_signal):
                    self._execute_hft_signal(buy_signal)
                
                if self._validate_hft_signal(sell_signal):
                    self._execute_hft_signal(sell_signal)
            
        except Exception as e:
            self.logger.error(f"Error in market making: {str(e)}")
    
    def _check_arbitrage(self, symbol: str, tick: Any):
        """Check for arbitrage opportunities (simplified)"""
        try:
            # In real implementation, this would compare prices across multiple brokers/exchanges
            # For now, implement statistical arbitrage based on price divergence
            
            if symbol not in self.tick_buffer or len(self.tick_buffer[symbol]) < 50:
                return
            
            ticks = list(self.tick_buffer[symbol])[-50:]
            
            # Calculate moving averages for mean reversion
            prices = [t.mid_price for t in ticks]
            short_ma = np.mean(prices[-5:])   # Very short average
            long_ma = np.mean(prices[-20:])   # Short average
            
            current_price = tick.bid + tick.ask / 2
            pip_size = self._get_pip_size(symbol)
            
            # Price divergence from moving average
            divergence_pips = (current_price - long_ma) / pip_size
            momentum_pips = (short_ma - long_ma) / pip_size
            
            # Arbitrage conditions (mean reversion)
            if (abs(divergence_pips) > 2.0 and  # Significant divergence
                abs(momentum_pips) > 1.0):       # Momentum confirmation
                
                # Determine reversion direction
                action = 'SELL' if divergence_pips > 0 else 'BUY'  # Bet on reversion
                
                lot_size = self._calculate_arbitrage_lot_size(symbol, abs(divergence_pips))
                
                signal = {
                    'strategy': 'hft_arbitrage',
                    'symbol': symbol,
                    'action': action,
                    'lot_size': lot_size,
                    'price': tick.ask if action == 'BUY' else tick.bid,
                    'stop_loss': self._calculate_arbitrage_stop_loss(symbol, tick, action, divergence_pips),
                    'take_profit': self._calculate_arbitrage_take_profit(symbol, tick, action, divergence_pips),
                    'urgency': 'high',
                    'reason': f'Statistical arbitrage: {divergence_pips:.1f} pip divergence'
                }
                
                if self._validate_hft_signal(signal):
                    self._execute_hft_signal(signal)
            
        except Exception as e:
            self.logger.error(f"Error in arbitrage check: {str(e)}")
    
    def _analyze_hft_opportunities(self, symbol: str) -> List[Dict[str, Any]]:
        """Analyze HFT opportunities and return signals"""
        signals = []
        
        try:
            # Get recent tick data
            if symbol not in self.tick_buffer or len(self.tick_buffer[symbol]) < 5:
                return signals
            
            recent_ticks = list(self.tick_buffer[symbol])[-5:]
            current_tick = recent_ticks[-1]
            
            # Order book imbalance analysis
            imbalance_signal = self._analyze_order_book_imbalance(symbol, current_tick)
            if imbalance_signal:
                signals.append(imbalance_signal)
            
            # Momentum scalping
            momentum_signal = self._analyze_momentum_scalping(symbol, recent_ticks)
            if momentum_signal:
                signals.append(momentum_signal)
            
            # Spread compression opportunities
            spread_signal = self._analyze_spread_compression(symbol, recent_ticks)
            if spread_signal:
                signals.append(spread_signal)
            
        except Exception as e:
            self.logger.error(f"Error analyzing HFT opportunities: {str(e)}")
        
        return signals
    
    def _analyze_order_book_imbalance(self, symbol: str, tick: TickData) -> Optional[Dict[str, Any]]:
        """Analyze order book imbalance for trading opportunities"""
        try:
            if symbol not in self.order_book_data:
                return None
            
            order_book = self.order_book_data[symbol]
            
            # Calculate bid/ask volume imbalance (simplified)
            bid_volume = sum(level.volume for level in order_book.get('bids', []))
            ask_volume = sum(level.volume for level in order_book.get('asks', []))
            
            if bid_volume + ask_volume == 0:
                return None
            
            imbalance_ratio = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Significant imbalance threshold
            if abs(imbalance_ratio) > 0.3:
                action = 'BUY' if imbalance_ratio > 0 else 'SELL'
                
                return {
                    'strategy': 'hft_imbalance',
                    'symbol': symbol,
                    'action': action,
                    'lot_size': settings.trading.default_lot_size,
                    'price': tick.ask if action == 'BUY' else tick.bid,
                    'confidence': abs(imbalance_ratio),
                    'urgency': 'immediate',
                    'reason': f'Order book imbalance: {imbalance_ratio:.2f}'
                }
            
        except Exception as e:
            self.logger.error(f"Error analyzing order book imbalance: {str(e)}")
        
        return None
    
    def _analyze_momentum_scalping(self, symbol: str, ticks: List[TickData]) -> Optional[Dict[str, Any]]:
        """Analyze momentum for scalping opportunities"""
        try:
            if len(ticks) < 3:
                return None
            
            # Calculate price momentum
            price_changes = []
            for i in range(1, len(ticks)):
                change = ticks[i].mid_price - ticks[i-1].mid_price
                price_changes.append(change)
            
            momentum = sum(price_changes[-2:])  # Last 2 price changes
            pip_size = self._get_pip_size(symbol)
            momentum_pips = momentum / pip_size
            
            # Momentum threshold
            if abs(momentum_pips) > 0.5:
                action = 'BUY' if momentum_pips > 0 else 'SELL'
                current_tick = ticks[-1]
                
                return {
                    'strategy': 'hft_momentum',
                    'symbol': symbol,
                    'action': action,
                    'lot_size': settings.trading.default_lot_size,
                    'price': current_tick.ask if action == 'BUY' else current_tick.bid,
                    'stop_loss': self._calculate_momentum_stop_loss(symbol, current_tick, action),
                    'take_profit': self._calculate_momentum_take_profit(symbol, current_tick, action),
                    'urgency': 'high',
                    'reason': f'Momentum scalping: {momentum_pips:.2f} pips'
                }
            
        except Exception as e:
            self.logger.error(f"Error analyzing momentum scalping: {str(e)}")
        
        return None
    
    def _analyze_spread_compression(self, symbol: str, ticks: List[TickData]) -> Optional[Dict[str, Any]]:
        """Analyze spread compression opportunities"""
        try:
            if len(ticks) < 3:
                return None
            
            current_spread = ticks[-1].spread
            avg_spread = np.mean([t.spread for t in ticks])
            pip_size = self._get_pip_size(symbol)
            
            spread_compression = (avg_spread - current_spread) / pip_size
            
            # Significant spread compression
            if spread_compression > 0.3:  # Spread compressed by 0.3 pips
                # Trade in direction of likely expansion
                current_tick = ticks[-1]
                
                # Simple direction determination
                action = 'BUY'  # Simplified logic
                
                return {
                    'strategy': 'hft_spread_compression',
                    'symbol': symbol,
                    'action': action,
                    'lot_size': settings.trading.default_lot_size * 0.5,  # Smaller size
                    'price': current_tick.ask if action == 'BUY' else current_tick.bid,
                    'urgency': 'normal',
                    'reason': f'Spread compression: {spread_compression:.2f} pips'
                }
            
        except Exception as e:
            self.logger.error(f"Error analyzing spread compression: {str(e)}")
        
        return None
    
    def _validate_hft_signal(self, signal: Dict[str, Any]) -> bool:
        """Validate HFT signal before execution"""
        try:
            # Check if we already have position in this symbol
            symbol = signal['symbol']
            if symbol in self.active_positions:
                return False
            
            # Check spread conditions
            tick = self.mt5_connector.get_tick(symbol)
            if not tick:
                return False
            
            spread = tick.ask - tick.bid
            pip_size = self._get_pip_size(symbol)
            spread_pips = spread / pip_size
            
            if spread_pips > self.max_spread_threshold:
                return False
            
            # Check risk limits
            risk_check, _ = self.risk_manager.validate_risk_limits(signal)
            if not risk_check:
                return False
            
            # Check execution time constraints
            if signal.get('urgency') == 'immediate':
                # For immediate execution, check if we can meet timing requirements
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating HFT signal: {str(e)}")
            return False
    
    def _execute_hft_signal(self, signal: Dict[str, Any]):
        """Execute HFT signal with ultra-fast execution"""
        try:
            execution_start = time.time()
            
            # Execute order
            result = self.order_manager.place_order(signal)
            
            execution_time = (time.time() - execution_start) * 1000  # Convert to milliseconds
            self.execution_times.append(execution_time)
            
            if result and result.retcode == 10009:  # Success
                # Track position
                self.active_positions[signal['symbol']] = {
                    'ticket': result.order,
                    'entry_time': datetime.now(),
                    'signal': signal,
                    'execution_time_ms': execution_time
                }
                
                self.trades_today += 1
                
                # Log execution
                self.logger.info(f"HFT order executed: {signal['symbol']} {signal['action']} "
                               f"in {execution_time:.2f}ms")
                
                # Check if execution time meets HFT standards
                if execution_time > self.max_execution_time * 1000:
                    self.logger.warning(f"Execution time {execution_time:.2f}ms exceeds "
                                      f"HFT threshold {self.max_execution_time * 1000:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error executing HFT signal: {str(e)}")
    
    def _manage_hft_positions(self):
        """Manage existing HFT positions"""
        try:
            current_time = datetime.now()
            positions_to_close = []
            
            for symbol, position_info in self.active_positions.items():
                entry_time = position_info['entry_time']
                hold_time = (current_time - entry_time).total_seconds()
                
                # Check if position should be closed
                should_close = False
                close_reason = ""
                
                # Maximum hold time exceeded
                if hold_time > self.max_position_hold_time:
                    should_close = True
                    close_reason = "Max hold time exceeded"
                
                # Check profit target or stop loss
                current_positions = self.mt5_connector.get_positions(symbol)
                for pos in current_positions:
                    if pos.ticket == position_info['ticket']:
                        # Simple profit/loss check
                        if pos.profit > 0:  # Any profit - take it quickly in HFT
                            should_close = True
                            close_reason = "Profit target reached"
                        elif pos.profit < -10:  # Small loss threshold
                            should_close = True
                            close_reason = "Stop loss triggered"
                
                if should_close:
                    positions_to_close.append((symbol, position_info, close_reason))
            
            # Close positions
            for symbol, position_info, reason in positions_to_close:
                self._close_hft_position(symbol, position_info, reason)
            
        except Exception as e:
            self.logger.error(f"Error managing HFT positions: {str(e)}")
    
    def _close_hft_position(self, symbol: str, position_info: Dict[str, Any], reason: str):
        """Close HFT position"""
        try:
            close_start = time.time()
            
            result = self.order_manager.close_position(position_info['ticket'])
            
            close_time = (time.time() - close_start) * 1000
            
            if result and result.retcode == 10009:
                # Calculate PnL and update statistics
                positions = self.mt5_connector.get_positions()
                for pos in positions:
                    if pos.ticket == position_info['ticket']:
                        pnl = pos.profit
                        self.total_pnl += pnl
                        self.pnl_tracking.append(pnl)
                        
                        if pnl > 0:
                            self.winning_trades += 1
                        
                        break
                
                # Remove from active positions
                del self.active_positions[symbol]
                
                self.logger.info(f"HFT position closed: {symbol} - {reason} "
                               f"(Close time: {close_time:.2f}ms)")
            
        except Exception as e:
            self.logger.error(f"Error closing HFT position: {str(e)}")
    
    def _cleanup_old_data(self):
        """Clean up old tick data and position info"""
        try:
            current_time = datetime.now()
            cleanup_threshold = current_time - timedelta(minutes=5)
            
            # Clean tick buffers
            for symbol in self.tick_buffer:
                # Remove old ticks
                while (self.tick_buffer[symbol] and 
                       self.tick_buffer[symbol][0].timestamp < cleanup_threshold):
                    self.tick_buffer[symbol].popleft()
            
            # Clean order book data
            for symbol in list(self.order_book_data.keys()):
                if self.order_book_data[symbol]['last_update'] < cleanup_threshold:
                    del self.order_book_data[symbol]
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
    
    # Helper methods for calculations
    def _get_pip_size(self, symbol: str) -> float:
        """Get pip size for symbol"""
        if 'JPY' in symbol:
            return 0.01
        else:
            return 0.0001
    
    def _calculate_hft_lot_size(self, symbol: str, spread_pips: float) -> float:
        """Calculate lot size for HFT trades"""
        base_lot = settings.trading.default_lot_size
        
        # Adjust based on spread
        if spread_pips < 1.0:
            return base_lot * 0.5  # Smaller size for tight spreads
        elif spread_pips > 2.0:
            return base_lot * 0.3  # Even smaller for wide spreads
        else:
            return base_lot
    
    def _calculate_market_making_lot_size(self, symbol: str, spread_pips: float) -> float:
        """Calculate lot size for market making"""
        return settings.trading.default_lot_size * 0.5  # Conservative size
    
    def _calculate_arbitrage_lot_size(self, symbol: str, divergence_pips: float) -> float:
        """Calculate lot size for arbitrage trades"""
        base_lot = settings.trading.default_lot_size
        
        # Larger size for larger divergence
        if divergence_pips > 3.0:
            return base_lot * 1.5
        else:
            return base_lot
    
    def _calculate_scalping_stop_loss(self, symbol: str, tick: Any, action: str) -> float:
        """Calculate stop loss for scalping"""
        pip_size = self._get_pip_size(symbol)
        stop_distance = pip_size * 3  # 3 pip stop loss
        
        if action == 'BUY':
            return tick.bid - stop_distance
        else:
            return tick.ask + stop_distance
    
    def _calculate_scalping_take_profit(self, symbol: str, tick: Any, action: str) -> float:
        """Calculate take profit for scalping"""
        pip_size = self._get_pip_size(symbol)
        profit_distance = pip_size * 2  # 2 pip take profit
        
        if action == 'BUY':
            return tick.ask + profit_distance
        else:
            return tick.bid - profit_distance
    
    def _calculate_arbitrage_stop_loss(self, symbol: str, tick: Any, action: str, divergence_pips: float) -> float:
        """Calculate stop loss for arbitrage"""
        pip_size = self._get_pip_size(symbol)
        stop_distance = pip_size * max(2, abs(divergence_pips) * 0.5)
        
        if action == 'BUY':
            return tick.bid - stop_distance
        else:
            return tick.ask + stop_distance
    
    def _calculate_arbitrage_take_profit(self, symbol: str, tick: Any, action: str, divergence_pips: float) -> float:
        """Calculate take profit for arbitrage"""
        pip_size = self._get_pip_size(symbol)
        profit_distance = pip_size * min(abs(divergence_pips) * 0.7, 5)  # Max 5 pips
        
        if action == 'BUY':
            return tick.ask + profit_distance
        else:
            return tick.bid - profit_distance
    
    def _calculate_momentum_stop_loss(self, symbol: str, tick: TickData, action: str) -> float:
        """Calculate stop loss for momentum trades"""
        pip_size = self._get_pip_size(symbol)
        stop_distance = pip_size * 2.5
        
        if action == 'BUY':
            return tick.bid - stop_distance
        else:
            return tick.ask + stop_distance
    
    def _calculate_momentum_take_profit(self, symbol: str, tick: TickData, action: str) -> float:
        """Calculate take profit for momentum trades"""
        pip_size = self._get_pip_size(symbol)
        profit_distance = pip_size * 1.5
        
        if action == 'BUY':
            return tick.ask + profit_distance
        else:
            return tick.bid - profit_distance
    
    def get_hft_performance(self) -> Dict[str, Any]:
        """Get HFT strategy performance metrics"""
        try:
            win_rate = (self.winning_trades / self.trades_today) * 100 if self.trades_today > 0 else 0
            
            avg_execution_time = np.mean(self.execution_times) if self.execution_times else 0
            max_execution_time = np.max(self.execution_times) if self.execution_times else 0
            
            hft_compliant_trades = sum(1 for t in self.execution_times if t < self.max_execution_time * 1000)
            hft_compliance_rate = (hft_compliant_trades / len(self.execution_times)) * 100 if self.execution_times else 0
            
            return {
                'trades_today': self.trades_today,
                'winning_trades': self.winning_trades,
                'win_rate': win_rate,
                'total_pnl': self.total_pnl,
                'avg_execution_time_ms': avg_execution_time,
                'max_execution_time_ms': max_execution_time,
                'hft_compliance_rate': hft_compliance_rate,
                'active_positions': len(self.active_positions),
                'avg_hold_time_seconds': self.max_position_hold_time,
                'tick_buffer_sizes': {symbol: len(buffer) for symbol, buffer in self.tick_buffer.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error getting HFT performance: {str(e)}")
            return {}
    
    def update_state(self):
        """Update strategy state"""
        pass  # State updated in real-time processing loop
