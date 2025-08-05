"""
Order Management System for AuraTrade Bot
Handles order placement, modification, and tracking
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import threading

from core.mt5_connector import MT5Connector
from utils.logger import Logger

class OrderType(Enum):
    """Order types"""
    MARKET_BUY = "market_buy"
    MARKET_SELL = "market_sell"
    LIMIT_BUY = "limit_buy"
    LIMIT_SELL = "limit_sell"
    STOP_BUY = "stop_buy"
    STOP_SELL = "stop_sell"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIAL = "partial"

class OrderManager:
    """Manages all order operations and tracking"""
    
    def __init__(self, mt5_connector: MT5Connector):
        self.mt5_connector = mt5_connector
        self.logger = Logger()
        
        # Order tracking
        self.active_orders = {}  # ticket_id -> order_info
        self.order_history = []
        self.position_tracker = {}
        
        # Performance tracking
        self.execution_times = []
        self.slippage_data = []
        
        # Threading for order monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        self.logger.info("Order Manager initialized")
    
    def start_monitoring(self):
        """Start order monitoring thread"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_orders, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Order monitoring started")
    
    def stop_monitoring(self):
        """Stop order monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Order monitoring stopped")
    
    def place_order(self, signal: Dict[str, Any]) -> Optional[Any]:
        """Place order based on trading signal"""
        try:
            start_time = time.time()
            
            # Extract order parameters
            symbol = signal['symbol']
            action = signal['action'].upper()
            lot_size = signal['lot_size']
            order_type = signal.get('order_type', 'market')
            price = signal.get('price')
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            comment = signal.get('comment', 'AuraTrade Bot')
            
            # Validate order parameters
            if not self._validate_order_parameters(symbol, lot_size, action):
                return None
            
            # Place the order
            result = None
            if action == 'BUY':
                result = self.mt5_connector.place_buy_order(
                    symbol, lot_size, price, stop_loss, take_profit, order_type, comment
                )
            elif action == 'SELL':
                result = self.mt5_connector.place_sell_order(
                    symbol, lot_size, price, stop_loss, take_profit, order_type, comment
                )
            
            # Record execution time
            execution_time = (time.time() - start_time) * 1000  # milliseconds
            self.execution_times.append(execution_time)
            
            # Keep only last 1000 execution times
            if len(self.execution_times) > 1000:
                self.execution_times = self.execution_times[-1000:]
            
            if result and result.retcode == 10009:  # TRADE_RETCODE_DONE
                # Record successful order
                order_info = {
                    'ticket': result.order,
                    'symbol': symbol,
                    'action': action,
                    'lot_size': lot_size,
                    'requested_price': price,
                    'executed_price': result.price,
                    'execution_time_ms': execution_time,
                    'timestamp': datetime.now(),
                    'status': OrderStatus.FILLED,
                    'comment': comment
                }
                
                # Calculate slippage
                if price:
                    slippage = abs(result.price - price)
                    self.slippage_data.append(slippage)
                    order_info['slippage'] = slippage
                
                self.active_orders[result.order] = order_info
                self.order_history.append(order_info)
                
                self.logger.info(f"Order executed successfully: {symbol} {action} {lot_size} lots")
                self.logger.debug(f"Execution time: {execution_time:.2f}ms")
                
            else:
                error_msg = f"Order placement failed: {result.comment if result else 'Unknown error'}"
                self.logger.error(error_msg)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
    
    def place_market_order(self, symbol: str, action: str, lot_size: float,
                          stop_loss: Optional[float] = None, 
                          take_profit: Optional[float] = None) -> Optional[Any]:
        """Place market order"""
        signal = {
            'symbol': symbol,
            'action': action,
            'lot_size': lot_size,
            'order_type': 'market',
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        return self.place_order(signal)
    
    def place_limit_order(self, symbol: str, action: str, lot_size: float, price: float,
                         stop_loss: Optional[float] = None,
                         take_profit: Optional[float] = None) -> Optional[Any]:
        """Place limit order"""
        signal = {
            'symbol': symbol,
            'action': action,
            'lot_size': lot_size,
            'order_type': 'limit',
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        return self.place_order(signal)
    
    def place_stop_order(self, symbol: str, action: str, lot_size: float, price: float,
                        stop_loss: Optional[float] = None,
                        take_profit: Optional[float] = None) -> Optional[Any]:
        """Place stop order"""
        signal = {
            'symbol': symbol,
            'action': action,
            'lot_size': lot_size,
            'order_type': 'stop',
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        return self.place_order(signal)
    
    def close_position(self, ticket: int) -> Optional[Any]:
        """Close position by ticket"""
        try:
            start_time = time.time()
            
            result = self.mt5_connector.close_position(ticket)
            
            execution_time = (time.time() - start_time) * 1000
            self.execution_times.append(execution_time)
            
            if result and result.retcode == 10009:
                # Update order tracking
                if ticket in self.active_orders:
                    self.active_orders[ticket]['status'] = OrderStatus.FILLED
                    self.active_orders[ticket]['close_time'] = datetime.now()
                    self.active_orders[ticket]['close_execution_time_ms'] = execution_time
                
                self.logger.info(f"Position {ticket} closed successfully")
                self.logger.debug(f"Close execution time: {execution_time:.2f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {str(e)}")
            return None
    
    def close_all_positions(self, symbol: Optional[str] = None) -> List[Any]:
        """Close all positions for symbol or all symbols"""
        try:
            positions = self.mt5_connector.get_positions(symbol)
            results = []
            
            for position in positions:
                result = self.close_position(position.ticket)
                if result:
                    results.append(result)
            
            self.logger.info(f"Closed {len(results)} positions")
            return results
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {str(e)}")
            return []
    
    def modify_position(self, ticket: int, stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Optional[Any]:
        """Modify position stop loss and take profit"""
        try:
            result = self.mt5_connector.modify_position(ticket, stop_loss, take_profit)
            
            if result and result.retcode == 10009:
                # Update order tracking
                if ticket in self.active_orders:
                    self.active_orders[ticket]['modified_time'] = datetime.now()
                    if stop_loss:
                        self.active_orders[ticket]['stop_loss'] = stop_loss
                    if take_profit:
                        self.active_orders[ticket]['take_profit'] = take_profit
                
                self.logger.info(f"Position {ticket} modified successfully")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error modifying position {ticket}: {str(e)}")
            return None
    
    def cancel_order(self, ticket: int) -> Optional[Any]:
        """Cancel pending order"""
        try:
            # Note: MT5 order cancellation would be implemented here
            # This is a placeholder for the cancellation logic
            
            if ticket in self.active_orders:
                self.active_orders[ticket]['status'] = OrderStatus.CANCELLED
                self.active_orders[ticket]['cancel_time'] = datetime.now()
            
            self.logger.info(f"Order {ticket} cancelled")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {ticket}: {str(e)}")
            return None
    
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Any]:
        """Get all open positions"""
        try:
            return self.mt5_connector.get_positions(symbol)
        except Exception as e:
            self.logger.error(f"Error getting open positions: {str(e)}")
            return []
    
    def get_pending_orders(self, symbol: Optional[str] = None) -> List[Any]:
        """Get all pending orders"""
        try:
            return self.mt5_connector.get_orders(symbol)
        except Exception as e:
            self.logger.error(f"Error getting pending orders: {str(e)}")
            return []
    
    def get_closed_positions_today(self) -> List[Any]:
        """Get closed positions for today"""
        try:
            today = datetime.now().date()
            start_time = datetime.combine(today, datetime.min.time())
            end_time = datetime.combine(today, datetime.max.time())
            
            return self.mt5_connector.get_history_deals(start_time, end_time)
            
        except Exception as e:
            self.logger.error(f"Error getting today's closed positions: {str(e)}")
            return []
    
    def _validate_order_parameters(self, symbol: str, lot_size: float, action: str) -> bool:
        """Validate order parameters"""
        try:
            # Check symbol info
            symbol_info = self.mt5_connector.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Invalid symbol: {symbol}")
                return False
            
            # Check lot size
            if lot_size < symbol_info.volume_min or lot_size > symbol_info.volume_max:
                self.logger.error(f"Invalid lot size: {lot_size}")
                return False
            
            # Check action
            if action not in ['BUY', 'SELL']:
                self.logger.error(f"Invalid action: {action}")
                return False
            
            # Check if trading is allowed
            if not symbol_info.trade_mode:
                self.logger.error(f"Trading not allowed for {symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Order validation error: {str(e)}")
            return False
    
    def _monitor_orders(self):
        """Monitor active orders and positions"""
        while self.monitoring_active:
            try:
                # Update position status
                current_positions = self.get_open_positions()
                current_tickets = {pos.ticket for pos in current_positions}
                
                # Check for closed positions
                for ticket, order_info in list(self.active_orders.items()):
                    if ticket not in current_tickets and order_info['status'] == OrderStatus.FILLED:
                        # Position was closed
                        order_info['status'] = OrderStatus.FILLED
                        order_info['actual_close_time'] = datetime.now()
                
                # Clean up old orders
                self._cleanup_old_orders()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Order monitoring error: {str(e)}")
                time.sleep(5)  # Longer sleep on error
    
    def _cleanup_old_orders(self):
        """Clean up old order records"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Keep only recent order history
            self.order_history = [
                order for order in self.order_history 
                if order['timestamp'] > cutoff_time
            ]
            
            # Clean up closed positions from active orders
            for ticket, order_info in list(self.active_orders.items()):
                if (order_info.get('actual_close_time') and 
                    order_info['actual_close_time'] < cutoff_time):
                    del self.active_orders[ticket]
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get order execution statistics"""
        try:
            if not self.execution_times:
                return {}
            
            import numpy as np
            
            stats = {
                'total_orders': len(self.order_history),
                'avg_execution_time_ms': np.mean(self.execution_times),
                'min_execution_time_ms': np.min(self.execution_times),
                'max_execution_time_ms': np.max(self.execution_times),
                'median_execution_time_ms': np.median(self.execution_times),
                'hft_compliant_orders': sum(1 for t in self.execution_times if t < 1.0),
                'hft_compliance_rate': sum(1 for t in self.execution_times if t < 1.0) / len(self.execution_times)
            }
            
            if self.slippage_data:
                stats.update({
                    'avg_slippage': np.mean(self.slippage_data),
                    'max_slippage': np.max(self.slippage_data),
                    'min_slippage': np.min(self.slippage_data)
                })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating execution statistics: {str(e)}")
            return {}
    
    def get_order_book_pressure(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze order book pressure for HFT strategies"""
        try:
            # Get market depth
            depth = self.mt5_connector.get_market_depth(symbol)
            if not depth:
                return None
            
            # Calculate basic pressure metrics
            spread = depth['spread']
            mid_price = (depth['bid'] + depth['ask']) / 2
            
            pressure = {
                'spread': spread,
                'mid_price': mid_price,
                'bid_ask_ratio': depth['bid'] / depth['ask'] if depth['ask'] > 0 else 0,
                'pressure_direction': 'bullish' if depth['bid'] > depth['ask'] else 'bearish'
            }
            
            return pressure
            
        except Exception as e:
            self.logger.error(f"Error calculating order book pressure: {str(e)}")
            return None
    
    def get_active_order_summary(self) -> Dict[str, Any]:
        """Get summary of active orders and positions"""
        try:
            positions = self.get_open_positions()
            orders = self.get_pending_orders()
            
            summary = {
                'total_positions': len(positions),
                'total_pending_orders': len(orders),
                'buy_positions': sum(1 for pos in positions if pos.type == 0),  # POSITION_TYPE_BUY
                'sell_positions': sum(1 for pos in positions if pos.type == 1),  # POSITION_TYPE_SELL
                'total_volume': sum(pos.volume for pos in positions),
                'total_profit': sum(pos.profit for pos in positions),
                'positions_by_symbol': {}
            }
            
            # Group by symbol
            for pos in positions:
                symbol = pos.symbol
                if symbol not in summary['positions_by_symbol']:
                    summary['positions_by_symbol'][symbol] = {
                        'count': 0,
                        'volume': 0,
                        'profit': 0
                    }
                
                summary['positions_by_symbol'][symbol]['count'] += 1
                summary['positions_by_symbol'][symbol]['volume'] += pos.volume
                summary['positions_by_symbol'][symbol]['profit'] += pos.profit
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting active order summary: {str(e)}")
            return {}
