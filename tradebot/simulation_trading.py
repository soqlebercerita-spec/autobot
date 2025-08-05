"""
Enhanced Trading Simulation with Realistic Market Conditions
Improved execution and position management
"""

import time
import random
from datetime import datetime
from config import config

# Handle numpy import gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create minimal numpy-like functions
    class MockNumpy:
        @staticmethod
        def array(data):
            return data
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
    np = MockNumpy()

class AccountInfo:
    """Simulated account information"""
    def __init__(self, balance=10000.0, leverage=100):
        self.balance = balance
        self.equity = balance
        self.margin = 0.0
        self.free_margin = balance
        self.margin_level = 0.0
        self.login = 999999  # Simulation account
        self.leverage = leverage
        self.profit = 0.0

class PositionInfo:
    """Simulated position information"""
    def __init__(self, ticket, symbol, type_order, volume, price_open, sl=0, tp=0):
        self.ticket = ticket
        self.symbol = symbol
        self.type = type_order  # 0=BUY, 1=SELL
        self.volume = volume
        self.price_open = price_open
        self.sl = sl
        self.tp = tp
        self.profit = 0.0
        self.swap = 0.0
        self.commission = 0.0
        self.time = int(time.time())

class OrderSendResult:
    """Order send result"""
    def __init__(self, retcode=10009, deal=None, order=None):
        self.retcode = retcode  # 10009 = TRADE_RETCODE_DONE
        self.deal = deal
        self.order = order

class SimulationTrading:
    """Enhanced trading simulation with realistic conditions"""
    
    def __init__(self, market_api):
        self.market_api = market_api
        self.account = AccountInfo()
        self.positions = {}
        self.next_ticket = 1000001
        self.is_initialized = False
        
        # Trading constants
        self.ORDER_TYPE_BUY = 0
        self.ORDER_TYPE_SELL = 1
        self.TRADE_ACTION_DEAL = 1
        self.ORDER_FILLING_IOC = 1
        self.ORDER_TIME_GTC = 0
        
        # Slippage and execution settings
        self.max_slippage = 3  # pips
        self.execution_delay = 0.1  # seconds
    
    def initialize(self):
        """Initialize the simulation"""
        try:
            self.is_initialized = True
            print("Trading simulation initialized successfully")
            return True
        except Exception as e:
            print(f"Simulation initialization error: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the simulation"""
        self.is_initialized = False
        print("Trading simulation shutdown")
    
    def account_info(self):
        """Get account information"""
        if not self.is_initialized:
            return None
        
        try:
            # Update account equity based on open positions
            total_profit = sum(pos.profit for pos in self.positions.values())
            self.account.equity = self.account.balance + total_profit
            self.account.profit = total_profit
            
            # Calculate margin (simplified)
            total_margin = 0
            for pos in self.positions.values():
                current_price = self.market_api.get_current_price(pos.symbol)
                position_value = pos.volume * current_price * 100000  # Standard lot size
                required_margin = position_value / self.account.leverage
                total_margin += required_margin
            
            self.account.margin = total_margin
            self.account.free_margin = self.account.equity - total_margin
            
            if total_margin > 0:
                self.account.margin_level = (self.account.equity / total_margin) * 100
            else:
                self.account.margin_level = 0
            
            return self.account
        except Exception as e:
            print(f"Account info error: {e}")
            return None
    
    def get_account_info(self):
        """Alias for account_info() for compatibility"""
        return self.account_info()
    
    @property
    def balance(self):
        """Get current account balance"""
        return self.account.balance if self.account else 0.0
    
    def symbol_info_tick(self, symbol):
        """Get symbol tick information"""
        try:
            current_price = self.market_api.get_current_price(symbol)
            spread = self.market_api.get_spread(symbol)
            
            # Create tick-like object
            class TickInfo:
                def __init__(self, bid, ask):
                    self.bid = bid
                    self.ask = ask
                    self.last = (bid + ask) / 2
                    self.time = int(time.time())
            
            bid = current_price - spread / 2
            ask = current_price + spread / 2
            
            return TickInfo(bid, ask)
        except Exception as e:
            print(f"Symbol tick info error: {e}")
            return None
    
    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        """Get historical rates"""
        try:
            data = self.market_api.get_market_data(symbol, timeframe, count)
            if not data:
                return None
            
            # Convert to numpy-like array structure
            import numpy as np
            
            rates = []
            for point in data:
                rate = {
                    'time': int(point['time'].timestamp()),
                    'open': point['open'],
                    'high': point['high'],
                    'low': point['low'],
                    'close': point['close'],
                    'tick_volume': point.get('volume', 100),
                    'spread': 3,
                    'real_volume': point.get('volume', 100)
                }
                rates.append(tuple(rate.values()))
            
            # Convert to structured numpy array
            dtype = [
                ('time', 'i8'),
                ('open', 'f8'),
                ('high', 'f8'),
                ('low', 'f8'),
                ('close', 'f8'),
                ('tick_volume', 'i8'),
                ('spread', 'i4'),
                ('real_volume', 'i8')
            ]
            
            return np.array(rates, dtype=dtype)
        except Exception as e:
            print(f"Copy rates error: {e}")
            return None
    
    def order_send(self, request):
        """Send trading order with realistic execution"""
        try:
            time.sleep(self.execution_delay)  # Simulate execution delay
            
            symbol = request.get('symbol', '')
            action = request.get('action', self.TRADE_ACTION_DEAL)
            volume = request.get('volume', 0.01)
            type_order = request.get('type', self.ORDER_TYPE_BUY)
            price = request.get('price', 0)
            sl = request.get('sl', 0)
            tp = request.get('tp', 0)
            
            # Get current market price
            tick = self.symbol_info_tick(symbol)
            if not tick:
                return OrderSendResult(retcode=10018)  # Market closed
            
            # Apply slippage
            if type_order == self.ORDER_TYPE_BUY:
                execution_price = tick.ask + random.uniform(0, self.max_slippage * 0.0001)
            else:
                execution_price = tick.bid - random.uniform(0, self.max_slippage * 0.0001)
            
            # Check margin requirements
            account = self.account_info()
            if not account:
                return OrderSendResult(retcode=10019)  # No money
            
            position_value = volume * execution_price * 100000
            required_margin = position_value / account.leverage
            
            if required_margin > account.free_margin:
                return OrderSendResult(retcode=10019)  # Not enough money
            
            # Create position
            ticket = self.next_ticket
            self.next_ticket += 1
            
            position = PositionInfo(
                ticket=ticket,
                symbol=symbol,
                type_order=type_order,
                volume=volume,
                price_open=execution_price,
                sl=sl,
                tp=tp
            )
            
            self.positions[ticket] = position
            
            # Simulate order execution success
            return OrderSendResult(
                retcode=10009,  # TRADE_RETCODE_DONE
                deal=ticket,
                order=ticket
            )
            
        except Exception as e:
            print(f"Order send error: {e}")
            return OrderSendResult(retcode=10013)  # Invalid request
    
    def positions_get(self, symbol=None):
        """Get open positions"""
        try:
            if symbol:
                return [pos for pos in self.positions.values() if pos.symbol == symbol]
            else:
                return list(self.positions.values())
        except Exception as e:
            print(f"Positions get error: {e}")
            return []
    
    def position_close(self, ticket):
        """Close position by ticket"""
        try:
            if ticket not in self.positions:
                return OrderSendResult(retcode=10013)  # Invalid request
            
            position = self.positions[ticket]
            
            # Get current price for closing
            tick = self.symbol_info_tick(position.symbol)
            if not tick:
                return OrderSendResult(retcode=10018)  # Market closed
            
            # Calculate profit
            if position.type == self.ORDER_TYPE_BUY:
                close_price = tick.bid
                profit = (close_price - position.price_open) * position.volume * 100000
            else:
                close_price = tick.ask
                profit = (position.price_open - close_price) * position.volume * 100000
            
            # Update account balance
            self.account.balance += profit
            
            # Remove position
            del self.positions[ticket]
            
            print(f"Position {ticket} closed with profit: ${profit:.2f}")
            
            return OrderSendResult(retcode=10009)
            
        except Exception as e:
            print(f"Position close error: {e}")
            return OrderSendResult(retcode=10013)
    
    def update_positions(self):
        """Update position profits and check TP/SL"""
        try:
            positions_to_close = []
            
            for ticket, position in self.positions.items():
                current_price = self.market_api.get_current_price(position.symbol)
                
                # Calculate current profit
                if position.type == self.ORDER_TYPE_BUY:
                    position.profit = (current_price - position.price_open) * position.volume * 100000
                    
                    # Check TP/SL
                    if position.tp > 0 and current_price >= position.tp:
                        positions_to_close.append(ticket)
                    elif position.sl > 0 and current_price <= position.sl:
                        positions_to_close.append(ticket)
                        
                else:  # SELL
                    position.profit = (position.price_open - current_price) * position.volume * 100000
                    
                    # Check TP/SL
                    if position.tp > 0 and current_price <= position.tp:
                        positions_to_close.append(ticket)
                    elif position.sl > 0 and current_price >= position.sl:
                        positions_to_close.append(ticket)
            
            # Close positions that hit TP/SL
            for ticket in positions_to_close:
                self.position_close(ticket)
                
        except Exception as e:
            print(f"Update positions error: {e}")
    
    def last_error(self):
        """Get last error (simulation always returns success)"""
        return (0, "Success")
