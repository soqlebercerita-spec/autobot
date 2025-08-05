"""
Simulation Trading Module
Simulates real trading without actual broker connection
"""

import time
import datetime
import numpy as np
from typing import Dict, List, Optional

class Position:
    def __init__(self, ticket, symbol, type_order, volume, open_price, tp, sl, magic, comment):
        self.ticket = ticket
        self.symbol = symbol
        self.type = type_order  # 0 = BUY, 1 = SELL
        self.volume = volume
        self.open_price = open_price
        self.current_price = open_price
        self.tp = tp
        self.sl = sl
        self.magic = magic
        self.comment = comment
        self.open_time = datetime.datetime.now()
        self.profit = 0.0
        self.swap = 0.0
        self.commission = 0.0

class AccountInfo:
    def __init__(self, login=12345, balance=10000.0):
        self.login = login
        self.balance = balance
        self.equity = balance
        self.margin = 0.0
        self.margin_free = balance
        self.margin_level = 0.0 if balance == 0 else 100000.0

class SymbolInfo:
    def __init__(self, symbol):
        self.symbol = symbol
        # Default specs based on symbol type
        if "XAU" in symbol or "GOLD" in symbol:
            self.point = 0.01
            self.digits = 2
            self.trade_tick_value = 1.0
            self.volume_min = 0.01
            self.volume_max = 100.0
            self.volume_step = 0.01
        elif any(forex in symbol for forex in ["EUR", "GBP", "USD", "JPY", "CAD", "AUD", "NZD", "CHF"]):
            self.point = 0.00001
            self.digits = 5
            self.trade_tick_value = 10.0
            self.volume_min = 0.01
            self.volume_max = 100.0
            self.volume_step = 0.01
        else:
            self.point = 0.00001
            self.digits = 5
            self.trade_tick_value = 1.0
            self.volume_min = 0.01
            self.volume_max = 100.0
            self.volume_step = 0.01

class TickInfo:
    def __init__(self, symbol, bid, ask):
        self.symbol = symbol
        self.bid = bid
        self.ask = ask
        self.last = (bid + ask) / 2
        self.time = datetime.datetime.now()

class OrderResult:
    def __init__(self, retcode, deal=None, order=None, volume=None, price=None, comment=""):
        self.retcode = retcode
        self.deal = deal
        self.order = order
        self.volume = volume
        self.price = price
        self.comment = comment

class SimulationTrading:
    # Trade return codes (compatible with MT5)
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_ERROR = 10013
    TRADE_RETCODE_INVALID_VOLUME = 10014
    TRADE_RETCODE_INVALID_PRICE = 10015
    TRADE_RETCODE_INVALID_STOPS = 10016
    TRADE_RETCODE_MARKET_CLOSED = 10018
    TRADE_RETCODE_NO_MONEY = 10019
    
    # Order types
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    
    # Trade actions
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    
    # Order time types
    ORDER_TIME_GTC = 0
    
    # Order filling types
    ORDER_FILLING_IOC = 2
    
    def __init__(self, market_data_api):
        self.market_api = market_data_api
        self.positions = {}
        self.account = AccountInfo()
        self.next_ticket = 1000001
        self.is_connected = False
        self.terminal_info_data = {"connected": False}
        
    def initialize(self):
        """Initialize trading simulation"""
        try:
            # Test market data connection
            test_price = self.market_api.get_current_price("XAUUSD")
            if test_price and test_price > 0:
                self.is_connected = True
                self.terminal_info_data["connected"] = True
                return True
            return False
        except Exception as e:
            print(f"Error initializing simulation: {e}")
            return False
    
    def shutdown(self):
        """Shutdown trading simulation"""
        self.is_connected = False
        self.terminal_info_data["connected"] = False
    
    def last_error(self):
        """Return last error (simulation always returns 0)"""
        return (0, "No error")
    
    def terminal_info(self):
        """Get terminal information"""
        return self.terminal_info_data if self.is_connected else None
    
    def account_info(self):
        """Get account information"""
        if not self.is_connected:
            return None
        
        # Update equity based on current positions
        self._update_positions_profit()
        total_profit = sum(pos.profit for pos in self.positions.values())
        self.account.equity = self.account.balance + total_profit
        
        return self.account
    
    def symbol_info(self, symbol):
        """Get symbol information"""
        if not self.is_connected:
            return None
        return SymbolInfo(symbol)
    
    def symbol_info_tick(self, symbol):
        """Get current tick information"""
        if not self.is_connected:
            return None
        
        try:
            current_price = self.market_api.get_current_price(symbol)
            spread = current_price * 0.0001  # 1 pip spread simulation
            bid = current_price - spread/2
            ask = current_price + spread/2
            
            return TickInfo(symbol, bid, ask)
        except Exception as e:
            print(f"Error getting tick info for {symbol}: {e}")
            return None
    
    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        """Get historical rates"""
        if not self.is_connected:
            return None
        
        try:
            data = self.market_api.get_historical_data(symbol, 'M1', count)
            if data is None:
                return None
            
            # Convert to MT5-like structure
            rates = []
            for i in range(len(data['close'])):
                rates.append({
                    'time': int(data['time'][i].timestamp()),
                    'open': data['open'][i],
                    'high': data['high'][i],
                    'low': data['low'][i],
                    'close': data['close'][i],
                    'tick_volume': int(data['volume'][i])
                })
            
            # Convert to numpy structured array
            dtype = [('time', 'i8'), ('open', 'f8'), ('high', 'f8'), 
                    ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8')]
            
            rates_array = np.array([(r['time'], r['open'], r['high'], r['low'], r['close'], r['tick_volume']) 
                                  for r in rates], dtype=dtype)
            
            return rates_array
        except Exception as e:
            print(f"Error getting rates for {symbol}: {e}")
            return None
    
    def positions_get(self, symbol=None):
        """Get current positions"""
        if not self.is_connected:
            return None
        
        self._update_positions_profit()
        
        if symbol:
            return [pos for pos in self.positions.values() if pos.symbol == symbol]
        else:
            return list(self.positions.values())
    
    def order_send(self, request):
        """Send trading order"""
        if not self.is_connected:
            return OrderResult(self.TRADE_RETCODE_ERROR, comment="Not connected")
        
        try:
            action = request.get("action")
            
            if action == self.TRADE_ACTION_DEAL:
                return self._execute_market_order(request)
            elif action == self.TRADE_ACTION_SLTP:
                return self._modify_position(request)
            else:
                return OrderResult(self.TRADE_RETCODE_ERROR, comment="Unsupported action")
                
        except Exception as e:
            print(f"Error executing order: {e}")
            return OrderResult(self.TRADE_RETCODE_ERROR, comment=str(e))
    
    def _execute_market_order(self, request):
        """Execute market order"""
        try:
            symbol = request.get("symbol")
            volume = request.get("volume")
            order_type = request.get("type")
            price = request.get("price")
            sl = request.get("sl", 0)
            tp = request.get("tp", 0)
            magic = request.get("magic", 0)
            comment = request.get("comment", "")
            
            # Validate order
            if not symbol or not volume or order_type is None:
                return OrderResult(self.TRADE_RETCODE_ERROR, comment="Invalid request parameters")
            
            symbol_info = self.symbol_info(symbol)
            if not symbol_info:
                return OrderResult(self.TRADE_RETCODE_ERROR, comment="Symbol not found")
            
            # Check volume
            if volume < symbol_info.volume_min or volume > symbol_info.volume_max:
                return OrderResult(self.TRADE_RETCODE_INVALID_VOLUME, comment="Invalid volume")
            
            # Get current prices
            tick = self.symbol_info_tick(symbol)
            if not tick:
                return OrderResult(self.TRADE_RETCODE_ERROR, comment="No price data")
            
            # Determine execution price
            if order_type == self.ORDER_TYPE_BUY:
                execution_price = tick.ask
            else:
                execution_price = tick.bid
            
            # Check account balance (simplified)
            required_margin = volume * 1000  # Simplified margin calculation
            if required_margin > self.account.margin_free:
                return OrderResult(self.TRADE_RETCODE_NO_MONEY, comment="Insufficient funds")
            
            # Create position
            ticket = self.next_ticket
            self.next_ticket += 1
            
            position = Position(
                ticket=ticket,
                symbol=symbol,
                type_order=order_type,
                volume=volume,
                open_price=execution_price,
                tp=tp,
                sl=sl,
                magic=magic,
                comment=comment
            )
            
            self.positions[ticket] = position
            
            # Update account
            self.account.margin += required_margin
            self.account.margin_free -= required_margin
            
            return OrderResult(
                retcode=self.TRADE_RETCODE_DONE,
                deal=ticket,
                order=ticket,
                volume=volume,
                price=execution_price,
                comment="Order executed"
            )
            
        except Exception as e:
            print(f"Error executing market order: {e}")
            return OrderResult(self.TRADE_RETCODE_ERROR, comment=str(e))
    
    def _modify_position(self, request):
        """Modify position SL/TP"""
        try:
            position_ticket = request.get("position")
            new_sl = request.get("sl")
            new_tp = request.get("tp")
            
            if position_ticket not in self.positions:
                return OrderResult(self.TRADE_RETCODE_ERROR, comment="Position not found")
            
            position = self.positions[position_ticket]
            
            if new_sl is not None:
                position.sl = new_sl
            if new_tp is not None:
                position.tp = new_tp
            
            return OrderResult(
                retcode=self.TRADE_RETCODE_DONE,
                comment="Position modified"
            )
            
        except Exception as e:
            print(f"Error modifying position: {e}")
            return OrderResult(self.TRADE_RETCODE_ERROR, comment=str(e))
    
    def _update_positions_profit(self):
        """Update profit for all positions"""
        try:
            for position in self.positions.values():
                tick = self.symbol_info_tick(position.symbol)
                if not tick:
                    continue
                
                # Update current price
                if position.type == self.ORDER_TYPE_BUY:
                    position.current_price = tick.bid
                    price_diff = tick.bid - position.open_price
                else:
                    position.current_price = tick.ask
                    price_diff = position.open_price - tick.ask
                
                # Calculate profit (simplified)
                symbol_info = self.symbol_info(position.symbol)
                if symbol_info:
                    position.profit = price_diff * position.volume * symbol_info.trade_tick_value
                
                # Check SL/TP
                self._check_stop_levels(position, tick)
                
        except Exception as e:
            print(f"Error updating positions profit: {e}")
    
    def _check_stop_levels(self, position, tick):
        """Check if stop loss or take profit should be triggered"""
        try:
            should_close = False
            close_price = 0
            close_reason = ""
            
            if position.type == self.ORDER_TYPE_BUY:
                current_price = tick.bid
                # Check SL
                if position.sl > 0 and current_price <= position.sl:
                    should_close = True
                    close_price = position.sl
                    close_reason = "Stop Loss"
                # Check TP
                elif position.tp > 0 and current_price >= position.tp:
                    should_close = True
                    close_price = position.tp
                    close_reason = "Take Profit"
            else:
                current_price = tick.ask
                # Check SL
                if position.sl > 0 and current_price >= position.sl:
                    should_close = True
                    close_price = position.sl
                    close_reason = "Stop Loss"
                # Check TP
                elif position.tp > 0 and current_price <= position.tp:
                    should_close = True
                    close_price = position.tp
                    close_reason = "Take Profit"
            
            if should_close:
                self._close_position(position.ticket, close_price, close_reason)
                
        except Exception as e:
            print(f"Error checking stop levels: {e}")
    
    def _close_position(self, ticket, close_price, reason):
        """Close position at specified price"""
        try:
            if ticket not in self.positions:
                return
            
            position = self.positions[ticket]
            
            # Calculate final profit
            if position.type == self.ORDER_TYPE_BUY:
                price_diff = close_price - position.open_price
            else:
                price_diff = position.open_price - close_price
            
            final_profit = 0.0
            symbol_info = self.symbol_info(position.symbol)
            if symbol_info:
                final_profit = price_diff * position.volume * symbol_info.trade_tick_value
                
                # Update account balance
                self.account.balance += final_profit
                
                # Free margin
                required_margin = position.volume * 1000
                self.account.margin -= required_margin
                self.account.margin_free += required_margin
            
            # Remove position
            del self.positions[ticket]
            
            print(f"Position {ticket} closed: {reason}, Profit: {final_profit:.2f}")
            
        except Exception as e:
            print(f"Error closing position: {e}")
    
    def close_position_by_ticket(self, ticket):
        """Manually close position by ticket"""
        try:
            if ticket not in self.positions:
                return False
            
            position = self.positions[ticket]
            tick = self.symbol_info_tick(position.symbol)
            
            if not tick:
                return False
            
            if position.type == self.ORDER_TYPE_BUY:
                close_price = tick.bid
            else:
                close_price = tick.ask
            
            self._close_position(ticket, close_price, "Manual Close")
            return True
            
        except Exception as e:
            print(f"Error manually closing position: {e}")
            return False
    
    def close_all_positions(self):
        """Close all open positions"""
        try:
            tickets_to_close = list(self.positions.keys())
            for ticket in tickets_to_close:
                self.close_position_by_ticket(ticket)
            return True
        except Exception as e:
            print(f"Error closing all positions: {e}")
            return False