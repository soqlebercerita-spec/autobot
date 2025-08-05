"""
MetaTrader 5 Connector for AuraTrade Bot
Handles all MT5 API interactions
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import time

from config.credentials import Credentials
from utils.logger import Logger

class MT5Connector:
    """MetaTrader 5 connection and data management"""
    
    def __init__(self):
        self.logger = Logger()
        self.connected = False
        self.login_info = None
        self.symbol_info_cache = {}
        
    def initialize(self) -> bool:
        """Initialize MT5 connection"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            # Get credentials
            creds = Credentials.get_mt5_credentials()
            
            # Validate credentials
            if not Credentials.validate_mt5_credentials():
                self.logger.error("MT5 credentials are invalid")
                return False
            
            # Login to MT5
            if not mt5.login(creds['login'], creds['password'], creds['server']):
                error = mt5.last_error()
                self.logger.error(f"MT5 login failed: {error}")
                return False
            
            # Store login info
            self.login_info = mt5.account_info()
            self.connected = True
            
            self.logger.info(f"MT5 connected successfully. Account: {self.login_info.login}")
            self.logger.info(f"Balance: {self.login_info.balance}, Equity: {self.login_info.equity}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"MT5 initialization error: {str(e)}")
            return False
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        try:
            mt5.shutdown()
            self.connected = False
            self.logger.info("MT5 connection closed")
        except Exception as e:
            self.logger.error(f"MT5 shutdown error: {str(e)}")
    
    def is_connected(self) -> bool:
        """Check if MT5 is connected"""
        try:
            if not self.connected:
                return False
            
            # Test connection by getting account info
            account_info = mt5.account_info()
            return account_info is not None
            
        except Exception:
            return False
    
    def get_account_info(self):
        """Get account information"""
        try:
            return mt5.account_info()
        except Exception as e:
            self.logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_symbol_info(self, symbol: str) -> Optional[Any]:
        """Get symbol information with caching"""
        try:
            if symbol not in self.symbol_info_cache:
                info = mt5.symbol_info(symbol)
                if info is None:
                    self.logger.error(f"Symbol {symbol} not found")
                    return None
                self.symbol_info_cache[symbol] = info
            
            return self.symbol_info_cache[symbol]
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}: {str(e)}")
            return None
    
    def get_tick(self, symbol: str) -> Optional[Any]:
        """Get latest tick for symbol"""
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                self.logger.error(f"No tick data for {symbol}")
                return None
            
            return tick
            
        except Exception as e:
            self.logger.error(f"Error getting tick for {symbol}: {str(e)}")
            return None
    
    def get_ticks(self, symbol: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical ticks"""
        try:
            ticks = mt5.copy_ticks_from_pos(symbol, 0, count)
            if ticks is None or len(ticks) == 0:
                self.logger.error(f"No tick history for {symbol}")
                return None
            
            df = pd.DataFrame(ticks)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting ticks for {symbol}: {str(e)}")
            return None
    
    def get_rates(self, symbol: str, timeframe: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """Get OHLC rates"""
        try:
            # Convert timeframe string to MT5 constant
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
            
            if timeframe not in timeframe_map:
                self.logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            mt5_timeframe = timeframe_map[timeframe]
            
            # Get rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            if rates is None or len(rates) == 0:
                self.logger.error(f"No rate data for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting rates for {symbol} {timeframe}: {str(e)}")
            return None
    
    def get_positions(self, symbol: Optional[str] = None) -> List[Any]:
        """Get open positions"""
        try:
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            return list(positions) if positions else []
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {str(e)}")
            return []
    
    def get_orders(self, symbol: Optional[str] = None) -> List[Any]:
        """Get pending orders"""
        try:
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()
            
            return list(orders) if orders else []
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_history_deals(self, date_from: datetime, date_to: datetime) -> List[Any]:
        """Get historical deals"""
        try:
            deals = mt5.history_deals_get(date_from, date_to)
            return list(deals) if deals else []
            
        except Exception as e:
            self.logger.error(f"Error getting history deals: {str(e)}")
            return []
    
    def place_buy_order(self, symbol: str, lot: float, price: Optional[float] = None, 
                       sl: Optional[float] = None, tp: Optional[float] = None,
                       order_type: str = "market", comment: str = "AuraTrade") -> Any:
        """Place buy order"""
        return self._place_order(symbol, lot, mt5.ORDER_TYPE_BUY, price, sl, tp, order_type, comment)
    
    def place_sell_order(self, symbol: str, lot: float, price: Optional[float] = None,
                        sl: Optional[float] = None, tp: Optional[float] = None,
                        order_type: str = "market", comment: str = "AuraTrade") -> Any:
        """Place sell order"""
        return self._place_order(symbol, lot, mt5.ORDER_TYPE_SELL, price, sl, tp, order_type, comment)
    
    def _place_order(self, symbol: str, lot: float, order_type: int, price: Optional[float],
                    sl: Optional[float], tp: Optional[float], execution_type: str, comment: str) -> Any:
        """Internal order placement method"""
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Get current price if not provided
            if price is None:
                tick = self.get_tick(symbol)
                if not tick:
                    return None
                price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            
            # Normalize price and levels
            price = self._normalize_price(symbol, price)
            if sl:
                sl = self._normalize_price(symbol, sl)
            if tp:
                tp = self._normalize_price(symbol, tp)
            
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot,
                "type": order_type,
                "price": price,
                "deviation": 10,  # Price deviation in points
                "magic": 123456,  # Magic number
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add stop loss and take profit if provided
            if sl:
                request["sl"] = sl
            if tp:
                request["tp"] = tp
            
            # Send order
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Order placed successfully: {symbol} {lot} lots at {price}")
            else:
                error_msg = f"Order failed: {result.comment if result else 'Unknown error'}"
                self.logger.error(error_msg)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            return None
    
    def close_position(self, ticket: int) -> Any:
        """Close position by ticket"""
        try:
            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                self.logger.error(f"Position {ticket} not found")
                return None
            
            position = positions[0]
            
            # Determine opposite order type
            order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
            
            # Get current price
            tick = self.get_tick(position.symbol)
            if not tick:
                return None
            
            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
            price = self._normalize_price(position.symbol, price)
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 10,
                "magic": 123456,
                "comment": "AuraTrade Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Send close order
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Position {ticket} closed successfully")
            else:
                error_msg = f"Close failed: {result.comment if result else 'Unknown error'}"
                self.logger.error(error_msg)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error closing position {ticket}: {str(e)}")
            return None
    
    def modify_position(self, ticket: int, sl: Optional[float] = None, tp: Optional[float] = None) -> Any:
        """Modify position stop loss and take profit"""
        try:
            # Get position info
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                self.logger.error(f"Position {ticket} not found")
                return None
            
            position = positions[0]
            
            # Normalize levels
            if sl:
                sl = self._normalize_price(position.symbol, sl)
            if tp:
                tp = self._normalize_price(position.symbol, tp)
            
            # Prepare modify request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
                "sl": sl if sl else position.sl,
                "tp": tp if tp else position.tp,
            }
            
            # Send modify order
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"Position {ticket} modified successfully")
            else:
                error_msg = f"Modify failed: {result.comment if result else 'Unknown error'}"
                self.logger.error(error_msg)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error modifying position {ticket}: {str(e)}")
            return None
    
    def _normalize_price(self, symbol: str, price: float) -> float:
        """Normalize price according to symbol digits"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                return round(price, symbol_info.digits)
            return price
        except:
            return price
    
    def get_spread(self, symbol: str) -> Optional[float]:
        """Get current spread for symbol"""
        try:
            tick = self.get_tick(symbol)
            if tick:
                symbol_info = self.get_symbol_info(symbol)
                if symbol_info:
                    spread_points = tick.ask - tick.bid
                    return spread_points / symbol_info.point
            return None
        except Exception as e:
            self.logger.error(f"Error getting spread for {symbol}: {str(e)}")
            return None
    
    def get_market_depth(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market depth (order book) for symbol"""
        try:
            # Note: Market depth requires special market data subscription
            # This is a simplified implementation
            tick = self.get_tick(symbol)
            if tick:
                return {
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'spread': tick.ask - tick.bid,
                    'volume_real': getattr(tick, 'volume_real', 0)
                }
            return None
        except Exception as e:
            self.logger.error(f"Error getting market depth for {symbol}: {str(e)}")
            return None
    
    def calculate_margin(self, symbol: str, lot_size: float, order_type: str) -> Optional[float]:
        """Calculate required margin for position"""
        try:
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
            
            # Get current price
            tick = self.get_tick(symbol)
            if not tick:
                return None
            
            # Calculate margin requirement
            if order_type.upper() == 'BUY':
                price = tick.ask
            else:
                price = tick.bid
            
            # Basic margin calculation (simplified)
            margin = lot_size * symbol_info.trade_contract_size * price / 100  # Assuming 1:100 leverage
            
            return margin
            
        except Exception as e:
            self.logger.error(f"Error calculating margin: {str(e)}")
            return None
    
    def get_trading_hours(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get trading hours for symbol"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if symbol_info:
                return {
                    'trade_mode': symbol_info.trade_mode,
                    'trade_stops_level': symbol_info.trade_stops_level,
                    'trade_freeze_level': symbol_info.trade_freeze_level
                }
            return None
        except Exception as e:
            self.logger.error(f"Error getting trading hours for {symbol}: {str(e)}")
            return None
