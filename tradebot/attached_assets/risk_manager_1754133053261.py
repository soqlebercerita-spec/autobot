"""
Risk Management Module for Trading Bot
"""

import MetaTrader5 as mt5
import datetime
import numpy as np

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.daily_trades = 0
        self.daily_loss = 0
        self.last_reset_date = datetime.date.today()
        self.max_drawdown_reached = False
        
    def reset_daily_counters(self):
        """Reset daily counters if new day"""
        current_date = datetime.date.today()
        if current_date != self.last_reset_date:
            self.daily_trades = 0
            self.daily_loss = 0
            self.last_reset_date = current_date
            self.max_drawdown_reached = False
            print(f"Daily counters reset for {current_date}")
    
    def can_place_order(self):
        """Check if we can place a new order based on risk parameters"""
        try:
            self.reset_daily_counters()
            
            # Check account info
            account_info = mt5.account_info()
            if not account_info:
                return False
            
            # Check minimum balance
            if account_info.balance < self.config.SALDO_MINIMAL:
                print(f"Balance too low: {account_info.balance} < {self.config.SALDO_MINIMAL}")
                return False
            
            # Check daily trade limit
            if self.daily_trades >= self.config.MAX_ORDER_PER_SESSION:
                print(f"Daily trade limit reached: {self.daily_trades}")
                return False
            
            # Check current open positions
            positions = mt5.positions_get()
            if positions and len(positions) >= self.config.MAX_ORDER_PER_SESSION:
                print(f"Max open positions reached: {len(positions)}")
                return False
            
            # Check drawdown
            if self.is_max_drawdown_reached(account_info):
                print("Maximum drawdown reached")
                return False
            
            # Check trading hours
            current_hour = datetime.datetime.now().hour
            if current_hour < self.config.TRADING_START_HOUR or current_hour >= self.config.TRADING_END_HOUR:
                return False
            
            return True
            
        except Exception as e:
            print(f"Error checking order conditions: {e}")
            return False
    
    def is_max_drawdown_reached(self, account_info):
        """Check if maximum drawdown limit is reached"""
        try:
            if not hasattr(self, 'initial_balance'):
                self.initial_balance = account_info.balance
            
            current_equity = account_info.equity
            drawdown_percent = ((self.initial_balance - current_equity) / self.initial_balance) * 100
            
            if drawdown_percent >= self.config.MAX_DRAWDOWN:
                self.max_drawdown_reached = True
                return True
            
            return False
            
        except Exception as e:
            print(f"Error calculating drawdown: {e}")
            return False
    
    def calculate_position_size(self, symbol, risk_percent=None):
        """Calculate optimal position size based on risk management"""
        try:
            if risk_percent is None:
                risk_percent = self.config.MAX_RISK_PER_TRADE
            
            account_info = mt5.account_info()
            if not account_info:
                return 0.01
            
            balance = account_info.balance
            risk_amount = balance * (risk_percent / 100)
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return 0.01
            
            # Calculate pip value
            point = symbol_info.point
            if symbol_info.digits == 5 or symbol_info.digits == 3:
                pip_size = point * 10
            else:
                pip_size = point
            
            # Estimate stop loss in pips (using default SL percentage)
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return 0.01
            
            sl_pips = (self.config.SL_PERSEN_DEFAULT * tick.ask) / pip_size
            
            # Calculate position size
            if sl_pips > 0:
                pip_value = symbol_info.trade_tick_value
                position_size = risk_amount / (sl_pips * pip_value)
                
                # Ensure within symbol limits
                min_lot = symbol_info.volume_min
                max_lot = symbol_info.volume_max
                step = symbol_info.volume_step
                
                # Round to nearest step
                position_size = round(position_size / step) * step
                position_size = max(min_lot, min(max_lot, position_size))
                
                return position_size
            
            return symbol_info.volume_min
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.01
    
    def add_position(self, ticket, signal, volume, entry_price, tp, sl):
        """Add position to tracking"""
        try:
            self.positions[ticket] = {
                'signal': signal,
                'volume': volume,
                'entry_price': entry_price,
                'tp': tp,
                'sl': sl,
                'entry_time': datetime.datetime.now(),
                'highest_price': entry_price if signal == "BUY" else entry_price,
                'lowest_price': entry_price if signal == "SELL" else entry_price
            }
            self.daily_trades += 1
            
        except Exception as e:
            print(f"Error adding position: {e}")
    
    def remove_position(self, ticket):
        """Remove position from tracking"""
        try:
            if ticket in self.positions:
                del self.positions[ticket]
        except Exception as e:
            print(f"Error removing position: {e}")
    
    def calculate_trailing_stop(self, position, current_price):
        """Calculate trailing stop loss"""
        try:
            ticket = position.ticket
            symbol = position.symbol
            
            if ticket not in self.positions:
                return None
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return None
            
            point = symbol_info.point
            trailing_distance = self.config.TRAILING_STOP_PIPS * point
            
            position_data = self.positions[ticket]
            
            if position.type == 0:  # Buy position
                # Update highest price
                if current_price > position_data['highest_price']:
                    position_data['highest_price'] = current_price
                
                # Calculate new stop loss
                new_sl = position_data['highest_price'] - trailing_distance
                
                # Only move SL up, never down
                if new_sl > position.sl:
                    return new_sl
                    
            else:  # Sell position
                # Update lowest price
                if current_price < position_data['lowest_price']:
                    position_data['lowest_price'] = current_price
                
                # Calculate new stop loss
                new_sl = position_data['lowest_price'] + trailing_distance
                
                # Only move SL down, never up
                if position.sl == 0 or new_sl < position.sl:
                    return new_sl
            
            return None
            
        except Exception as e:
            print(f"Error calculating trailing stop: {e}")
            return None
    
    def check_correlation(self, symbol1, symbol2, period=50):
        """Check correlation between two symbols"""
        try:
            rates1 = mt5.copy_rates_from_pos(symbol1, mt5.TIMEFRAME_H1, 0, period)
            rates2 = mt5.copy_rates_from_pos(symbol2, mt5.TIMEFRAME_H1, 0, period)
            
            if rates1 is None or rates2 is None:
                return 0
            
            if len(rates1) != len(rates2):
                min_len = min(len(rates1), len(rates2))
                rates1 = rates1[:min_len]
                rates2 = rates2[:min_len]
            
            prices1 = rates1['close']
            prices2 = rates2['close']
            
            correlation = np.corrcoef(prices1, prices2)[0, 1]
            return correlation if not np.isnan(correlation) else 0
            
        except Exception as e:
            print(f"Error calculating correlation: {e}")
            return 0
    
    def get_position_exposure(self):
        """Get total position exposure"""
        try:
            positions = mt5.positions_get()
            if not positions:
                return 0, 0
            
            total_buy_volume = 0
            total_sell_volume = 0
            
            for position in positions:
                if position.type == 0:  # Buy
                    total_buy_volume += position.volume
                else:  # Sell
                    total_sell_volume += position.volume
            
            return total_buy_volume, total_sell_volume
            
        except Exception as e:
            print(f"Error calculating exposure: {e}")
            return 0, 0
    
    def is_news_time(self):
        """Check if it's during high-impact news time (basic implementation)"""
        try:
            now = datetime.datetime.now()
            
            # Avoid trading during typical news hours (example times)
            news_hours = [
                (8, 30, 9, 30),   # London open
                (13, 30, 14, 30), # US open
                (20, 30, 21, 30), # US close
            ]
            
            current_time = now.hour * 60 + now.minute
            
            for start_h, start_m, end_h, end_m in news_hours:
                start_time = start_h * 60 + start_m
                end_time = end_h * 60 + end_m
                
                if start_time <= current_time <= end_time:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking news time: {e}")
            return False
    
    def calculate_risk_reward_ratio(self, entry_price, tp_price, sl_price):
        """Calculate risk-reward ratio"""
        try:
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)
            
            if risk == 0:
                return 0
            
            return reward / risk
            
        except Exception as e:
            print(f"Error calculating risk-reward ratio: {e}")
            return 0
    
    def should_avoid_trading(self):
        """Determine if trading should be avoided due to risk conditions"""
        try:
            # Check if max drawdown reached
            if self.max_drawdown_reached:
                return True, "Maximum drawdown reached"
            
            # Check if during news time
            if self.is_news_time():
                return True, "High-impact news time"
            
            # Check account health
            account_info = mt5.account_info()
            if account_info:
                margin_level = account_info.margin_level
                if margin_level < 200:  # Less than 200% margin level
                    return True, f"Low margin level: {margin_level}%"
            
            # Check if too many losing trades today
            if self.daily_loss > self.config.SALDO_MINIMAL * 0.05:  # 5% of minimum balance
                return True, "Daily loss limit reached"
            
            return False, "Safe to trade"
            
        except Exception as e:
            print(f"Error checking trading conditions: {e}")
            return True, "Error in risk assessment"
    
    def update_daily_loss(self, loss_amount):
        """Update daily loss tracking"""
        try:
            self.reset_daily_counters()
            self.daily_loss += loss_amount
        except Exception as e:
            print(f"Error updating daily loss: {e}")
    
    def get_risk_status(self):
        """Get current risk status summary"""
        try:
            account_info = mt5.account_info()
            positions = mt5.positions_get()
            
            status = {
                'daily_trades': self.daily_trades,
                'max_trades': self.config.MAX_ORDER_PER_SESSION,
                'daily_loss': self.daily_loss,
                'open_positions': len(positions) if positions else 0,
                'balance': account_info.balance if account_info else 0,
                'equity': account_info.equity if account_info else 0,
                'margin_level': account_info.margin_level if account_info else 0,
                'can_trade': self.can_place_order()
            }
            
            return status
            
        except Exception as e:
            print(f"Error getting risk status: {e}")
            return {}
