"""
Enhanced Risk Management Module for Trading Bot
Optimized for Windows MT5 integration
"""

import datetime
import numpy as np
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

class RiskManager:
    def __init__(self, config):
        self.config = config
        self.positions = {}
        self.daily_trades = 0
        self.daily_loss = 0
        self.daily_profit = 0
        self.last_reset_date = datetime.date.today()
        self.max_drawdown_reached = False
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        
    def reset_daily_counters(self):
        """Reset daily counters if new day"""
        current_date = datetime.date.today()
        if current_date != self.last_reset_date:
            self.daily_trades = 0
            self.daily_loss = 0
            self.daily_profit = 0
            self.last_reset_date = current_date
            self.max_drawdown_reached = False
            self.consecutive_losses = 0
            print(f"✅ Daily counters reset for {current_date}")
    
    def can_place_order(self):
        """Check if we can place a new order based on risk parameters"""
        try:
            self.reset_daily_counters()
            
            if not MT5_AVAILABLE:
                # Simulation mode - more lenient rules
                return self.daily_trades < self.config.MAX_ORDER_PER_SESSION
            
            # Check account info
            account_info = mt5.account_info()
            if not account_info:
                print("❌ Cannot get account info")
                return False
            
            # Check minimum balance
            if account_info.balance < self.config.SALDO_MINIMAL:
                print(f"❌ Balance too low: {account_info.balance} < {self.config.SALDO_MINIMAL}")
                return False
            
            # Check daily trade limit
            if self.daily_trades >= self.config.MAX_ORDER_PER_SESSION:
                print(f"⚠️ Daily trade limit reached: {self.daily_trades}")
                return False
            
            # Check consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                print(f"⚠️ Too many consecutive losses: {self.consecutive_losses}")
                return False
            
            # Check current open positions
            positions = mt5.positions_get()
            if positions and len(positions) >= 3:  # Max 3 open positions
                print("⚠️ Too many open positions")
                return False
            
            # Check drawdown
            if account_info.equity < account_info.balance * (1 - self.config.MAX_DRAWDOWN / 100):
                print(f"⚠️ Maximum drawdown reached")
                self.max_drawdown_reached = True
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Risk check error: {e}")
            return False
    
    def calculate_position_size(self, balance, risk_percent=None):
        """Calculate optimal position size based on risk management"""
        try:
            if risk_percent is None:
                risk_percent = self.config.MAX_RISK_PER_TRADE
            
            # Base calculation
            risk_amount = balance * (risk_percent / 100)
            
            # Adjust based on consecutive losses
            if self.consecutive_losses > 2:
                risk_amount *= 0.5  # Reduce risk after losses
            
            # Adjust based on daily performance
            if self.daily_loss > balance * 0.05:  # If daily loss > 5%
                risk_amount *= 0.3  # Significantly reduce risk
            
            # Calculate lot size (for forex/gold)
            # Assuming $10 per 0.01 lot for gold
            lot_size = max(0.01, min(1.0, risk_amount / 100))
            
            return round(lot_size, 2)
            
        except Exception as e:
            print(f"❌ Position size calculation error: {e}")
            return 0.01  # Default minimum
    
    def record_trade_result(self, profit_loss):
        """Record trade result for risk tracking"""
        try:
            self.daily_trades += 1
            
            if profit_loss > 0:
                self.daily_profit += profit_loss
                self.consecutive_losses = 0
                print(f"✅ Profit: {profit_loss:.2f}")
            else:
                self.daily_loss += abs(profit_loss)
                self.consecutive_losses += 1
                print(f"❌ Loss: {profit_loss:.2f} (Consecutive: {self.consecutive_losses})")
            
            # Log daily stats
            net_daily = self.daily_profit - self.daily_loss
            print(f"📊 Daily Stats - Trades: {self.daily_trades}, Net: {net_daily:.2f}")
            
        except Exception as e:
            print(f"❌ Error recording trade result: {e}")
    
    def get_risk_status(self):
        """Get current risk status summary"""
        try:
            if not MT5_AVAILABLE:
                return {
                    'status': 'SIMULATION',
                    'daily_trades': self.daily_trades,
                    'max_trades': self.config.MAX_ORDER_PER_SESSION,
                    'can_trade': self.can_place_order()
                }
            
            account_info = mt5.account_info()
            if not account_info:
                return {'status': 'ERROR', 'message': 'Cannot get account info'}
            
            drawdown_pct = (1 - account_info.equity / account_info.balance) * 100
            
            return {
                'status': 'LIVE' if not self.max_drawdown_reached else 'PAUSED',
                'balance': account_info.balance,
                'equity': account_info.equity,
                'daily_trades': self.daily_trades,
                'max_trades': self.config.MAX_ORDER_PER_SESSION,
                'consecutive_losses': self.consecutive_losses,
                'drawdown_pct': drawdown_pct,
                'can_trade': self.can_place_order()
            }
            
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}