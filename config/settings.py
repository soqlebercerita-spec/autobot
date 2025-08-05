
"""
Trading Settings and Parameters for AuraTrade
Dynamic settings that can be adjusted during runtime
"""

class TradingSettings:
    """Dynamic trading settings"""
    
    def __init__(self):
        # Current Trading Mode
        self.trading_mode = "NORMAL"  # NORMAL, SCALPING, HFT
        self.auto_trading_enabled = False
        self.simulation_mode = True
        
        # Current Session Settings
        self.max_trades_per_session = 100
        self.current_trades_count = 0
        self.session_profit_target = 5.0  # 5% daily target
        self.session_loss_limit = 2.0  # 2% daily loss limit
        
        # Dynamic Risk Adjustments
        self.current_risk_multiplier = 1.0
        self.volatility_adjustment = 1.0
        self.market_condition = "NORMAL"  # TRENDING, RANGING, VOLATILE
        
        # Performance Tracking
        self.win_rate = 0.0
        self.total_trades = 0
        self.profitable_trades = 0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
    def update_performance(self, trade_result: dict):
        """Update performance metrics"""
        self.total_trades += 1
        if trade_result.get("profit", 0) > 0:
            self.profitable_trades += 1
        
        self.win_rate = (self.profitable_trades / self.total_trades) * 100
        
        # Update drawdown
        current_dd = trade_result.get("drawdown", 0)
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        self.current_drawdown = current_dd
    
    def should_continue_trading(self) -> bool:
        """Check if trading should continue based on current performance"""
        # Check trade limits 
        if self.current_trades_count >= self.max_trades_per_session:
            return False
        
        # Check drawdown limits
        if self.current_drawdown >= self.session_loss_limit:
            return False
        
        # Check if profit target reached
        # (could stop or continue based on strategy)
        
        return True
    
    def adjust_risk_for_market_condition(self):
        """Adjust risk based on current market conditions"""
        if self.market_condition == "VOLATILE":
            self.current_risk_multiplier = 0.5  # Reduce risk
        elif self.market_condition == "TRENDING":
            self.current_risk_multiplier = 1.2  # Increase risk slightly
        else:
            self.current_risk_multiplier = 1.0  # Normal risk

# Global settings instance
Settings = TradingSettings()
