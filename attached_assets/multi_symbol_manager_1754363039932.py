"""
Multi-Symbol Portfolio Management
Trade multiple symbols simultaneously with correlation analysis
"""

import numpy as np
from typing import Dict, List, Optional
import datetime

class MultiSymbolManager:
    def __init__(self):
        self.symbols = ['XAUUSDm', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD']
        self.price_history = {}
        self.correlation_matrix = {}
        self.portfolio_weights = {}
        self.max_symbols_active = 3
        
    def add_symbol(self, symbol: str):
        """Add new symbol to portfolio"""
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.price_history[symbol] = []
    
    def update_price_history(self, symbol: str, price: float):
        """Update price history for correlation analysis"""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append({
            'price': price,
            'timestamp': datetime.datetime.now()
        })
        
        # Keep only last 100 prices
        if len(self.price_history[symbol]) > 100:
            self.price_history[symbol] = self.price_history[symbol][-100:]
    
    def calculate_correlation_matrix(self) -> Dict:
        """Calculate correlation matrix between symbols"""
        correlations = {}
        
        for symbol1 in self.symbols:
            correlations[symbol1] = {}
            for symbol2 in self.symbols:
                if symbol1 == symbol2:
                    correlations[symbol1][symbol2] = 1.0
                else:
                    corr = self.calculate_correlation(symbol1, symbol2)
                    correlations[symbol1][symbol2] = corr
        
        self.correlation_matrix = correlations
        return correlations
    
    def calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        if symbol1 not in self.price_history or symbol2 not in self.price_history:
            return 0.0
        
        prices1 = [p['price'] for p in self.price_history[symbol1]]
        prices2 = [p['price'] for p in self.price_history[symbol2]]
        
        if len(prices1) < 10 or len(prices2) < 10:
            return 0.0
        
        # Align lengths
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]
        
        try:
            correlation = np.corrcoef(prices1, prices2)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def get_portfolio_allocation(self, balance: float) -> Dict:
        """Calculate optimal portfolio allocation"""
        self.calculate_correlation_matrix()
        
        # Simple equal weight allocation with correlation adjustment
        base_weight = 1.0 / len(self.symbols)
        weights = {}
        
        for symbol in self.symbols:
            # Reduce weight for highly correlated assets
            correlation_penalty = 0
            for other_symbol in self.symbols:
                if symbol != other_symbol:
                    corr = abs(self.correlation_matrix.get(symbol, {}).get(other_symbol, 0))
                    if corr > 0.7:  # High correlation threshold
                        correlation_penalty += corr * 0.1
            
            weights[symbol] = max(base_weight - correlation_penalty, 0.1)
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {symbol: weight/total_weight for symbol, weight in weights.items()}
        
        # Convert to position sizes
        allocation = {}
        for symbol, weight in weights.items():
            allocation[symbol] = {
                'weight': weight,
                'allocation_amount': balance * weight,
                'max_risk_per_trade': min(weight * 2, 0.02)  # Max 2% per trade
            }
        
        self.portfolio_weights = allocation
        return allocation
    
    def should_trade_symbol(self, symbol: str, active_positions: List[str]) -> bool:
        """Check if symbol should be traded based on portfolio rules"""
        # Don't exceed max active symbols
        if len(active_positions) >= self.max_symbols_active and symbol not in active_positions:
            return False
        
        # Check correlation with active positions
        for active_symbol in active_positions:
            if symbol != active_symbol:
                corr = abs(self.correlation_matrix.get(symbol, {}).get(active_symbol, 0))
                if corr > 0.8:  # Very high correlation
                    return False
        
        return True
    
    def get_position_size(self, symbol: str, balance: float, risk_percent: float) -> float:
        """Calculate position size for symbol considering portfolio allocation"""
        if symbol not in self.portfolio_weights:
            return balance * risk_percent / 100
        
        allocation = self.portfolio_weights[symbol]
        max_risk = allocation['max_risk_per_trade']
        
        return balance * min(risk_percent / 100, max_risk)
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary statistics"""
        if not self.correlation_matrix:
            self.calculate_correlation_matrix()
        
        summary = {
            'total_symbols': len(self.symbols),
            'correlation_matrix': self.correlation_matrix,
            'portfolio_weights': self.portfolio_weights,
            'diversification_score': self.calculate_diversification_score()
        }
        
        return summary
    
    def calculate_diversification_score(self) -> float:
        """Calculate portfolio diversification score (0-1)"""
        if not self.correlation_matrix:
            return 0.0
        
        total_correlations = 0
        count = 0
        
        for symbol1 in self.symbols:
            for symbol2 in self.symbols:
                if symbol1 != symbol2:
                    corr = abs(self.correlation_matrix.get(symbol1, {}).get(symbol2, 0))
                    total_correlations += corr
                    count += 1
        
        if count == 0:
            return 1.0
        
        avg_correlation = total_correlations / count
        diversification_score = 1.0 - avg_correlation
        
        return max(0.0, min(1.0, diversification_score))