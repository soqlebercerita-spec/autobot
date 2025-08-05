"""
Advanced Backtesting Engine
Historical strategy validation and optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import datetime

class BacktestingEngine:
    def __init__(self):
        self.historical_data = {}
        self.strategies = []
        self.results = {}
        
    def load_historical_data(self, symbol: str, timeframe: str = 'H1', bars: int = 1000):
        """Load historical data for backtesting"""
        # In real implementation, this would connect to MT5 or data provider
        # For now, generate sample data
        dates = pd.date_range(
            start=datetime.datetime.now() - datetime.timedelta(days=bars//24),
            periods=bars,
            freq='H'
        )
        
        # Generate realistic price data
        np.random.seed(42)
        price_changes = np.random.normal(0, 0.001, bars)
        prices = [2000.0]  # Starting price for Gold
        
        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        self.historical_data[symbol] = pd.DataFrame({
            'datetime': dates,
            'open': prices[:-1],
            'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices[:-1]],
            'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices[:-1]],
            'close': prices[1:],
            'volume': np.random.randint(100, 1000, bars)
        })
        
        return self.historical_data[symbol]
    
    def add_strategy(self, strategy_config: Dict):
        """Add strategy configuration for backtesting"""
        self.strategies.append(strategy_config)
    
    def run_backtest(self, symbol: str, strategy_config: Dict, start_balance: float = 10000) -> Dict:
        """Run backtest for specific strategy"""
        if symbol not in self.historical_data:
            self.load_historical_data(symbol)
        
        data = self.historical_data[symbol].copy()
        
        # Initialize tracking variables
        balance = start_balance
        positions = []
        trades = []
        equity_curve = [balance]
        
        # Strategy parameters
        tp_percent = strategy_config.get('tp_percent', 1.0)
        sl_percent = strategy_config.get('sl_percent', 2.0)
        lot_size = strategy_config.get('lot_size', 0.01)
        
        for i in range(len(data)):
            current_price = data.iloc[i]['close']
            current_time = data.iloc[i]['datetime']
            
            # Check existing positions for TP/SL
            for position in positions[:]:
                if position['type'] == 'BUY':
                    pnl_percent = (current_price - position['entry_price']) / position['entry_price'] * 100
                    if pnl_percent >= tp_percent or pnl_percent <= -sl_percent:
                        # Close position
                        pnl = balance * (pnl_percent / 100)
                        balance += pnl
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'type': position['type'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_percent': pnl_percent
                        })
                        
                        positions.remove(position)
                
                elif position['type'] == 'SELL':
                    pnl_percent = (position['entry_price'] - current_price) / position['entry_price'] * 100
                    if pnl_percent >= tp_percent or pnl_percent <= -sl_percent:
                        # Close position
                        pnl = balance * (pnl_percent / 100)
                        balance += pnl
                        
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': current_time,
                            'type': position['type'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_percent': pnl_percent
                        })
                        
                        positions.remove(position)
            
            # Generate trading signals (simplified)
            if i > 20 and len(positions) == 0:  # Only one position at a time
                signal = self.generate_signal(data.iloc[max(0, i-20):i+1])
                
                if signal['action'] in ['BUY', 'SELL']:
                    positions.append({
                        'type': signal['action'],
                        'entry_price': current_price,
                        'entry_time': current_time,
                        'lot_size': lot_size
                    })
            
            equity_curve.append(balance)
        
        # Calculate performance metrics
        results = self.calculate_performance_metrics(trades, start_balance, balance, equity_curve)
        results['trades'] = trades
        results['equity_curve'] = equity_curve
        
        return results
    
    def generate_signal(self, data_slice) -> Dict:
        """Generate simplified trading signal for backtesting"""
        if len(data_slice) < 10:
            return {'action': 'HOLD', 'confidence': 0}
        
        # Simple moving average crossover strategy
        short_ma = data_slice['close'].rolling(5).mean().iloc[-1]
        long_ma = data_slice['close'].rolling(10).mean().iloc[-1]
        current_price = data_slice['close'].iloc[-1]
        
        if short_ma > long_ma and current_price > short_ma:
            return {'action': 'BUY', 'confidence': 0.7}
        elif short_ma < long_ma and current_price < short_ma:
            return {'action': 'SELL', 'confidence': 0.7}
        else:
            return {'action': 'HOLD', 'confidence': 0}
    
    def calculate_performance_metrics(self, trades: List[Dict], start_balance: float, 
                                    end_balance: float, equity_curve: List[float]) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Basic metrics
        total_return = ((end_balance - start_balance) / start_balance) * 100
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Maximum drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        for balance in equity_curve:
            if balance > peak:
                peak = balance
            drawdown = ((peak - balance) / peak) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Sharpe ratio (simplified)
        returns = [equity_curve[i] - equity_curve[i-1] for i in range(1, len(equity_curve))]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        }
    
    def optimize_strategy(self, symbol: str, parameter_ranges: Dict) -> Dict:
        """Optimize strategy parameters using grid search"""
        best_result = None
        best_params = None
        best_return = -float('inf')
        
        # Generate parameter combinations
        param_combinations = self.generate_param_combinations(parameter_ranges)
        
        for params in param_combinations[:50]:  # Limit to 50 combinations
            result = self.run_backtest(symbol, params)
            if result['total_return'] > best_return:
                best_return = result['total_return']
                best_result = result
                best_params = params
        
        return {
            'best_params': best_params,
            'best_result': best_result,
            'optimization_summary': {
                'combinations_tested': len(param_combinations[:50]),
                'best_return': best_return
            }
        }
    
    def generate_param_combinations(self, parameter_ranges: Dict) -> List[Dict]:
        """Generate parameter combinations for optimization"""
        combinations = []
        
        tp_values = parameter_ranges.get('tp_percent', [0.5, 1.0, 1.5, 2.0])
        sl_values = parameter_ranges.get('sl_percent', [1.0, 2.0, 3.0, 4.0])
        lot_values = parameter_ranges.get('lot_size', [0.01, 0.02, 0.05])
        
        for tp in tp_values:
            for sl in sl_values:
                for lot in lot_values:
                    combinations.append({
                        'tp_percent': tp,
                        'sl_percent': sl,
                        'lot_size': lot
                    })
        
        return combinations