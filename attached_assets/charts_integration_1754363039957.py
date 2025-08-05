"""
Enhanced Charts Integration
Real-time charts with candlestick display and technical analysis overlay
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import numpy as np
import datetime
from typing import List, Dict, Optional

class ChartsIntegration:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = None
        self.chart_data = {}
        self.indicators_data = {}
        self.setup_chart()
        
    def setup_chart(self):
        """Setup the main chart interface"""
        # Create chart frame
        chart_frame = ttk.LabelFrame(self.parent_frame, text="📈 Real-time Trading Charts", padding=10)
        chart_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        chart_frame.grid_rowconfigure(1, weight=1)
        chart_frame.grid_columnconfigure(0, weight=1)
        
        # Chart controls
        controls_frame = ttk.Frame(chart_frame)
        controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # Symbol selection
        ttk.Label(controls_frame, text="Symbol:").grid(row=0, column=0, padx=5)
        self.symbol_var = tk.StringVar(value="XAUUSDm")
        symbol_combo = ttk.Combobox(controls_frame, textvariable=self.symbol_var, 
                                   values=["XAUUSDm", "EURUSD", "GBPUSD", "USDJPY", "BTCUSD"],
                                   state="readonly")
        symbol_combo.grid(row=0, column=1, padx=5)
        
        # Timeframe selection
        ttk.Label(controls_frame, text="Timeframe:").grid(row=0, column=2, padx=5)
        self.timeframe_var = tk.StringVar(value="M15")
        timeframe_combo = ttk.Combobox(controls_frame, textvariable=self.timeframe_var,
                                      values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                                      state="readonly")
        timeframe_combo.grid(row=0, column=3, padx=5)
        
        # Indicator toggles
        self.show_ma_var = tk.BooleanVar(value=True)
        self.show_rsi_var = tk.BooleanVar(value=True)
        self.show_bb_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(controls_frame, text="MA", variable=self.show_ma_var,
                       command=self.update_chart).grid(row=0, column=4, padx=5)
        ttk.Checkbutton(controls_frame, text="RSI", variable=self.show_rsi_var,
                       command=self.update_chart).grid(row=0, column=5, padx=5)
        ttk.Checkbutton(controls_frame, text="Bollinger", variable=self.show_bb_var,
                       command=self.update_chart).grid(row=0, column=6, padx=5)
        
        # Update button
        ttk.Button(controls_frame, text="🔄 Update", 
                  command=self.update_chart).grid(row=0, column=7, padx=10)
        
        # Create matplotlib canvas
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        
        # Initialize chart
        self.create_initial_chart()
    
    def create_initial_chart(self):
        """Create initial chart layout"""
        self.figure.clear()
        
        # Main price chart
        self.ax1 = self.figure.add_subplot(3, 1, (1, 2))
        self.ax1.set_title("Price Chart with Technical Indicators")
        self.ax1.grid(True, alpha=0.3)
        
        # RSI subplot
        self.ax2 = self.figure.add_subplot(3, 1, 3)
        self.ax2.set_title("RSI (14)")
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_ylim(0, 100)
        self.ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        self.ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def add_price_data(self, symbol: str, price_data: Dict):
        """Add price data for charting"""
        if symbol not in self.chart_data:
            self.chart_data[symbol] = []
        
        self.chart_data[symbol].append({
            'timestamp': datetime.datetime.now(),
            'open': price_data.get('open', price_data.get('price', 0)),
            'high': price_data.get('high', price_data.get('price', 0)),
            'low': price_data.get('low', price_data.get('price', 0)),
            'close': price_data.get('close', price_data.get('price', 0)),
            'volume': price_data.get('volume', 100)
        })
        
        # Keep only last 100 candles
        if len(self.chart_data[symbol]) > 100:
            self.chart_data[symbol] = self.chart_data[symbol][-100:]
    
    def add_indicators_data(self, symbol: str, indicators: Dict):
        """Add technical indicators data"""
        if symbol not in self.indicators_data:
            self.indicators_data[symbol] = []
        
        self.indicators_data[symbol].append({
            'timestamp': datetime.datetime.now(),
            'ma_short': indicators.get('ma_short'),
            'ma_long': indicators.get('ma_long'),
            'rsi': indicators.get('rsi'),
            'bb_upper': indicators.get('bb_upper'),
            'bb_middle': indicators.get('bb_middle'),
            'bb_lower': indicators.get('bb_lower')
        })
        
        # Keep only last 100 data points
        if len(self.indicators_data[symbol]) > 100:
            self.indicators_data[symbol] = self.indicators_data[symbol][-100:]
    
    def update_chart(self):
        """Update chart with latest data"""
        symbol = self.symbol_var.get()
        
        if symbol not in self.chart_data or len(self.chart_data[symbol]) < 2:
            return
        
        self.figure.clear()
        
        # Get data
        data = self.chart_data[symbol][-50:]  # Last 50 candles
        timestamps = [d['timestamp'] for d in data]
        opens = [d['open'] for d in data]
        highs = [d['high'] for d in data]
        lows = [d['low'] for d in data]
        closes = [d['close'] for d in data]
        
        # Main price chart
        self.ax1 = self.figure.add_subplot(3, 1, (1, 2))
        
        # Create candlestick chart
        self.plot_candlesticks(self.ax1, timestamps, opens, highs, lows, closes)
        
        # Add technical indicators
        if symbol in self.indicators_data:
            indicators = self.indicators_data[symbol][-50:]
            
            if self.show_ma_var.get():
                self.plot_moving_averages(self.ax1, timestamps, indicators)
            
            if self.show_bb_var.get():
                self.plot_bollinger_bands(self.ax1, timestamps, indicators)
        
        self.ax1.set_title(f"{symbol} - {self.timeframe_var.get()}")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.tick_params(axis='x', rotation=45)
        
        # RSI subplot
        if self.show_rsi_var.get():
            self.ax2 = self.figure.add_subplot(3, 1, 3)
            if symbol in self.indicators_data:
                self.plot_rsi(self.ax2, timestamps, indicators)
            self.ax2.set_title("RSI (14)")
            self.ax2.grid(True, alpha=0.3)
            self.ax2.set_ylim(0, 100)
            self.ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
            self.ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
            self.ax2.tick_params(axis='x', rotation=45)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_candlesticks(self, ax, timestamps, opens, highs, lows, closes):
        """Plot candlestick chart"""
        for i in range(len(timestamps)):
            color = 'green' if closes[i] >= opens[i] else 'red'
            
            # Body
            body_height = abs(closes[i] - opens[i])
            body_bottom = min(opens[i], closes[i])
            
            ax.bar(timestamps[i], body_height, bottom=body_bottom, 
                  color=color, alpha=0.8, width=0.0003)
            
            # Wick
            ax.plot([timestamps[i], timestamps[i]], [lows[i], highs[i]], 
                   color=color, linewidth=1)
    
    def plot_moving_averages(self, ax, timestamps, indicators):
        """Plot moving averages"""
        ma_short = [ind.get('ma_short') for ind in indicators if ind.get('ma_short')]
        ma_long = [ind.get('ma_long') for ind in indicators if ind.get('ma_long')]
        
        if ma_short and len(ma_short) == len(timestamps):
            ax.plot(timestamps, ma_short, color='blue', linewidth=1, 
                   label='MA Short', alpha=0.7)
        
        if ma_long and len(ma_long) == len(timestamps):
            ax.plot(timestamps, ma_long, color='orange', linewidth=1, 
                   label='MA Long', alpha=0.7)
        
        ax.legend()
    
    def plot_bollinger_bands(self, ax, timestamps, indicators):
        """Plot Bollinger Bands"""
        bb_upper = [ind.get('bb_upper') for ind in indicators if ind.get('bb_upper')]
        bb_middle = [ind.get('bb_middle') for ind in indicators if ind.get('bb_middle')]
        bb_lower = [ind.get('bb_lower') for ind in indicators if ind.get('bb_lower')]
        
        if bb_upper and bb_lower and len(bb_upper) == len(timestamps):
            ax.plot(timestamps, bb_upper, color='purple', linewidth=1, 
                   alpha=0.5, label='BB Upper')
            ax.plot(timestamps, bb_middle, color='purple', linewidth=1, 
                   alpha=0.7, label='BB Middle')
            ax.plot(timestamps, bb_lower, color='purple', linewidth=1, 
                   alpha=0.5, label='BB Lower')
            
            # Fill between bands
            ax.fill_between(timestamps, bb_upper, bb_lower, 
                           color='purple', alpha=0.1)
    
    def plot_rsi(self, ax, timestamps, indicators):
        """Plot RSI indicator"""
        rsi_values = [ind.get('rsi') for ind in indicators if ind.get('rsi')]
        
        if rsi_values and len(rsi_values) == len(timestamps):
            ax.plot(timestamps, rsi_values, color='purple', linewidth=2)
            
            # Color fill for overbought/oversold
            for i in range(len(rsi_values)):
                if rsi_values[i] > 70:
                    ax.scatter(timestamps[i], rsi_values[i], color='red', s=20)
                elif rsi_values[i] < 30:
                    ax.scatter(timestamps[i], rsi_values[i], color='green', s=20)
    
    def add_trade_marker(self, timestamp: datetime.datetime, price: float, 
                        trade_type: str, symbol: str):
        """Add trade marker to chart"""
        if symbol == self.symbol_var.get():
            color = 'green' if trade_type == 'BUY' else 'red'
            marker = '^' if trade_type == 'BUY' else 'v'
            
            self.ax1.scatter(timestamp, price, color=color, marker=marker, 
                           s=100, zorder=5, label=f'{trade_type} Signal')
            self.canvas.draw()
    
    def get_chart_summary(self) -> Dict:
        """Get chart data summary"""
        summary = {}
        
        for symbol in self.chart_data:
            if self.chart_data[symbol]:
                latest = self.chart_data[symbol][-1]
                summary[symbol] = {
                    'last_price': latest['close'],
                    'data_points': len(self.chart_data[symbol]),
                    'last_update': latest['timestamp'].isoformat()
                }
        
        return summary
    
    def export_chart_data(self, symbol: str, filename: str):
        """Export chart data to CSV"""
        if symbol not in self.chart_data:
            return False
        
        try:
            import csv
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
                
                for data in self.chart_data[symbol]:
                    writer.writerow([
                        data['timestamp'].isoformat(),
                        data['open'],
                        data['high'],
                        data['low'],
                        data['close'],
                        data['volume']
                    ])
            
            return True
        except Exception as e:
            print(f"Export error: {e}")
            return False

class HeatMapIntegration:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.correlation_data = {}
        self.setup_heatmap()
    
    def setup_heatmap(self):
        """Setup correlation heatmap"""
        heatmap_frame = ttk.LabelFrame(self.parent_frame, text="🔥 Market Correlation Heatmap", padding=10)
        heatmap_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.figure_heatmap = Figure(figsize=(8, 6), dpi=100)
        self.canvas_heatmap = FigureCanvasTkAgg(self.figure_heatmap, heatmap_frame)
        self.canvas_heatmap.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # Update button
        ttk.Button(heatmap_frame, text="🔄 Update Heatmap", 
                  command=self.update_heatmap).grid(row=1, column=0, pady=10)
    
    def update_correlation_data(self, correlations: Dict):
        """Update correlation data for heatmap"""
        self.correlation_data = correlations
        self.update_heatmap()
    
    def update_heatmap(self):
        """Update correlation heatmap"""
        if not self.correlation_data:
            return
        
        self.figure_heatmap.clear()
        ax = self.figure_heatmap.add_subplot(1, 1, 1)
        
        # Prepare data for heatmap
        symbols = list(self.correlation_data.keys())
        correlation_matrix = []
        
        for symbol1 in symbols:
            row = []
            for symbol2 in symbols:
                correlation = self.correlation_data.get(symbol1, {}).get(symbol2, 0)
                row.append(correlation)
            correlation_matrix.append(row)
        
        # Create heatmap
        im = ax.imshow(correlation_matrix, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(symbols)))
        ax.set_yticks(range(len(symbols)))
        ax.set_xticklabels(symbols, rotation=45)
        ax.set_yticklabels(symbols)
        
        # Add correlation values as text
        for i in range(len(symbols)):
            for j in range(len(symbols)):
                text = ax.text(j, i, f'{correlation_matrix[i][j]:.2f}',
                             ha="center", va="center", color="black")
        
        ax.set_title("Symbol Correlation Matrix")
        self.figure_heatmap.colorbar(im, ax=ax)
        self.figure_heatmap.tight_layout()
        self.canvas_heatmap.draw()