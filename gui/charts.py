"""
Chart Widget for AuraTrade Bot
Real-time financial charts using matplotlib
"""

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading

from core.mt5_connector import MT5Connector
from analysis.technical_analysis import TechnicalAnalysis
from utils.logger import Logger

class ChartWidget:
    """Advanced chart widget for financial data visualization"""
    
    def __init__(self, parent, mt5_connector: MT5Connector):
        self.parent = parent
        self.mt5_connector = mt5_connector
        self.logger = Logger()
        self.technical_analysis = TechnicalAnalysis()
        
        # Chart configuration
        self.current_symbol = "EURUSD"
        self.current_timeframe = "H1"
        self.chart_bars = 200
        
        # Data storage
        self.price_data = None
        self.indicators = {}
        
        # Chart elements
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.ax_price = None
        self.ax_volume = None
        self.ax_indicators = []
        
        # Update control
        self.update_lock = threading.Lock()
        self.last_update = None
        
        # Style configuration
        plt.style.use('dark_background')
        self.colors = {
            'background': '#1e1e1e',
            'grid': '#404040',
            'bullish': '#00ff00',
            'bearish': '#ff0000',
            'volume': '#4080ff',
            'ma': '#ffff00',
            'signal': '#ff8000'
        }
        
        # Initialize chart
        self._create_chart_widget()
        self._setup_chart()
        
        self.logger.info("Chart Widget initialized")
    
    def _create_chart_widget(self):
        """Create chart widget components"""
        try:
            # Main container
            chart_frame = ttk.Frame(self.parent)
            chart_frame.pack(fill=tk.BOTH, expand=True)
            
            # Control panel
            control_frame = ttk.Frame(chart_frame)
            control_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Symbol selection
            ttk.Label(control_frame, text="Symbol:").pack(side=tk.LEFT, padx=(0, 5))
            self.symbol_var = tk.StringVar(value=self.current_symbol)
            symbol_combo = ttk.Combobox(control_frame, textvariable=self.symbol_var, 
                                      values=["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"],
                                      width=10, state="readonly")
            symbol_combo.pack(side=tk.LEFT, padx=(0, 10))
            symbol_combo.bind('<<ComboboxSelected>>', self._on_symbol_change)
            
            # Timeframe selection
            ttk.Label(control_frame, text="Timeframe:").pack(side=tk.LEFT, padx=(0, 5))
            self.timeframe_var = tk.StringVar(value=self.current_timeframe)
            timeframe_combo = ttk.Combobox(control_frame, textvariable=self.timeframe_var,
                                         values=["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
                                         width=8, state="readonly")
            timeframe_combo.pack(side=tk.LEFT, padx=(0, 10))
            timeframe_combo.bind('<<ComboboxSelected>>', self._on_timeframe_change)
            
            # Indicator controls
            ttk.Label(control_frame, text="Indicators:").pack(side=tk.LEFT, padx=(10, 5))
            
            self.show_ma_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(control_frame, text="MA", variable=self.show_ma_var,
                           command=self._on_indicator_change).pack(side=tk.LEFT, padx=(0, 5))
            
            self.show_bb_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(control_frame, text="Bollinger", variable=self.show_bb_var,
                           command=self._on_indicator_change).pack(side=tk.LEFT, padx=(0, 5))
            
            self.show_rsi_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(control_frame, text="RSI", variable=self.show_rsi_var,
                           command=self._on_indicator_change).pack(side=tk.LEFT, padx=(0, 5))
            
            self.show_macd_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(control_frame, text="MACD", variable=self.show_macd_var,
                           command=self._on_indicator_change).pack(side=tk.LEFT, padx=(0, 5))
            
            # Refresh button
            ttk.Button(control_frame, text="Refresh", 
                      command=self._refresh_chart).pack(side=tk.RIGHT, padx=(10, 0))
            
            # Chart container
            self.chart_container = ttk.Frame(chart_frame)
            self.chart_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
            
        except Exception as e:
            self.logger.error(f"Error creating chart widget: {str(e)}")
    
    def _setup_chart(self):
        """Setup matplotlib chart"""
        try:
            # Create figure with subplots
            self.figure = Figure(figsize=(12, 8), dpi=100, facecolor=self.colors['background'])
            
            # Create subplots
            gs = self.figure.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.1)
            
            # Main price chart
            self.ax_price = self.figure.add_subplot(gs[0])
            self.ax_price.set_facecolor(self.colors['background'])
            self.ax_price.grid(True, color=self.colors['grid'], alpha=0.3)
            
            # Volume chart
            self.ax_volume = self.figure.add_subplot(gs[1], sharex=self.ax_price)
            self.ax_volume.set_facecolor(self.colors['background'])
            self.ax_volume.grid(True, color=self.colors['grid'], alpha=0.3)
            
            # Indicator charts
            self.ax_rsi = self.figure.add_subplot(gs[2], sharex=self.ax_price)
            self.ax_rsi.set_facecolor(self.colors['background'])
            self.ax_rsi.grid(True, color=self.colors['grid'], alpha=0.3)
            self.ax_rsi.set_ylim(0, 100)
            
            self.ax_macd = self.figure.add_subplot(gs[3], sharex=self.ax_price)
            self.ax_macd.set_facecolor(self.colors['background'])
            self.ax_macd.grid(True, color=self.colors['grid'], alpha=0.3)
            
            # Create canvas
            self.canvas = FigureCanvasTkAgg(self.figure, self.chart_container)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Create navigation toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.chart_container)
            self.toolbar.update()
            
            # Configure date formatting
            self.ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.ax_price.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            
            # Labels
            self.ax_price.set_ylabel('Price', color='white')
            self.ax_volume.set_ylabel('Volume', color='white')
            self.ax_rsi.set_ylabel('RSI', color='white')
            self.ax_macd.set_ylabel('MACD', color='white')
            
            # Initial data load
            self.update_data()
            
        except Exception as e:
            self.logger.error(f"Error setting up chart: {str(e)}")
    
    def update_data(self):
        """Update chart data"""
        try:
            with self.update_lock:
                # Get OHLC data
                rates = self.mt5_connector.get_rates(self.current_symbol, self.current_timeframe, self.chart_bars)
                
                if rates is None or len(rates) == 0:
                    return
                
                self.price_data = rates
                
                # Calculate indicators
                self._calculate_indicators()
                
                # Update chart
                self._update_chart_display()
                
                self.last_update = datetime.now()
                
        except Exception as e:
            self.logger.error(f"Error updating chart data: {str(e)}")
    
    def _calculate_indicators(self):
        """Calculate technical indicators"""
        try:
            if self.price_data is None or len(self.price_data) == 0:
                return
            
            close_prices = self.price_data['close'].values
            high_prices = self.price_data['high'].values
            low_prices = self.price_data['low'].values
            volumes = self.price_data.get('tick_volume', self.price_data.get('real_volume', np.ones(len(close_prices)))).values
            
            # Moving Averages
            self.indicators['sma_20'] = self.technical_analysis.calculate_sma(close_prices, 20)
            self.indicators['sma_50'] = self.technical_analysis.calculate_sma(close_prices, 50)
            self.indicators['ema_12'] = self.technical_analysis.calculate_ema(close_prices, 12)
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self.technical_analysis.calculate_bollinger_bands(close_prices, 20, 2.0)
            self.indicators['bb_upper'] = bb_upper
            self.indicators['bb_middle'] = bb_middle
            self.indicators['bb_lower'] = bb_lower
            
            # RSI
            self.indicators['rsi'] = self.technical_analysis.calculate_rsi(close_prices, 14)
            
            # MACD
            macd_line, signal_line, histogram = self.technical_analysis.calculate_macd(close_prices)
            self.indicators['macd_line'] = macd_line
            self.indicators['macd_signal'] = signal_line
            self.indicators['macd_histogram'] = histogram
            
            # Volume indicators
            volume_analysis = self.technical_analysis.calculate_volume_indicators(close_prices, volumes)
            self.indicators['volume_sma'] = volume_analysis['volume_sma']
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
    
    def _update_chart_display(self):
        """Update chart display with current data"""
        try:
            # Clear all axes
            self.ax_price.clear()
            self.ax_volume.clear()
            self.ax_rsi.clear()
            self.ax_macd.clear()
            
            if self.price_data is None or len(self.price_data) == 0:
                return
            
            # Prepare data
            dates = self.price_data.index
            opens = self.price_data['open'].values
            highs = self.price_data['high'].values
            lows = self.price_data['low'].values
            closes = self.price_data['close'].values
            volumes = self.price_data.get('tick_volume', self.price_data.get('real_volume', np.ones(len(closes)))).values
            
            # Plot candlesticks
            self._plot_candlesticks(dates, opens, highs, lows, closes)
            
            # Plot indicators
            self._plot_moving_averages(dates)
            self._plot_bollinger_bands(dates)
            self._plot_volume(dates, volumes)
            self._plot_rsi(dates)
            self._plot_macd(dates)
            
            # Configure axes
            self._configure_axes()
            
            # Refresh canvas
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"Error updating chart display: {str(e)}")
    
    def _plot_candlesticks(self, dates, opens, highs, lows, closes):
        """Plot candlestick chart"""
        try:
            # Calculate colors
            colors = []
            for i in range(len(closes)):
                if closes[i] >= opens[i]:
                    colors.append(self.colors['bullish'])
                else:
                    colors.append(self.colors['bearish'])
            
            # Plot candlesticks using line plots (simplified)
            for i in range(len(dates)):
                # High-low line
                self.ax_price.plot([dates[i], dates[i]], [lows[i], highs[i]], 
                                 color='white', linewidth=0.8, alpha=0.8)
                
                # Body
                body_height = abs(closes[i] - opens[i])
                if body_height > 0:
                    self.ax_price.bar(dates[i], body_height, 
                                    bottom=min(opens[i], closes[i]),
                                    color=colors[i], alpha=0.8, width=0.8)
            
            # Plot close line
            self.ax_price.plot(dates, closes, color='white', linewidth=1, alpha=0.9)
            
        except Exception as e:
            self.logger.error(f"Error plotting candlesticks: {str(e)}")
    
    def _plot_moving_averages(self, dates):
        """Plot moving averages"""
        try:
            if not self.show_ma_var.get():
                return
            
            if 'sma_20' in self.indicators:
                valid_indices = ~np.isnan(self.indicators['sma_20'])
                if np.any(valid_indices):
                    self.ax_price.plot(dates[valid_indices], self.indicators['sma_20'][valid_indices], 
                                     color='yellow', linewidth=1, label='SMA 20', alpha=0.8)
            
            if 'sma_50' in self.indicators:
                valid_indices = ~np.isnan(self.indicators['sma_50'])
                if np.any(valid_indices):
                    self.ax_price.plot(dates[valid_indices], self.indicators['sma_50'][valid_indices], 
                                     color='orange', linewidth=1, label='SMA 50', alpha=0.8)
            
            if 'ema_12' in self.indicators:
                valid_indices = ~np.isnan(self.indicators['ema_12'])
                if np.any(valid_indices):
                    self.ax_price.plot(dates[valid_indices], self.indicators['ema_12'][valid_indices], 
                                     color='cyan', linewidth=1, label='EMA 12', alpha=0.8)
            
        except Exception as e:
            self.logger.error(f"Error plotting moving averages: {str(e)}")
    
    def _plot_bollinger_bands(self, dates):
        """Plot Bollinger Bands"""
        try:
            if not self.show_bb_var.get():
                return
            
            if all(key in self.indicators for key in ['bb_upper', 'bb_middle', 'bb_lower']):
                upper = self.indicators['bb_upper']
                middle = self.indicators['bb_middle']
                lower = self.indicators['bb_lower']
                
                valid_indices = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
                
                if np.any(valid_indices):
                    valid_dates = dates[valid_indices]
                    
                    self.ax_price.plot(valid_dates, upper[valid_indices], 
                                     color='purple', linewidth=1, alpha=0.6, label='BB Upper')
                    self.ax_price.plot(valid_dates, middle[valid_indices], 
                                     color='purple', linewidth=1, alpha=0.8, label='BB Middle')
                    self.ax_price.plot(valid_dates, lower[valid_indices], 
                                     color='purple', linewidth=1, alpha=0.6, label='BB Lower')
                    
                    # Fill between bands
                    self.ax_price.fill_between(valid_dates, upper[valid_indices], lower[valid_indices],
                                             color='purple', alpha=0.1)
            
        except Exception as e:
            self.logger.error(f"Error plotting Bollinger Bands: {str(e)}")
    
    def _plot_volume(self, dates, volumes):
        """Plot volume chart"""
        try:
            # Plot volume bars
            colors = ['green' if i % 2 == 0 else 'red' for i in range(len(volumes))]  # Simplified coloring
            self.ax_volume.bar(dates, volumes, color=colors, alpha=0.6, width=0.8)
            
            # Plot volume moving average if available
            if 'volume_sma' in self.indicators:
                valid_indices = ~np.isnan(self.indicators['volume_sma'])
                if np.any(valid_indices):
                    self.ax_volume.plot(dates[valid_indices], self.indicators['volume_sma'][valid_indices],
                                      color='white', linewidth=1, alpha=0.8)
            
        except Exception as e:
            self.logger.error(f"Error plotting volume: {str(e)}")
    
    def _plot_rsi(self, dates):
        """Plot RSI indicator"""
        try:
            if not self.show_rsi_var.get():
                return
            
            if 'rsi' in self.indicators:
                rsi = self.indicators['rsi']
                self.ax_rsi.plot(dates, rsi, color='yellow', linewidth=1)
                
                # Plot RSI levels
                self.ax_rsi.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                self.ax_rsi.axhline(y=30, color='green', linestyle='--', alpha=0.5)
                self.ax_rsi.axhline(y=50, color='white', linestyle='-', alpha=0.3)
                
                self.ax_rsi.set_ylim(0, 100)
            
        except Exception as e:
            self.logger.error(f"Error plotting RSI: {str(e)}")
    
    def _plot_macd(self, dates):
        """Plot MACD indicator"""
        try:
            if not self.show_macd_var.get():
                return
            
            if all(key in self.indicators for key in ['macd_line', 'macd_signal', 'macd_histogram']):
                macd_line = self.indicators['macd_line']
                signal_line = self.indicators['macd_signal']
                histogram = self.indicators['macd_histogram']
                
                # Plot MACD line and signal line
                valid_macd = ~np.isnan(macd_line)
                valid_signal = ~np.isnan(signal_line)
                
                if np.any(valid_macd):
                    self.ax_macd.plot(dates[valid_macd], macd_line[valid_macd], 
                                    color='blue', linewidth=1, label='MACD')
                
                if np.any(valid_signal):
                    self.ax_macd.plot(dates[valid_signal], signal_line[valid_signal], 
                                    color='red', linewidth=1, label='Signal')
                
                # Plot histogram
                valid_hist = ~np.isnan(histogram)
                if np.any(valid_hist):
                    colors = ['green' if h >= 0 else 'red' for h in histogram[valid_hist]]
                    self.ax_macd.bar(dates[valid_hist], histogram[valid_hist], 
                                   color=colors, alpha=0.6, width=0.8)
                
                # Zero line
                self.ax_macd.axhline(y=0, color='white', linestyle='-', alpha=0.3)
            
        except Exception as e:
            self.logger.error(f"Error plotting MACD: {str(e)}")
    
    def _configure_axes(self):
        """Configure chart axes"""
        try:
            # Configure price axis
            self.ax_price.set_facecolor(self.colors['background'])
            self.ax_price.grid(True, color=self.colors['grid'], alpha=0.3)
            self.ax_price.set_ylabel('Price', color='white')
            self.ax_price.tick_params(colors='white')
            
            # Show legend if indicators are displayed
            if (self.show_ma_var.get() or self.show_bb_var.get()):
                self.ax_price.legend(loc='upper left', fontsize=8, facecolor='black', edgecolor='white')
            
            # Configure volume axis
            self.ax_volume.set_facecolor(self.colors['background'])
            self.ax_volume.grid(True, color=self.colors['grid'], alpha=0.3)
            self.ax_volume.set_ylabel('Volume', color='white')
            self.ax_volume.tick_params(colors='white')
            
            # Configure RSI axis
            self.ax_rsi.set_facecolor(self.colors['background'])
            self.ax_rsi.grid(True, color=self.colors['grid'], alpha=0.3)
            self.ax_rsi.set_ylabel('RSI', color='white')
            self.ax_rsi.tick_params(colors='white')
            self.ax_rsi.set_ylim(0, 100)
            
            # Configure MACD axis
            self.ax_macd.set_facecolor(self.colors['background'])
            self.ax_macd.grid(True, color=self.colors['grid'], alpha=0.3)
            self.ax_macd.set_ylabel('MACD', color='white')
            self.ax_macd.set_xlabel('Time', color='white')
            self.ax_macd.tick_params(colors='white')
            
            # Format dates
            self.figure.autofmt_xdate()
            
            # Set title
            title = f"{self.current_symbol} - {self.current_timeframe}"
            if self.last_update:
                title += f" (Updated: {self.last_update.strftime('%H:%M:%S')})"
            
            self.ax_price.set_title(title, color='white', fontsize=12)
            
            # Hide indicator axes if not shown
            if not self.show_rsi_var.get():
                self.ax_rsi.set_visible(False)
            else:
                self.ax_rsi.set_visible(True)
            
            if not self.show_macd_var.get():
                self.ax_macd.set_visible(False)
            else:
                self.ax_macd.set_visible(True)
            
        except Exception as e:
            self.logger.error(f"Error configuring axes: {str(e)}")
    
    # Event handlers
    def _on_symbol_change(self, event=None):
        """Handle symbol change"""
        try:
            self.current_symbol = self.symbol_var.get()
            self.update_data()
        except Exception as e:
            self.logger.error(f"Error changing symbol: {str(e)}")
    
    def _on_timeframe_change(self, event=None):
        """Handle timeframe change"""
        try:
            self.current_timeframe = self.timeframe_var.get()
            self.update_data()
        except Exception as e:
            self.logger.error(f"Error changing timeframe: {str(e)}")
    
    def _on_indicator_change(self):
        """Handle indicator display change"""
        try:
            self._update_chart_display()
        except Exception as e:
            self.logger.error(f"Error changing indicators: {str(e)}")
    
    def _refresh_chart(self):
        """Refresh chart data"""
        try:
            self.update_data()
        except Exception as e:
            self.logger.error(f"Error refreshing chart: {str(e)}")
    
    def set_symbol(self, symbol: str):
        """Set chart symbol programmatically"""
        try:
            self.symbol_var.set(symbol)
            self.current_symbol = symbol
            self.update_data()
        except Exception as e:
            self.logger.error(f"Error setting symbol: {str(e)}")
    
    def set_timeframe(self, timeframe: str):
        """Set chart timeframe programmatically"""
        try:
            self.timeframe_var.set(timeframe)
            self.current_timeframe = timeframe
            self.update_data()
        except Exception as e:
            self.logger.error(f"Error setting timeframe: {str(e)}")
    
    def add_signal_marker(self, timestamp: datetime, price: float, signal_type: str, color: str = 'yellow'):
        """Add trading signal marker to chart"""
        try:
            if self.ax_price and self.price_data is not None:
                # Find nearest data point
                nearest_idx = self.price_data.index.get_loc(timestamp, method='nearest')
                marker_time = self.price_data.index[nearest_idx]
                
                # Add marker
                marker = 'o' if signal_type.upper() == 'BUY' else 'v'
                self.ax_price.scatter(marker_time, price, c=color, s=100, marker=marker, 
                                    alpha=0.8, edgecolors='black', linewidth=1, zorder=5)
                
                # Add text label
                self.ax_price.annotate(signal_type, xy=(marker_time, price), 
                                     xytext=(10, 10), textcoords='offset points',
                                     color=color, fontsize=8, fontweight='bold')
                
                self.canvas.draw()
                
        except Exception as e:
            self.logger.error(f"Error adding signal marker: {str(e)}")
    
    def get_current_data(self) -> Optional[Dict[str, Any]]:
        """Get current chart data"""
        try:
            if self.price_data is None:
                return None
            
            return {
                'symbol': self.current_symbol,
                'timeframe': self.current_timeframe,
                'data': self.price_data,
                'indicators': self.indicators,
                'last_update': self.last_update
            }
            
        except Exception as e:
            self.logger.error(f"Error getting current data: {str(e)}")
            return None
