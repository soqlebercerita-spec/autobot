# Overview

AuraTrade Bot is an institutional-level automated trading system designed for high-frequency trading (HFT) and advanced market analysis. The system integrates with MetaTrader5 for live trading capabilities while providing comprehensive simulation modes for testing and development. The bot features advanced technical analysis, machine learning-powered predictions, real-time risk management, and multiple trading strategies optimized for different market conditions.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Component Architecture Pattern
The system follows a modular microservices-inspired architecture with clear separation of concerns. Each module is designed to be independent and communicates through well-defined interfaces, allowing for easy maintenance and scaling.

## Core Engine Layer
- **Trading Engine**: Central orchestrator that coordinates all trading activities and strategies (`core/trading_engine.py`)
- **MT5 Connector**: Handles all MetaTrader5 API interactions with fallback to simulation mode (`core/mt5_connector.py`)
- **Order Manager**: Manages order placement, modification, and tracking with performance monitoring (`core/order_manager.py`)
- **Risk Manager**: Implements comprehensive risk management with VaR calculations and portfolio protection (`core/risk_manager.py`)
- **Position Sizing**: Advanced position sizing algorithms including Kelly Criterion and volatility adjustment (`core/position_sizing.py`)

## Trading Strategy Framework
- **HFT Strategy**: Ultra-fast execution targeting <1ms with market making and arbitrage capabilities (`strategies/hft_strategy.py`)
- **Scalping Strategy**: Short-term trading optimized for quick profits with tight TP/SL parameters (`strategies/scalping_strategy.py`)
- **Arbitrage Strategy**: Statistical and cross-market arbitrage with mean reversion detection (`strategies/arbitrage_strategy.py`)
- **Strategy Engine**: Pluggable architecture allowing multiple strategies to run simultaneously

## Analysis and Intelligence Layer
- **Technical Analysis**: Custom implementation of technical indicators without TA-Lib dependency (`analysis/technical_analysis.py`)
- **Pattern Recognition**: Advanced price pattern detection for chart patterns and candlestick formations (`analysis/pattern_recognition.py`)
- **Market Conditions**: Real-time market regime detection and trading session classification (`analysis/market_conditions.py`)
- **Sentiment Analyzer**: News and social media sentiment analysis for trading decisions (`analysis/sentiment_analyzer.py`)
- **Machine Learning Engine**: LSTM/CNN/Transformer models for market prediction and pattern recognition (`core/ml_engine.py`)

## Data Management Layer
- **Data Manager**: Centralized data storage and retrieval with SQLite backend and intelligent caching (`data/data_manager.py`)
- **Tick Data Cache**: High-performance in-memory tick data storage with configurable retention
- **OHLC Data Cache**: Multi-timeframe price data caching with automatic refresh mechanisms
- **Performance Tracking**: Comprehensive logging of execution times, slippage, and cache performance

## User Interface Layer
- **Main Window**: Primary GUI interface built with Tkinter for cross-platform compatibility (`gui/main_window.py`)
- **Dashboard**: Real-time trading dashboard with performance metrics and position monitoring (`gui/dashboard.py`)
- **Charts**: Advanced financial charts using matplotlib with technical indicator overlays (`gui/charts.py`)
- **Multi-tab Interface**: Organized controls for basic settings, risk management, and performance monitoring

## Configuration and Settings
- **Hierarchical Config System**: Layered configuration with environment variables, files, and GUI overrides
- **Trading Mode Support**: Normal, Scalping, and HFT modes with specialized parameters
- **Balance-Based TP/SL**: All profit/loss calculations based on account balance percentage rather than market price
- **Risk Parameters**: Configurable daily limits, drawdown protection, and exposure controls

## Cross-Platform Compatibility
- **MT5 Wrapper**: Provides fallback functionality when MetaTrader5 is not available (`tradebot/mt5_wrapper.py`)
- **Simulation Trading**: Complete trading simulation environment for testing and development
- **Graceful Degradation**: System continues to operate in simulation mode when live trading is unavailable

## Performance Optimization
- **Threading Model**: Separate threads for GUI updates, data collection, and trading execution
- **Caching Strategy**: Multi-level caching for market data, technical indicators, and trading signals
- **Memory Management**: Automatic cleanup and garbage collection for long-running operations
- **HFT Optimization**: Sub-millisecond execution targeting with optimized data structures

# External Dependencies

## Core Trading Platform
- **MetaTrader5**: Primary trading platform integration for live market data and order execution
- **MetaTrader5 Python API**: Official MT5 Python library for platform communication

## Data Processing and Analysis
- **NumPy**: Mathematical operations and array processing for technical analysis
- **Pandas**: Data manipulation and time series analysis for market data
- **SciKit-Learn**: Machine learning models and preprocessing for market prediction

## GUI and Visualization
- **Tkinter**: Cross-platform GUI framework (built-in with Python)
- **Matplotlib**: Financial charting and data visualization
- **Pillow**: Image processing for GUI enhancements

## External APIs and Services
- **Requests**: HTTP library for external API communication
- **Telegram Bot API**: Real-time notifications and alerts (optional)
- **News/Sentiment APIs**: External news sources for sentiment analysis (configurable)

## System and Performance
- **Threading**: Built-in Python threading for concurrent operations
- **SQLite**: Embedded database for data storage and trade logging
- **PSUtil**: System monitoring and performance optimization
- **Schedule**: Task scheduling for automated operations

## Optional Windows-Specific
- **PyWin32**: Windows-specific features and optimizations when available
- **Plyer**: Cross-platform notifications when supported
- **Colorama**: Enhanced console output formatting