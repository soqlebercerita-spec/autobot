# Trading Bot Launcher Files

## Available .bat Files

### 1. Start_Multi_Mode_Trading.bat
- Launches the main trading mode selector
- Choose between Normal, Scalping, and HFT modes
- Each mode opens in a separate window

### 2. Start_Windows_Trading_Bot.bat  
- Launches the enhanced Windows MT5 trading bot
- Full-featured trading interface
- Works with simulation and real MT5

### 3. Start_Real_MT5_Trader.bat
- **FOR REAL MONEY TRADING ONLY**
- Requires MT5 installed and connected
- Balance-based TP/SL calculations
- Enhanced safety features

### 4. Start_Trading_Bot_Integrated.bat
- Launches the integrated trading bot
- Cross-platform compatibility
- Good for testing and learning

### 5. Quick_Launch_HFT.bat
- Direct launch to HFT mode
- Ultra-fast 1-second scanning
- 0.3% TP, 1.5% SL of account balance

## Requirements

- Python 3.8+
- All required packages installed
- For real trading: MT5 installed and configured

## Safety Notice

⚠️ **IMPORTANT**: Files marked for real trading use actual money. Always test in simulation mode first.

## Usage

1. Double-click any .bat file to launch
2. The console window will show startup progress
3. Trading GUI will open automatically
4. Press any key in console to close when done

## Trading Modes

- **Normal**: Conservative (1% TP, 3% SL)
- **Scalping**: Quick trades (0.5% TP, 2% SL)  
- **HFT**: Ultra-fast (0.3% TP, 1.5% SL)

All TP/SL percentages are based on account balance, not market price.