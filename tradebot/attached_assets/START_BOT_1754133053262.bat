@echo off
title Trading Bot - Auto Trading MT5
color 0A

echo ========================================
echo 🤖 TRADING BOT - WINDOWS MT5
echo    Auto Trading ^& Scalping Bot
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install requirements if needed
echo 📦 Checking requirements...
pip install --quiet MetaTrader5 numpy requests >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Installing requirements...
    pip install MetaTrader5 numpy requests
)

echo ✅ Requirements ready
echo.

REM Check if MT5 files exist
if exist "trading_bot_windows.py" (
    echo 🚀 Starting Windows MT5 version...
    python trading_bot_windows.py
) else if exist "trading_bot_integrated.py" (
    echo 🚀 Starting integrated version...
    python trading_bot_integrated.py
) else if exist "START_TRADING_BOT.py" (
    echo 🚀 Starting launcher...
    python START_TRADING_BOT.py
) else (
    echo ❌ Bot files not found!
    echo Please ensure you have the trading bot files in this folder.
)

echo.
echo 👋 Trading session ended
pause