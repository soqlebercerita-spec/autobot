@echo off
title Enhanced Trading Bot - Windows MT5
color 0A
cls

echo ========================================
echo 🚀 ENHANCED TRADING BOT - WINDOWS MT5
echo    Fixed Price Retrieval ^& Signal Generation
echo    Optimized Market Opportunity Capture
echo ========================================
echo.

REM Check if running from correct directory
if not exist "trading_bot_windows.py" (
    if not exist "trading_bot_integrated.py" (
        if not exist "START_TRADING_BOT.py" (
            echo ❌ Trading bot files not found!
            echo 📁 Please run this from the bot directory
            pause
            exit /b 1
        )
    )
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo 📥 Install Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Quick dependency check
echo 🔍 Checking dependencies...
python -c "import numpy, requests" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Dependencies missing! Running installer...
    call INSTALL_WINDOWS.bat
    if errorlevel 1 (
        echo ❌ Installation failed
        pause
        exit /b 1
    )
)
echo ✅ Dependencies OK
echo.

REM Check MetaTrader5 (for Windows version)
python -c "import MetaTrader5" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  MetaTrader5 not available - using simulation mode
    if exist "trading_bot_integrated.py" (
        echo 🔬 Starting Enhanced Simulation Bot...
        python trading_bot_integrated.py
    ) else (
        echo 🚀 Starting Bot Launcher...
        python START_TRADING_BOT.py
    )
) else (
    echo ✅ MetaTrader5 available
    if exist "trading_bot_windows.py" (
        echo 🪟 Starting Enhanced Windows MT5 Bot...
        python trading_bot_windows.py
    ) else if exist "trading_bot_integrated.py" (
        echo 🔬 Starting Enhanced Simulation Bot...
        python trading_bot_integrated.py
    ) else (
        echo 🚀 Starting Bot Launcher...
        python START_TRADING_BOT.py
    )
)

echo.
echo 👋 Trading session ended
echo 📊 Check logs folder for detailed information
pause