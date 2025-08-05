@echo off
title Trading Bot - Windows Installation
color 0A
cls

echo ========================================
echo 🚀 TRADING BOT - WINDOWS INSTALLER
echo    Enhanced MT5 Integration Setup
echo ========================================
echo.

REM Check Administrator privileges
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ Running as Administrator
) else (
    echo ⚠️  Recommended to run as Administrator for best results
)
echo.

REM Check if Python is installed
echo 🔍 Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! 
    echo 📥 Please install Python 3.8+ from: https://www.python.org/downloads/
    echo    Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found
python --version
echo.

REM Check pip
echo 🔍 Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip not found! Please reinstall Python with pip included
    pause
    exit /b 1
)
echo ✅ pip found
echo.

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install requirements
echo 📦 Installing Windows Trading Bot requirements...
echo    This may take a few minutes...
echo.

pip install MetaTrader5
if errorlevel 1 (
    echo ⚠️  MetaTrader5 installation failed - you may need to install manually
) else (
    echo ✅ MetaTrader5 installed
)

pip install numpy requests pandas matplotlib Pillow
if errorlevel 1 (
    echo ❌ Core packages installation failed
    pause
    exit /b 1
) else (
    echo ✅ Core packages installed
)

pip install psutil schedule pywin32 plyer colorama
if errorlevel 1 (
    echo ⚠️  Some optional packages failed - continuing anyway
) else (
    echo ✅ Optional packages installed
)

echo.
echo ========================================
echo ✅ INSTALLATION COMPLETE!
echo ========================================
echo.
echo 📋 Next Steps:
echo 1. Install and login to MetaTrader5
echo 2. Enable "Allow automated trading" in MT5
echo 3. Run: START_TRADING_BOT.py
echo.
echo 🔧 Configuration:
echo - Edit config.py for trading parameters
echo - Add Telegram credentials for notifications
echo.
echo 📖 Files created:
echo - Trading bot files are ready
echo - Logs folder will be created automatically
echo - Configuration files are set
echo.
pause