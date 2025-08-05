
@echo off
title AuraTrade - Install Requirements
color 0E
cls

echo ========================================
echo 📦 AURA TRADE - REQUIREMENTS INSTALLER
echo    Installing Python packages...
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
python --version
echo.

echo 📦 Installing required packages...
echo This may take a few minutes...
echo.

REM Install basic requirements
pip install schedule colorama requests numpy pandas matplotlib pillow psutil

if errorlevel 1 (
    echo ❌ Installation failed
    pause
    exit /b 1
)

echo.
echo ✅ Installation completed successfully!
echo.
echo 📋 Next steps:
echo 1. Run start_bot.bat to launch the main bot
echo 2. Or run start_trading_integrated.bat for simulation mode
echo 3. Or run quick_start_hft.bat for HFT mode
echo.
pause
