
@echo off
title AuraTrade - System Check
color 0F
cls

echo ========================================
echo 🔍 AURA TRADE - SYSTEM CHECK
echo    Checking system requirements...
echo ========================================
echo.

echo 🖥️  Platform: %OS%
echo 📁 Current Directory: %CD%
echo.

REM Check Python
echo 🔍 Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found
    echo Please install Python 3.8+
) else (
    echo ✅ Python found
    python --version
)
echo.

REM Check pip
echo 🔍 Checking pip...
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip not found
) else (
    echo ✅ pip found
    pip --version
)
echo.

REM Check key packages
echo 🔍 Checking key packages...
python -c "import numpy; print('✅ numpy installed')" 2>nul || echo "❌ numpy not found"
python -c "import pandas; print('✅ pandas installed')" 2>nul || echo "❌ pandas not found"
python -c "import requests; print('✅ requests installed')" 2>nul || echo "❌ requests not found"
python -c "import matplotlib; print('✅ matplotlib installed')" 2>nul || echo "❌ matplotlib not found"
echo.

REM Check files
echo 🔍 Checking bot files...
if exist "bot.py" (
    echo ✅ Main bot file found
) else (
    echo ❌ bot.py not found
)

if exist "tradebot\trading_bot_integrated.py" (
    echo ✅ Integrated bot file found
) else (
    echo ❌ Integrated bot file not found
)

if exist "config\config.py" (
    echo ✅ Config file found
) else (
    echo ❌ Config file not found
)

echo.
echo ========================================
echo 🎯 System check completed
echo ========================================
echo.
echo 💡 Tips:
echo - If packages missing: run install_requirements.bat
echo - For trading: run start_bot.bat
echo - For simulation: run start_trading_integrated.bat
echo.
pause
