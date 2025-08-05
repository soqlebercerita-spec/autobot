
@echo off
title AuraTrade - Simulation Mode
color 0A
cls

echo ========================================
echo 🔬 AURA TRADE - SIMULATION MODE
echo    Safe Testing • No Real Money Risk
echo ========================================
echo.

echo 💡 Simulation Mode Features:
echo    ✅ Real market data simulation
echo    ✅ All trading strategies active
echo    ✅ Risk-free environment
echo    ✅ Perfect for learning and testing
echo.

echo 🚀 Starting Simulation Mode...
echo.

cd /d "%~dp0"

REM Set environment variable for simulation
set TRADING_MODE=SIMULATION
python bot.py

echo.
echo ========================================
echo 🔬 Simulation session ended
echo ========================================
pause
