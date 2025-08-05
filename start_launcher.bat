
@echo off
title AuraTrade - Bot Launcher GUI
color 0D
cls

echo ========================================
echo 🎮 TRADING BOT LAUNCHER GUI
echo    Choose your trading mode
echo ========================================
echo.

echo 🚀 Opening Bot Launcher...
echo 📊 Available modes:
echo    • Integrated Trading Bot
echo    • HFT Mode
echo    • Windows MT5 (if available)
echo.

cd /d "%~dp0"
cd tradebot
python START_TRADING_BOT.py

echo.
echo ========================================
echo 👋 Launcher closed
echo ========================================
pause
