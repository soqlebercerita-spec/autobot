
@echo off
title AuraTrade - Integrated Trading Bot
color 0B
cls

echo ========================================
echo 🔬 INTEGRATED TRADING BOT
echo    Safe Simulation • Real Market Data
echo ========================================
echo.

echo 🚀 Starting Integrated Trading Bot...
echo 💰 Using Virtual Balance: $10,000
echo 📊 Real Market Data: Enabled
echo 🛡️  Risk Management: Active
echo.

cd /d "%~dp0"
cd tradebot
python trading_bot_integrated.py

echo.
echo ========================================
echo 👋 Bot session ended
echo ========================================
pause
