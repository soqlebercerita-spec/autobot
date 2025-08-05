
@echo off
title AuraTrade - HFT Mode
color 0C
cls

echo ========================================
echo ⚡ HIGH FREQUENCY TRADING MODE
echo    Ultra-Fast • 1-Second Scanning
echo ========================================
echo.

echo 🚀 Starting HFT Trading Bot...
echo ⚡ Scan Interval: 1 second
echo 🎯 TP: 0.3%% • SL: 1.5%%
echo 🔥 Max Speed: 10 trades/second
echo.

cd /d "%~dp0"
cd tradebot
python trading_bot_hft.py

echo.
echo ========================================
echo ⚡ HFT session ended
echo ========================================
pause
