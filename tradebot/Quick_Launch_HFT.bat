@echo off
title HFT Quick Launch
echo.
echo ========================================
echo         HFT QUICK LAUNCH
echo ========================================
echo.
echo Starting HFT Mode directly...
echo TP: 0.3%% of balance
echo SL: 1.5%% of balance  
echo Ultra-fast 1-second scanning
echo.

cd /d "%~dp0"
python -c "from trading_modes import HFTTradingBot; bot = HFTTradingBot(); bot.run()"

pause