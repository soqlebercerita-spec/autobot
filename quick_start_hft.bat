
@echo off
title AuraTrade - HFT Mode
color 0C
cls

echo ========================================
echo ⚡ HIGH-FREQUENCY TRADING MODE
echo    Ultra-Fast • 1s Scan • 0.3%% TP
echo ========================================
echo.

echo ⚠️  WARNING: HFT Mode aktif!
echo    - Scanning setiap 1 detik
echo    - TP: 0.3%% dari modal
echo    - SL: 1.5%% dari modal
echo.

set /p confirm="Continue with HFT mode? (Y/N): "
if /i "%confirm%" NEQ "Y" (
    echo Operation cancelled.
    pause
    exit /b
)

echo.
echo ⚡ Activating HFT Mode...
echo.

cd /d "%~dp0"
cd tradebot

python trading_bot_hft.py

echo.
echo ========================================
echo ⚡ HFT session ended
echo ========================================
pause
