@echo off
title Multi-Mode Trading Bot Launcher
echo.
echo ========================================
echo    Multi-Mode Trading Bot Launcher
echo ========================================
echo.
echo Starting Multi-Mode Trading System...
echo.

cd /d "%~dp0"
python trading_modes.py

pause