@echo off
title Real MT5 Trading Bot - LIVE TRADING
echo.
echo ========================================
echo   REAL MT5 TRADING BOT - LIVE TRADING
echo ========================================
echo.
echo WARNING: This is for REAL MONEY trading!
echo Make sure you have:
echo - MT5 installed and running
echo - Real trading account connected
echo - Sufficient account balance
echo.
echo Starting Real MT5 Trading Bot...
echo.

cd /d "%~dp0"
python real_mt5_trader.py

pause