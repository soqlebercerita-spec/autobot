
@echo off
title AuraTrade - Institutional Trading Bot
color 0A
cls

echo ========================================
echo 🤖 AURA TRADE - INSTITUTIONAL TRADING BOT
echo    Advanced HFT • Multi-Symbol • AI-Powered
echo ========================================
echo.

echo 🚀 Starting AuraTrade Bot...
echo.

REM Change to script directory
cd /d "%~dp0"

REM Run the main bot
python bot.py

echo.
echo ========================================
echo 👋 Bot session ended
echo ========================================
pause
