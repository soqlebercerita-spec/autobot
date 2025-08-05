
@echo off
title AuraTrade - Main Launcher
color 0A
cls

echo ========================================
echo 🤖 AURA TRADE - MAIN LAUNCHER
echo    Institutional Trading Bot Platform
echo ========================================
echo.

echo 🚀 Starting AuraTrade Bot...
echo ⚡ Mode: Integrated Simulation
echo 🔒 Safe Mode: Virtual Money Only
echo.

cd /d "%~dp0"
python bot.py

echo.
echo ========================================
echo 👋 Trading session ended
echo ========================================
pause
