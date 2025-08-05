
@echo off
title AuraTrade - System Check & Start
color 0A
cls

echo ========================================
echo 🔍 SYSTEM CHECK & AUTO START
echo    Checking requirements then launching
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo 📥 Please install Python from: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo ✅ Python: OK

REM Check basic packages
echo 🔍 Checking packages...
python -c "import requests, numpy" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Some packages missing - continuing anyway...
) else (
    echo ✅ Packages: OK
)

echo.
echo 🚀 Starting AuraTrade Bot...
echo.

cd /d "%~dp0"
python bot.py

echo.
echo ========================================
echo 👋 Session ended
echo ========================================
pause
