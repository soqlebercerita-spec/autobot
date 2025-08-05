#!/usr/bin/env python3
"""
Launcher untuk Trading Bot
Menjalankan trading bot dengan GUI
"""

import os
import sys
import subprocess

def main():
    print("=" * 50)
    print("🤖 TRADING BOT AMAN CUAN")
    print("=" * 50)
    print("Auto Trading & Scalping Bot")
    print("Dengan Indikator Teknikal & Risk Management")
    print("=" * 50)
    
    # Set display untuk GUI
    os.environ['DISPLAY'] = ':0'
    
    # Import dan jalankan bot
    try:
        from trading_bot_integrated import TradingBot
        
        print("✅ Modul berhasil dimuat")
        print("🚀 Meluncurkan GUI...")
        print("\nCara menggunakan:")
        print("1. Klik 'Connect' untuk koneksi data pasar")
        print("2. Atur parameter trading (Symbol, Lot, TP/SL)")
        print("3. Centang 'Enable Scalping Mode' jika mau scalping")
        print("4. Klik 'Start' untuk mulai auto trading")
        print("5. Klik 'Stop' untuk menghentikan bot")
        print("\n" + "=" * 50)
        
        # Jalankan bot
        bot = TradingBot()
        bot.run()
        
    except ImportError as e:
        print(f"❌ Error import: {e}")
        print("🔧 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy", "requests"])
        
        # Coba lagi
        from trading_bot_integrated import TradingBot
        bot = TradingBot()
        bot.run()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("📞 Hubungi developer jika masalah berlanjut")
        input("Tekan Enter untuk keluar...")

if __name__ == "__main__":
    main()