# 🚀 Cara Install Trading Bot di Windows

## 📋 Persyaratan

1. **Windows 10/11** (64-bit)
2. **Python 3.8 atau lebih baru**
3. **MetaTrader5** terinstall dan login ke akun trading
4. **Koneksi internet** untuk real-time data

## 🔧 Langkah Instalasi

### 1. Download dan Ekstrak Files
```
- Download semua file dari project ini
- Ekstrak ke folder, misal: C:\TradingBot\
```

### 2. Install Python Dependencies
Buka Command Prompt sebagai Administrator, lalu jalankan:
```bash
cd C:\TradingBot\
pip install -r install_requirements.txt
```

Atau install manual:
```bash
pip install MetaTrader5 numpy requests pandas matplotlib
```

### 3. Setup MetaTrader5
- Buka MetaTrader5
- Login ke akun trading Anda
- Di menu **Tools → Options → Expert Advisors**:
  - ✅ Centang "Allow automated trading"
  - ✅ Centang "Allow DLL imports"
  - ✅ Centang "Allow import of external experts"

### 4. Konfigurasi Telegram (Opsional)
Edit file `config.py` atau set environment variables:
```python
TELEGRAM_BOT_TOKEN = "TOKEN_BOT_TELEGRAM_ANDA"
TELEGRAM_CHAT_ID = "CHAT_ID_TELEGRAM_ANDA"
```

## 🚀 Menjalankan Bot

### Untuk Windows dengan MT5 (Live Trading):
```bash
python trading_bot_windows.py
```

### Untuk Testing/Simulasi:
```bash
python trading_bot_integrated.py
```

## ⚙️ Pengaturan Bot

### Trading Settings:
- **Symbol**: XAUUSDm, EURUSD, BTCUSD, dll
- **Lot**: 0.01 - 1.0 (sesuai modal)
- **Interval**: 10-60 detik
- **TP%**: Take Profit dalam persen (0.5-5%)
- **SL%**: Stop Loss dalam persen (0.5-5%)

### Scalping Mode:
- ✅ Centang untuk trading scalping (TP/SL lebih ketat)
- TP Scalping: 0.1-0.5%
- SL Scalping: 0.1-0.5%

### Risk Management:
- Max 10 order per session
- Trading hours: 07:00-21:00
- Auto trailing stop: 50 pips
- Daily loss limit: 10%

## 📊 Cara Penggunaan

1. **Jalankan aplikasi**
2. **Klik "Connect MT5"** - bot akan cek koneksi ke MetaTrader5
3. **Set parameter trading** sesuai modal dan risk appetite
4. **Klik "Start Bot"** untuk mulai auto trading
5. **Monitor log** untuk melihat aktivitas trading
6. **Klik "Stop Bot"** untuk menghentikan
7. **Klik "Close All"** untuk tutup semua posisi

## 🔧 Troubleshooting

### Error "MetaTrader5 not found":
- Pastikan MT5 sudah terinstall
- Pastikan MT5 sedang running dan login
- Install ulang package: `pip install --upgrade MetaTrader5`

### Error "Connection failed":
- Cek koneksi internet
- Restart MetaTrader5
- Pastikan automated trading enabled di MT5

### Bot tidak buka posisi:
- Cek symbol tersedia di Market Watch
- Cek saldo minimum (>$500)
- Cek spread tidak terlalu besar
- Cek trading hours (07:00-21:00)

### Error permission:
- Jalankan Command Prompt sebagai Administrator
- Cek antivirus tidak block aplikasi

## 📈 Strategi Trading

Bot menggunakan kombinasi indikator:
- **MA10**: Simple Moving Average 10 period
- **EMA9, EMA21, EMA50**: Exponential Moving Average
- **WMA5, WMA10**: Weighted Moving Average  
- **RSI14**: Relative Strength Index
- **Bollinger Bands**: 20 period, 2 deviation

### Signal BUY:
- Harga < MA10
- RSI > 20
- Harga ≤ Bollinger Lower Band
- WMA5 < WMA10
- EMA trend bearish (EMA9 < EMA21 < EMA50)

### Signal SELL:
- Harga > MA10
- RSI < 80
- Harga ≥ Bollinger Upper Band
- WMA5 > WMA10
- EMA trend bullish (EMA9 > EMA21 > EMA50)

## 📝 Log Files

Bot akan membuat file log:
- `trading_log.txt`: Log aktivitas umum
- `trade_log.csv`: Data trading untuk analisis
- `performance_log.json`: Statistik performance

## ⚠️ Warning

- **TRADING FOREX/CFD BERISIKO TINGGI**
- **Bisa kehilangan seluruh modal**
- **Test dulu dengan akun demo/cent**
- **Gunakan money management yang ketat**
- **Monitor bot secara berkala**

## 📞 Support

Jika ada masalah:
1. Cek file log untuk error detail
2. Pastikan semua requirements sudah terinstall
3. Test dengan akun demo dulu
4. Gunakan lot kecil untuk testing

**Happy Trading! 📈💰**