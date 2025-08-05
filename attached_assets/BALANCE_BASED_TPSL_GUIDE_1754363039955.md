# 💰 BALANCE-BASED TP/SL SYSTEM

## ✅ **SUDAH DIUPDATE: TP/SL BERDASARKAN PERSENTASE MODAL**

Trading bot sekarang menggunakan sistem TP/SL berdasarkan persentase dari modal/balance, bukan dari harga market.

## 🎯 **CARA KERJA SISTEM BARU:**

### **Konsep:**
- **TP (Take Profit)**: Berdasarkan % dari total modal
- **SL (Stop Loss)**: Berdasarkan % dari total modal
- **Real Money Management**: Lebih realistis untuk risk management

### **Contoh Perhitungan:**
```
Modal: Rp 5,000,000
TP 1%: Target profit Rp 50,000
SL 5%: Max loss Rp 250,000

Bot akan menghitung berapa pips yang diperlukan 
untuk mencapai target profit/loss tersebut
```

## ⚙️ **KONFIGURASI GUI:**

### **Panel "Balance-Based TP/SL":**
- **TP % of Balance**: Persentase target profit dari modal (default 1%)
- **SL % of Balance**: Persentase max loss dari modal (default 5%)
- **Real-time calculation**: Bot otomatis hitung harga TP/SL

### **Setting Modes:**

#### **Normal Mode:**
- TP: 1% dari modal
- SL: 5% dari modal

#### **Scalping Mode:**
- TP: 0.5% dari modal  
- SL: 2% dari modal

#### **HFT Mode:**
- Menggunakan setting GUI atau default scalping

## 🔧 **CARA MENGGUNAKAN:**

### **1. Via GUI Settings:**
1. Buka trading bot
2. Pergi ke tab "Basic Settings"  
3. Lihat section "💰 Balance-Based TP/SL"
4. Set persentase yang diinginkan:
   - **TP % of Balance**: Misal 0.5% untuk konservatif
   - **SL % of Balance**: Misal 3% untuk moderate risk

### **2. Automatic Calculation:**
Bot akan otomatis:
- Ambil balance saat ini dari account
- Hitung target profit dalam rupiah/dollar
- Convert ke pips berdasarkan symbol dan lot size
- Set TP/SL price yang tepat

## 📊 **CONTOH REAL:**

### **Modal 5 Juta:**
```
Balance: Rp 5,000,000
TP 1% = Rp 50,000 target profit
SL 5% = Rp 250,000 max loss

Untuk XAUUSD lot 0.01:
- Pip value ≈ 100
- TP pips = 50,000/100 = 500 pips
- SL pips = 250,000/100 = 2500 pips

Jika harga gold 2030:
- TP = 2030 + (500 × 0.1) = 2080
- SL = 2030 - (2500 × 0.1) = 1780
```

### **Modal 10 Juta:**
```
Balance: Rp 10,000,000  
TP 1% = Rp 100,000 target
SL 5% = Rp 500,000 max loss

Untuk EURUSD lot 0.01:
- Pip value ≈ 10,000
- TP pips = 100,000/10,000 = 10 pips
- SL pips = 500,000/10,000 = 50 pips

Jika EUR 1.0850:
- TP = 1.0850 + (10 × 0.0001) = 1.0860
- SL = 1.0850 - (50 × 0.0001) = 1.0800
```

## 🔥 **KEUNGGULAN SISTEM BARU:**

### **1. Risk Management Realistis:**
- TP/SL berdasarkan kemampuan modal
- Tidak terpengaruh volatilitas harga
- Konsisten dengan money management

### **2. Flexible Configuration:**
- Bisa diatur via GUI
- Different modes (normal, scalping, HFT)
- Real-time adjustment

### **3. Smart Calculation:**
- Auto pip value calculation per symbol
- Support major pairs, gold, crypto
- Error handling dengan fallback

### **4. Professional Approach:**
- Sesuai dengan standar trading professional
- Risk reward ratio yang jelas
- Portfolio management yang proper

## 📝 **LOG MONITORING:**

Bot akan menampilkan log seperti ini:
```
💰 Balance-based TP/SL calculated:
   • Balance: $10,000.00
   • Target Profit: $100.00 (1.0%)
   • Max Loss: $500.00 (5.0%)
   • TP Price: 2080.50000 | SL Price: 1980.50000
```

## ✅ **STATUS IMPLEMENTASI:**

- ✅ Config updated dengan balance-based parameters
- ✅ GUI panel tersedia untuk setting TP/SL %
- ✅ Automatic calculation system ready
- ✅ Support all symbols (forex, gold, crypto)
- ✅ Integration dengan normal, scalping, dan HFT mode
- ✅ Real-time balance detection
- ✅ Error handling dan fallback system

## 🎯 **KESIMPULAN:**

**Sistem TP/SL sudah sepenuhnya diubah menjadi berbasis persentase modal!**

Sekarang bot akan:
- Menghitung TP/SL berdasarkan % dari modal Anda
- Lebih realistis untuk risk management
- Sesuai dengan prinsip money management yang benar
- Dapat dikonfigurasi melalui GUI

**Trading dengan risk management yang proper! 💪**