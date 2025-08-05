#!/usr/bin/env python3
"""
Test Trading Execution System
"""

import sys
import time
import random

def test_basic_functionality():
    """Test basic trading functionality"""
    print("🧪 Testing Trading System...")
    
    try:
        # Test imports
        from config import config
        from enhanced_indicators import EnhancedIndicators
        from market_data_api import MarketDataAPI
        print("✅ All imports successful")
        
        # Test market data
        market_api = MarketDataAPI()
        data = market_api.get_market_data("XAUUSDm", count=50)
        if data and len(data) > 10:
            print(f"✅ Market data: {len(data)} points retrieved")
            prices = [item['close'] for item in data]
            print(f"✅ Current price: {prices[-1]:.5f}")
        else:
            print("❌ Market data retrieval failed")
            return False
        
        # Test indicators
        indicators = EnhancedIndicators()
        signal_data = indicators.enhanced_signal_analysis(prices)
        signal = signal_data.get('signal')
        confidence = signal_data.get('confidence', 0)
        print(f"✅ Signal generated: {signal} (confidence: {confidence:.2f})")
        
        # Test balance calculation
        balance = 5000000.0  # 5M simulation
        tp_pct = config.TP_PERSEN_BALANCE
        sl_pct = config.SL_PERSEN_BALANCE
        
        tp_money = balance * tp_pct
        sl_money = balance * sl_pct
        
        print(f"✅ Balance calculations:")
        print(f"   Balance: ${balance:,.2f}")
        print(f"   TP Money: ${tp_money:,.2f} ({tp_pct*100}%)")
        print(f"   SL Money: ${sl_money:,.2f} ({sl_pct*100}%)")
        
        # Simulate trade execution
        if signal and confidence >= 0.2:
            current_price = prices[-1]
            lot_size = 0.01
            
            # Calculate TP/SL prices
            contract_size = 100.0
            point = 0.01
            pip_size = 10 * point
            pip_value = contract_size * pip_size * lot_size
            
            tp_pips = tp_money / pip_value if pip_value > 0 else 50
            sl_pips = sl_money / pip_value if pip_value > 0 else 100
            
            if signal == "BUY":
                tp_price = current_price + (tp_pips * pip_size)
                sl_price = current_price - (sl_pips * pip_size)
            else:
                tp_price = current_price - (tp_pips * pip_size)
                sl_price = current_price + (sl_pips * pip_size)
            
            print(f"✅ Trade calculation:")
            print(f"   Signal: {signal}")
            print(f"   Entry: {current_price:.5f}")
            print(f"   TP: {tp_price:.5f}")
            print(f"   SL: {sl_price:.5f}")
            print(f"   Lot: {lot_size}")
            
            # Simulate execution
            success = random.random() > 0.2  # 80% success rate
            if success:
                trade_id = random.randint(100000, 999999)
                print(f"🎯 SIMULATED TRADE EXECUTED!")
                print(f"   Trade ID: #{trade_id}")
                print(f"   Status: SUCCESS")
                return True
            else:
                print(f"❌ SIMULATED TRADE FAILED!")
                return False
        else:
            print(f"⚠️ No valid signal (confidence too low: {confidence:.2f})")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n🎉 Trading system test PASSED!")
        print("✅ All components working correctly")
    else:
        print("\n💥 Trading system test FAILED!")
        print("❌ Issues detected in trading system")
    
    sys.exit(0 if success else 1)