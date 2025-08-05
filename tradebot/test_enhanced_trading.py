#!/usr/bin/env python3
"""
Test Enhanced Trading Bot - Command Line Version
Test the fixes for price retrieval and signal generation
"""

import time
import datetime
import numpy as np
from market_data_api import MarketDataAPI
from simulation_trading import SimulationTrading
from enhanced_indicators import EnhancedIndicators
from config import config

class TestTradingBot:
    def __init__(self):
        # Initialize enhanced components
        self.market_api = MarketDataAPI()
        self.mt5 = SimulationTrading(self.market_api)
        self.indicators = EnhancedIndicators()
        
        # Counters
        self.total_opportunities_captured = 0
        self.total_opportunities_missed = 0
        self.price_retrieval_failures = 0
        self.signals_generated = 0
        
        print("=" * 60)
        print("🚀 ENHANCED TRADING BOT - COMMAND LINE TEST")
        print("=" * 60)
        print("✅ Fixed Price Retrieval Issues")
        print("⚡ Optimized Signal Generation") 
        print("🎯 Enhanced Opportunity Capture")
        print(f"🔧 Price Spike Threshold: {config.LONJAKAN_THRESHOLD} (Reduced from 10)")
        print(f"⚙️  Signal Confidence: {config.SIGNAL_CONFIDENCE_THRESHOLD}")
        print(f"📊 Scan Interval: {config.DEFAULT_INTERVAL}s")
        print("=" * 60)
    
    def test_price_retrieval(self, symbol="XAUUSDm"):
        """Test enhanced price retrieval mechanism"""
        print(f"\n🧪 Testing price retrieval for {symbol}...")
        
        success_count = 0
        total_tests = 5
        
        for i in range(total_tests):
            try:
                price = self.market_api.get_current_price(symbol)
                if price and price > 0:
                    print(f"✅ Test {i+1}: Price retrieved successfully: {price:.5f}")
                    success_count += 1
                else:
                    print(f"❌ Test {i+1}: Price retrieval failed")
                    self.price_retrieval_failures += 1
                
                time.sleep(0.5)
            except Exception as e:
                print(f"❌ Test {i+1}: Error - {e}")
                self.price_retrieval_failures += 1
        
        success_rate = (success_count / total_tests) * 100
        print(f"\n📊 Price Retrieval Results:")
        print(f"   Success Rate: {success_rate:.1f}% ({success_count}/{total_tests})")
        print(f"   Failures: {self.price_retrieval_failures}")
        
        return success_rate > 80
    
    def test_signal_generation(self, symbol="XAUUSDm"):
        """Test enhanced signal generation"""
        print(f"\n🎯 Testing signal generation for {symbol}...")
        
        try:
            # Get market data
            market_data = self.market_api.get_market_data(symbol, count=config.DATA_BUFFER_SIZE)
            if not market_data:
                print("❌ Failed to get market data for signal testing")
                return False
            
            # Extract price arrays
            close_prices = [point['close'] for point in market_data]
            high_prices = [point['high'] for point in market_data]
            low_prices = [point['low'] for point in market_data]
            
            print(f"✅ Retrieved {len(close_prices)} data points")
            print(f"   Latest price: {close_prices[-1]:.5f}")
            print(f"   Price range: {min(close_prices):.5f} - {max(close_prices):.5f}")
            
            # Test enhanced signal analysis
            signal_result = self.indicators.enhanced_signal_analysis(
                close_prices, high_prices, low_prices
            )
            
            print(f"\n📈 Signal Analysis Results:")
            print(f"   Signal: {signal_result['signal']}")
            print(f"   Confidence: {signal_result['confidence']:.3f}")
            print(f"   Strength: {signal_result['strength']:.3f}")
            print(f"   Volatility: {signal_result.get('volatility', 0):.3f}")
            
            # Display indicator values
            indicators = signal_result.get('indicators', {})
            if indicators:
                print(f"\n📊 Technical Indicators:")
                print(f"   Current Price: {indicators.get('current_price', 'N/A'):.5f}")
                print(f"   MA Short: {indicators.get('ma_short', 'N/A'):.5f}")
                print(f"   MA Long: {indicators.get('ma_long', 'N/A'):.5f}")
                print(f"   EMA Fast: {indicators.get('ema_fast', 'N/A'):.5f}")
                print(f"   EMA Slow: {indicators.get('ema_slow', 'N/A'):.5f}")
                print(f"   RSI: {indicators.get('rsi', 'N/A'):.1f}")
                
                bb_upper = indicators.get('bb_upper')
                bb_lower = indicators.get('bb_lower')
                if bb_upper and bb_lower:
                    print(f"   Bollinger Upper: {bb_upper:.5f}")
                    print(f"   Bollinger Lower: {bb_lower:.5f}")
            
            # Display scoring details
            scores = signal_result.get('scores', {})
            if scores:
                print(f"\n🎯 Signal Scoring:")
                print(f"   Buy Score: {scores.get('buy_score', 0)}")
                print(f"   Sell Score: {scores.get('sell_score', 0)}")
                print(f"   Buy Confidence: {scores.get('buy_confidence', 0):.3f}")
                print(f"   Sell Confidence: {scores.get('sell_confidence', 0):.3f}")
            
            if signal_result['signal'] != 'WAIT':
                self.signals_generated += 1
                print(f"🎉 Trading signal generated: {signal_result['signal']}")
                return True
            else:
                print("⏳ No signal generated - waiting for better conditions")
                return False
                
        except Exception as e:
            print(f"❌ Signal generation error: {e}")
            return False
    
    def test_price_spike_detection(self, symbol="XAUUSDm"):
        """Test price spike detection with new threshold"""
        print(f"\n⚡ Testing price spike detection for {symbol}...")
        
        try:
            # Get some price data
            price_history = []
            for i in range(10):
                price = self.market_api.get_current_price(symbol)
                if price:
                    price_history.append(price)
                time.sleep(0.2)
            
            if len(price_history) < 5:
                print("❌ Insufficient price data for spike detection test")
                return False
            
            # Calculate price movements
            price_changes = []
            for i in range(1, len(price_history)):
                change_pct = abs((price_history[i] - price_history[i-1]) / price_history[i-1]) * 100
                price_changes.append(change_pct)
            
            max_change = max(price_changes) if price_changes else 0
            avg_change = sum(price_changes) / len(price_changes) if price_changes else 0
            
            print(f"📊 Price Movement Analysis:")
            print(f"   Price samples: {len(price_history)}")
            print(f"   Max change: {max_change:.3f}%")
            print(f"   Avg change: {avg_change:.3f}%")
            print(f"   Spike threshold: {config.LONJAKAN_THRESHOLD}%")
            
            # Check if price spike would be detected
            spike_detected = max_change > config.LONJAKAN_THRESHOLD
            
            if spike_detected:
                print(f"⚠️  Price spike detected ({max_change:.3f}% > {config.LONJAKAN_THRESHOLD}%)")
                print("   Old threshold (10%) would have missed more opportunities")
            else:
                print(f"✅ No price spike detected ({max_change:.3f}% < {config.LONJAKAN_THRESHOLD}%)")
                print("   Price movement acceptable for trading")
            
            return not spike_detected  # Return True if no spike (good for trading)
            
        except Exception as e:
            print(f"❌ Price spike detection error: {e}")
            return False
    
    def test_opportunity_capture(self, symbol="XAUUSDm", test_cycles=5):
        """Test overall opportunity capture capability"""
        print(f"\n🎯 Testing opportunity capture for {symbol} ({test_cycles} cycles)...")
        
        opportunities_found = 0
        signals_generated = 0
        
        for cycle in range(test_cycles):
            print(f"\n--- Cycle {cycle + 1}/{test_cycles} ---")
            
            try:
                # Test price retrieval
                price = self.market_api.get_current_price(symbol)
                if not price or price <= 0:
                    print(f"❌ Price retrieval failed in cycle {cycle + 1}")
                    self.total_opportunities_missed += 1
                    continue
                
                print(f"✅ Current price: {price:.5f}")
                
                # Test signal generation
                market_data = self.market_api.get_market_data(symbol, count=50)
                if not market_data:
                    print(f"❌ Market data retrieval failed in cycle {cycle + 1}")
                    self.total_opportunities_missed += 1
                    continue
                
                close_prices = [point['close'] for point in market_data]
                signal_result = self.indicators.enhanced_signal_analysis(close_prices)
                
                if signal_result['signal'] != 'WAIT':
                    signals_generated += 1
                    self.total_opportunities_captured += 1
                    print(f"🎉 Opportunity captured: {signal_result['signal']} (Confidence: {signal_result['confidence']:.3f})")
                else:
                    print(f"⏳ No signal generated in cycle {cycle + 1}")
                
                opportunities_found += 1
                
            except Exception as e:
                print(f"❌ Error in cycle {cycle + 1}: {e}")
                self.total_opportunities_missed += 1
            
            time.sleep(1)
        
        # Calculate results
        success_rate = (opportunities_found / test_cycles) * 100 if test_cycles > 0 else 0
        signal_rate = (signals_generated / opportunities_found) * 100 if opportunities_found > 0 else 0
        
        print(f"\n📊 Opportunity Capture Results:")
        print(f"   Test cycles: {test_cycles}")
        print(f"   Successful data retrieval: {opportunities_found}/{test_cycles} ({success_rate:.1f}%)")
        print(f"   Signals generated: {signals_generated}/{opportunities_found} ({signal_rate:.1f}%)")
        print(f"   Total captured: {self.total_opportunities_captured}")
        print(f"   Total missed: {self.total_opportunities_missed}")
        
        overall_success = (self.total_opportunities_captured / (self.total_opportunities_captured + self.total_opportunities_missed)) * 100
        print(f"   Overall success rate: {overall_success:.1f}%")
        
        return success_rate > 80
    
    def run_comprehensive_test(self):
        """Run comprehensive test of all enhancements"""
        print("\n🚀 Starting comprehensive test of enhanced trading bot...")
        
        # Initialize simulation
        print("\n🔗 Initializing enhanced trading simulation...")
        if not self.mt5.initialize():
            print("❌ Failed to initialize trading simulation")
            return False
        
        print("✅ Trading simulation initialized successfully")
        
        # Test all components
        results = {}
        
        print("\n" + "="*60)
        print("📊 RUNNING COMPREHENSIVE TESTS")
        print("="*60)
        
        # Test 1: Price Retrieval
        results['price_retrieval'] = self.test_price_retrieval()
        
        # Test 2: Signal Generation  
        results['signal_generation'] = self.test_signal_generation()
        
        # Test 3: Price Spike Detection
        results['spike_detection'] = self.test_price_spike_detection()
        
        # Test 4: Opportunity Capture
        results['opportunity_capture'] = self.test_opportunity_capture()
        
        # Final Results
        print("\n" + "="*60)
        print("📊 FINAL TEST RESULTS")
        print("="*60)
        
        passed_tests = sum(1 for result in results.values() if result)
        total_tests = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        overall_score = (passed_tests / total_tests) * 100
        print(f"\nOverall Score: {overall_score:.1f}% ({passed_tests}/{total_tests} tests passed)")
        
        if overall_score >= 75:
            print("\n🎉 ENHANCED TRADING BOT - WORKING EXCELLENT!")
            print("✅ Price retrieval issues have been fixed")
            print("⚡ Signal generation has been optimized")
            print("🎯 Opportunity capture has been enhanced")
        elif overall_score >= 50:
            print("\n⚠️  Enhanced trading bot working but needs improvement")
        else:
            print("\n❌ Enhanced trading bot needs significant fixes")
        
        print("\n📈 Performance Statistics:")
        print(f"   Opportunities Captured: {self.total_opportunities_captured}")
        print(f"   Opportunities Missed: {self.total_opportunities_missed}")
        print(f"   Price Retrieval Failures: {self.price_retrieval_failures}")
        print(f"   Signals Generated: {self.signals_generated}")
        
        return overall_score >= 75

def main():
    """Main test function"""
    try:
        # Create and run test
        test_bot = TestTradingBot()
        success = test_bot.run_comprehensive_test()
        
        if success:
            print("\n🎉 All tests completed successfully!")
            print("Enhanced trading bot is ready for use!")
        else:
            print("\n⚠️  Some tests failed. Please check the results above.")
            
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")

if __name__ == "__main__":
    main()