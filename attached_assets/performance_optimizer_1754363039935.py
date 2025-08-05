"""
Performance Optimizer for Windows Trading Bot
Optimizes system resources and trading performance
"""

import os
import gc
import threading
import time
import psutil
from datetime import datetime, timedelta
from config import config

class PerformanceOptimizer:
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.last_cleanup = datetime.now()
        self.performance_stats = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_usage': 0,
            'network_usage': 0,
            'last_update': datetime.now()
        }
        
        print("✅ Performance optimizer initialized")
    
    def start_monitoring(self):
        """Start performance monitoring in background"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        print("📊 Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                self._update_stats()
                self._check_cleanup_needed()
                self._optimize_if_needed()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                print(f"⚠️ Monitoring error: {e}")
                time.sleep(60)
    
    def _update_stats(self):
        """Update current system statistics"""
        try:
            # Get current process
            process = psutil.Process()
            
            # Update stats
            self.performance_stats.update({
                'cpu_usage': process.cpu_percent(),
                'memory_usage': process.memory_info().rss / 1024 / 1024,  # MB
                'memory_percent': process.memory_percent(),
                'threads': process.num_threads(),
                'last_update': datetime.now()
            })
            
            # System stats
            system_stats = {
                'system_cpu': psutil.cpu_percent(),
                'system_memory': psutil.virtual_memory().percent,
                'system_disk': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent
            }
            
            self.performance_stats.update(system_stats)
            
        except Exception as e:
            print(f"⚠️ Stats update error: {e}")
    
    def _check_cleanup_needed(self):
        """Check if cleanup is needed"""
        try:
            now = datetime.now()
            
            # Cleanup every 5 minutes
            if now - self.last_cleanup > timedelta(minutes=5):
                self._cleanup_resources()
                self.last_cleanup = now
                
        except Exception as e:
            print(f"⚠️ Cleanup check error: {e}")
    
    def _cleanup_resources(self):
        """Cleanup resources to free memory"""
        try:
            # Force garbage collection
            collected = gc.collect()
            
            if collected > 0:
                print(f"🧹 Cleaned up {collected} objects")
            
            # Clean up old log files (keep last 7 days)
            self._cleanup_old_logs()
            
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")
    
    def _cleanup_old_logs(self):
        """Remove old log files"""
        try:
            logs_dir = "logs"
            if not os.path.exists(logs_dir):
                return
            
            cutoff_date = datetime.now() - timedelta(days=7)
            cleaned_count = 0
            
            for filename in os.listdir(logs_dir):
                if filename.endswith('.log'):
                    file_path = os.path.join(logs_dir, filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    
                    if file_time < cutoff_date:
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                        except:
                            pass
            
            if cleaned_count > 0:
                print(f"🧹 Cleaned {cleaned_count} old log files")
                
        except Exception as e:
            print(f"⚠️ Log cleanup error: {e}")
    
    def _optimize_if_needed(self):
        """Optimize performance if needed"""
        try:
            # If memory usage is high, suggest optimization
            if self.performance_stats.get('memory_percent', 0) > 80:
                print("⚠️ High memory usage detected - running cleanup")
                self._cleanup_resources()
            
            # If CPU usage is consistently high, suggest reducing scan frequency
            if self.performance_stats.get('cpu_usage', 0) > 90:
                print("⚠️ High CPU usage detected - consider reducing scan frequency")
            
        except Exception as e:
            print(f"⚠️ Optimization error: {e}")
    
    def get_performance_report(self):
        """Get current performance report"""
        try:
            stats = self.performance_stats.copy()
            
            # Add interpretation
            report = {
                'timestamp': datetime.now().isoformat(),
                'process_stats': {
                    'cpu_usage': f"{stats.get('cpu_usage', 0):.1f}%",
                    'memory_usage': f"{stats.get('memory_usage', 0):.1f} MB",
                    'memory_percent': f"{stats.get('memory_percent', 0):.1f}%",
                    'threads': stats.get('threads', 0)
                },
                'system_stats': {
                    'cpu_usage': f"{stats.get('system_cpu', 0):.1f}%",
                    'memory_usage': f"{stats.get('system_memory', 0):.1f}%",
                    'disk_usage': f"{stats.get('system_disk', 0):.1f}%"
                },
                'recommendations': self._get_recommendations()
            }
            
            return report
            
        except Exception as e:
            print(f"⚠️ Report generation error: {e}")
            return {}
    
    def _get_recommendations(self):
        """Get performance recommendations"""
        recommendations = []
        
        try:
            stats = self.performance_stats
            
            if stats.get('memory_percent', 0) > 70:
                recommendations.append("Consider reducing scan frequency to lower memory usage")
            
            if stats.get('system_cpu', 0) > 80:
                recommendations.append("High system CPU usage - close unnecessary programs")
            
            if stats.get('system_memory', 0) > 85:
                recommendations.append("High system memory usage - restart system if needed")
            
            if stats.get('threads', 0) > 20:
                recommendations.append("High thread count - check for resource leaks")
            
            if not recommendations:
                recommendations.append("Performance looks good!")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def optimize_windows_settings(self):
        """Optimize Windows settings for trading (requires admin)"""
        try:
            import winreg
            
            optimizations = []
            
            # Try to set high performance power plan (if admin)
            try:
                os.system('powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c')
                optimizations.append("Set High Performance power plan")
            except:
                pass
            
            # Set process priority to high
            try:
                import psutil
                process = psutil.Process()
                process.nice(psutil.HIGH_PRIORITY_CLASS)
                optimizations.append("Set high process priority")
            except:
                pass
            
            if optimizations:
                print(f"✅ Applied {len(optimizations)} Windows optimizations:")
                for opt in optimizations:
                    print(f"   • {opt}")
            else:
                print("ℹ️ No Windows optimizations could be applied (may require admin rights)")
                
        except Exception as e:
            print(f"⚠️ Windows optimization error: {e}")
    
    def get_system_info(self):
        """Get detailed system information"""
        try:
            import platform
            
            info = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': f"{psutil.virtual_memory().total / 1024**3:.1f} GB",
                'disk_total': f"{psutil.disk_usage('C:' if os.name == 'nt' else '/').total / 1024**3:.1f} GB",
                'uptime': str(timedelta(seconds=int(time.time() - psutil.boot_time())))
            }
            
            return info
            
        except Exception as e:
            print(f"⚠️ System info error: {e}")
            return {}

# Global instance
performance_optimizer = PerformanceOptimizer()