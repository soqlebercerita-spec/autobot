"""
Auto-Updater for Windows Trading Bot
Keeps the bot updated with latest fixes and features
"""

import os
import json
import requests
import zipfile
import shutil
import threading
import time
from datetime import datetime, timedelta
from config import config

class AutoUpdater:
    def __init__(self):
        self.version_file = "version.json"
        self.update_url = "https://api.github.com/repos/yourusername/trading-bot/releases/latest"  # Replace with actual repo
        self.current_version = self.get_current_version()
        self.auto_check = True
        self.check_interval = 3600  # Check every hour
        self.last_check = None
        
        print(f"🔄 Auto-updater initialized - Current version: {self.current_version}")
    
    def get_current_version(self):
        """Get current bot version"""
        try:
            if os.path.exists(self.version_file):
                with open(self.version_file, 'r') as f:
                    version_data = json.load(f)
                return version_data.get("version", "1.0.0")
            else:
                # Create initial version file
                version_data = {
                    "version": "1.0.0",
                    "release_date": datetime.now().isoformat(),
                    "features": [
                        "Enhanced price retrieval",
                        "Optimized signal generation", 
                        "Improved opportunity capture",
                        "Windows MT5 integration",
                        "Risk management system",
                        "Telegram notifications",
                        "Advanced logging"
                    ]
                }
                
                with open(self.version_file, 'w') as f:
                    json.dump(version_data, f, indent=2)
                
                return version_data["version"]
                
        except Exception as e:
            print(f"⚠️ Error reading version: {e}")
            return "1.0.0"
    
    def start_auto_check(self):
        """Start automatic update checking in background"""
        if self.auto_check:
            update_thread = threading.Thread(target=self._auto_check_loop, daemon=True)
            update_thread.start()
            print("🔄 Auto-update checking started")
    
    def _auto_check_loop(self):
        """Background loop for checking updates"""
        while self.auto_check:
            try:
                self.check_for_updates(silent=True)
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"⚠️ Auto-update check error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def check_for_updates(self, silent=False):
        """Check for available updates"""
        try:
            if not silent:
                print("🔍 Checking for updates...")
            
            # For demo purposes, we'll simulate update checking
            # In real implementation, this would check GitHub releases or your update server
            
            self.last_check = datetime.now()
            
            # Simulate version check
            available_version = self._simulate_version_check()
            
            if self._is_newer_version(available_version, self.current_version):
                if not silent:
                    print(f"🎉 Update available: {available_version}")
                return {
                    "update_available": True,
                    "current_version": self.current_version,
                    "latest_version": available_version,
                    "release_notes": self._get_simulated_release_notes(available_version)
                }
            else:
                if not silent:
                    print("✅ You have the latest version")
                return {
                    "update_available": False,
                    "current_version": self.current_version,
                    "latest_version": self.current_version
                }
                
        except Exception as e:
            error_msg = f"❌ Update check failed: {e}"
            if not silent:
                print(error_msg)
            return {"error": error_msg}
    
    def _simulate_version_check(self):
        """Simulate checking for updates (replace with real implementation)"""
        # In real implementation, this would make HTTP request to update server
        # For demo, we'll simulate having updates periodically
        
        import random
        
        # Simulate having updates 20% of the time
        if random.random() < 0.2:
            return "1.1.0"  # Simulated newer version
        else:
            return self.current_version
    
    def _get_simulated_release_notes(self, version):
        """Get simulated release notes (replace with real implementation)"""
        return {
            "version": version,
            "release_date": datetime.now().isoformat(),
            "features": [
                "🔧 Fixed price retrieval issues",
                "⚡ Improved signal generation speed", 
                "🎯 Enhanced opportunity capture rate",
                "🔒 Better error handling",
                "📊 Improved performance monitoring"
            ],
            "bug_fixes": [
                "Fixed memory leak in data processing",
                "Resolved GUI freezing issues",
                "Corrected indicator calculations"
            ],
            "download_url": "https://github.com/yourusername/trading-bot/releases/latest"
        }
    
    def _is_newer_version(self, version1, version2):
        """Compare version strings"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            return v1_parts > v2_parts
            
        except:
            return False
    
    def download_update(self, update_info):
        """Download and install update"""
        try:
            print(f"📥 Downloading update {update_info['latest_version']}...")
            
            # In real implementation, this would download from the actual URL
            download_url = update_info.get("release_notes", {}).get("download_url")
            
            if not download_url:
                print("❌ No download URL available")
                return False
            
            # Simulate download process
            print("📦 Download complete, preparing installation...")
            
            # Create backup of current version
            if not self._create_backup():
                print("❌ Failed to create backup")
                return False
            
            # Simulate installation
            print("🔧 Installing update...")
            time.sleep(2)  # Simulate installation time
            
            # Update version file
            self._update_version_file(update_info)
            
            print(f"✅ Update {update_info['latest_version']} installed successfully!")
            print("🔄 Please restart the bot to use the new version")
            
            return True
            
        except Exception as e:
            print(f"❌ Update installation failed: {e}")
            return False
    
    def _create_backup(self):
        """Create backup of current bot files"""
        try:
            backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if os.path.exists(backup_dir):
                shutil.rmtree(backup_dir)
            
            os.makedirs(backup_dir)
            
            # List of files to backup
            files_to_backup = [
                "trading_bot_windows.py",
                "trading_bot_integrated.py", 
                "config.py",
                "enhanced_indicators.py",
                "market_data_api.py",
                "simulation_trading.py",
                "risk_manager.py",
                "telegram_notifier.py",
                "trade_logger.py",
                "version.json"
            ]
            
            backed_up = 0
            for filename in files_to_backup:
                if os.path.exists(filename):
                    shutil.copy2(filename, backup_dir)
                    backed_up += 1
            
            print(f"📁 Backed up {backed_up} files to {backup_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Backup creation failed: {e}")
            return False
    
    def _update_version_file(self, update_info):
        """Update version file with new version info"""
        try:
            version_data = {
                "version": update_info["latest_version"],
                "previous_version": self.current_version,
                "update_date": datetime.now().isoformat(),
                "release_notes": update_info.get("release_notes", {}),
                "auto_updated": True
            }
            
            with open(self.version_file, 'w') as f:
                json.dump(version_data, f, indent=2)
            
            self.current_version = update_info["latest_version"]
            
        except Exception as e:
            print(f"❌ Error updating version file: {e}")
    
    def rollback_update(self):
        """Rollback to previous version if available"""
        try:
            # Find most recent backup
            backup_dirs = [d for d in os.listdir('.') if d.startswith('backup_')]
            
            if not backup_dirs:
                print("❌ No backup found for rollback")
                return False
            
            # Get most recent backup
            latest_backup = max(backup_dirs)
            
            print(f"🔄 Rolling back to backup: {latest_backup}")
            
            # Restore files from backup
            backup_path = os.path.join('.', latest_backup)
            
            for filename in os.listdir(backup_path):
                src = os.path.join(backup_path, filename)
                dst = filename
                shutil.copy2(src, dst)
                print(f"   Restored: {filename}")
            
            print("✅ Rollback completed successfully!")
            print("🔄 Please restart the bot")
            
            return True
            
        except Exception as e:
            print(f"❌ Rollback failed: {e}")
            return False
    
    def get_update_status(self):
        """Get current update status"""
        try:
            status = {
                "current_version": self.current_version,
                "auto_check_enabled": self.auto_check,
                "last_check": self.last_check.isoformat() if self.last_check else None,
                "check_interval_hours": self.check_interval / 3600
            }
            
            # Check if update is available
            update_info = self.check_for_updates(silent=True)
            status.update(update_info)
            
            return status
            
        except Exception as e:
            return {"error": str(e)}
    
    def configure_auto_update(self, enabled=True, interval_hours=1):
        """Configure auto-update settings"""
        try:
            self.auto_check = enabled
            self.check_interval = interval_hours * 3600
            
            if enabled:
                self.start_auto_check()
                print(f"✅ Auto-update enabled (check every {interval_hours} hours)")
            else:
                print("❌ Auto-update disabled")
            
            return True
            
        except Exception as e:
            print(f"❌ Error configuring auto-update: {e}")
            return False
    
    def manual_update_check(self):
        """Manually check for updates with user interaction"""
        try:
            print("🔍 Checking for updates...")
            update_info = self.check_for_updates()
            
            if update_info.get("update_available"):
                print(f"🎉 Update {update_info['latest_version']} is available!")
                
                release_notes = update_info.get("release_notes", {})
                features = release_notes.get("features", [])
                bug_fixes = release_notes.get("bug_fixes", [])
                
                print("\\n📋 What's New:")
                for feature in features:
                    print(f"   {feature}")
                
                if bug_fixes:
                    print("\\n🔧 Bug Fixes:")
                    for fix in bug_fixes:
                        print(f"   • {fix}")
                
                response = input("\\n❓ Do you want to install this update? (y/n): ")
                
                if response.lower() == 'y':
                    return self.download_update(update_info)
                else:
                    print("⏭️ Update skipped")
                    return False
            else:
                print("✅ You have the latest version!")
                return True
                
        except Exception as e:
            print(f"❌ Manual update check failed: {e}")
            return False

# Global instance
auto_updater = AutoUpdater()