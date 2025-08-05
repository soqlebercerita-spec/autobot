
"""
Credentials Management for AuraTrade
Secure handling of sensitive information
"""

import os
from typing import Optional

class Credentials:
    """Secure credentials management"""
    
    @staticmethod
    def get_mt5_credentials() -> dict:
        """Get MT5 credentials from environment"""
        return {
            "login": os.getenv("MT5_LOGIN", ""),
            "password": os.getenv("MT5_PASSWORD", ""),
            "server": os.getenv("MT5_SERVER", ""),
            "timeout": int(os.getenv("MT5_TIMEOUT", "60000")),
            "portable": os.getenv("MT5_PORTABLE", "false").lower() == "true"
        }
    
    @staticmethod
    def get_telegram_credentials() -> dict:
        """Get Telegram credentials from environment"""
        return {
            "bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
            "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
            "enabled": os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
        }
    
    @staticmethod
    def validate_credentials() -> dict:
        """Validate all credentials and return status"""
        status = {
            "mt5": False,
            "telegram": False,
            "errors": []
        }
        
        # Check MT5
        mt5_creds = Credentials.get_mt5_credentials()
        if mt5_creds["login"] and mt5_creds["password"] and mt5_creds["server"]:
            status["mt5"] = True
        else:
            status["errors"].append("MT5 credentials incomplete (will use simulation)")
        
        # Check Telegram
        tg_creds = Credentials.get_telegram_credentials()
        if tg_creds["enabled"]:
            if tg_creds["bot_token"] and tg_creds["chat_id"]:
                status["telegram"] = True
            else:
                status["errors"].append("Telegram enabled but credentials missing")
        
        return status
