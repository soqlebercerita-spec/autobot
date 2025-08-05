"""
News & Economic Calendar Integration
Auto trading halt during high-impact news events
"""

import requests
import datetime
import json
from typing import List, Dict, Optional

class NewsIntegration:
    def __init__(self):
        self.forex_factory_url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        self.high_impact_events = []
        self.trading_halt_periods = []
        
    def fetch_economic_calendar(self) -> List[Dict]:
        """Fetch economic calendar from Forex Factory"""
        try:
            response = requests.get(self.forex_factory_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            print(f"News fetch error: {e}")
            return []
    
    def is_high_impact_event(self, event: Dict) -> bool:
        """Check if event is high impact"""
        impact = event.get('impact', '').lower()
        return impact in ['high', 'red']
    
    def should_halt_trading(self, current_time: datetime.datetime = None) -> bool:
        """Check if trading should be halted due to news"""
        if current_time is None:
            current_time = datetime.datetime.now()
        
        events = self.fetch_economic_calendar()
        
        for event in events:
            if self.is_high_impact_event(event):
                try:
                    event_time = datetime.datetime.strptime(event['date'], '%Y-%m-%d %H:%M:%S')
                    # Halt trading 15 minutes before and after high impact news
                    start_halt = event_time - datetime.timedelta(minutes=15)
                    end_halt = event_time + datetime.timedelta(minutes=15)
                    
                    if start_halt <= current_time <= end_halt:
                        return True
                except:
                    continue
        
        return False
    
    def get_upcoming_events(self, hours_ahead: int = 24) -> List[Dict]:
        """Get upcoming high impact events"""
        events = self.fetch_economic_calendar()
        upcoming = []
        current_time = datetime.datetime.now()
        
        for event in events:
            if self.is_high_impact_event(event):
                try:
                    event_time = datetime.datetime.strptime(event['date'], '%Y-%m-%d %H:%M:%S')
                    if current_time <= event_time <= current_time + datetime.timedelta(hours=hours_ahead):
                        upcoming.append(event)
                except:
                    continue
        
        return upcoming
    
    def get_market_sentiment(self) -> Dict:
        """Get basic market sentiment analysis"""
        events = self.fetch_economic_calendar()
        sentiment = {'bullish': 0, 'bearish': 0, 'neutral': 0}
        
        for event in events:
            if self.is_high_impact_event(event):
                # Simple sentiment based on event type
                title = event.get('title', '').lower()
                if any(word in title for word in ['rate', 'gdp', 'employment', 'jobs']):
                    if 'increase' in title or 'up' in title:
                        sentiment['bullish'] += 1
                    elif 'decrease' in title or 'down' in title:
                        sentiment['bearish'] += 1
                    else:
                        sentiment['neutral'] += 1
        
        total = sum(sentiment.values())
        if total > 0:
            sentiment = {k: v/total for k, v in sentiment.items()}
        
        return sentiment