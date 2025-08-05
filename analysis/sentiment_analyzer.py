"""
Sentiment Analysis Module for AuraTrade Bot
News and social media sentiment analysis for trading decisions
"""

import re
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import statistics
import time

from utils.logger import Logger

@dataclass
class NewsItem:
    """News item data structure"""
    title: str
    content: str
    source: str
    timestamp: datetime
    sentiment_score: float
    impact_level: str
    relevant_symbols: List[str]
    category: str

@dataclass
class SentimentSignal:
    """Sentiment-based trading signal"""
    symbol: str
    sentiment_score: float
    signal_strength: float
    signal_direction: str
    confidence: float
    sources: List[str]
    reasoning: str
    timestamp: datetime

class SentimentAnalyzer:
    """Advanced Sentiment Analysis for Financial Markets"""
    
    def __init__(self):
        self.logger = Logger()
        
        # Configuration
        self.update_interval_minutes = 15
        self.sentiment_history_hours = 24
        self.max_news_items = 100
        
        # Sentiment thresholds
        self.bullish_threshold = 0.6
        self.bearish_threshold = -0.6
        self.strong_sentiment_threshold = 0.8
        
        # Data storage
        self.news_cache = deque(maxlen=self.max_news_items)
        self.sentiment_history = {}  # symbol -> deque of sentiment scores
        self.last_update = None
        
        # Symbol-keyword mapping
        self.symbol_keywords = {
            'EURUSD': ['euro', 'eur', 'european central bank', 'ecb', 'eurozone', 'europe'],
            'GBPUSD': ['pound', 'gbp', 'sterling', 'bank of england', 'boe', 'brexit', 'uk', 'britain'],
            'USDJPY': ['dollar', 'usd', 'federal reserve', 'fed', 'japan', 'yen', 'jpy', 'boj'],
            'USDCHF': ['dollar', 'usd', 'swiss', 'franc', 'chf', 'switzerland', 'snb'],
            'AUDUSD': ['australian', 'aussie', 'aud', 'rba', 'australia', 'commodity'],
            'USDCAD': ['canadian', 'cad', 'boc', 'canada', 'oil', 'crude'],
            'NZDUSD': ['new zealand', 'kiwi', 'nzd', 'rbnz'],
            'XAUUSD': ['gold', 'xau', 'precious metals', 'inflation', 'safe haven'],
            'BTCUSD': ['bitcoin', 'btc', 'crypto', 'cryptocurrency', 'digital currency']
        }
        
        # Economic event impact levels
        self.high_impact_events = [
            'nonfarm payrolls', 'nfp', 'employment', 'unemployment', 'jobs',
            'inflation', 'cpi', 'ppi', 'pce',
            'gdp', 'growth', 'recession',
            'interest rate', 'fed decision', 'fomc', 'monetary policy',
            'trade war', 'tariffs', 'trade deal',
            'election', 'political', 'government',
            'war', 'conflict', 'geopolitical'
        ]
        
        # Sentiment keywords
        self.positive_keywords = [
            'bullish', 'optimistic', 'positive', 'growth', 'strong', 'rise', 'increase',
            'boost', 'surge', 'rally', 'gain', 'improve', 'better', 'good', 'excellent',
            'promising', 'confident', 'upbeat', 'soar', 'jump', 'advance'
        ]
        
        self.negative_keywords = [
            'bearish', 'pessimistic', 'negative', 'decline', 'fall', 'drop', 'weak',
            'crash', 'plunge', 'collapse', 'loss', 'worse', 'bad', 'terrible',
            'concerning', 'worried', 'fear', 'panic', 'slide', 'tumble', 'slump'
        ]
        
        # Free news sources (APIs that don't require paid subscriptions)
        self.news_sources = {
            'newsapi': {
                'url': 'https://newsapi.org/v2/everything',
                'api_key_env': 'NEWSAPI_KEY',
                'enabled': True
            },
            'alpha_vantage': {
                'url': 'https://www.alphavantage.co/query',
                'api_key_env': 'ALPHA_VANTAGE_KEY', 
                'enabled': True
            },
            'reddit': {
                'url': 'https://www.reddit.com/r/investing.json',
                'api_key_env': None,
                'enabled': True
            }
        }
        
        self.logger.info("Sentiment Analyzer initialized")
    
    def analyze_market_sentiment(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Analyze overall market sentiment"""
        try:
            if symbols is None:
                symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
            
            # Update news data if needed
            self._update_news_data()
            
            analysis = {
                'timestamp': datetime.now(),
                'overall_sentiment': 0.0,
                'symbol_sentiments': {},
                'sentiment_signals': [],
                'news_summary': {},
                'market_mood': 'neutral',
                'confidence_level': 0.0,
                'trend_analysis': {},
                'risk_sentiment': {}
            }
            
            # Analyze sentiment for each symbol
            symbol_scores = []
            for symbol in symbols:
                sentiment_data = self._analyze_symbol_sentiment(symbol)
                analysis['symbol_sentiments'][symbol] = sentiment_data
                
                if sentiment_data['score'] is not None:
                    symbol_scores.append(sentiment_data['score'])
                
                # Generate trading signals based on sentiment
                signals = self._generate_sentiment_signals(symbol, sentiment_data)
                analysis['sentiment_signals'].extend(signals)
            
            # Calculate overall sentiment
            if symbol_scores:
                analysis['overall_sentiment'] = statistics.mean(symbol_scores)
                analysis['confidence_level'] = self._calculate_confidence(symbol_scores)
            
            # Determine market mood
            analysis['market_mood'] = self._determine_market_mood(analysis['overall_sentiment'])
            
            # News summary
            analysis['news_summary'] = self._create_news_summary()
            
            # Sentiment trend analysis
            analysis['trend_analysis'] = self._analyze_sentiment_trends(symbols)
            
            # Risk sentiment analysis
            analysis['risk_sentiment'] = self._analyze_risk_sentiment()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing market sentiment: {str(e)}")
            return self._get_default_sentiment_analysis()
    
    def _update_news_data(self):
        """Update news data from various sources"""
        try:
            current_time = datetime.now()
            
            # Check if update is needed
            if (self.last_update and 
                (current_time - self.last_update).total_seconds() < self.update_interval_minutes * 60):
                return
            
            self.logger.info("Updating news data...")
            
            # Fetch from NewsAPI
            self._fetch_newsapi_data()
            
            # Fetch from Alpha Vantage
            self._fetch_alpha_vantage_news()
            
            # Fetch from Reddit (free)
            self._fetch_reddit_sentiment()
            
            # Clean old news
            self._cleanup_old_news()
            
            self.last_update = current_time
            self.logger.info(f"News update completed. Total items: {len(self.news_cache)}")
            
        except Exception as e:
            self.logger.error(f"Error updating news data: {str(e)}")
    
    def _fetch_newsapi_data(self):
        """Fetch news from NewsAPI"""
        try:
            import os
            api_key = os.getenv('NEWSAPI_KEY')
            if not api_key:
                self.logger.info("NewsAPI key not found, skipping NewsAPI")
                return
            
            # Financial keywords for news search
            keywords = 'forex OR currency OR "central bank" OR "interest rate" OR economy OR GDP OR inflation'
            
            params = {
                'q': keywords,
                'apiKey': api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20,
                'from': (datetime.now() - timedelta(hours=6)).isoformat()
            }
            
            response = requests.get(self.news_sources['newsapi']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                for article in articles:
                    news_item = self._process_news_article(article, 'NewsAPI')
                    if news_item:
                        self.news_cache.append(news_item)
                        
                self.logger.info(f"Fetched {len(articles)} articles from NewsAPI")
            else:
                self.logger.warning(f"NewsAPI request failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error fetching NewsAPI data: {str(e)}")
    
    def _fetch_alpha_vantage_news(self):
        """Fetch news from Alpha Vantage"""
        try:
            import os
            api_key = os.getenv('ALPHA_VANTAGE_KEY')
            if not api_key:
                self.logger.info("Alpha Vantage key not found, skipping Alpha Vantage news")
                return
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': 'FOREX:EUR,FOREX:GBP,FOREX:JPY,FOREX:CHF',
                'apikey': api_key,
                'limit': 20
            }
            
            response = requests.get(self.news_sources['alpha_vantage']['url'], params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                feed = data.get('feed', [])
                
                for item in feed:
                    news_item = self._process_alpha_vantage_item(item)
                    if news_item:
                        self.news_cache.append(news_item)
                
                self.logger.info(f"Fetched {len(feed)} items from Alpha Vantage")
            else:
                self.logger.warning(f"Alpha Vantage request failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error fetching Alpha Vantage news: {str(e)}")
    
    def _fetch_reddit_sentiment(self):
        """Fetch sentiment from Reddit (free API)"""
        try:
            # Fetch from investing subreddit
            subreddits = ['investing', 'forex', 'SecurityAnalysis']
            
            for subreddit in subreddits:
                try:
                    url = f'https://www.reddit.com/r/{subreddit}/hot.json?limit=10'
                    headers = {'User-Agent': 'AuraTrade Bot 1.0'}
                    
                    response = requests.get(url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        posts = data.get('data', {}).get('children', [])
                        
                        for post in posts:
                            post_data = post.get('data', {})
                            news_item = self._process_reddit_post(post_data, subreddit)
                            if news_item:
                                self.news_cache.append(news_item)
                        
                        self.logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
                    
                    # Avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching from r/{subreddit}: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error fetching Reddit sentiment: {str(e)}")
    
    def _process_news_article(self, article: Dict, source: str) -> Optional[NewsItem]:
        """Process a news article into NewsItem"""
        try:
            title = article.get('title', '')
            description = article.get('description', '')
            content = f"{title} {description}".strip()
            
            if not content or len(content) < 20:
                return None
            
            # Parse timestamp
            published_at = article.get('publishedAt', '')
            if published_at:
                timestamp = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            else:
                timestamp = datetime.now()
            
            # Calculate sentiment
            sentiment_score = self._calculate_text_sentiment(content)
            
            # Determine impact level
            impact_level = self._determine_impact_level(content)
            
            # Find relevant symbols
            relevant_symbols = self._find_relevant_symbols(content)
            
            # Determine category
            category = self._categorize_news(content)
            
            return NewsItem(
                title=title,
                content=content,
                source=source,
                timestamp=timestamp,
                sentiment_score=sentiment_score,
                impact_level=impact_level,
                relevant_symbols=relevant_symbols,
                category=category
            )
            
        except Exception as e:
            self.logger.error(f"Error processing news article: {str(e)}")
            return None
    
    def _process_alpha_vantage_item(self, item: Dict) -> Optional[NewsItem]:
        """Process Alpha Vantage news item"""
        try:
            title = item.get('title', '')
            summary = item.get('summary', '')
            content = f"{title} {summary}".strip()
            
            if not content:
                return None
            
            # Parse timestamp
            time_published = item.get('time_published', '')
            if time_published:
                # Alpha Vantage format: YYYYMMDDTHHMMSS
                timestamp = datetime.strptime(time_published, '%Y%m%dT%H%M%S')
            else:
                timestamp = datetime.now()
            
            # Use Alpha Vantage sentiment if available
            sentiment_score = 0.0
            sentiment_data = item.get('overall_sentiment_score')
            if sentiment_data:
                sentiment_score = float(sentiment_data)
            else:
                sentiment_score = self._calculate_text_sentiment(content)
            
            # Get ticker sentiment
            relevant_symbols = []
            ticker_sentiment = item.get('ticker_sentiment', [])
            for ticker in ticker_sentiment:
                symbol = ticker.get('ticker', '')
                if symbol.startswith('FOREX:'):
                    # Convert forex ticker to standard format
                    currency = symbol.replace('FOREX:', '')
                    if currency in ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'CAD', 'NZD']:
                        relevant_symbols.append(f"{currency}USD")
            
            if not relevant_symbols:
                relevant_symbols = self._find_relevant_symbols(content)
            
            impact_level = self._determine_impact_level(content)
            category = self._categorize_news(content)
            
            return NewsItem(
                title=title,
                content=content,
                source='Alpha Vantage',
                timestamp=timestamp,
                sentiment_score=sentiment_score,
                impact_level=impact_level,
                relevant_symbols=relevant_symbols,
                category=category
            )
            
        except Exception as e:
            self.logger.error(f"Error processing Alpha Vantage item: {str(e)}")
            return None
    
    def _process_reddit_post(self, post_data: Dict, subreddit: str) -> Optional[NewsItem]:
        """Process Reddit post"""
        try:
            title = post_data.get('title', '')
            selftext = post_data.get('selftext', '')
            content = f"{title} {selftext}".strip()
            
            if not content or len(content) < 30:
                return None
            
            # Skip deleted/removed posts
            if '[deleted]' in content or '[removed]' in content:
                return None
            
            # Parse timestamp
            created_utc = post_data.get('created_utc', 0)
            timestamp = datetime.fromtimestamp(created_utc) if created_utc else datetime.now()
            
            # Calculate sentiment
            sentiment_score = self._calculate_text_sentiment(content)
            
            # Determine relevance and impact
            relevant_symbols = self._find_relevant_symbols(content)
            if not relevant_symbols:
                return None  # Skip if not relevant to any symbols
            
            impact_level = 'low'  # Reddit posts generally have lower impact
            category = f'social_media_{subreddit}'
            
            return NewsItem(
                title=title,
                content=content,
                source=f'Reddit r/{subreddit}',
                timestamp=timestamp,
                sentiment_score=sentiment_score,
                impact_level=impact_level,
                relevant_symbols=relevant_symbols,
                category=category
            )
            
        except Exception as e:
            self.logger.error(f"Error processing Reddit post: {str(e)}")
            return None
    
    def _calculate_text_sentiment(self, text: str) -> float:
        """Calculate sentiment score for text (-1 to 1)"""
        try:
            if not text:
                return 0.0
            
            # Clean and lowercase text
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            words = text.split()
            
            if not words:
                return 0.0
            
            # Count positive and negative words
            positive_count = sum(1 for word in words if word in self.positive_keywords)
            negative_count = sum(1 for word in words if word in self.negative_keywords)
            
            # Calculate raw sentiment
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words == 0:
                return 0.0
            
            sentiment_score = (positive_count - negative_count) / len(words)
            
            # Normalize to -1 to 1 range
            sentiment_score = max(-1.0, min(1.0, sentiment_score * 10))
            
            # Apply intensity multipliers
            intensity_words = ['very', 'extremely', 'highly', 'significantly', 'massively', 'dramatically']
            intensity_count = sum(1 for word in words if word in intensity_words)
            if intensity_count > 0:
                sentiment_score *= (1 + intensity_count * 0.2)
            
            return max(-1.0, min(1.0, sentiment_score))
            
        except Exception as e:
            self.logger.error(f"Error calculating text sentiment: {str(e)}")
            return 0.0
    
    def _determine_impact_level(self, text: str) -> str:
        """Determine impact level of news"""
        try:
            text_lower = text.lower()
            
            # Check for high impact keywords
            high_impact_count = sum(1 for keyword in self.high_impact_events 
                                  if keyword in text_lower)
            
            if high_impact_count >= 2:
                return 'high'
            elif high_impact_count == 1:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'low'
    
    def _find_relevant_symbols(self, text: str) -> List[str]:
        """Find symbols relevant to the text"""
        try:
            text_lower = text.lower()
            relevant_symbols = []
            
            for symbol, keywords in self.symbol_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        if symbol not in relevant_symbols:
                            relevant_symbols.append(symbol)
                        break
            
            return relevant_symbols
            
        except Exception:
            return []
    
    def _categorize_news(self, text: str) -> str:
        """Categorize news content"""
        try:
            text_lower = text.lower()
            
            if any(word in text_lower for word in ['rate', 'fed', 'ecb', 'boe', 'monetary']):
                return 'monetary_policy'
            elif any(word in text_lower for word in ['employment', 'jobs', 'unemployment', 'nfp']):
                return 'employment'
            elif any(word in text_lower for word in ['inflation', 'cpi', 'ppi', 'pce']):
                return 'inflation'
            elif any(word in text_lower for word in ['gdp', 'growth', 'recession', 'economy']):
                return 'economic_growth'
            elif any(word in text_lower for word in ['trade', 'tariff', 'export', 'import']):
                return 'trade'
            elif any(word in text_lower for word in ['election', 'political', 'government', 'policy']):
                return 'political'
            elif any(word in text_lower for word in ['war', 'conflict', 'geopolitical', 'sanctions']):
                return 'geopolitical'
            else:
                return 'general'
                
        except Exception:
            return 'general'
    
    def _analyze_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment for a specific symbol"""
        try:
            # Filter news items relevant to symbol
            relevant_news = [item for item in self.news_cache 
                           if symbol in item.relevant_symbols]
            
            if not relevant_news:
                return {
                    'symbol': symbol,
                    'score': None,
                    'confidence': 0.0,
                    'news_count': 0,
                    'recent_trend': 'neutral',
                    'key_themes': [],
                    'impact_distribution': {}
                }
            
            # Calculate weighted sentiment score
            total_weight = 0
            weighted_sentiment = 0
            
            for news in relevant_news:
                # Weight by recency and impact
                age_hours = (datetime.now() - news.timestamp).total_seconds() / 3600
                recency_weight = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours
                
                impact_weights = {'high': 3.0, 'medium': 2.0, 'low': 1.0}
                impact_weight = impact_weights.get(news.impact_level, 1.0)
                
                weight = recency_weight * impact_weight
                weighted_sentiment += news.sentiment_score * weight
                total_weight += weight
            
            avg_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
            
            # Calculate confidence based on news volume and consistency
            confidence = min(1.0, len(relevant_news) / 10)  # Max confidence with 10+ news items
            
            # Analyze sentiment consistency
            sentiments = [news.sentiment_score for news in relevant_news]
            if len(sentiments) > 1:
                sentiment_std = statistics.stdev(sentiments)
                consistency_factor = max(0.1, 1.0 - sentiment_std)
                confidence *= consistency_factor
            
            # Determine recent trend
            recent_trend = self._calculate_sentiment_trend(symbol, relevant_news)
            
            # Extract key themes
            key_themes = self._extract_key_themes(relevant_news)
            
            # Impact distribution
            impact_distribution = {}
            for news in relevant_news:
                impact = news.impact_level
                impact_distribution[impact] = impact_distribution.get(impact, 0) + 1
            
            # Update sentiment history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = deque(maxlen=100)
            
            self.sentiment_history[symbol].append({
                'timestamp': datetime.now(),
                'score': avg_sentiment,
                'confidence': confidence
            })
            
            return {
                'symbol': symbol,
                'score': avg_sentiment,
                'confidence': confidence,
                'news_count': len(relevant_news),
                'recent_trend': recent_trend,
                'key_themes': key_themes,
                'impact_distribution': impact_distribution,
                'latest_news': relevant_news[:3]  # Latest 3 news items
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return {'symbol': symbol, 'score': None, 'confidence': 0.0}
    
    def _calculate_sentiment_trend(self, symbol: str, recent_news: List[NewsItem]) -> str:
        """Calculate sentiment trend for symbol"""
        try:
            if len(recent_news) < 3:
                return 'neutral'
            
            # Sort by timestamp
            sorted_news = sorted(recent_news, key=lambda x: x.timestamp)
            
            # Split into recent and older
            mid_point = len(sorted_news) // 2
            older_news = sorted_news[:mid_point]
            recent_news_subset = sorted_news[mid_point:]
            
            older_sentiment = statistics.mean([news.sentiment_score for news in older_news])
            recent_sentiment = statistics.mean([news.sentiment_score for news in recent_news_subset])
            
            sentiment_change = recent_sentiment - older_sentiment
            
            if sentiment_change > 0.1:
                return 'improving'
            elif sentiment_change < -0.1:
                return 'deteriorating'
            else:
                return 'stable'
                
        except Exception:
            return 'neutral'
    
    def _extract_key_themes(self, news_items: List[NewsItem]) -> List[str]:
        """Extract key themes from news items"""
        try:
            theme_counts = {}
            
            for news in news_items:
                category = news.category
                theme_counts[category] = theme_counts.get(category, 0) + 1
            
            # Sort by frequency and return top themes
            sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
            return [theme for theme, count in sorted_themes[:5]]
            
        except Exception:
            return []
    
    def _generate_sentiment_signals(self, symbol: str, sentiment_data: Dict[str, Any]) -> List[SentimentSignal]:
        """Generate trading signals based on sentiment"""
        signals = []
        
        try:
            sentiment_score = sentiment_data.get('score')
            confidence = sentiment_data.get('confidence', 0.0)
            
            if sentiment_score is None or confidence < 0.3:
                return signals
            
            # Generate signals based on sentiment strength
            if sentiment_score > self.bullish_threshold:
                signal_strength = min(1.0, abs(sentiment_score) * confidence)
                
                if sentiment_score > self.strong_sentiment_threshold:
                    signal_direction = 'strong_bullish'
                else:
                    signal_direction = 'bullish'
                
                signals.append(SentimentSignal(
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    signal_strength=signal_strength,
                    signal_direction=signal_direction,
                    confidence=confidence,
                    sources=[news.source for news in sentiment_data.get('latest_news', [])],
                    reasoning=f"Positive sentiment ({sentiment_score:.2f}) from {sentiment_data.get('news_count', 0)} news items",
                    timestamp=datetime.now()
                ))
            
            elif sentiment_score < self.bearish_threshold:
                signal_strength = min(1.0, abs(sentiment_score) * confidence)
                
                if sentiment_score < -self.strong_sentiment_threshold:
                    signal_direction = 'strong_bearish'
                else:
                    signal_direction = 'bearish'
                
                signals.append(SentimentSignal(
                    symbol=symbol,
                    sentiment_score=sentiment_score,
                    signal_strength=signal_strength,
                    signal_direction=signal_direction,
                    confidence=confidence,
                    sources=[news.source for news in sentiment_data.get('latest_news', [])],
                    reasoning=f"Negative sentiment ({sentiment_score:.2f}) from {sentiment_data.get('news_count', 0)} news items",
                    timestamp=datetime.now()
                ))
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment signals: {str(e)}")
            return []
    
    def _determine_market_mood(self, overall_sentiment: float) -> str:
        """Determine overall market mood"""
        if overall_sentiment > 0.4:
            return 'optimistic'
        elif overall_sentiment > 0.1:
            return 'slightly_positive'
        elif overall_sentiment < -0.4:
            return 'pessimistic'
        elif overall_sentiment < -0.1:
            return 'slightly_negative'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, scores: List[float]) -> float:
        """Calculate overall confidence based on score consistency"""
        try:
            if len(scores) < 2:
                return 0.5
            
            # Higher confidence when scores are consistent
            std_dev = statistics.stdev(scores)
            consistency = max(0.0, 1.0 - std_dev)
            
            # Higher confidence with more data points
            volume_factor = min(1.0, len(scores) / 5)
            
            return consistency * volume_factor
            
        except Exception:
            return 0.5
    
    def _create_news_summary(self) -> Dict[str, Any]:
        """Create summary of recent news"""
        try:
            if not self.news_cache:
                return {'total_items': 0, 'by_impact': {}, 'by_category': {}}
            
            # Count by impact level
            impact_counts = {}
            category_counts = {}
            source_counts = {}
            
            recent_items = [item for item in self.news_cache 
                          if (datetime.now() - item.timestamp).total_seconds() < 6 * 3600]  # Last 6 hours
            
            for item in recent_items:
                # Impact distribution
                impact = item.impact_level
                impact_counts[impact] = impact_counts.get(impact, 0) + 1
                
                # Category distribution
                category = item.category
                category_counts[category] = category_counts.get(category, 0) + 1
                
                # Source distribution
                source = item.source
                source_counts[source] = source_counts.get(source, 0) + 1
            
            return {
                'total_items': len(recent_items),
                'by_impact': impact_counts,
                'by_category': category_counts,
                'by_source': source_counts,
                'time_range': '6 hours'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating news summary: {str(e)}")
            return {'total_items': 0, 'error': str(e)}
    
    def _analyze_sentiment_trends(self, symbols: List[str]) -> Dict[str, Any]:
        """Analyze sentiment trends over time"""
        try:
            trend_analysis = {}
            
            for symbol in symbols:
                if symbol not in self.sentiment_history:
                    continue
                
                history = list(self.sentiment_history[symbol])
                if len(history) < 3:
                    continue
                
                # Calculate trend
                recent_scores = [item['score'] for item in history[-5:]]
                older_scores = [item['score'] for item in history[-10:-5]] if len(history) >= 10 else [item['score'] for item in history[:-5]]
                
                if recent_scores and older_scores:
                    recent_avg = statistics.mean(recent_scores)
                    older_avg = statistics.mean(older_scores)
                    
                    change = recent_avg - older_avg
                    
                    if change > 0.1:
                        trend = 'improving'
                    elif change < -0.1:
                        trend = 'declining'
                    else:
                        trend = 'stable'
                    
                    trend_analysis[symbol] = {
                        'trend': trend,
                        'change': change,
                        'current_score': recent_scores[-1] if recent_scores else 0,
                        'data_points': len(history)
                    }
            
            return trend_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment trends: {str(e)}")
            return {}
    
    def _analyze_risk_sentiment(self) -> Dict[str, Any]:
        """Analyze risk-on/risk-off sentiment"""
        try:
            risk_indicators = {
                'risk_on_keywords': ['growth', 'optimism', 'recovery', 'expansion', 'investment'],
                'risk_off_keywords': ['uncertainty', 'volatility', 'crisis', 'recession', 'fear']
            }
            
            risk_on_count = 0
            risk_off_count = 0
            
            recent_news = [item for item in self.news_cache 
                          if (datetime.now() - item.timestamp).total_seconds() < 12 * 3600]  # Last 12 hours
            
            for news in recent_news:
                content_lower = news.content.lower()
                
                for keyword in risk_indicators['risk_on_keywords']:
                    if keyword in content_lower:
                        risk_on_count += 1
                
                for keyword in risk_indicators['risk_off_keywords']:
                    if keyword in content_lower:
                        risk_off_count += 1
            
            total_risk_mentions = risk_on_count + risk_off_count
            
            if total_risk_mentions == 0:
                sentiment = 'neutral'
                confidence = 0.0
            else:
                risk_on_ratio = risk_on_count / total_risk_mentions
                
                if risk_on_ratio > 0.65:
                    sentiment = 'risk_on'
                elif risk_on_ratio < 0.35:
                    sentiment = 'risk_off'
                else:
                    sentiment = 'neutral'
                
                confidence = abs(risk_on_ratio - 0.5) * 2  # 0 to 1 scale
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'risk_on_mentions': risk_on_count,
                'risk_off_mentions': risk_off_count,
                'analysis_period': '12 hours'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk sentiment: {str(e)}")
            return {'sentiment': 'neutral', 'confidence': 0.0}
    
    def _cleanup_old_news(self):
        """Remove old news items"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.sentiment_history_hours)
            
            # Filter out old news
            recent_news = deque(maxlen=self.max_news_items)
            for news in self.news_cache:
                if news.timestamp > cutoff_time:
                    recent_news.append(news)
            
            self.news_cache = recent_news
            
            # Clean sentiment history
            for symbol in self.sentiment_history:
                while (self.sentiment_history[symbol] and 
                       self.sentiment_history[symbol][0]['timestamp'] < cutoff_time):
                    self.sentiment_history[symbol].popleft()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old news: {str(e)}")
    
    def _get_default_sentiment_analysis(self) -> Dict[str, Any]:
        """Get default sentiment analysis when errors occur"""
        return {
            'timestamp': datetime.now(),
            'overall_sentiment': 0.0,
            'symbol_sentiments': {},
            'sentiment_signals': [],
            'market_mood': 'neutral',
            'confidence_level': 0.0,
            'error': 'Failed to analyze sentiment'
        }
    
    def get_sentiment_summary(self) -> str:
        """Get brief sentiment summary"""
        try:
            if not self.news_cache:
                return "No sentiment data available"
            
            # Quick analysis of recent sentiment
            recent_news = [item for item in self.news_cache 
                          if (datetime.now() - item.timestamp).total_seconds() < 3 * 3600]
            
            if not recent_news:
                return "No recent sentiment data"
            
            avg_sentiment = statistics.mean([news.sentiment_score for news in recent_news])
            
            if avg_sentiment > 0.3:
                mood = "Positive"
            elif avg_sentiment < -0.3:
                mood = "Negative"
            else:
                mood = "Neutral"
            
            return f"Market Sentiment: {mood} ({avg_sentiment:.2f}) based on {len(recent_news)} recent news items"
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment summary: {str(e)}")
            return "Error generating sentiment summary"
    
    def get_symbol_sentiment_score(self, symbol: str) -> Optional[float]:
        """Get current sentiment score for a symbol"""
        try:
            if symbol in self.sentiment_history and self.sentiment_history[symbol]:
                return self.sentiment_history[symbol][-1]['score']
            return None
        except Exception:
            return None
