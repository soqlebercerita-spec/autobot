"""
Data Manager for AuraTrade Bot
Handles data storage, retrieval, and management for trading operations
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import threading
import os
import json
from collections import deque
import pickle

from core.mt5_connector import MT5Connector
from utils.logger import Logger

class DataManager:
    """Comprehensive data management system for trading bot"""
    
    def __init__(self, mt5_connector: MT5Connector):
        self.mt5_connector = mt5_connector
        self.logger = Logger()
        
        # Database configuration
        self.db_path = "data/trading_bot.db"
        self.backup_path = "data/backups/"
        
        # Data caches
        self.tick_data_cache = {}  # symbol -> deque of tick data
        self.ohlc_data_cache = {}  # (symbol, timeframe) -> DataFrame
        self.indicator_cache = {}  # (symbol, timeframe, indicator) -> array
        
        # Cache settings
        self.max_tick_cache_size = 10000
        self.max_ohlc_cache_bars = 5000
        self.cache_refresh_interval = 60  # seconds
        
        # Threading
        self.data_lock = threading.RLock()
        self.cache_thread = None
        self.stop_caching = False
        
        # Performance tracking
        self.data_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize database and caches
        self._initialize_database()
        self._initialize_caches()
        
        self.logger.info("Data Manager initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            # Create data directory
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            os.makedirs(self.backup_path, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Tick data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tick_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        bid REAL NOT NULL,
                        ask REAL NOT NULL,
                        volume REAL DEFAULT 0,
                        flags INTEGER DEFAULT 0,
                        created_at INTEGER DEFAULT (strftime('%s', 'now'))
                    )
                ''')
                
                # OHLC data table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ohlc_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL DEFAULT 0,
                        created_at INTEGER DEFAULT (strftime('%s', 'now')),
                        UNIQUE(symbol, timeframe, timestamp)
                    )
                ''')
                
                # Trading signals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        action TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        reason TEXT,
                        timestamp INTEGER NOT NULL,
                        executed INTEGER DEFAULT 0,
                        created_at INTEGER DEFAULT (strftime('%s', 'now'))
                    )
                ''')
                
                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        strategy TEXT,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        timestamp INTEGER NOT NULL,
                        created_at INTEGER DEFAULT (strftime('%s', 'now'))
                    )
                ''')
                
                # System events table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        timestamp INTEGER NOT NULL,
                        created_at INTEGER DEFAULT (strftime('%s', 'now'))
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tick_symbol_time ON tick_data(symbol, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_tf_time ON ohlc_data(symbol, timeframe, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON trading_signals(symbol, timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_time ON performance_metrics(timestamp)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_time ON system_events(timestamp)')
                
                conn.commit()
                
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def _initialize_caches(self):
        """Initialize data caches"""
        try:
            # Initialize tick data caches
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']  # Default symbols
            for symbol in symbols:
                self.tick_data_cache[symbol] = deque(maxlen=self.max_tick_cache_size)
            
            # Load recent OHLC data into cache
            self._load_recent_ohlc_data()
            
            self.logger.info("Data caches initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing caches: {str(e)}")
    
    def start_caching(self):
        """Start background caching thread"""
        if not self.cache_thread or not self.cache_thread.is_alive():
            self.stop_caching = False
            self.cache_thread = threading.Thread(target=self._cache_worker, daemon=True)
            self.cache_thread.start()
            self.logger.info("Data caching thread started")
    
    def stop_caching_thread(self):
        """Stop background caching thread"""
        self.stop_caching = True
        if self.cache_thread and self.cache_thread.is_alive():
            self.cache_thread.join(timeout=5)
        self.logger.info("Data caching thread stopped")
    
    def _cache_worker(self):
        """Background worker for data caching"""
        while not self.stop_caching:
            try:
                # Update tick data
                self._update_all_tick_data()
                
                # Update OHLC data
                self._update_all_ohlc_data()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep before next update
                for _ in range(self.cache_refresh_interval):
                    if self.stop_caching:
                        break
                    threading.Event().wait(1)
                    
            except Exception as e:
                self.logger.error(f"Cache worker error: {str(e)}")
                threading.Event().wait(5)  # Wait longer on error
    
    def update_tick_data(self, symbol: str) -> bool:
        """Update tick data for a symbol"""
        try:
            tick = self.mt5_connector.get_tick(symbol)
            if not tick:
                return False
            
            with self.data_lock:
                # Add to cache
                if symbol not in self.tick_data_cache:
                    self.tick_data_cache[symbol] = deque(maxlen=self.max_tick_cache_size)
                
                tick_data = {
                    'timestamp': tick.time,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'volume': getattr(tick, 'volume', 0),
                    'flags': getattr(tick, 'flags', 0)
                }
                
                self.tick_data_cache[symbol].append(tick_data)
                
                # Store in database periodically
                if len(self.tick_data_cache[symbol]) % 100 == 0:
                    self._store_tick_data_batch(symbol)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating tick data for {symbol}: {str(e)}")
            return False
    
    def update_ohlc_data(self, symbol: str, timeframe: str) -> bool:
        """Update OHLC data for symbol and timeframe"""
        try:
            rates = self.mt5_connector.get_rates(symbol, timeframe, 100)
            if rates is None or len(rates) == 0:
                return False
            
            cache_key = (symbol, timeframe)
            
            with self.data_lock:
                # Update cache
                self.ohlc_data_cache[cache_key] = rates
                
                # Store in database
                self._store_ohlc_data(symbol, timeframe, rates)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating OHLC data for {symbol} {timeframe}: {str(e)}")
            return False
    
    def get_tick_data(self, symbol: str, count: int = 1000) -> Optional[List[Dict]]:
        """Get tick data from cache or database"""
        try:
            self.data_requests += 1
            
            with self.data_lock:
                # Try cache first
                if symbol in self.tick_data_cache and len(self.tick_data_cache[symbol]) >= count:
                    self.cache_hits += 1
                    return list(self.tick_data_cache[symbol])[-count:]
                
                # Fallback to database
                self.cache_misses += 1
                return self._get_tick_data_from_db(symbol, count)
            
        except Exception as e:
            self.logger.error(f"Error getting tick data for {symbol}: {str(e)}")
            return None
    
    def get_ohlc_data(self, symbol: str, timeframe: str, count: int = 1000) -> Optional[pd.DataFrame]:
        """Get OHLC data from cache or database"""
        try:
            self.data_requests += 1
            cache_key = (symbol, timeframe)
            
            with self.data_lock:
                # Try cache first
                if cache_key in self.ohlc_data_cache:
                    cached_data = self.ohlc_data_cache[cache_key]
                    if len(cached_data) >= count:
                        self.cache_hits += 1
                        return cached_data.tail(count).copy()
                
                # Try MT5 connector
                rates = self.mt5_connector.get_rates(symbol, timeframe, count)
                if rates is not None and len(rates) > 0:
                    self.ohlc_data_cache[cache_key] = rates
                    self.cache_hits += 1
                    return rates
                
                # Fallback to database
                self.cache_misses += 1
                return self._get_ohlc_data_from_db(symbol, timeframe, count)
            
        except Exception as e:
            self.logger.error(f"Error getting OHLC data for {symbol} {timeframe}: {str(e)}")
            return None
    
    def get_real_time_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get real-time market data for symbol"""
        try:
            # Get latest tick
            tick = self.mt5_connector.get_tick(symbol)
            if not tick:
                return None
            
            # Get latest OHLC data
            rates_m1 = self.get_ohlc_data(symbol, 'M1', 1)
            
            real_time_data = {
                'symbol': symbol,
                'timestamp': datetime.fromtimestamp(tick.time),
                'bid': tick.bid,
                'ask': tick.ask,
                'spread': tick.ask - tick.bid,
                'mid_price': (tick.bid + tick.ask) / 2.0,
                'volume': getattr(tick, 'volume', 0)
            }
            
            # Add OHLC data if available
            if rates_m1 is not None and len(rates_m1) > 0:
                latest_bar = rates_m1.iloc[-1]
                real_time_data.update({
                    'open': latest_bar['open'],
                    'high': latest_bar['high'],
                    'low': latest_bar['low'],
                    'close': latest_bar['close'],
                    'bar_volume': latest_bar.get('tick_volume', latest_bar.get('real_volume', 0))
                })
            
            return real_time_data
            
        except Exception as e:
            self.logger.error(f"Error getting real-time data for {symbol}: {str(e)}")
            return None
    
    def store_trading_signal(self, signal: Dict[str, Any]) -> bool:
        """Store trading signal in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trading_signals 
                    (symbol, strategy, action, confidence, price, stop_loss, take_profit, reason, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.get('symbol', ''),
                    signal.get('strategy', ''),
                    signal.get('action', ''),
                    signal.get('confidence', 0.0),
                    signal.get('price', 0.0),
                    signal.get('stop_loss'),
                    signal.get('take_profit'),
                    signal.get('reason', ''),
                    int(signal.get('timestamp', datetime.now()).timestamp())
                ))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing trading signal: {str(e)}")
            return False
    
    def store_performance_metric(self, symbol: str, strategy: str, metric_name: str, 
                                metric_value: float) -> bool:
        """Store performance metric"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (symbol, strategy, metric_name, metric_value, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    strategy,
                    metric_name,
                    metric_value,
                    int(datetime.now().timestamp())
                ))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing performance metric: {str(e)}")
            return False
    
    def store_system_event(self, event_type: str, severity: str, message: str, 
                          details: str = None) -> bool:
        """Store system event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_events 
                    (event_type, severity, message, details, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    event_type,
                    severity,
                    message,
                    details,
                    int(datetime.now().timestamp())
                ))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing system event: {str(e)}")
            return False
    
    def get_trading_signals(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """Get trading signals from database"""
        try:
            cutoff_time = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if symbol:
                    cursor.execute('''
                        SELECT * FROM trading_signals 
                        WHERE symbol = ? AND timestamp > ?
                        ORDER BY timestamp DESC
                    ''', (symbol, cutoff_time))
                else:
                    cursor.execute('''
                        SELECT * FROM trading_signals 
                        WHERE timestamp > ?
                        ORDER BY timestamp DESC
                    ''', (cutoff_time,))
                
                columns = [desc[0] for desc in cursor.description]
                signals = []
                
                for row in cursor.fetchall():
                    signal = dict(zip(columns, row))
                    signal['timestamp'] = datetime.fromtimestamp(signal['timestamp'])
                    signals.append(signal)
                
                return signals
            
        except Exception as e:
            self.logger.error(f"Error getting trading signals: {str(e)}")
            return []
    
    def get_performance_metrics(self, symbol: str = None, strategy: str = None, 
                               hours: int = 24) -> List[Dict]:
        """Get performance metrics from database"""
        try:
            cutoff_time = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = 'SELECT * FROM performance_metrics WHERE timestamp > ?'
                params = [cutoff_time]
                
                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)
                
                if strategy:
                    query += ' AND strategy = ?'
                    params.append(strategy)
                
                query += ' ORDER BY timestamp DESC'
                
                cursor.execute(query, params)
                
                columns = [desc[0] for desc in cursor.description]
                metrics = []
                
                for row in cursor.fetchall():
                    metric = dict(zip(columns, row))
                    metric['timestamp'] = datetime.fromtimestamp(metric['timestamp'])
                    metrics.append(metric)
                
                return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return []
    
    def get_system_events(self, event_type: str = None, severity: str = None, 
                         hours: int = 24) -> List[Dict]:
        """Get system events from database"""
        try:
            cutoff_time = int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = 'SELECT * FROM system_events WHERE timestamp > ?'
                params = [cutoff_time]
                
                if event_type:
                    query += ' AND event_type = ?'
                    params.append(event_type)
                
                if severity:
                    query += ' AND severity = ?'
                    params.append(severity)
                
                query += ' ORDER BY timestamp DESC'
                
                cursor.execute(query, params)
                
                columns = [desc[0] for desc in cursor.description]
                events = []
                
                for row in cursor.fetchall():
                    event = dict(zip(columns, row))
                    event['timestamp'] = datetime.fromtimestamp(event['timestamp'])
                    events.append(event)
                
                return events
            
        except Exception as e:
            self.logger.error(f"Error getting system events: {str(e)}")
            return []
    
    def _update_all_tick_data(self):
        """Update tick data for all cached symbols"""
        for symbol in self.tick_data_cache.keys():
            self.update_tick_data(symbol)
    
    def _update_all_ohlc_data(self):
        """Update OHLC data for all cached symbols"""
        timeframes = ['M1', 'M5', 'M15', 'H1']
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
        
        for symbol in symbols:
            for timeframe in timeframes:
                self.update_ohlc_data(symbol, timeframe)
    
    def _store_tick_data_batch(self, symbol: str):
        """Store batch of tick data to database"""
        try:
            if symbol not in self.tick_data_cache:
                return
            
            # Get recent tick data
            recent_ticks = list(self.tick_data_cache[symbol])[-100:]  # Last 100 ticks
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for tick in recent_ticks:
                    cursor.execute('''
                        INSERT OR IGNORE INTO tick_data 
                        (symbol, timestamp, bid, ask, volume, flags)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        tick['timestamp'],
                        tick['bid'],
                        tick['ask'],
                        tick['volume'],
                        tick['flags']
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing tick data batch for {symbol}: {str(e)}")
    
    def _store_ohlc_data(self, symbol: str, timeframe: str, rates: pd.DataFrame):
        """Store OHLC data to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for _, row in rates.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO ohlc_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        timeframe,
                        int(row.name.timestamp()),  # Index is datetime
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        row.get('tick_volume', row.get('real_volume', 0))
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error storing OHLC data: {str(e)}")
    
    def _get_tick_data_from_db(self, symbol: str, count: int) -> List[Dict]:
        """Get tick data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT timestamp, bid, ask, volume, flags
                    FROM tick_data 
                    WHERE symbol = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (symbol, count))
                
                tick_data = []
                for row in cursor.fetchall():
                    tick_data.append({
                        'timestamp': row[0],
                        'bid': row[1],
                        'ask': row[2],
                        'volume': row[3],
                        'flags': row[4]
                    })
                
                return list(reversed(tick_data))  # Return in chronological order
            
        except Exception as e:
            self.logger.error(f"Error getting tick data from DB: {str(e)}")
            return []
    
    def _get_ohlc_data_from_db(self, symbol: str, timeframe: str, count: int) -> Optional[pd.DataFrame]:
        """Get OHLC data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlc_data 
                    WHERE symbol = ? AND timeframe = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                
                df = pd.read_sql_query(query, conn, params=(symbol, timeframe, count))
                
                if len(df) == 0:
                    return None
                
                # Convert timestamp to datetime index
                df['time'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('time', inplace=True)
                df.drop('timestamp', axis=1, inplace=True)
                
                # Reverse to get chronological order
                df = df.iloc[::-1]
                
                return df
            
        except Exception as e:
            self.logger.error(f"Error getting OHLC data from DB: {str(e)}")
            return None
    
    def _load_recent_ohlc_data(self):
        """Load recent OHLC data into cache"""
        try:
            symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD']
            timeframes = ['M1', 'M5', 'M15', 'H1']
            
            for symbol in symbols:
                for timeframe in timeframes:
                    data = self._get_ohlc_data_from_db(symbol, timeframe, 1000)
                    if data is not None:
                        cache_key = (symbol, timeframe)
                        self.ohlc_data_cache[cache_key] = data
                        
        except Exception as e:
            self.logger.error(f"Error loading recent OHLC data: {str(e)}")
    
    def _cleanup_old_data(self):
        """Clean up old data from database and cache"""
        try:
            # Clean database (keep last 30 days)
            cutoff_time = int((datetime.now() - timedelta(days=30)).timestamp())
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Clean old tick data
                cursor.execute('DELETE FROM tick_data WHERE timestamp < ?', (cutoff_time,))
                
                # Clean old OHLC data (keep more for analysis)
                ohlc_cutoff = int((datetime.now() - timedelta(days=90)).timestamp())
                cursor.execute('DELETE FROM ohlc_data WHERE timestamp < ?', (ohlc_cutoff,))
                
                # Clean old signals
                signal_cutoff = int((datetime.now() - timedelta(days=7)).timestamp())
                cursor.execute('DELETE FROM trading_signals WHERE timestamp < ?', (signal_cutoff,))
                
                # Clean old events
                event_cutoff = int((datetime.now() - timedelta(days=7)).timestamp())
                cursor.execute('DELETE FROM system_events WHERE timestamp < ?', (event_cutoff,))
                
                conn.commit()
                
                # Vacuum database to reclaim space
                cursor.execute('VACUUM')
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {str(e)}")
    
    def create_backup(self) -> bool:
        """Create database backup"""
        try:
            backup_filename = f"trading_bot_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            backup_path = os.path.join(self.backup_path, backup_filename)
            
            # Create backup directory
            os.makedirs(self.backup_path, exist_ok=True)
            
            # Copy database
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            self.logger.info(f"Database backup created: {backup_path}")
            
            # Clean old backups (keep last 10)
            self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {str(e)}")
            return False
    
    def _cleanup_old_backups(self):
        """Clean up old backup files"""
        try:
            if not os.path.exists(self.backup_path):
                return
            
            # Get all backup files
            backup_files = []
            for file in os.listdir(self.backup_path):
                if file.startswith('trading_bot_backup_') and file.endswith('.db'):
                    file_path = os.path.join(self.backup_path, file)
                    backup_files.append((file_path, os.path.getmtime(file_path)))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the 10 most recent backups
            for file_path, _ in backup_files[10:]:
                os.remove(file_path)
                self.logger.info(f"Removed old backup: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old backups: {str(e)}")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.data_requests
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate_percent': hit_rate,
            'tick_cache_symbols': len(self.tick_data_cache),
            'ohlc_cache_keys': len(self.ohlc_data_cache),
            'indicator_cache_keys': len(self.indicator_cache)
        }
    
    def cleanup_old_data(self):
        """Public method to trigger data cleanup"""
        self._cleanup_old_data()
    
    def export_data(self, symbol: str, timeframe: str, start_date: datetime, 
                   end_date: datetime, format: str = 'csv') -> Optional[str]:
        """Export data to file"""
        try:
            start_timestamp = int(start_date.timestamp())
            end_timestamp = int(end_date.timestamp())
            
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT timestamp, open, high, low, close, volume
                    FROM ohlc_data 
                    WHERE symbol = ? AND timeframe = ? 
                    AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                '''
                
                df = pd.read_sql_query(query, conn, params=(symbol, timeframe, start_timestamp, end_timestamp))
                
                if len(df) == 0:
                    return None
                
                # Convert timestamp to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
                
                # Create export filename
                export_filename = f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.{format}"
                export_path = os.path.join("data/exports", export_filename)
                
                # Create export directory
                os.makedirs(os.path.dirname(export_path), exist_ok=True)
                
                # Export data
                if format.lower() == 'csv':
                    df.to_csv(export_path, index=False)
                elif format.lower() == 'json':
                    df.to_json(export_path, orient='records', date_format='iso')
                elif format.lower() == 'excel':
                    df.to_excel(export_path, index=False)
                else:
                    raise ValueError(f"Unsupported export format: {format}")
                
                self.logger.info(f"Data exported to: {export_path}")
                return export_path
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {str(e)}")
            return None
