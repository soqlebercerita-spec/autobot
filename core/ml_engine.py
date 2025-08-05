"""
Machine Learning Engine for AuraTrade Bot
Advanced ML models for market prediction and pattern recognition
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pickle
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from config.config import config
from config.settings import settings
from analysis.technical_analysis import TechnicalAnalysis
from utils.logger import Logger

class LSTMModel:
    """Simplified LSTM-like model using sklearn"""
    
    def __init__(self, sequence_length: int = 60, features: int = 10):
        self.sequence_length = sequence_length
        self.features = features
        self.model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            activation='tanh',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i].flatten())
            y.append(data[i, 0])  # Predict close price
        return np.array(X), np.array(y)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class CNNModel:
    """Simplified CNN-like model for pattern recognition"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = MinMaxScaler()
        self.is_trained = False
    
    def prepare_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for CNN-like processing"""
        X, y = [], []
        for i in range(self.window_size, len(data)):
            # Create feature maps similar to CNN
            window = data[i-self.window_size:i]
            features = []
            
            # Price features
            features.extend(window[:, 0])  # Close prices
            features.extend(window[:, 1])  # Volume
            
            # Technical features
            features.append(np.mean(window[:, 0]))  # Average price
            features.append(np.std(window[:, 0]))   # Price volatility
            features.append(np.max(window[:, 0]) - np.min(window[:, 0]))  # Range
            
            X.append(features)
            y.append(data[i, 0])  # Next close price
        
        return np.array(X), np.array(y)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

class TransformerModel:
    """Simplified Transformer-like model using ensemble methods"""
    
    def __init__(self, attention_heads: int = 8):
        self.attention_heads = attention_heads
        self.models = []
        for _ in range(attention_heads):
            self.models.append(GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ))
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_attention_features(self, data: np.ndarray, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features with attention-like mechanism"""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            sequence = data[i-sequence_length:i]
            
            # Multi-head attention simulation
            attention_features = []
            
            for head in range(self.attention_heads):
                # Different attention patterns for each head
                if head == 0:  # Recent price attention
                    weights = np.exp(np.linspace(-2, 0, sequence_length))
                elif head == 1:  # Volume attention
                    weights = sequence[:, 1] / np.sum(sequence[:, 1])
                elif head == 2:  # Volatility attention
                    price_changes = np.diff(sequence[:, 0])
                    weights = np.abs(np.concatenate([[0], price_changes]))
                else:  # Uniform attention
                    weights = np.ones(sequence_length) / sequence_length
                
                weights = weights / np.sum(weights)
                
                # Weighted features
                weighted_price = np.sum(sequence[:, 0] * weights)
                weighted_volume = np.sum(sequence[:, 1] * weights)
                
                attention_features.extend([weighted_price, weighted_volume])
            
            X.append(attention_features)
            y.append(data[i, 0])
        
        return np.array(X), np.array(y)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the ensemble models"""
        X_scaled = self.scaler.fit_transform(X)
        
        # Split features for different attention heads
        features_per_head = X_scaled.shape[1] // self.attention_heads
        
        for i, model in enumerate(self.models):
            start_idx = i * features_per_head
            end_idx = (i + 1) * features_per_head
            head_features = X_scaled[:, start_idx:end_idx]
            model.fit(head_features, y)
        
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            return np.array([])
        
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        features_per_head = X_scaled.shape[1] // self.attention_heads
        
        for i, model in enumerate(self.models):
            start_idx = i * features_per_head
            end_idx = (i + 1) * features_per_head
            head_features = X_scaled[:, start_idx:end_idx]
            pred = model.predict(head_features)
            predictions.append(pred)
        
        # Ensemble average
        return np.mean(predictions, axis=0)

class MLEngine:
    """Main Machine Learning Engine"""
    
    def __init__(self):
        self.logger = Logger()
        self.technical_analysis = TechnicalAnalysis()
        
        # Models
        self.lstm_model = LSTMModel()
        self.cnn_model = CNNModel()
        self.transformer_model = TransformerModel()
        
        # Model states
        self.models_trained = {
            'lstm': False,
            'cnn': False,
            'transformer': False
        }
        
        # Prediction cache
        self.prediction_cache = {}
        self.cache_timeout = 60  # seconds
        
        # Model performance tracking
        self.model_performance = {
            'lstm': {'accuracy': 0.0, 'mse': float('inf'), 'last_update': None},
            'cnn': {'accuracy': 0.0, 'mse': float('inf'), 'last_update': None},
            'transformer': {'accuracy': 0.0, 'mse': float('inf'), 'last_update': None}
        }
        
        # Model paths for persistence
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.logger.info("ML Engine initialized")
    
    def prepare_training_data(self, symbol: str, timeframe: str = 'H1', 
                            bars: int = 5000) -> Optional[np.ndarray]:
        """Prepare training data from market data"""
        try:
            # This would get actual market data
            # For now, create realistic synthetic data structure
            dates = pd.date_range(
                start=datetime.now() - timedelta(days=bars//24), 
                periods=bars, 
                freq='H'
            )
            
            # Create realistic OHLCV data structure
            np.random.seed(42)  # For reproducible "realistic" data
            base_price = 1.1000
            
            data = []
            for i in range(bars):
                # Simulate realistic price movement
                change = np.random.normal(0, 0.0001)
                base_price += change
                
                high = base_price + abs(np.random.normal(0, 0.0002))
                low = base_price - abs(np.random.normal(0, 0.0002))
                close = base_price + np.random.normal(0, 0.0001)
                volume = np.random.randint(100, 1000)
                
                data.append([close, volume, high, low, base_price])  # [close, volume, high, low, open]
                base_price = close
            
            # Add technical indicators
            prices = np.array([d[0] for d in data])
            volumes = np.array([d[1] for d in data])
            
            # Calculate technical indicators
            sma_20 = self._calculate_sma(prices, 20)
            ema_12 = self._calculate_ema(prices, 12)
            rsi = self._calculate_rsi(prices, 14)
            macd_line, macd_signal = self._calculate_macd(prices)
            
            # Combine all features
            training_data = []
            for i in range(len(data)):
                features = [
                    data[i][0],  # close
                    data[i][1],  # volume
                    sma_20[i] if i < len(sma_20) else prices[i],
                    ema_12[i] if i < len(ema_12) else prices[i],
                    rsi[i] if i < len(rsi) else 50.0,
                    macd_line[i] if i < len(macd_line) else 0.0,
                    macd_signal[i] if i < len(macd_signal) else 0.0,
                    data[i][2],  # high
                    data[i][3],  # low
                    data[i][4]   # open
                ]
                training_data.append(features)
            
            return np.array(training_data)
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            return None
    
    def train_models(self, symbol: str, retrain: bool = False) -> bool:
        """Train all ML models"""
        try:
            # Load or prepare training data
            training_data = self.prepare_training_data(symbol)
            if training_data is None or len(training_data) < 100:
                self.logger.error("Insufficient training data")
                return False
            
            self.logger.info(f"Training ML models for {symbol} with {len(training_data)} samples")
            
            # Split data
            train_size = int(0.8 * len(training_data))
            train_data = training_data[:train_size]
            test_data = training_data[train_size:]
            
            # Train LSTM
            try:
                X_lstm, y_lstm = self.lstm_model.prepare_sequences(train_data)
                if len(X_lstm) > 0:
                    self.lstm_model.fit(X_lstm, y_lstm)
                    self.models_trained['lstm'] = True
                    
                    # Evaluate
                    X_test_lstm, y_test_lstm = self.lstm_model.prepare_sequences(test_data)
                    if len(X_test_lstm) > 0:
                        pred_lstm = self.lstm_model.predict(X_test_lstm)
                        mse_lstm = mean_squared_error(y_test_lstm, pred_lstm)
                        self.model_performance['lstm'] = {
                            'accuracy': r2_score(y_test_lstm, pred_lstm),
                            'mse': mse_lstm,
                            'last_update': datetime.now()
                        }
                    
                    self.logger.info("LSTM model trained successfully")
            except Exception as e:
                self.logger.error(f"LSTM training error: {str(e)}")
            
            # Train CNN
            try:
                X_cnn, y_cnn = self.cnn_model.prepare_features(train_data)
                if len(X_cnn) > 0:
                    self.cnn_model.fit(X_cnn, y_cnn)
                    self.models_trained['cnn'] = True
                    
                    # Evaluate
                    X_test_cnn, y_test_cnn = self.cnn_model.prepare_features(test_data)
                    if len(X_test_cnn) > 0:
                        pred_cnn = self.cnn_model.predict(X_test_cnn)
                        mse_cnn = mean_squared_error(y_test_cnn, pred_cnn)
                        self.model_performance['cnn'] = {
                            'accuracy': r2_score(y_test_cnn, pred_cnn),
                            'mse': mse_cnn,
                            'last_update': datetime.now()
                        }
                    
                    self.logger.info("CNN model trained successfully")
            except Exception as e:
                self.logger.error(f"CNN training error: {str(e)}")
            
            # Train Transformer
            try:
                X_transformer, y_transformer = self.transformer_model.prepare_attention_features(train_data)
                if len(X_transformer) > 0:
                    self.transformer_model.fit(X_transformer, y_transformer)
                    self.models_trained['transformer'] = True
                    
                    # Evaluate
                    X_test_transformer, y_test_transformer = self.transformer_model.prepare_attention_features(test_data)
                    if len(X_test_transformer) > 0:
                        pred_transformer = self.transformer_model.predict(X_test_transformer)
                        mse_transformer = mean_squared_error(y_test_transformer, pred_transformer)
                        self.model_performance['transformer'] = {
                            'accuracy': r2_score(y_test_transformer, pred_transformer),
                            'mse': mse_transformer,
                            'last_update': datetime.now()
                        }
                    
                    self.logger.info("Transformer model trained successfully")
            except Exception as e:
                self.logger.error(f"Transformer training error: {str(e)}")
            
            # Save models
            self.save_models(symbol)
            
            return any(self.models_trained.values())
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return False
    
    def predict_price(self, symbol: str, timeframe: str = 'H1', 
                     periods_ahead: int = 1) -> Dict[str, Any]:
        """Generate price predictions using ensemble of models"""
        try:
            cache_key = f"{symbol}_{timeframe}_{periods_ahead}"
            
            # Check cache
            if cache_key in self.prediction_cache:
                cache_time, cached_result = self.prediction_cache[cache_key]
                if (datetime.now() - cache_time).seconds < self.cache_timeout:
                    return cached_result
            
            # Get recent data for prediction
            recent_data = self.prepare_training_data(symbol, timeframe, 100)
            if recent_data is None or len(recent_data) < 60:
                return self._get_default_prediction()
            
            predictions = {}
            confidences = {}
            
            # LSTM Prediction
            if self.models_trained['lstm']:
                try:
                    X_lstm, _ = self.lstm_model.prepare_sequences(recent_data[-70:])
                    if len(X_lstm) > 0:
                        pred_lstm = self.lstm_model.predict(X_lstm[-1:])
                        predictions['lstm'] = pred_lstm[0] if len(pred_lstm) > 0 else recent_data[-1, 0]
                        confidences['lstm'] = self._calculate_confidence('lstm')
                except Exception as e:
                    self.logger.error(f"LSTM prediction error: {str(e)}")
            
            # CNN Prediction
            if self.models_trained['cnn']:
                try:
                    X_cnn, _ = self.cnn_model.prepare_features(recent_data[-30:])
                    if len(X_cnn) > 0:
                        pred_cnn = self.cnn_model.predict(X_cnn[-1:])
                        predictions['cnn'] = pred_cnn[0] if len(pred_cnn) > 0 else recent_data[-1, 0]
                        confidences['cnn'] = self._calculate_confidence('cnn')
                except Exception as e:
                    self.logger.error(f"CNN prediction error: {str(e)}")
            
            # Transformer Prediction
            if self.models_trained['transformer']:
                try:
                    X_transformer, _ = self.transformer_model.prepare_attention_features(recent_data[-40:])
                    if len(X_transformer) > 0:
                        pred_transformer = self.transformer_model.predict(X_transformer[-1:])
                        predictions['transformer'] = pred_transformer[0] if len(pred_transformer) > 0 else recent_data[-1, 0]
                        confidences['transformer'] = self._calculate_confidence('transformer')
                except Exception as e:
                    self.logger.error(f"Transformer prediction error: {str(e)}")
            
            if not predictions:
                return self._get_default_prediction()
            
            # Ensemble prediction with confidence weighting
            total_weight = sum(confidences.values())
            if total_weight > 0:
                ensemble_pred = sum(pred * confidences[model] for model, pred in predictions.items()) / total_weight
            else:
                ensemble_pred = np.mean(list(predictions.values()))
            
            current_price = recent_data[-1, 0]
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': current_price,
                'predicted_price': ensemble_pred,
                'price_change': ensemble_pred - current_price,
                'price_change_pct': ((ensemble_pred - current_price) / current_price) * 100,
                'individual_predictions': predictions,
                'model_confidences': confidences,
                'prediction_strength': self._calculate_prediction_strength(predictions, confidences),
                'timestamp': datetime.now(),
                'periods_ahead': periods_ahead
            }
            
            # Cache result
            self.prediction_cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {str(e)}")
            return self._get_default_prediction()
    
    def detect_patterns(self, symbol: str, pattern_types: List[str] = None) -> Dict[str, Any]:
        """Detect price patterns using ML models"""
        try:
            if pattern_types is None:
                pattern_types = ['trend_reversal', 'breakout', 'consolidation', 'momentum']
            
            recent_data = self.prepare_training_data(symbol, 'H1', 200)
            if recent_data is None or len(recent_data) < 50:
                return {}
            
            patterns = {}
            
            # Use CNN model for pattern recognition
            if self.models_trained['cnn']:
                try:
                    # Prepare features for pattern detection
                    X_pattern, _ = self.cnn_model.prepare_features(recent_data[-50:])
                    
                    if len(X_pattern) > 0:
                        # Get model confidence for different patterns
                        recent_features = X_pattern[-1:]
                        
                        # Simulate pattern detection
                        price_data = recent_data[-20:, 0]  # Last 20 close prices
                        
                        # Trend detection
                        trend_slope = np.polyfit(range(len(price_data)), price_data, 1)[0]
                        patterns['trend'] = {
                            'direction': 'bullish' if trend_slope > 0 else 'bearish',
                            'strength': min(abs(trend_slope) * 10000, 1.0),
                            'confidence': 0.8
                        }
                        
                        # Volatility pattern
                        volatility = np.std(price_data)
                        patterns['volatility'] = {
                            'level': 'high' if volatility > np.mean(np.std(recent_data[:, 0])) else 'low',
                            'value': volatility,
                            'confidence': 0.7
                        }
                        
                        # Support/Resistance levels
                        high_prices = recent_data[-20:, 7]  # High prices
                        low_prices = recent_data[-20:, 8]   # Low prices
                        
                        patterns['support_resistance'] = {
                            'resistance': np.max(high_prices),
                            'support': np.min(low_prices),
                            'current_position': 'middle',  # Simplified
                            'confidence': 0.6
                        }
                
                except Exception as e:
                    self.logger.error(f"Pattern detection error: {str(e)}")
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {str(e)}")
            return {}
    
    def generate_trading_signals(self, symbol: str) -> Dict[str, Any]:
        """Generate ML-based trading signals"""
        try:
            # Get price prediction
            prediction = self.predict_price(symbol)
            
            # Get pattern analysis
            patterns = self.detect_patterns(symbol)
            
            if not prediction or prediction.get('price_change_pct', 0) == 0:
                return {'signal': 'HOLD', 'confidence': 0.0, 'reason': 'No prediction available'}
            
            signal_strength = 0.0
            signal_direction = 'HOLD'
            reasons = []
            
            # Price prediction signal
            price_change_pct = prediction.get('price_change_pct', 0)
            prediction_strength = prediction.get('prediction_strength', 0)
            
            if abs(price_change_pct) > 0.1:  # 0.1% threshold
                if price_change_pct > 0:
                    signal_direction = 'BUY'
                    signal_strength += prediction_strength * 0.4
                    reasons.append(f"Price prediction: +{price_change_pct:.2f}%")
                else:
                    signal_direction = 'SELL'
                    signal_strength += prediction_strength * 0.4
                    reasons.append(f"Price prediction: {price_change_pct:.2f}%")
            
            # Pattern-based signals
            if patterns:
                trend = patterns.get('trend', {})
                if trend.get('strength', 0) > 0.5:
                    if trend.get('direction') == 'bullish':
                        if signal_direction in ['HOLD', 'BUY']:
                            signal_direction = 'BUY'
                            signal_strength += trend.get('confidence', 0) * 0.3
                            reasons.append(f"Bullish trend detected")
                    else:
                        if signal_direction in ['HOLD', 'SELL']:
                            signal_direction = 'SELL'
                            signal_strength += trend.get('confidence', 0) * 0.3
                            reasons.append(f"Bearish trend detected")
                
                # Volatility consideration
                volatility = patterns.get('volatility', {})
                if volatility.get('level') == 'high':
                    signal_strength *= 0.8  # Reduce confidence in high volatility
                    reasons.append("High volatility detected")
            
            # Ensemble confidence
            model_confidences = prediction.get('model_confidences', {})
            avg_confidence = np.mean(list(model_confidences.values())) if model_confidences else 0.5
            signal_strength *= avg_confidence
            
            # Final signal
            if signal_strength < 0.3:
                signal_direction = 'HOLD'
            
            return {
                'signal': signal_direction,
                'confidence': min(signal_strength, 1.0),
                'strength': signal_strength,
                'reasons': reasons,
                'prediction': prediction,
                'patterns': patterns,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {str(e)}")
            return {'signal': 'HOLD', 'confidence': 0.0, 'reason': f'Error: {str(e)}'}
    
    def _calculate_confidence(self, model_name: str) -> float:
        """Calculate model confidence based on performance"""
        try:
            performance = self.model_performance.get(model_name, {})
            accuracy = performance.get('accuracy', 0.0)
            mse = performance.get('mse', float('inf'))
            
            # Normalize accuracy (R2 score can be negative)
            normalized_accuracy = max(0.0, min(1.0, accuracy))
            
            # Inverse of MSE (lower MSE = higher confidence)
            mse_confidence = 1.0 / (1.0 + mse * 1000) if mse != float('inf') else 0.0
            
            # Combined confidence
            confidence = (normalized_accuracy + mse_confidence) / 2.0
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _calculate_prediction_strength(self, predictions: Dict[str, float], 
                                     confidences: Dict[str, float]) -> float:
        """Calculate overall prediction strength"""
        try:
            if not predictions or not confidences:
                return 0.0
            
            # Agreement between models
            pred_values = list(predictions.values())
            if len(pred_values) > 1:
                agreement = 1.0 - (np.std(pred_values) / np.mean(pred_values))
                agreement = max(0.0, min(1.0, agreement))
            else:
                agreement = 1.0
            
            # Average confidence
            avg_confidence = np.mean(list(confidences.values()))
            
            # Combined strength
            strength = agreement * avg_confidence
            
            return min(strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction strength: {str(e)}")
            return 0.0
    
    def _get_default_prediction(self) -> Dict[str, Any]:
        """Get default prediction when models fail"""
        return {
            'symbol': 'UNKNOWN',
            'timeframe': 'H1',
            'current_price': 0.0,
            'predicted_price': 0.0,
            'price_change': 0.0,
            'price_change_pct': 0.0,
            'individual_predictions': {},
            'model_confidences': {},
            'prediction_strength': 0.0,
            'timestamp': datetime.now(),
            'periods_ahead': 1
        }
    
    def save_models(self, symbol: str):
        """Save trained models to disk"""
        try:
            model_file = os.path.join(self.model_dir, f"{symbol}_models.pkl")
            
            models_data = {
                'lstm': self.lstm_model if self.models_trained['lstm'] else None,
                'cnn': self.cnn_model if self.models_trained['cnn'] else None,
                'transformer': self.transformer_model if self.models_trained['transformer'] else None,
                'performance': self.model_performance,
                'trained_states': self.models_trained,
                'timestamp': datetime.now()
            }
            
            with open(model_file, 'wb') as f:
                pickle.dump(models_data, f)
            
            self.logger.info(f"Models saved for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, symbol: str) -> bool:
        """Load trained models from disk"""
        try:
            model_file = os.path.join(self.model_dir, f"{symbol}_models.pkl")
            
            if not os.path.exists(model_file):
                return False
            
            with open(model_file, 'rb') as f:
                models_data = pickle.load(f)
            
            # Load models
            if models_data['lstm']:
                self.lstm_model = models_data['lstm']
                self.models_trained['lstm'] = True
            
            if models_data['cnn']:
                self.cnn_model = models_data['cnn']
                self.models_trained['cnn'] = True
            
            if models_data['transformer']:
                self.transformer_model = models_data['transformer']
                self.models_trained['transformer'] = True
            
            # Load performance data
            self.model_performance = models_data.get('performance', self.model_performance)
            
            self.logger.info(f"Models loaded for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            return False
    
    # Helper methods for technical indicators
    def _calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        sma = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])
        return sma
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        ema = np.full(len(prices), np.nan)
        multiplier = 2.0 / (period + 1)
        
        # Initialize with SMA
        ema[period - 1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i - 1] * (1 - multiplier))
        
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        rsi = np.full(len(prices), 50.0)
        
        if len(prices) < period + 1:
            return rsi
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(prices) - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD"""
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = self._calculate_ema(macd_line, signal)
        
        return macd_line, signal_line
    
    def get_ml_status(self) -> Dict[str, Any]:
        """Get current ML engine status"""
        return {
            'models_trained': self.models_trained,
            'model_performance': self.model_performance,
            'cache_size': len(self.prediction_cache),
            'total_predictions': sum(1 for perf in self.model_performance.values() 
                                   if perf['last_update'] is not None),
            'last_training': max([perf.get('last_update', datetime.min) 
                                for perf in self.model_performance.values()]),
            'available_models': ['LSTM', 'CNN', 'Transformer']
        }
