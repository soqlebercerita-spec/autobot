"""
Machine Learning & AI Engine for Trading Bot
Neural Networks, Pattern Recognition, Market Analysis, Sentiment Analysis
"""

import numpy as np
import json
import pickle
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import requests
from config import config

class ModelType(Enum):
    LSTM = "lstm"
    CNN = "cnn"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"

class MarketRegime(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    CRISIS = "crisis"

@dataclass
class MLPrediction:
    symbol: str
    model_type: ModelType
    prediction: str  # BUY/SELL/HOLD
    confidence: float
    price_target: float
    time_horizon: int  # minutes
    features_used: List[str]
    timestamp: datetime

@dataclass
class MarketCondition:
    regime: MarketRegime
    volatility: float
    trend_strength: float
    liquidity_score: float
    sentiment_score: float
    confidence: float
    timestamp: datetime

class MLEngine:
    def __init__(self, market_data_api, indicators):
        self.market_api = market_data_api
        self.indicators = indicators
        
        # Model configurations
        self.models_config = {
            ModelType.LSTM: {
                'enabled': True,
                'lookback_period': 60,
                'prediction_horizon': 15,  # 15 minutes
                'retrain_frequency': 1440,  # 24 hours
                'features': ['price', 'volume', 'volatility', 'momentum']
            },
            ModelType.CNN: {
                'enabled': True,
                'lookback_period': 100,
                'prediction_horizon': 5,  # 5 minutes
                'features': ['price_patterns', 'volume_patterns', 'candlestick_patterns']
            },
            ModelType.TRANSFORMER: {
                'enabled': True,
                'lookback_period': 200,
                'prediction_horizon': 30,  # 30 minutes
                'attention_heads': 8,
                'features': ['multi_timeframe', 'cross_asset', 'macro_indicators']
            }
        }
        
        # Mock models (in production, load real trained models)
        self.models = {
            model_type: self._initialize_mock_model(model_type) 
            for model_type in ModelType if model_type != ModelType.ENSEMBLE
        }
        
        # Pattern recognition templates
        self.patterns = self._initialize_patterns()
        
        # Market regime detection
        self.current_regime = MarketRegime.RANGING
        self.regime_confidence = 0.5
        
        # Sentiment data sources
        self.sentiment_sources = {
            'news': True,
            'social_media': True,
            'options_flow': False,  # Requires premium API
            'fear_greed_index': True
        }
        
        # Performance tracking
        self.model_performance = {
            model_type: {
                'predictions': 0,
                'correct_predictions': 0,
                'accuracy': 0.0,
                'last_retrain': datetime.now(),
                'avg_confidence': 0.0
            } for model_type in ModelType
        }
        
        # Adaptive parameters
        self.adaptive_params = {
            'volatility_threshold': 0.002,
            'trend_sensitivity': 0.5,
            'pattern_confidence_min': 0.6,
            'ensemble_weight_lstm': 0.4,
            'ensemble_weight_cnn': 0.3,
            'ensemble_weight_transformer': 0.3
        }
        
        print("🤖 ML Engine initialized")
        print(f"   • LSTM: {'✅' if self.models_config[ModelType.LSTM]['enabled'] else '❌'}")
        print(f"   • CNN: {'✅' if self.models_config[ModelType.CNN]['enabled'] else '❌'}")
        print(f"   • Transformer: {'✅' if self.models_config[ModelType.TRANSFORMER]['enabled'] else '❌'}")
        print(f"   • Pattern Recognition: ✅")
        print(f"   • Sentiment Analysis: ✅")
        print(f"   • Market Regime Detection: ✅")
    
    def _initialize_mock_model(self, model_type: ModelType) -> Dict:
        """Initialize mock model (replace with real model loading)"""
        return {
            'type': model_type,
            'trained': True,
            'last_update': datetime.now(),
            'accuracy': np.random.uniform(0.6, 0.8),  # Mock accuracy
            'parameters': np.random.randn(100),  # Mock parameters
            'feature_importance': np.random.uniform(0, 1, 10)
        }
    
    def _initialize_patterns(self) -> Dict:
        """Initialize chart pattern recognition templates"""
        return {
            'head_shoulders': {
                'template': [1.0, 1.2, 1.0, 1.5, 1.0, 1.2, 1.0],
                'tolerance': 0.1,
                'signal': 'SELL',
                'reliability': 0.8
            },
            'double_top': {
                'template': [1.0, 1.3, 1.0, 1.3, 1.0],
                'tolerance': 0.05,
                'signal': 'SELL',
                'reliability': 0.7
            },
            'double_bottom': {
                'template': [1.0, 0.7, 1.0, 0.7, 1.0],
                'tolerance': 0.05,
                'signal': 'BUY',
                'reliability': 0.7
            },
            'cup_handle': {
                'template': [1.0, 0.8, 0.7, 0.8, 1.0, 0.95, 1.0],
                'tolerance': 0.1,
                'signal': 'BUY',
                'reliability': 0.75
            },
            'ascending_triangle': {
                'template': [1.0, 1.05, 0.9, 1.05, 0.95, 1.05],
                'tolerance': 0.03,
                'signal': 'BUY',
                'reliability': 0.65
            },
            'descending_triangle': {
                'template': [1.0, 0.95, 1.1, 0.95, 1.05, 0.95],
                'tolerance': 0.03,
                'signal': 'SELL',
                'reliability': 0.65
            }
        }
    
    def generate_ml_predictions(self, symbol: str) -> List[MLPrediction]:
        """Generate predictions from all ML models"""
        predictions = []
        
        try:
            # Get historical data
            prices = self.market_api.get_recent_prices(symbol, count=200)
            if len(prices) < 100:
                return predictions
            
            # LSTM Prediction
            if self.models_config[ModelType.LSTM]['enabled']:
                lstm_pred = self._lstm_prediction(symbol, prices)
                if lstm_pred:
                    predictions.append(lstm_pred)
            
            # CNN Prediction
            if self.models_config[ModelType.CNN]['enabled']:
                cnn_pred = self._cnn_prediction(symbol, prices)
                if cnn_pred:
                    predictions.append(cnn_pred)
            
            # Transformer Prediction
            if self.models_config[ModelType.TRANSFORMER]['enabled']:
                transformer_pred = self._transformer_prediction(symbol, prices)
                if transformer_pred:
                    predictions.append(transformer_pred)
            
            # Ensemble Prediction
            if len(predictions) > 1:
                ensemble_pred = self._ensemble_prediction(predictions)
                if ensemble_pred:
                    predictions.append(ensemble_pred)
        
        except Exception as e:
            print(f"❌ ML prediction error: {e}")
        
        return predictions
    
    def _lstm_prediction(self, symbol: str, prices: List[float]) -> Optional[MLPrediction]:
        """LSTM-based time series prediction"""
        try:
            lookback = self.models_config[ModelType.LSTM]['lookback_period']
            if len(prices) < lookback:
                return None
            
            # Prepare features
            price_changes = np.diff(prices[-lookback:])
            volatility = np.std(price_changes)
            momentum = np.mean(price_changes[-10:])
            
            # Simulate LSTM prediction (replace with real model)
            trend_score = momentum / volatility if volatility > 0 else 0
            
            # Pattern in price changes
            recent_pattern = price_changes[-20:]
            pattern_momentum = np.polyfit(range(len(recent_pattern)), recent_pattern, 1)[0]
            
            confidence = min(abs(trend_score) * 2, 0.95)
            
            if trend_score > 0.1 and pattern_momentum > 0:
                prediction = "BUY"
                price_target = prices[-1] * (1 + volatility * 2)
            elif trend_score < -0.1 and pattern_momentum < 0:
                prediction = "SELL"
                price_target = prices[-1] * (1 - volatility * 2)
            else:
                prediction = "HOLD"
                price_target = prices[-1]
            
            return MLPrediction(
                symbol=symbol,
                model_type=ModelType.LSTM,
                prediction=prediction,
                confidence=confidence,
                price_target=price_target,
                time_horizon=self.models_config[ModelType.LSTM]['prediction_horizon'],
                features_used=['price_momentum', 'volatility', 'trend_score'],
                timestamp=datetime.now()
            )
        
        except Exception as e:
            print(f"❌ LSTM prediction error: {e}")
            return None
    
    def _cnn_prediction(self, symbol: str, prices: List[float]) -> Optional[MLPrediction]:
        """CNN-based pattern recognition prediction"""
        try:
            lookback = self.models_config[ModelType.CNN]['lookback_period']
            if len(prices) < lookback:
                return None
            
            # Create price image/pattern
            price_matrix = np.array(prices[-lookback:]).reshape(-1, 10)  # 10x10 pattern
            
            # Detect patterns using template matching
            detected_patterns = self.detect_chart_patterns(prices[-lookback:])
            
            if not detected_patterns:
                return None
            
            # Get strongest pattern
            best_pattern = max(detected_patterns, key=lambda x: x['confidence'])
            
            confidence = best_pattern['confidence']
            prediction = best_pattern['signal']
            
            # Calculate price target based on pattern
            if prediction == "BUY":
                price_target = prices[-1] * (1 + 0.01)  # 1% target
            elif prediction == "SELL":
                price_target = prices[-1] * (1 - 0.01)  # 1% target
            else:
                price_target = prices[-1]
            
            return MLPrediction(
                symbol=symbol,
                model_type=ModelType.CNN,
                prediction=prediction,
                confidence=confidence,
                price_target=price_target,
                time_horizon=self.models_config[ModelType.CNN]['prediction_horizon'],
                features_used=['chart_patterns', best_pattern['pattern']],
                timestamp=datetime.now()
            )
        
        except Exception as e:
            print(f"❌ CNN prediction error: {e}")
            return None
    
    def _transformer_prediction(self, symbol: str, prices: List[float]) -> Optional[MLPrediction]:
        """Transformer-based multi-timeframe prediction"""
        try:
            lookback = self.models_config[ModelType.TRANSFORMER]['lookback_period']
            if len(prices) < lookback:
                return None
            
            # Multi-timeframe analysis
            short_term = prices[-20:]
            medium_term = prices[-60:]
            long_term = prices[-lookback:]
            
            # Attention mechanism simulation
            short_trend = np.polyfit(range(len(short_term)), short_term, 1)[0]
            medium_trend = np.polyfit(range(len(medium_term)), medium_term, 1)[0]
            long_trend = np.polyfit(range(len(long_term)), long_term, 1)[0]
            
            # Weighted attention
            attention_weights = [0.5, 0.3, 0.2]  # Short, medium, long
            combined_trend = (short_trend * attention_weights[0] + 
                            medium_trend * attention_weights[1] + 
                            long_trend * attention_weights[2])
            
            # Cross-asset correlation (simplified)
            market_regime = self.detect_market_regime(symbol)
            regime_multiplier = 1.0
            
            if market_regime.regime == MarketRegime.TRENDING:
                regime_multiplier = 1.2
            elif market_regime.regime == MarketRegime.VOLATILE:
                regime_multiplier = 0.8
            
            final_signal = combined_trend * regime_multiplier
            confidence = min(abs(final_signal) * 10, 0.9)
            
            if final_signal > 0.0001:
                prediction = "BUY"
                price_target = prices[-1] * (1 + abs(final_signal) * 100)
            elif final_signal < -0.0001:
                prediction = "SELL"
                price_target = prices[-1] * (1 - abs(final_signal) * 100)
            else:
                prediction = "HOLD"
                price_target = prices[-1]
            
            return MLPrediction(
                symbol=symbol,
                model_type=ModelType.TRANSFORMER,
                prediction=prediction,
                confidence=confidence,
                price_target=price_target,
                time_horizon=self.models_config[ModelType.TRANSFORMER]['prediction_horizon'],
                features_used=['multi_timeframe', 'market_regime', 'attention_weights'],
                timestamp=datetime.now()
            )
        
        except Exception as e:
            print(f"❌ Transformer prediction error: {e}")
            return None
    
    def _ensemble_prediction(self, predictions: List[MLPrediction]) -> Optional[MLPrediction]:
        """Combine multiple model predictions using ensemble method"""
        try:
            if len(predictions) < 2:
                return None
            
            # Weight predictions by confidence and model performance
            weights = {
                ModelType.LSTM: self.adaptive_params['ensemble_weight_lstm'],
                ModelType.CNN: self.adaptive_params['ensemble_weight_cnn'],
                ModelType.TRANSFORMER: self.adaptive_params['ensemble_weight_transformer']
            }
            
            weighted_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            total_weight = 0
            price_targets = []
            
            for pred in predictions:
                model_weight = weights.get(pred.model_type, 0.3)
                confidence_weight = pred.confidence
                final_weight = model_weight * confidence_weight
                
                weighted_scores[pred.prediction] += final_weight
                total_weight += final_weight
                price_targets.append(pred.price_target)
            
            # Normalize scores
            if total_weight > 0:
                for key in weighted_scores:
                    weighted_scores[key] /= total_weight
            
            # Get final prediction
            final_prediction = max(weighted_scores, key=weighted_scores.get)
            final_confidence = weighted_scores[final_prediction]
            final_price_target = np.mean(price_targets)
            
            return MLPrediction(
                symbol=predictions[0].symbol,
                model_type=ModelType.ENSEMBLE,
                prediction=final_prediction,
                confidence=final_confidence,
                price_target=final_price_target,
                time_horizon=15,  # Average horizon
                features_used=['ensemble_of_models'],
                timestamp=datetime.now()
            )
        
        except Exception as e:
            print(f"❌ Ensemble prediction error: {e}")
            return None
    
    def detect_chart_patterns(self, prices: List[float]) -> List[Dict]:
        """Detect chart patterns using template matching"""
        detected_patterns = []
        
        try:
            # Normalize prices for pattern matching
            normalized_prices = np.array(prices) / prices[0]
            
            for pattern_name, pattern_info in self.patterns.items():
                template = np.array(pattern_info['template'])
                tolerance = pattern_info['tolerance']
                
                # Sliding window pattern matching
                for i in range(len(normalized_prices) - len(template) + 1):
                    window = normalized_prices[i:i+len(template)]
                    
                    # Calculate pattern similarity
                    similarity = 1 - np.mean(np.abs(window - template))
                    
                    if similarity > (1 - tolerance):
                        confidence = similarity * pattern_info['reliability']
                        
                        detected_patterns.append({
                            'pattern': pattern_name,
                            'signal': pattern_info['signal'],
                            'confidence': confidence,
                            'start_index': i,
                            'end_index': i + len(template),
                            'similarity': similarity
                        })
        
        except Exception as e:
            print(f"❌ Pattern detection error: {e}")
        
        return detected_patterns
    
    def detect_market_regime(self, symbol: str) -> MarketCondition:
        """Detect current market regime using AI"""
        try:
            prices = self.market_api.get_recent_prices(symbol, count=100)
            if len(prices) < 50:
                return MarketCondition(
                    regime=MarketRegime.RANGING,
                    volatility=0.01,
                    trend_strength=0.0,
                    liquidity_score=0.5,
                    sentiment_score=0.0,
                    confidence=0.3,
                    timestamp=datetime.now()
                )
            
            # Calculate regime indicators
            volatility = np.std(np.diff(prices)) / np.mean(prices)
            
            # Trend strength
            trend_coef = np.polyfit(range(len(prices)), prices, 1)[0]
            trend_strength = abs(trend_coef) / np.mean(prices)
            
            # Range detection
            price_range = (max(prices) - min(prices)) / np.mean(prices)
            
            # Regime classification
            if volatility > 0.02:
                if trend_strength > 0.01:
                    regime = MarketRegime.VOLATILE
                else:
                    regime = MarketRegime.CRISIS
            elif trend_strength > 0.005:
                regime = MarketRegime.TRENDING
            elif price_range < 0.05:
                regime = MarketRegime.CALM
            else:
                regime = MarketRegime.RANGING
            
            # Liquidity score (simplified)
            liquidity_score = min(1.0, len(prices) / 100.0)
            
            # Get sentiment
            sentiment_score = self.get_market_sentiment(symbol)
            
            # Confidence based on data quality
            confidence = min(0.9, len(prices) / 100.0)
            
            return MarketCondition(
                regime=regime,
                volatility=volatility,
                trend_strength=trend_strength,
                liquidity_score=liquidity_score,
                sentiment_score=sentiment_score,
                confidence=confidence,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            print(f"❌ Market regime detection error: {e}")
            return MarketCondition(
                regime=MarketRegime.RANGING,
                volatility=0.01,
                trend_strength=0.0,
                liquidity_score=0.5,
                sentiment_score=0.0,
                confidence=0.3,
                timestamp=datetime.now()
            )
    
    def get_market_sentiment(self, symbol: str) -> float:
        """Analyze market sentiment from multiple sources"""
        try:
            sentiment_scores = []
            
            # News sentiment (simulated)
            if self.sentiment_sources['news']:
                news_sentiment = self._analyze_news_sentiment(symbol)
                sentiment_scores.append(news_sentiment)
            
            # Social media sentiment (simulated)
            if self.sentiment_sources['social_media']:
                social_sentiment = self._analyze_social_sentiment(symbol)
                sentiment_scores.append(social_sentiment)
            
            # Fear & Greed Index
            if self.sentiment_sources['fear_greed_index']:
                fear_greed = self._get_fear_greed_index()
                sentiment_scores.append(fear_greed)
            
            # Average sentiment
            if sentiment_scores:
                return np.mean(sentiment_scores)
            else:
                return 0.0
        
        except Exception as e:
            print(f"❌ Sentiment analysis error: {e}")
            return 0.0
    
    def _analyze_news_sentiment(self, symbol: str) -> float:
        """Analyze news sentiment (mock implementation)"""
        # In production, this would connect to news APIs and use NLP
        # Mock sentiment based on time and randomness
        base_sentiment = np.sin(time.time() / 3600) * 0.3  # Hourly cycle
        noise = np.random.normal(0, 0.1)
        return np.clip(base_sentiment + noise, -1, 1)
    
    def _analyze_social_sentiment(self, symbol: str) -> float:
        """Analyze social media sentiment (mock implementation)"""
        # Mock social sentiment
        return np.random.uniform(-0.5, 0.5)
    
    def _get_fear_greed_index(self) -> float:
        """Get Fear & Greed Index (mock implementation)"""
        # Mock fear & greed index
        return np.random.uniform(-1, 1)
    
    def update_adaptive_parameters(self, market_condition: MarketCondition):
        """Update ML parameters based on market conditions"""
        try:
            # Adjust volatility threshold
            if market_condition.regime == MarketRegime.VOLATILE:
                self.adaptive_params['volatility_threshold'] *= 1.2
            elif market_condition.regime == MarketRegime.CALM:
                self.adaptive_params['volatility_threshold'] *= 0.8
            
            # Adjust trend sensitivity
            if market_condition.regime == MarketRegime.TRENDING:
                self.adaptive_params['trend_sensitivity'] *= 1.1
            else:
                self.adaptive_params['trend_sensitivity'] *= 0.95
            
            # Adjust ensemble weights based on recent performance
            for model_type in [ModelType.LSTM, ModelType.CNN, ModelType.TRANSFORMER]:
                accuracy = self.model_performance[model_type]['accuracy']
                weight_key = f'ensemble_weight_{model_type.value}'
                
                if accuracy > 0.7:
                    self.adaptive_params[weight_key] *= 1.05
                elif accuracy < 0.5:
                    self.adaptive_params[weight_key] *= 0.95
            
            # Normalize ensemble weights
            total_weight = (self.adaptive_params['ensemble_weight_lstm'] + 
                          self.adaptive_params['ensemble_weight_cnn'] + 
                          self.adaptive_params['ensemble_weight_transformer'])
            
            if total_weight > 0:
                self.adaptive_params['ensemble_weight_lstm'] /= total_weight
                self.adaptive_params['ensemble_weight_cnn'] /= total_weight
                self.adaptive_params['ensemble_weight_transformer'] /= total_weight
            
        except Exception as e:
            print(f"❌ Adaptive parameter update error: {e}")
    
    def get_ml_summary(self) -> Dict:
        """Get comprehensive ML engine summary"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'models': {},
                'market_regime': {
                    'current': self.current_regime.value,
                    'confidence': self.regime_confidence
                },
                'adaptive_parameters': self.adaptive_params.copy(),
                'performance': {}
            }
            
            # Model performance
            for model_type, perf in self.model_performance.items():
                summary['models'][model_type.value] = {
                    'enabled': self.models_config.get(model_type, {}).get('enabled', False),
                    'predictions': perf['predictions'],
                    'accuracy': perf['accuracy'],
                    'avg_confidence': perf['avg_confidence'],
                    'last_retrain': perf['last_retrain'].isoformat()
                }
            
            return summary
        
        except Exception as e:
            print(f"❌ ML summary error: {e}")
            return {}

# Global instance
ml_engine = None  # Will be initialized with dependencies