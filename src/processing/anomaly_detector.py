"""
AnomalyDetector module for SessionSentry.
Implements machine learning-based anomaly detection for login/logout events.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Configure logger
logger = logging.getLogger("SessionSentry.AnomalyDetector")

class AnomalyDetector:
    """
    Detects anomalies in Windows login/logout events using machine learning.
    Uses Isolation Forest algorithm for unsupervised anomaly detection.
    """
    
    def __init__(self, model_path="models/anomaly_model.pkl"):
        """
        Initialize the anomaly detector.
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.features = [
            'hour_of_day', 
            'day_of_week',
            'is_weekend',
            'login_count_1h',
            'login_count_24h',
            'login_type_encoded',
            'source_address_encoded'
        ]
        
        # Load model if it exists, otherwise it will be trained later
        self._load_model()
        
        # Historical data for building user profiles
        self.user_history = {}  # username -> list of events
        
        logger.info("AnomalyDetector initialized")
    
    def _load_model(self) -> bool:
        """
        Load a previously trained model if available.
        
        Returns:
            True if model was loaded, False otherwise
        """
        if os.path.exists(self.model_path):
            try:
                logger.info(f"Loading existing model from {self.model_path}")
                loaded_data = joblib.load(self.model_path)
                self.model = loaded_data.get('model')
                self.scaler = loaded_data.get('scaler')
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}", exc_info=True)
                return False
        else:
            logger.info("No existing model found. Model will be trained on first run.")
            return False
    
    def _save_model(self):
        """Save the trained model to disk."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Save model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}", exc_info=True)
    
    def train_model(self, historical_events: List[Dict[str, Any]]):
        """
        Train the anomaly detection model using historical login/logout events.
        
        Args:
            historical_events: List of event dictionaries from the past
        """
        if not historical_events:
            logger.warning("No historical events provided for training")
            return
        
        logger.info(f"Training anomaly detection model with {len(historical_events)} events")
        
        try:
            # Convert events to DataFrame
            df = pd.DataFrame(historical_events)
            
            # Extract features
            X = self._extract_features(df)
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Isolation Forest model
            self.model = IsolationForest(
                n_estimators=100, 
                contamination=0.05,  # Expect 5% of logins to be anomalous
                random_state=42,
                max_samples='auto'
            )
            self.model.fit(X_scaled)
            
            # Save the trained model
            self._save_model()
            
            logger.info("Anomaly detection model trained successfully")
        except Exception as e:
            logger.error(f"Error training model: {e}", exc_info=True)
    
    def _extract_features(self, events_df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from events for anomaly detection.
        
        Args:
            events_df: DataFrame containing events
            
        Returns:
            NumPy array of features
        """
        # Create new dataframe for features
        features_df = pd.DataFrame()
        
        # Time-based features
        features_df['hour_of_day'] = events_df['timestamp'].dt.hour
        features_df['day_of_week'] = events_df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        
        # Frequency features (would normally compute from historical data)
        # For now we'll use dummy values
        features_df['login_count_1h'] = 1
        features_df['login_count_24h'] = 5
        
        # Categorical features - simple encoding for now
        # In a real system, would use more sophisticated encoding
        features_df['login_type_encoded'] = events_df['login_type'].astype('category').cat.codes.fillna(-1)
        features_df['source_address_encoded'] = events_df['source_address'].astype('category').cat.codes.fillna(-1)
        
        # Select and return the feature columns
        return features_df[self.features].values
    
    def detect_anomalies(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect anomalies in a batch of login/logout events.
        
        Args:
            events: List of event dictionaries
            
        Returns:
            List of anomalous events with anomaly scores
        """
        if not events:
            logger.info("No events to analyze")
            return []
        
        # Update user history with new events
        self._update_user_history(events)
        
        # If model not trained yet, train it with available history
        if self.model is None:
            all_history = []
            for user_events in self.user_history.values():
                all_history.extend(user_events)
            
            if len(all_history) > 10:  # Need minimum events to train
                self.train_model(all_history)
            else:
                logger.info("Not enough historical data to train model yet")
                return []
        
        try:
            # Convert events to DataFrame
            df = pd.DataFrame(events)
            
            # Extract features
            X = self._extract_features(df)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Predict anomalies
            # -1 for anomalies, 1 for normal samples
            predictions = self.model.predict(X_scaled)
            
            # Get anomaly scores (negative score = more anomalous)
            scores = self.model.decision_function(X_scaled)
            
            # Identify anomalous events
            anomalies = []
            for i, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == -1:  # Anomaly detected
                    event = events[i].copy()
                    event['anomaly_score'] = float(score)
                    event['anomaly_reason'] = self._get_anomaly_reason(events[i], X[i])
                    anomalies.append(event)
            
            logger.info(f"Detected {len(anomalies)} anomalies out of {len(events)} events")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}", exc_info=True)
            return []
    
    def _update_user_history(self, events: List[Dict[str, Any]]):
        """
        Update the history of events for each user.
        
        Args:
            events: New events to add to history
        """
        # Group events by username
        for event in events:
            username = event.get('username')
            if not username:
                continue
                
            if username not in self.user_history:
                self.user_history[username] = []
                
            # Add event to user history
            self.user_history[username].append(event)
            
            # Limit history size to avoid memory issues
            if len(self.user_history[username]) > 1000:
                # Keep only the most recent events
                self.user_history[username] = self.user_history[username][-1000:]
    
    def _get_anomaly_reason(self, event: Dict[str, Any], features: np.ndarray) -> str:
        """
        Determine the reason why an event was flagged as anomalous.
        
        Args:
            event: The event that was flagged
            features: Feature vector for the event
            
        Returns:
            String describing the reason for the anomaly
        """
        reasons = []
        
        # Time-based anomalies
        hour = features[0]
        is_weekend = features[2]
        
        if hour < 6 or hour > 20:
            reasons.append("Unusual login time (outside business hours)")
        
        if is_weekend:
            reasons.append("Weekend login")
            
        # Login type anomalies
        login_type = event.get('login_type')
        if login_type and login_type not in [2, 3, 7, 8, 9, 10, 11]:
            reasons.append(f"Unusual login type ({login_type})")
            
        # Source address anomalies
        source = event.get('source_address')
        if source and source != "127.0.0.1" and source != "::1":
            reasons.append(f"Remote login from {source}")
            
        if not reasons:
            reasons.append("Statistical outlier based on user behavior patterns")
            
        return "; ".join(reasons) 