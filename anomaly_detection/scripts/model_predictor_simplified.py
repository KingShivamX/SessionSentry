# src/anomaly_detection/scripts/model_predictor_simplified.py
import pandas as pd
import numpy as np
import joblib
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def create_features(data):
    """
    Create features to improve model performance with robust handling of NaN values.
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Fill NaN values in numeric columns
    for col in ['login_hour', 'session_duration', 'failed_attempts', 'day_of_week']:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # Time-based features
    df['is_business_hours'] = ((df['login_hour'] >= 8) & (df['login_hour'] <= 18)).astype(int)
    df['is_night'] = ((df['login_hour'] >= 23) | (df['login_hour'] <= 5)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Session features
    df['short_session'] = (df['session_duration'] <= 5).astype(int)
    df['long_session'] = (df['session_duration'] >= 120).astype(int)
    
    # Login attempt features
    df['high_failed_attempts'] = (df['failed_attempts'] >= 3).astype(int)
    
    # Feature combinations
    df['night_weekend'] = df['is_night'] * df['is_weekend']
    df['night_weekday'] = (df['is_night'] == 1) & (df['is_weekend'] == 0)
    df['night_weekday'] = df['night_weekday'].astype(int)
    df['failed_night'] = df['failed_attempts'] * df['is_night']
    
    # IP address risk (simplified)
    if 'ip_address' in df.columns:
        df['ip_last_octet'] = df['ip_address'].str.split('.').str[-1].fillna('0').astype(int)
        df['ip_risk'] = ((df['ip_last_octet'] > 200) | (df['ip_last_octet'] < 10)).astype(int)
    else:
        df['ip_risk'] = 0
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    return df

def load_model(model_path):
    """
    Load the trained model and preprocessing components.
    """
    try:
        model_data = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_anomalies(model_data, data):
    """
    Use the model to predict anomalies in the input data.
    """
    # Apply feature engineering
    df = create_features(data)
    
    # Get model components
    ensemble = model_data['ensemble']
    scaler = model_data['scaler']
    features = model_data['features']
    
    # Scale features
    scaled_data = pd.DataFrame(
        scaler.transform(df[features]),
        columns=features
    )
    
    # Get predictions from IsolationForest
    if_model = ensemble['isolation_forest']['model']
    if_features = ensemble['isolation_forest']['features']
    if_preds = if_model.predict(scaled_data[if_features])
    if_scores = if_model.decision_function(scaled_data[if_features])
    
    # Get predictions from SVM
    svm_model = ensemble['svm']['model']
    svm_features = ensemble['svm']['features']
    svm_preds = svm_model.predict(scaled_data[svm_features])
    svm_scores = svm_model.decision_function(scaled_data[svm_features])
    
    # Store results
    df['if_anomaly'] = np.where(if_preds == -1, 1, 0)
    df['svm_anomaly'] = np.where(svm_preds == -1, 1, 0)
    df['if_score'] = if_scores
    df['svm_score'] = svm_scores
    
    # Ensemble prediction
    df['predicted_anomaly'] = np.where(
        (df['if_anomaly'] == 1) | (df['svm_anomaly'] == 1),
        1, 0
    )
    
    # Combined score
    df['anomaly_score'] = (0.7 * df['if_score']) + (0.3 * df['svm_score'])
    
    # Apply custom rules for known anomaly patterns
    custom_conditions = (
        (df['is_night'] == 1) & (df['failed_attempts'] >= 2) |  # Night login with failures
        (df['failed_attempts'] >= 5) |  # Many failed attempts
        (df['session_duration'] >= 180) & (df['is_night'] == 1) |  # Long night sessions
        (df['session_duration'] <= 3) & (df['failed_attempts'] >= 1)  # Short sessions with failures
    )
    
    # Final anomaly flag
    df['final_anomaly'] = np.where(
        df['predicted_anomaly'] | custom_conditions,
        1, 0
    )
    
    return df

def identify_anomaly_reasons(row):
    """
    Identify reasons why a session is flagged as anomalous.
    """
    reasons = []
    
    # Time-based anomalies
    if row['is_night'] == 1:
        reasons.append("unusual login hour (night time: 11 PM to 5 AM)")
    
    # Session anomalies
    if row['short_session'] == 1:
        reasons.append("unusually short session duration (≤ 5 minutes)")
    if row['session_duration'] >= 180:
        reasons.append("unusually long session duration (≥ 180 minutes)")
    
    # Login attempt anomalies
    if row['failed_attempts'] >= 5:
        reasons.append("very high number of failed attempts (≥ 5)")
    elif row['failed_attempts'] >= 3:
        reasons.append("multiple failed attempts (≥ 3)")
    
    # Combined anomalies
    if row['night_weekday'] == 1:
        reasons.append("night login during weekday")
    if row['failed_night'] > 0:
        reasons.append("failed login attempts during night hours")
    
    # Model predictions
    if 'if_anomaly' in row and row['if_anomaly'] == 1:
        reasons.append("flagged by Isolation Forest algorithm")
    if 'svm_anomaly' in row and row['svm_anomaly'] == 1:
        reasons.append("flagged by One-Class SVM algorithm")
    
    # If no specific reasons were found
    if not reasons and row['final_anomaly'] == 1:
        reasons.append("combination of unusual patterns detected by the model")
    
    return reasons

def sample_usage_demo():
    """
    Demonstrate how to use the model for prediction.
    """
    # Load the model
    model_path = '../data/models/simplified_anomaly_model.joblib'
    model_data = load_model(model_path)
    
    if model_data is None:
        print("Failed to load the model. Attempting to load fallback model...")
        model_path = '../data/models/improved_isolation_forest_model.joblib'
        model_data = load_model(model_path)
        
        if model_data is None:
            print("Failed to load any model. Please check the model paths.")
            return
    
    # Create sample data
    print("Creating sample data for demonstration...")
    sample_data = pd.DataFrame({
        'user_id': ['user_1', 'user_2', 'user_3', 'user_4', 'user_5', 'user_6'],
        'login_hour': [9, 14, 2, 10, 23, 8],
        'session_duration': [60, 30, 5, 180, 45, 2],
        'ip_address': ['192.168.1.100', '192.168.1.101', '192.168.1.220', 
                      '192.168.1.103', '10.0.0.1', '192.168.1.105'],
        'failed_attempts': [0, 1, 3, 0, 2, 0],
        'day_of_week': [1, 2, 6, 4, 3, 5],
        'event_type': ['login', 'login', 'login', 'login', 'login', 'login'],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 6
    })
    
    # Make predictions
    print("Making predictions on sample data...")
    results = predict_anomalies(model_data, sample_data)
    
    # Display results
    print("\nPrediction Results:")
    display_cols = ['user_id', 'login_hour', 'session_duration', 'failed_attempts', 
                   'predicted_anomaly', 'final_anomaly']
    print(results[display_cols])
    
    # Summarize anomalies
    anomalies = results[results['final_anomaly'] == 1]
    normal = results[results['final_anomaly'] == 0]
    
    print(f"\nDetected {len(anomalies)} anomalies out of {len(results)} sessions.")
    
    if len(anomalies) > 0:
        print("\nAnomalous Sessions:")
        for _, row in anomalies.iterrows():
            print(f"User: {row['user_id']}")
            print(f"  Login Hour: {row['login_hour']}")
            print(f"  Session Duration: {row['session_duration']} minutes")
            print(f"  Failed Attempts: {row['failed_attempts']}")
            
            reasons = identify_anomaly_reasons(row)
            if reasons:
                print(f"  Anomaly detected due to: {', '.join(reasons)}")
            print("-" * 50)

def load_from_file(input_file):
    """
    Load data from a CSV file and make predictions.
    """
    try:
        # Load data
        data = pd.read_csv(input_file)
        print(f"Loaded data from {input_file}, shape: {data.shape}")
        
        # Load model
        model_path = '../data/models/simplified_anomaly_model.joblib'
        model_data = load_model(model_path)
        
        if model_data is None:
            print("Failed to load simplified model. Attempting to load fallback model...")
            model_path = '../data/models/improved_isolation_forest_model.joblib'
            model_data = load_model(model_path)
            
            if model_data is None:
                print("Failed to load any model. Please check the model paths.")
                return
        
        # Make predictions
        results = predict_anomalies(model_data, data)
        
        # Save results
        output_file = os.path.join(os.path.dirname(input_file), 'simplified_prediction_results.csv')
        results.to_csv(output_file, index=False)
        print(f"Prediction results saved to {output_file}")
        
        # Print summary
        anomalies = results[results['final_anomaly'] == 1]
        print(f"Detected {len(anomalies)} anomalies out of {len(results)} sessions.")
        
        # Display sample of anomalies
        if len(anomalies) > 0:
            sample_size = min(3, len(anomalies))
            print(f"\nSample of {sample_size} anomalous sessions:")
            for i, (_, row) in enumerate(anomalies.head(sample_size).iterrows()):
                print(f"User: {row['user_id']}")
                
                reasons = identify_anomaly_reasons(row)
                if reasons:
                    print(f"  Anomaly detected due to: {', '.join(reasons)}")
                print("-" * 50)
                
                if i >= sample_size - 1:
                    break
            
            if len(anomalies) > sample_size:
                print(f"... and {len(anomalies) - sample_size} more anomalies.")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict anomalies in session data')
    parser.add_argument('--file', type=str, help='Path to the input CSV file')
    parser.add_argument('--demo', action='store_true', help='Run the demonstration with sample data')
    
    args = parser.parse_args()
    
    if args.file:
        load_from_file(args.file)
    elif args.demo:
        sample_usage_demo()
    else:
        sample_usage_demo()  # Default to demo if no arguments provided 