# src/anomaly_detection/scripts/model_predictor.py
import pandas as pd
import numpy as np
import joblib
import os
import argparse
from datetime import datetime, timedelta

def create_additional_features(data):
    """
    Create additional features to improve model performance.
    
    Parameters:
    -----------
    data : DataFrame
        The input data
        
    Returns:
    --------
    DataFrame
        Data with additional features
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # 1. Business hours flag (8 AM to 6 PM)
    df['is_business_hours'] = ((df['login_hour'] >= 8) & (df['login_hour'] <= 18)).astype(int)
    
    # 2. Weekend flag (5 = Sat, 6 = Sun)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 3. Night hours flag (11 PM to 5 AM)
    df['is_night'] = ((df['login_hour'] >= 23) | (df['login_hour'] <= 5)).astype(int)
    
    # 4. Session duration categories
    df['short_session'] = (df['session_duration'] <= 5).astype(int)
    df['long_session'] = (df['session_duration'] >= 120).astype(int)
    
    # 5. Feature interactions
    df['night_weekend'] = df['is_night'] * df['is_weekend']
    df['failed_night'] = df['failed_attempts'] * df['is_night']
    
    # 6. IP address risk (simplified version)
    # In real scenario, we would analyze IP reputation or geolocation
    df['ip_last_octet'] = df['ip_address'].apply(lambda x: int(x.split('.')[-1]))
    df['ip_risk'] = ((df['ip_last_octet'] > 200) | (df['ip_last_octet'] < 10)).astype(int)
    
    return df

def load_model(model_path):
    """
    Load the trained model and preprocessing components from the specified path.
    
    Parameters:
    -----------
    model_path : str
        Path to the model file
        
    Returns:
    --------
    dict
        Dictionary containing the model and preprocessing components
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
    Predict anomalies in the input data using the trained model.
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing the model and preprocessing components
    data : DataFrame
        Input data for prediction
        
    Returns:
    --------
    DataFrame
        Data with anomaly predictions and scores
    """
    # Apply feature engineering
    enhanced_data = create_additional_features(data)
    
    # Extract components from model_data
    model = model_data['model']
    scaler = model_data['scaler']
    features = model_data['features']
    
    # Scale the features
    X = pd.DataFrame(
        scaler.transform(enhanced_data[features]), 
        columns=features
    )
    
    # Predict anomalies
    raw_predictions = model.predict(X)
    enhanced_data['anomaly'] = np.where(raw_predictions == -1, 1, 0)
    
    # Calculate anomaly scores
    enhanced_data['anomaly_score'] = model.decision_function(X)
    
    return enhanced_data

def sample_usage_demo():
    """
    Demonstrate how to use the model for prediction with sample data.
    """
    # Load the model
    model_path = '../data/models/improved_isolation_forest_model.joblib'
    model_data = load_model(model_path)
    
    if model_data is None:
        print("Failed to load the model. Please check the model path.")
        return
    
    # Create sample data
    print("Creating sample data for demonstration...")
    sample_data = pd.DataFrame({
        'user_id': ['user_1', 'user_2', 'user_3', 'user_4'],
        'login_hour': [9, 14, 2, 10],  # 2 AM should be flagged as unusual
        'session_duration': [60, 30, 5, 180],  # 5 min and 180 min might be unusual
        'ip_address': ['192.168.1.100', '192.168.1.101', '192.168.1.220', '192.168.1.103'],
        'failed_attempts': [0, 1, 6, 0],  # 6 failed attempts should be flagged
        'day_of_week': [1, 2, 6, 4],  # 6 is Sunday
        'event_type': ['login', 'login', 'login', 'login'],
        'timestamp': [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ]
    })
    
    # Make predictions
    print("Making predictions on sample data...")
    results = predict_anomalies(model_data, sample_data)
    
    # Display results
    print("\nPrediction Results:")
    print(results[['user_id', 'login_hour', 'failed_attempts', 'anomaly', 'anomaly_score']])
    
    # Identify anomalies
    anomalies = results[results['anomaly'] == 1]
    normal = results[results['anomaly'] == 0]
    
    print(f"\nDetected {len(anomalies)} anomalies out of {len(results)} sessions.")
    
    if len(anomalies) > 0:
        print("\nAnomalous Sessions:")
        for _, row in anomalies.iterrows():
            print(f"User: {row['user_id']}, Login Hour: {row['login_hour']}, "
                  f"Failed Attempts: {row['failed_attempts']}, "
                  f"Anomaly Score: {row['anomaly_score']:.4f}")
            
            # Explain why it might be anomalous
            reasons = []
            if row['login_hour'] in [0, 1, 2, 3, 4, 5, 23]:
                reasons.append("unusual login hour (night time)")
            if row['session_duration'] <= 5 or row['session_duration'] >= 120:
                reasons.append("unusual session duration")
            if row['failed_attempts'] >= 5:
                reasons.append("high number of failed attempts")
            if row['is_night'] == 1 and row['day_of_week'] < 5:
                reasons.append("night login on weekday")
            
            if reasons:
                print(f"Possible reasons: {', '.join(reasons)}")
            print("-" * 50)

def load_from_file(input_file):
    """
    Load data from a CSV file and make predictions.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file
    """
    try:
        # Load data
        data = pd.read_csv(input_file)
        print(f"Loaded data from {input_file}, shape: {data.shape}")
        
        # Load model
        model_path = '../data/models/improved_isolation_forest_model.joblib'
        model_data = load_model(model_path)
        
        if model_data is None:
            print("Failed to load the model. Please check the model path.")
            return
        
        # Make predictions
        results = predict_anomalies(model_data, data)
        
        # Save results
        output_file = os.path.join(os.path.dirname(input_file), 'prediction_results.csv')
        results.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Print summary
        anomalies = results[results['anomaly'] == 1]
        print(f"Detected {len(anomalies)} anomalies out of {len(results)} sessions.")
        
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