# src/anomaly_detection/scripts/model_predictor.py
import pandas as pd
import numpy as np
import joblib
import os
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def create_advanced_features(data):
    """
    Create advanced features to significantly improve model performance.
    
    Parameters:
    -----------
    data : DataFrame
        The input data
        
    Returns:
    --------
    DataFrame
        Data with advanced features
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # 1. Time-based features
    # Business hours flag (8 AM to 6 PM)
    df['is_business_hours'] = ((df['login_hour'] >= 8) & (df['login_hour'] <= 18)).astype(int)
    
    # Indian working hours flag (9 AM to 5 PM)
    df['is_indian_working_hours'] = ((df['login_hour'] >= 9) & (df['login_hour'] <= 17)).astype(int)
    
    # Early morning flag (5 AM to 8 AM)
    df['is_early_morning'] = ((df['login_hour'] >= 5) & (df['login_hour'] < 8)).astype(int)
    
    # Late evening flag (6 PM to 11 PM)
    df['is_late_evening'] = ((df['login_hour'] >= 18) & (df['login_hour'] < 23)).astype(int)
    
    # Night hours flag (11 PM to 5 AM)
    df['is_night'] = ((df['login_hour'] >= 23) | (df['login_hour'] <= 5)).astype(int)
    
    # Weekend flag (5 = Sat, 6 = Sun)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Workday flag (1-5 are workdays)
    df['is_workday'] = (df['day_of_week'] <= 4).astype(int)
    
    # 2. Session duration features
    df['short_session'] = (df['session_duration'] <= 5).astype(int)
    df['medium_session'] = ((df['session_duration'] > 5) & (df['session_duration'] < 60)).astype(int)
    df['long_session'] = ((df['session_duration'] >= 60) & (df['session_duration'] < 180)).astype(int)
    df['very_long_session'] = (df['session_duration'] >= 180).astype(int)
    
    # Create session duration bins (better for pattern detection)
    # Handle NaN values by filling with a default value
    df['session_duration_filled'] = df['session_duration'].fillna(0)
    df['session_duration_bin'] = pd.cut(
        df['session_duration_filled'], 
        bins=[0, 5, 15, 30, 60, 120, 240, float('inf')],
        labels=[0, 1, 2, 3, 4, 5, 6]
    ).astype(int)
    
    # Clean up temporary column
    df.drop('session_duration_filled', axis=1, inplace=True)
    
    # 3. Login attempt features
    df['has_failed_attempts'] = (df['failed_attempts'] > 0).astype(int)
    df['moderate_failed_attempts'] = ((df['failed_attempts'] >= 3) & (df['failed_attempts'] < 10)).astype(int)
    df['high_failed_attempts'] = (df['failed_attempts'] >= 10).astype(int)
    
    # 4. IP address features
    # Extract parts of IP for analysis
    if 'ip_address' in df.columns:
        # First, filter out any rows with NaN IP addresses
        ip_mask = df['ip_address'].notna()
        
        # Initialize IP-related columns with zeros
        df['ip_class'] = 0
        df['ip_last_octet'] = 0
        df['ip_first_octet'] = 0
        df['ip_second_octet'] = 0
        
        # Only process valid IP addresses
        if ip_mask.any():
            df.loc[ip_mask, 'ip_octets'] = df.loc[ip_mask, 'ip_address'].apply(lambda x: x.split('.'))
            df.loc[ip_mask, 'ip_class'] = df.loc[ip_mask, 'ip_octets'].apply(lambda x: int(x[0]))
            df.loc[ip_mask, 'ip_last_octet'] = df.loc[ip_mask, 'ip_octets'].apply(lambda x: int(x[3]))
            df.loc[ip_mask, 'ip_first_octet'] = df.loc[ip_mask, 'ip_octets'].apply(lambda x: int(x[0]))
            df.loc[ip_mask, 'ip_second_octet'] = df.loc[ip_mask, 'ip_octets'].apply(lambda x: int(x[1]))
        
        # Potentially suspicious IP characteristics
        df['ip_risk'] = (
            (df['ip_last_octet'] > 230) | 
            (df['ip_last_octet'] < 10) | 
            (df['ip_first_octet'] == 10) |
            ((df['ip_first_octet'] == 192) & (df['ip_second_octet'] == 168))
        ).astype(int)
        
        # Remove intermediate columns
        if 'ip_octets' in df.columns:
            df.drop('ip_octets', axis=1, inplace=True)
    else:
        # If ip_address column doesn't exist, create dummy ip_risk column
        df['ip_risk'] = 0
    
    # 5. High-value composite features
    # Risky scenarios combinations
    df['night_weekend'] = df['is_night'] * df['is_weekend']
    df['night_weekday'] = df['is_night'] * df['is_workday']
    df['failed_night'] = df['failed_attempts'] * df['is_night']
    df['failed_weekend'] = df['failed_attempts'] * df['is_weekend']
    df['short_session_high_failed'] = df['short_session'] * df['high_failed_attempts']
    df['working_hours_long_session'] = df['is_indian_working_hours'] * df['very_long_session']
    df['working_hours_failed_attempts'] = df['is_indian_working_hours'] * df['moderate_failed_attempts']
    
    # 6. Statistical anomaly indicators
    # Login hour deviation from typical 9-5 workday
    df['hour_deviation'] = np.minimum(
        np.abs(df['login_hour'] - 9),  # Deviation from 9 AM
        np.abs(df['login_hour'] - 17)   # Deviation from 5 PM
    )
    
    # Unusual combinations (weighted anomaly indicators)
    df['unusual_time_duration'] = df['hour_deviation'] * df['session_duration'].fillna(0) / 60
    df['unusual_time_failures'] = df['hour_deviation'] * df['failed_attempts'].fillna(0)
    
    # Handle potential division by zero
    df['failure_rate'] = df['failed_attempts'].fillna(0) / np.maximum(df['session_duration'].fillna(1), 1)
    
    # 7. Combined suspicious activity indicator
    df['suspicious_combo'] = ((df['very_long_session'] == 1) & 
                             (df['high_failed_attempts'] == 1) & 
                             (df['is_indian_working_hours'] == 1)).astype(int)
    
    # 8. Advanced risk score (composite weighted score)
    df['risk_score'] = (
        df['is_night'] * 2.5 +                   # Night time is risky
        df['high_failed_attempts'] * 4 +         # Failed attempts are highly suspicious
        df['short_session'] * 1.5 +              # Very short sessions can be suspicious
        df['very_long_session'] * 1 +            # Very long sessions can be unusual
        df['night_weekday'] * 3 +                # Night login on weekday is unusual
        df['is_weekend'] * 0.5 +                 # Weekend logins might be slightly unusual
        df['ip_risk'] * 2                        # Suspicious IP is risky
    )
    
    # 9. Normalize risk score to 0-10 scale
    max_possible_score = 14.5  # Sum of all weights above
    df['normalized_risk'] = (df['risk_score'] / max_possible_score) * 10
    
    # Fill any remaining NaN values with 0
    df = df.fillna(0)
    
    return df

def load_advanced_model(model_path):
    """
    Load the trained advanced ensemble model and preprocessing components.
    
    Parameters:
    -----------
    model_path : str
        Path to the model file
        
    Returns:
    --------
    dict
        Dictionary containing the ensemble model and preprocessing components
    """
    try:
        model_data = joblib.load(model_path)
        print(f"Advanced ensemble model loaded successfully from {model_path}")
        return model_data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_with_ensemble(ensemble, data, scaler, features):
    """
    Make predictions using the advanced ensemble model.
    
    Parameters:
    -----------
    ensemble : dict
        Dictionary of trained models
    data : DataFrame
        Input data for prediction
    scaler : StandardScaler
        Scaler used to preprocess data
    features : list
        List of features used by the model
        
    Returns:
    --------
    DataFrame
        Data with anomaly predictions and scores
    """
    # Scale features
    X = pd.DataFrame(scaler.transform(data[features]), columns=features)
    
    # Make predictions with each model
    if_model = ensemble['isolation_forest']['model']
    if_features = ensemble['isolation_forest']['features']
    if_preds = if_model.predict(X[if_features])
    if_scores = if_model.decision_function(X[if_features])
    
    # Some features might be subset for SVM
    svm_model = ensemble['one_class_svm']['model']
    svm_features = ensemble['one_class_svm']['features']
    svm_preds = svm_model.predict(X[svm_features])
    svm_scores = svm_model.decision_function(X[svm_features])
    
    lof_model = ensemble['lof']['model']
    lof_features = ensemble['lof']['features']
    lof_preds = lof_model.predict(X[lof_features])
    lof_scores = lof_model.decision_function(X[lof_features])
    
    # Convert predictions to binary (1 for anomaly, 0 for normal)
    data['if_anomaly'] = np.where(if_preds == -1, 1, 0)
    data['svm_anomaly'] = np.where(svm_preds == -1, 1, 0)
    data['lof_anomaly'] = np.where(lof_preds == -1, 1, 0)
    
    # Add scores
    data['if_score'] = if_scores
    data['svm_score'] = svm_scores
    data['lof_score'] = lof_scores
    
    # Create weighted ensemble prediction
    data['ensemble_score'] = (
        0.5 * data['if_score'] + 
        0.25 * data['svm_score'] + 
        0.25 * data['lof_score']
    )
    
    # Final ensemble prediction (majority voting)
    data['ensemble_votes'] = (
        data['if_anomaly'] + 
        data['svm_anomaly'] + 
        data['lof_anomaly']
    )
    data['predicted_anomaly'] = np.where(data['ensemble_votes'] >= 2, 1, 0)  # Majority vote
    
    # Calculate a normalized confidence score
    max_score = data['ensemble_score'].max()
    min_score = data['ensemble_score'].min()
    range_score = max_score - min_score
    
    if range_score > 0:
        data['anomaly_score'] = 1 - ((data['ensemble_score'] - min_score) / range_score)
    else:
        data['anomaly_score'] = 0.5  # Default if no range
        
    # Final anomaly flag 
    data['final_anomaly'] = data['predicted_anomaly']
    
    # Add custom rules for Indian working hours criteria
    data['indian_criteria_anomaly'] = np.where(
        (data['very_long_session'] == 1) & 
        (data['high_failed_attempts'] == 1) & 
        (data['is_indian_working_hours'] == 1),
        1, 0
    )
    
    # Combine model anomaly with custom anomaly
    data['final_anomaly'] = np.where(
        (data['predicted_anomaly'] == 1) | (data['indian_criteria_anomaly'] == 1),
        1, 0
    )
    
    return data

def predict_anomalies(model_data, data):
    """
    Predict anomalies in the input data using the advanced ensemble model.
    
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
    # Apply advanced feature engineering
    enhanced_data = create_advanced_features(data)
    
    # Extract components from model_data
    ensemble = model_data['ensemble']
    scaler = model_data['scaler']
    features = model_data['features']
    
    # Use ensemble to make predictions
    results = predict_with_ensemble(ensemble, enhanced_data, scaler, features)
    
    return results

def identify_anomaly_reasons(row):
    """
    Identify reasons why a session is flagged as anomalous.
    
    Parameters:
    -----------
    row : Series
        A row of data representing a session
        
    Returns:
    --------
    list
        List of reasons why the session is anomalous
    """
    reasons = []
    
    # Time-based anomalies
    if row['is_night'] == 1:
        reasons.append("unusual login hour (night time: 11 PM to 5 AM)")
    elif row['is_early_morning'] == 1:
        reasons.append("early morning login (5 AM to 8 AM)")
    elif row['is_late_evening'] == 1:
        reasons.append("late evening login (6 PM to 11 PM)")
        
    # Session duration anomalies
    if row['short_session'] == 1:
        reasons.append("unusually short session duration (≤ 5 minutes)")
    elif row['very_long_session'] == 1:
        reasons.append("unusually long session duration (≥ 180 minutes)")
        
    # Login attempt anomalies
    if row['moderate_failed_attempts'] == 1:
        reasons.append("moderate number of failed attempts (3-9)")
    elif row['high_failed_attempts'] == 1:
        reasons.append("high number of failed attempts (≥ 10)")
        
    # Time-day combination anomalies
    if row['night_weekday'] == 1:
        reasons.append("night login on a weekday")
    if row['night_weekend'] == 1:
        reasons.append("night login on weekend")
        
    # High risk combinations
    if row['short_session_high_failed'] == 1:
        reasons.append("short session with high failed attempts (highly suspicious)")
    if row['working_hours_long_session'] == 1 and row['high_failed_attempts'] == 1:
        reasons.append("very long session with high failed attempts during working hours")
        
    # IP-based anomalies
    if 'ip_risk' in row and row['ip_risk'] == 1:
        reasons.append("suspicious IP address pattern")
        
    # Model-specific anomalies
    if 'if_anomaly' in row and row['if_anomaly'] == 1:
        reasons.append("unusual pattern detected by Isolation Forest")
    if 'svm_anomaly' in row and row['svm_anomaly'] == 1:
        reasons.append("unusual pattern detected by One-Class SVM")
    if 'lof_anomaly' in row and row['lof_anomaly'] == 1:
        reasons.append("unusual pattern detected by Local Outlier Factor")
        
    # If we couldn't identify specific reasons but it's still anomalous
    if len(reasons) == 0 and row['predicted_anomaly'] == 1:
        reasons.append("unusual combination of features detected by ensemble model")
        
    # Add risk score information if it's high
    if 'normalized_risk' in row and row['normalized_risk'] > 7:
        reasons.append(f"overall high risk score ({row['normalized_risk']:.1f}/10)")
    
    return reasons

def sample_usage_demo():
    """
    Demonstrate how to use the advanced ensemble model for prediction with sample data.
    """
    # Load the model
    model_path = '../data/models/advanced_anomaly_detection_model.joblib'
    model_data = load_advanced_model(model_path)
    
    if model_data is None:
        print("Failed to load the advanced model. Falling back to previous model.")
        # Try loading the older model as fallback
        model_path = '../data/models/improved_isolation_forest_model.joblib'
        model_data = load_advanced_model(model_path)
        
        if model_data is None:
            print("Failed to load any model. Please check model paths.")
            return
    
    # Create sample data
    print("Creating sample data for demonstration...")
    sample_data = pd.DataFrame({
        'user_id': ['user_1', 'user_2', 'user_3', 'user_4', 'user_5', 'user_6', 'user_7'],
        'login_hour': [9, 14, 2, 10, 12, 22, 7],  # Various login hours
        'session_duration': [60, 30, 5, 180, 200, 150, 3],  # Various session durations
        'ip_address': ['192.168.1.100', '192.168.1.101', '192.168.1.220', '192.168.1.103', 
                      '192.168.1.150', '10.0.0.1', '192.168.0.240'],
        'failed_attempts': [0, 1, 6, 0, 12, 4, 8],  # Various failed attempt counts
        'day_of_week': [1, 2, 6, 4, 3, 5, 1],  # Various days
        'event_type': ['login', 'login', 'login', 'login', 'login', 'login', 'login'],
        'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] * 7
    })
    
    # Make predictions
    print("Making predictions with advanced ensemble model...")
    results = predict_anomalies(model_data, sample_data)
    
    # Display results
    print("\nPrediction Results:")
    display_cols = ['user_id', 'login_hour', 'session_duration', 'failed_attempts', 
                   'predicted_anomaly', 'anomaly_score', 'indian_criteria_anomaly', 'final_anomaly']
    
    # Ensure all columns exist
    display_cols = [col for col in display_cols if col in results.columns]
    print(results[display_cols])
    
    # Identify anomalies using the final anomaly flag
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
            print(f"  Anomaly Score: {row['anomaly_score']:.4f}")
            
            # Get detailed reasons for anomaly
            reasons = identify_anomaly_reasons(row)
            if reasons:
                print(f"  Possible reasons: {', '.join(reasons)}")
            print("-" * 60)

def load_from_file(input_file):
    """
    Load data from a CSV file and make predictions using the advanced ensemble model.
    
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
        model_path = '../data/models/advanced_anomaly_detection_model.joblib'
        model_data = load_advanced_model(model_path)
        
        if model_data is None:
            print("Failed to load the advanced model. Falling back to previous model.")
            # Try loading the older model as fallback
            model_path = '../data/models/improved_isolation_forest_model.joblib'
            model_data = load_advanced_model(model_path)
            
            if model_data is None:
                print("Failed to load any model. Please check model paths.")
                return
        
        # Make predictions
        results = predict_anomalies(model_data, data)
        
        # Save results
        output_file = os.path.join(os.path.dirname(input_file), 'advanced_prediction_results.csv')
        results.to_csv(output_file, index=False)
        print(f"Advanced predictions saved to {output_file}")
        
        # Print summary
        anomalies = results[results['final_anomaly'] == 1]
        print(f"Detected {len(anomalies)} anomalies out of {len(results)} sessions.")
        
        # Display sample of anomalies
        if len(anomalies) > 0:
            sample_size = min(5, len(anomalies))
            print(f"\nSample of {sample_size} anomalous sessions:")
            for i, (_, row) in enumerate(anomalies.head(sample_size).iterrows()):
                print(f"User: {row['user_id']}")
                print(f"  Anomaly Score: {row['anomaly_score']:.4f}")
                
                # Get detailed reasons for anomaly
                reasons = identify_anomaly_reasons(row)
                if reasons:
                    print(f"  Possible reasons: {', '.join(reasons)}")
                print("-" * 60)
                
                if i >= sample_size - 1:
                    break
            
            if len(anomalies) > sample_size:
                print(f"... and {len(anomalies) - sample_size} more anomalies.")
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict anomalies in session data using advanced ensemble model')
    parser.add_argument('--file', type=str, help='Path to the input CSV file')
    parser.add_argument('--demo', action='store_true', help='Run the demonstration with sample data')
    
    args = parser.parse_args()
    
    if args.file:
        load_from_file(args.file)
    elif args.demo:
        sample_usage_demo()
    else:
        sample_usage_demo()  # Default to demo if no arguments provided 