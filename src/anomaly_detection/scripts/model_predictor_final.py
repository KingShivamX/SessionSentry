# src/anomaly_detection/scripts/model_predictor_final.py
import pandas as pd
import numpy as np
import joblib
import os
import argparse
import datetime
import warnings
warnings.filterwarnings('ignore')

def create_features(data):
    """
    Create features with robust handling of NaN values.
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
    df['very_long_session'] = (df['session_duration'] >= 180).astype(int)
    
    # Login attempt features
    df['has_failed_attempts'] = (df['failed_attempts'] > 0).astype(int)
    
    # Feature combinations
    df['night_weekday'] = (df['is_night'] == 1) & (df['is_weekend'] == 0)
    df['night_weekday'] = df['night_weekday'].astype(int)
    df['failed_night'] = df['has_failed_attempts'] * df['is_night']
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    return df

def load_model(model_path):
    """
    Load the trained model and preprocessing components.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model file.
        
    Returns:
    --------
    dict
        Dictionary containing the model components.
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
    Predict anomalies in the input data.
    
    Parameters:
    -----------
    model_data : dict
        Dictionary containing the model components.
    data : DataFrame
        Input data to predict anomalies on.
        
    Returns:
    --------
    DataFrame
        Data with anomaly predictions and explanations.
    """
    if model_data is None:
        print("No model data available. Cannot make predictions.")
        return data
    
    # Extract model components
    isolation_forest = model_data['isolation_forest']
    scaler = model_data['scaler']
    features = model_data['features']
    rules = model_data['rules']
    
    # Apply feature engineering
    print("Applying feature engineering...")
    df = create_features(data)
    
    # Apply rule-based anomaly detection
    print("Applying rule-based anomaly detection...")
    
    # Create a list to store rule-based flags and explanations
    df['anomaly_reasons'] = ""
    
    # Rule 1: Night logins
    night_login_mask = (df['is_night'] == 1) & rules['night_login']
    df.loc[night_login_mask, 'anomaly_reasons'] += "Unusual login hour (night time); "
    
    # Rule 2: Failed attempts above threshold
    failed_attempts_mask = df['failed_attempts'] >= rules['failed_attempts_threshold']
    df.loc[failed_attempts_mask, 'anomaly_reasons'] += f"High number of failed attempts ({rules['failed_attempts_threshold']}+); "
    
    # Rule 3: Very short sessions
    short_session_mask = df['session_duration'] <= rules['short_session_threshold']
    df.loc[short_session_mask, 'anomaly_reasons'] += f"Suspiciously short session (≤{rules['short_session_threshold']} minutes); "
    
    # Rule 4: Very long sessions
    long_session_mask = df['session_duration'] >= rules['long_session_threshold']
    df.loc[long_session_mask, 'anomaly_reasons'] += f"Unusually long session (≥{rules['long_session_threshold']} minutes); "
    
    # Rule 5: Night sessions with failures
    night_failures_mask = (df['failed_attempts'] >= 1) & (df['is_night'] == 1) & rules['night_with_failures']
    df.loc[night_failures_mask, 'anomaly_reasons'] += "Failed login attempts during night hours; "
    
    # Rule 6: Short sessions with failures
    short_failures_mask = (df['failed_attempts'] >= 1) & (df['session_duration'] <= rules['short_session_with_failures'])
    df.loc[short_failures_mask, 'anomaly_reasons'] += f"Failed attempts during short session (≤{rules['short_session_with_failures']} minutes); "
    
    # Rule 7: Multiple failures outside business hours
    outside_failures_mask = (df['failed_attempts'] >= rules['multiple_failures_outside_hours']) & (df['is_business_hours'] == 0)
    df.loc[outside_failures_mask, 'anomaly_reasons'] += f"{rules['multiple_failures_outside_hours']}+ failed attempts outside business hours; "
    
    # Combine all rules
    rule_based_anomaly = (
        night_login_mask | 
        failed_attempts_mask | 
        short_session_mask | 
        long_session_mask | 
        night_failures_mask | 
        short_failures_mask | 
        outside_failures_mask
    )
    df['rule_based_anomaly'] = rule_based_anomaly.astype(int)
    
    # Apply Isolation Forest as backup
    print("Applying Isolation Forest model...")
    df_scaled = pd.DataFrame(
        scaler.transform(df[features]),
        columns=features
    )
    
    iso_predictions = isolation_forest.predict(df_scaled)
    iso_scores = isolation_forest.decision_function(df_scaled)
    df['iso_anomaly'] = np.where(iso_predictions == -1, 1, 0)
    df['anomaly_score'] = iso_scores
    
    # Combine rule-based and machine learning predictions
    df['is_anomaly'] = np.where(
        (df['rule_based_anomaly'] == 1) | 
        ((df['iso_anomaly'] == 1) & (df['has_failed_attempts'] == 1)),
        1, 0
    )
    
    # Add ML-based anomaly reasons
    ml_anomaly_mask = (df['iso_anomaly'] == 1) & (df['rule_based_anomaly'] == 0) & (df['has_failed_attempts'] == 1)
    df.loc[ml_anomaly_mask, 'anomaly_reasons'] += "ML model detected unusual pattern with failed attempts; "
    
    # Clean up anomaly reasons (remove trailing semicolon and space)
    df['anomaly_reasons'] = df['anomaly_reasons'].str.rstrip('; ')
    
    # Keep only relevant columns for the output
    output_columns = list(data.columns) + ['is_anomaly', 'anomaly_score', 'anomaly_reasons']
    
    # Summarize anomalies
    anomaly_count = df['is_anomaly'].sum()
    print(f"\nDetected {anomaly_count} anomalous sessions out of {len(df)} total sessions.")
    
    if anomaly_count > 0:
        print("\nSummary of anomalies:")
        anomalies = df[df['is_anomaly'] == 1]
        for idx, row in anomalies.iterrows():
            user = row.get('username', f"User at index {idx}")
            reasons = row['anomaly_reasons']
            print(f"- {user}: {reasons}")
    
    return df[output_columns]

def sample_usage_demo():
    """
    Demonstrate how to use the model for predictions with sample data.
    """
    print("\n=== SAMPLE USAGE DEMONSTRATION ===")
    
    # Step 1: Load the model
    model_path = '../data/models/final_anomaly_model.joblib'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model_data = load_model(model_path)
    
    # Step 2: Create sample data (similar to what might be in a real application)
    print("\nCreating sample data...")
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    
    sample_data = pd.DataFrame({
        'username': ['user_1', 'user_2', 'user_3', 'user_4', 'user_5'],
        'date': [current_date] * 5,
        'login_hour': [9, 14, 2, 10, 16],  # user_3 has unusual hour (2 AM)
        'session_duration': [45, 60, 5, 240, 30],  # user_3 has short session, user_4 has very long
        'failed_attempts': [0, 1, 6, 0, 0],  # user_3 has many failed attempts
        'day_of_week': [1, 2, 3, 3, 5]  # 0-6, where 0 is Monday
    })
    
    print("Sample data created:")
    print(sample_data)
    
    # Step 3: Make predictions
    print("\nMaking predictions...")
    results = predict_anomalies(model_data, sample_data)
    
    # Step 4: Display results with more details
    print("\nDetailed prediction results:")
    results_display = results[['username', 'login_hour', 'session_duration', 
                               'failed_attempts', 'day_of_week', 
                               'is_anomaly', 'anomaly_score', 'anomaly_reasons']]
    
    # Better display for clear understanding
    for idx, row in results_display.iterrows():
        print("\n" + "-"*50)
        print(f"User: {row['username']}")
        print(f"Login hour: {row['login_hour']}")
        print(f"Session duration: {row['session_duration']} minutes")
        print(f"Failed attempts: {row['failed_attempts']}")
        print(f"Day of week: {row['day_of_week']} (0=Monday, 6=Sunday)")
        print(f"Anomaly detected: {'YES' if row['is_anomaly'] == 1 else 'NO'}")
        print(f"Anomaly score: {row['anomaly_score']:.4f}")
        
        if row['is_anomaly'] == 1:
            print(f"Reasons: {row['anomaly_reasons']}")
    
    return results

def load_from_file(input_file):
    """
    Load data from a CSV file, make predictions, and save results.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file.
        
    Returns:
    --------
    DataFrame
        Results with anomaly predictions.
    """
    print(f"\nLoading data from {input_file}...")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return None
    
    try:
        data = pd.read_csv(input_file)
        print(f"Loaded {len(data)} rows from {input_file}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
    # Load the model
    model_path = '../data/models/final_anomaly_model.joblib'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    model_data = load_model(model_path)
    
    # Make predictions
    results = predict_anomalies(model_data, data)
    
    # Save results
    output_file = input_file.replace('.csv', '_anomalies.csv')
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Summarize anomalies
    anomaly_count = results['is_anomaly'].sum()
    total_count = len(results)
    anomaly_percent = (anomaly_count / total_count) * 100
    
    print(f"\nSummary of Analysis:")
    print(f"- Total sessions analyzed: {total_count}")
    print(f"- Anomalies detected: {anomaly_count} ({anomaly_percent:.2f}%)")
    
    if anomaly_count > 0:
        # Group by reasons to understand most common anomalies
        reasons = results[results['is_anomaly'] == 1]['anomaly_reasons'].value_counts()
        print("\nTop anomaly patterns:")
        for reason, count in reasons.items():
            print(f"- {reason}: {count} occurrences")
    
    return results

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Predict anomalies using the final trained model')
    
    # Add arguments - make the group optional
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--file', help='Path to input CSV file')
    group.add_argument('--demo', action='store_true', help='Run a demonstration with sample data')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no arguments are provided, default to demo mode
    if not args.file and not args.demo:
        print("No arguments provided. Running in demo mode by default.")
        sample_usage_demo()
    # Run based on arguments
    elif args.demo:
        sample_usage_demo()
    elif args.file:
        load_from_file(args.file) 