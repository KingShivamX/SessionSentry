# src/anomaly_detection/scripts/model_trainer.py
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.svm import OneClassSVM
import sklearn.neighbors
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
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

def create_true_anomaly_labels(data):
    """
    Create more accurate ground truth labels for anomaly detection.
    
    Parameters:
    -----------
    data : DataFrame
        The dataset with features
        
    Returns:
    --------
    Series
        Binary labels where 1 indicates a true anomaly
    """
    # Define comprehensive conditions for what constitutes an anomaly
    conditions = (
        # Time-based anomalies
        (data['is_night'] == 1) |  # Night logins (11 PM to 5 AM)
        
        # Session anomalies
        (data['session_duration'] <= 2) |  # Very short sessions
        (data['session_duration'] >= 180) |  # Very long sessions
        
        # Login attempt anomalies
        (data['failed_attempts'] >= 3) |  # Multiple failed attempts
        
        # Composite anomalies (combinations of factors)
        ((data['is_weekend'] == 1) & (data['session_duration'] >= 120)) |  # Long weekend sessions
        ((data['is_night'] == 1) & (data['failed_attempts'] >= 2)) |  # Night logins with failed attempts
        ((data['is_weekend'] == 0) & (data['is_night'] == 1) & (data['session_duration'] < 15)) |  # Short night sessions on weekdays
        
        # Very suspicious combinations
        ((data['failed_attempts'] >= 5) & (data['session_duration'] <= 10)) |  # High failures in short sessions
        ((data['is_night'] == 1) & (data['session_duration'] >= 90))  # Long night sessions
    )
    
    return conditions.astype(int)

def train_advanced_isolation_forest(train_data, features, random_state=42):
    """
    Train an advanced Isolation Forest model with optimal parameters.
    
    Parameters:
    -----------
    train_data : DataFrame
        Training data
    features : list
        Features to use
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    model : IsolationForest
        Trained model
    """
    # Define advanced parameters
    model = IsolationForest(
        n_estimators=200,
        max_samples='auto',
        contamination=0.15,  # Optimized contamination rate
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        random_state=random_state,
        verbose=0
    )
    
    # Train model
    model.fit(train_data[features])
    
    return model

def train_one_class_svm(train_data, features):
    """
    Train a One-Class SVM model for anomaly detection.
    
    Parameters:
    -----------
    train_data : DataFrame
        Training data
    features : list
        Features to use
        
    Returns:
    --------
    model : OneClassSVM
        Trained model
    """
    # One-Class SVM works better with smaller feature sets
    # so we'll use a subset of the most important features
    model = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.1,  # Contamination parameter
        shrinking=True
    )
    
    # Train model
    model.fit(train_data[features])
    
    return model

def train_local_outlier_factor(train_data, features):
    """
    Train a Local Outlier Factor model for anomaly detection.
    
    Parameters:
    -----------
    train_data : DataFrame
        Training data
    features : list
        Features to use
        
    Returns:
    --------
    model : LocalOutlierFactor
        Trained model
    """
    model = sklearn.neighbors.LocalOutlierFactor(
        n_neighbors=20,
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        contamination=0.15,
        novelty=True
    )
    
    # Train model
    model.fit(train_data[features])
    
    return model

def create_anomaly_ensemble(train_data, features):
    """
    Create an ensemble of multiple anomaly detection algorithms.
    
    Parameters:
    -----------
    train_data : DataFrame
        Training data
    features : list
        Features to use
        
    Returns:
    --------
    dict
        Dictionary containing all trained models
    """
    print("Training ensemble of anomaly detection models...")
    
    # Train individual models
    isolation_forest = train_advanced_isolation_forest(train_data, features)
    
    # Use a subset of features for One-Class SVM (it works better with fewer features)
    # Selecting 5 most important features - this would need to be optimized in practice
    svm_features = features[:min(5, len(features))]
    one_class_svm = train_one_class_svm(train_data[svm_features], svm_features)
    
    # LOF for local density-based detection
    lof = train_local_outlier_factor(train_data, features)
    
    return {
        'isolation_forest': {
            'model': isolation_forest,
            'features': features
        },
        'one_class_svm': {
            'model': one_class_svm,
            'features': svm_features
        },
        'lof': {
            'model': lof,
            'features': features
        }
    }

def predict_with_ensemble(ensemble, test_data):
    """
    Make predictions using the ensemble of models.
    
    Parameters:
    -----------
    ensemble : dict
        Dictionary of trained models
    test_data : DataFrame
        Test data for prediction
        
    Returns:
    --------
    DataFrame
        Test data with predictions and ensemble scores
    """
    # Make predictions with each model
    if_model = ensemble['isolation_forest']['model']
    if_features = ensemble['isolation_forest']['features']
    if_preds = if_model.predict(test_data[if_features])
    if_scores = if_model.decision_function(test_data[if_features])
    
    svm_model = ensemble['one_class_svm']['model']
    svm_features = ensemble['one_class_svm']['features']
    svm_preds = svm_model.predict(test_data[svm_features])
    svm_scores = svm_model.decision_function(test_data[svm_features])
    
    lof_model = ensemble['lof']['model']
    lof_features = ensemble['lof']['features']
    lof_preds = lof_model.predict(test_data[lof_features])
    lof_scores = lof_model.decision_function(test_data[lof_features])
    
    # Convert predictions to binary (1 for anomaly, 0 for normal)
    test_data['if_anomaly'] = np.where(if_preds == -1, 1, 0)
    test_data['svm_anomaly'] = np.where(svm_preds == -1, 1, 0)
    test_data['lof_anomaly'] = np.where(lof_preds == -1, 1, 0)
    
    # Add scores
    test_data['if_score'] = if_scores
    test_data['svm_score'] = svm_scores
    test_data['lof_score'] = lof_scores
    
    # Create weighted ensemble prediction
    # More weight to Isolation Forest as it's typically more reliable
    test_data['ensemble_score'] = (
        0.5 * test_data['if_score'] + 
        0.25 * test_data['svm_score'] + 
        0.25 * test_data['lof_score']
    )
    
    # Final ensemble prediction (majority voting)
    test_data['ensemble_votes'] = (
        test_data['if_anomaly'] + 
        test_data['svm_anomaly'] + 
        test_data['lof_anomaly']
    )
    test_data['predicted_anomaly'] = np.where(test_data['ensemble_votes'] >= 2, 1, 0)  # Majority vote
    
    # Calculate a normalized confidence score
    max_score = test_data['ensemble_score'].max()
    min_score = test_data['ensemble_score'].min()
    range_score = max_score - min_score
    
    if range_score > 0:
        test_data['anomaly_score'] = 1 - ((test_data['ensemble_score'] - min_score) / range_score)
    else:
        test_data['anomaly_score'] = 0.5  # Default if no range
    
    return test_data

def train_improved_isolation_forest(train_data_path, test_data_path, output_model_path):
    """
    Train an improved ensemble model with advanced feature engineering.
    
    Parameters:
    -----------
    train_data_path : str
        Path to the training data CSV file
    test_data_path : str
        Path to the test data CSV file
    output_model_path : str
        Path to save the trained model
    
    Returns:
    --------
    dict
        Dictionary containing the ensemble models
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    # Load the training and test data
    print(f"Loading training data from {train_data_path}")
    train_data = pd.read_csv(train_data_path)
    
    print(f"Loading test data from {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    
    # Advanced feature engineering
    print("Applying advanced feature engineering...")
    train_data_advanced = create_advanced_features(train_data)
    test_data_advanced = create_advanced_features(test_data)
    
    # Create "true" labels for supervised evaluation
    test_data_advanced['true_anomaly'] = create_true_anomaly_labels(test_data_advanced)
    
    # Define the feature sets
    base_features = ['login_hour', 'session_duration', 'failed_attempts', 'day_of_week']
    
    # Create comprehensive feature list, excluding target and ID columns
    all_features = train_data_advanced.columns.tolist()
    excluded_columns = ['event_type', 'timestamp', 'user_id', 'ip_address', 
                       'true_anomaly', 'predicted_anomaly', 'anomaly_score']
    
    advanced_features = [f for f in all_features if f not in excluded_columns]
    
    # Scale features to improve model performance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data_advanced[advanced_features])
    X_test_scaled = scaler.transform(test_data_advanced[advanced_features])
    
    # Convert back to DataFrame for easier handling
    X_train = pd.DataFrame(X_train_scaled, columns=advanced_features)
    X_test = pd.DataFrame(X_test_scaled, columns=advanced_features)
    
    # Create and train ensemble
    ensemble = create_anomaly_ensemble(X_train, advanced_features)
    
    # Make predictions with ensemble
    results = predict_with_ensemble(ensemble, X_test)
    
    # Add predictions back to original test data
    test_data_advanced['predicted_anomaly'] = results['predicted_anomaly']
    test_data_advanced['anomaly_score'] = results['anomaly_score']
    
    # Save the ensemble model and preprocessing components
    model_data = {
        'ensemble': ensemble,
        'scaler': scaler,
        'features': advanced_features,
        'base_features': base_features
    }
    joblib.dump(model_data, output_model_path)
    print(f"Advanced ensemble model and components saved to {output_model_path}")
    
    # Evaluate the model
    evaluate_advanced_model(test_data_advanced)
    
    return ensemble

def evaluate_advanced_model(results_df):
    """
    Evaluate the advanced model using true anomaly labels.
    
    Parameters:
    -----------
    results_df : DataFrame
        Test data with predictions and true labels
    """
    print("\nEvaluating advanced model performance")
    
    # Calculate metrics
    y_true = results_df['true_anomaly']
    y_pred = results_df['predicted_anomaly']
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Save results
    results_path = '../data/processed/advanced_evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"Advanced evaluation results saved to {results_path}")
    
    # Create visualizations
    visualize_advanced_results(results_df)
    
    # Print metrics
    print("\nAdvanced Model Performance Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create and plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Advanced Model Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('../data/visuals/advanced/advanced_confusion_matrix.png')
    plt.close()
    
    print("Advanced confusion matrix saved to '../data/visuals/advanced/advanced_confusion_matrix.png'")

def visualize_advanced_results(results_df):
    """
    Create advanced visualizations for model evaluation.
    
    Parameters:
    -----------
    results_df : DataFrame
        Results data with predictions
    """
    # Create output directory
    output_dir = '../data/visuals/advanced'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ROC Curve
    if 'true_anomaly' in results_df.columns and 'anomaly_score' in results_df.columns:
        # Compute ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(results_df['true_anomaly'], results_df['anomaly_score'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'{output_dir}/roc_curve.png')
        plt.close()
    
    # 2. Anomaly score distribution by true class
    if 'true_anomaly' in results_df.columns and 'anomaly_score' in results_df.columns:
        plt.figure(figsize=(12, 8))
        sns.histplot(
            data=results_df, 
            x='anomaly_score', 
            hue='true_anomaly',
            bins=50,
            kde=True,
            palette=['green', 'red']
        )
        plt.title('Anomaly Score Distribution by True Class')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        plt.savefig(f'{output_dir}/score_distribution_by_class.png')
        plt.close()
    
    # 3. Feature importance using permutation importance
    # This requires having the model, which we don't in this function
    # So we'll approximate using correlation to anomaly score
    feature_cols = [col for col in results_df.columns if col not in 
                   ['true_anomaly', 'predicted_anomaly', 'anomaly_score', 
                    'user_id', 'event_type', 'timestamp', 'ip_address']]
    
    importance_dict = {}
    for feat in feature_cols:
        if results_df[feat].dtype in ['int64', 'float64']:
            corr = abs(np.corrcoef(results_df['anomaly_score'], results_df[feat])[0,1])
            if not np.isnan(corr):
                importance_dict[feat] = corr
    
    # Sort and plot
    if importance_dict:
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(14, 10))
        sns.barplot(data=importance_df.head(20), x='importance', y='feature')
        plt.title('Top 20 Features by Correlation with Anomaly Score')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png')
        plt.close()
    
    print(f"Advanced visualizations saved to {output_dir}")

if __name__ == "__main__":
    # Paths
    train_data_path = '../data/processed/train_data.csv'
    test_data_path = '../data/processed/test_data.csv'
    model_output_path = '../data/models/advanced_anomaly_detection_model.joblib'
    
    # Create directories if they don't exist
    os.makedirs('../data/models', exist_ok=True)
    os.makedirs('../data/visuals/advanced', exist_ok=True)
    
    # Train the advanced model
    ensemble = train_improved_isolation_forest(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        output_model_path=model_output_path
    )
    
    print("\nAdvanced anomaly detection model training and evaluation complete!")
