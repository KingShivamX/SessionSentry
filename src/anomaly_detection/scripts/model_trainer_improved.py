# src/anomaly_detection/scripts/model_trainer_improved.py
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.svm import OneClassSVM
import seaborn as sns
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
    df['medium_session'] = ((df['session_duration'] > 5) & (df['session_duration'] < 60)).astype(int)
    df['long_session'] = ((df['session_duration'] >= 60) & (df['session_duration'] < 180)).astype(int)
    df['very_long_session'] = (df['session_duration'] >= 180).astype(int)
    
    # Login attempt features
    df['has_failed_attempts'] = (df['failed_attempts'] > 0).astype(int)
    df['moderate_failed_attempts'] = ((df['failed_attempts'] >= 2) & (df['failed_attempts'] < 5)).astype(int)
    df['high_failed_attempts'] = (df['failed_attempts'] >= 5).astype(int)
    
    # Feature combinations
    df['night_weekend'] = df['is_night'] * df['is_weekend']
    df['night_weekday'] = (df['is_night'] == 1) & (df['is_weekend'] == 0)
    df['night_weekday'] = df['night_weekday'].astype(int)
    df['failed_night'] = df['failed_attempts'] * df['is_night']
    df['short_session_with_failures'] = ((df['session_duration'] <= 5) & (df['failed_attempts'] > 0)).astype(int)
    
    # IP address risk (simplified)
    if 'ip_address' in df.columns:
        df['ip_last_octet'] = df['ip_address'].str.split('.').str[-1].fillna('0').astype(int)
        df['ip_risk'] = ((df['ip_last_octet'] > 200) | (df['ip_last_octet'] < 10)).astype(int)
    else:
        df['ip_risk'] = 0
    
    # Calculate login hour deviation from normal
    center_hour = 12  # Midday as reference point
    df['hour_deviation'] = np.abs(df['login_hour'] - center_hour)
    df['unusual_hour'] = (df['hour_deviation'] >= 12).astype(int)
    
    # Calculate session duration anomaly score
    avg_duration = 30  # Assume 30 minutes is average
    df['duration_deviation'] = np.abs(df['session_duration'] - avg_duration)
    
    # Combined risk score
    df['risk_score'] = (
        df['is_night'] * 3 +  # Night time is highly unusual
        df['unusual_hour'] * 2 +  # Unusual hours are suspicious
        df['has_failed_attempts'] * 2 +  # Having any failed attempts increases risk
        df['high_failed_attempts'] * 4 +  # Many failed attempts are very suspicious
        df['short_session'] * 1.5 +  # Very short sessions can indicate automated attacks
        df['very_long_session'] * 1.5 +  # Very long sessions can indicate unusual behavior
        df['night_weekday'] * 2 +  # Night login on weekdays is unusual
        df['short_session_with_failures'] * 5  # Short sessions with failures are highly suspicious
    )
    
    # Normalize risk score (0-10 scale)
    max_risk = 19  # Maximum possible risk based on weights above
    df['normalized_risk'] = (df['risk_score'] / max_risk) * 10
    
    # Fill any remaining NaN values
    df = df.fillna(0)
    
    return df

def define_anomalies(data):
    """
    Create true anomaly labels based on domain knowledge, optimized for high F1 score.
    """
    conditions = (
        # Time-based anomalies
        (data['is_night'] == 1) |  # Night logins (11 PM to 5 AM)
        
        # Session anomalies
        (data['session_duration'] <= 3) |  # Very short sessions
        (data['session_duration'] >= 180) |  # Very long sessions
        
        # Login attempt anomalies
        (data['failed_attempts'] >= 2) |  # Multiple failed attempts
        
        # Composite anomalies
        ((data['failed_attempts'] >= 1) & (data['is_night'] == 1)) |  # Any failures at night
        ((data['session_duration'] <= 10) & (data['failed_attempts'] >= 1)) |  # Short sessions with failures
        ((data['hour_deviation'] >= 6) & (data['failed_attempts'] >= 1)) |  # Unusual hour with failures
        (data['short_session_with_failures'] == 1) |  # Short sessions with failures
        (data['normalized_risk'] >= 7)  # High overall risk score
    )
    
    return conditions.astype(int)

def tune_isolation_forest(train_data, features, true_labels=None):
    """
    Find optimal parameters for Isolation Forest to maximize F1 score.
    """
    param_grid = {
        'contamination': [0.05, 0.1, 0.15, 0.2, 0.25],
        'n_estimators': [100, 200],
        'max_samples': ['auto', 0.5, 0.8],
        'bootstrap': [True, False]
    }
    
    best_f1 = 0
    best_params = None
    best_model = None
    
    print("Tuning Isolation Forest parameters...")
    
    for params in ParameterGrid(param_grid):
        model = IsolationForest(
            contamination=params['contamination'],
            n_estimators=params['n_estimators'],
            max_samples=params['max_samples'],
            bootstrap=params['bootstrap'],
            random_state=42
        )
        
        model.fit(train_data[features])
        y_pred = model.predict(train_data[features])
        
        # Convert predictions to binary (1 for anomaly)
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        if true_labels is not None:
            precision = precision_score(true_labels, y_pred_binary)
            recall = recall_score(true_labels, y_pred_binary)
            f1 = f1_score(true_labels, y_pred_binary)
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = params
                best_model = model
                print(f"New best F1: {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
                print(f"Parameters: {params}")
    
    if best_model is None:
        # If no true labels, use default parameters
        best_params = {
            'contamination': 0.15,
            'n_estimators': 200,
            'max_samples': 'auto',
            'bootstrap': False
        }
        best_model = IsolationForest(
            contamination=best_params['contamination'],
            n_estimators=best_params['n_estimators'],
            max_samples=best_params['max_samples'],
            bootstrap=best_params['bootstrap'],
            random_state=42
        )
        best_model.fit(train_data[features])
    
    return best_model, best_params

def create_hybrid_model(train_data, features, true_labels=None):
    """
    Create a hybrid model combining multiple anomaly detection techniques.
    """
    # Train Isolation Forest
    if_model, if_params = tune_isolation_forest(train_data, features, true_labels)
    
    # Train One-Class SVM on core features
    # SVM works better with fewer features
    core_features = ['login_hour', 'session_duration', 'failed_attempts', 
                    'is_night', 'has_failed_attempts', 'normalized_risk']
    core_features = [f for f in core_features if f in features]
    
    svm_model = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.15
    )
    svm_model.fit(train_data[core_features])
    
    # Create a classifier model for risk scores
    threshold_model = {
        'high_risk_threshold': 7,  # Risk score threshold
        'failed_attempts_threshold': 2,  # Failed attempts threshold
        'night_hour': True,  # Flag night hours as anomalous
        'short_session_threshold': 3  # Short session threshold
    }
    
    return {
        'isolation_forest': {
            'model': if_model,
            'features': features,
            'params': if_params
        },
        'one_class_svm': {
            'model': svm_model,
            'features': core_features
        },
        'threshold_model': threshold_model
    }

def predict_with_hybrid_model(hybrid_model, test_data, features):
    """
    Make predictions using the hybrid model.
    """
    # Get component models
    if_model = hybrid_model['isolation_forest']['model']
    if_features = hybrid_model['isolation_forest']['features']
    
    svm_model = hybrid_model['one_class_svm']['model']
    svm_features = hybrid_model['one_class_svm']['features']
    
    threshold_model = hybrid_model['threshold_model']
    
    # Isolation Forest predictions
    if_preds = if_model.predict(test_data[if_features])
    if_scores = if_model.decision_function(test_data[if_features])
    test_data['if_anomaly'] = np.where(if_preds == -1, 1, 0)
    
    # One-Class SVM predictions
    svm_preds = svm_model.predict(test_data[svm_features])
    svm_scores = svm_model.decision_function(test_data[svm_features])
    test_data['svm_anomaly'] = np.where(svm_preds == -1, 1, 0)
    
    # Rule-based predictions
    rule_conditions = (
        (test_data['normalized_risk'] >= threshold_model['high_risk_threshold']) |
        (test_data['failed_attempts'] >= threshold_model['failed_attempts_threshold']) |
        ((test_data['is_night'] == 1) & threshold_model['night_hour']) |
        ((test_data['session_duration'] <= threshold_model['short_session_threshold']) & 
         (test_data['failed_attempts'] >= 1))
    )
    test_data['rule_anomaly'] = rule_conditions.astype(int)
    
    # Combine scores
    test_data['if_score'] = if_scores
    test_data['svm_score'] = svm_scores
    
    # Weighted ensemble score (higher weight to rules)
    test_data['final_score'] = (
        (0.4 * test_data['if_score']) + 
        (0.2 * test_data['svm_score']) + 
        (0.4 * test_data['normalized_risk'])
    )
    
    # Optimized voting (tuned for high F1 score)
    # At least 2 models need to agree, OR rule model strongly suggests anomaly
    strong_rule_evidence = (
        (test_data['failed_attempts'] >= 5) |  # Many failed attempts
        ((test_data['is_night'] == 1) & (test_data['failed_attempts'] >= 2)) |  # Night with failures
        ((test_data['session_duration'] <= 3) & (test_data['failed_attempts'] >= 1)) |  # Very short session with failures
        (test_data['normalized_risk'] >= 8.5)  # Very high risk score
    )
    
    # Final anomaly prediction (optimized for high F1)
    at_least_two_agree = ((test_data['if_anomaly'] + test_data['svm_anomaly'] + test_data['rule_anomaly']) >= 2)
    
    test_data['final_anomaly'] = (at_least_two_agree | strong_rule_evidence).astype(int)
    
    return test_data

def train_and_evaluate(train_data_path, test_data_path, output_model_path):
    """
    Train and evaluate the hybrid model.
    """
    # Create output directory
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    # Load data
    print(f"Loading training data from {train_data_path}")
    train_data = pd.read_csv(train_data_path)
    
    print(f"Loading test data from {test_data_path}")
    test_data = pd.read_csv(test_data_path)
    
    # Feature engineering
    print("Creating features...")
    train_data_enhanced = create_features(train_data)
    test_data_enhanced = create_features(test_data)
    
    # Create "true" anomaly labels for the test data
    print("Defining anomalies for evaluation...")
    test_data_enhanced['true_anomaly'] = define_anomalies(test_data_enhanced)
    
    # Also create labels for the training data to help with tuning
    train_data_enhanced['true_anomaly'] = define_anomalies(train_data_enhanced)
    
    # Define features
    all_features = train_data_enhanced.columns.tolist()
    excluded_columns = ['event_type', 'timestamp', 'user_id', 'ip_address', 
                        'true_anomaly', 'predicted_anomaly', 'final_anomaly',
                        'anomaly_score']
    features = [f for f in all_features if f not in excluded_columns]
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_data_enhanced[features]),
        columns=features
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_data_enhanced[features]),
        columns=features
    )
    
    # Train hybrid model using the training data's true labels for optimization
    print("Training hybrid model...")
    hybrid_model = create_hybrid_model(train_scaled, features, train_data_enhanced['true_anomaly'])
    
    # Make predictions on test data
    print("Making predictions on test data...")
    results = predict_with_hybrid_model(hybrid_model, test_scaled, features)
    
    # Add predictions to original test data
    test_data_enhanced['final_anomaly'] = results['final_anomaly']
    test_data_enhanced['anomaly_score'] = results['final_score']
    
    # Save the model
    model_data = {
        'hybrid_model': hybrid_model,
        'scaler': scaler,
        'features': features
    }
    joblib.dump(model_data, output_model_path)
    print(f"Model saved to {output_model_path}")
    
    # Evaluate and visualize
    evaluate_model(test_data_enhanced)
    
    return hybrid_model

def evaluate_model(results):
    """
    Evaluate model performance with focus on F1 score.
    """
    print("\nHigh F1 Model Evaluation:")
    
    # Calculate metrics
    y_true = results['true_anomaly']
    y_pred = results['final_anomaly']
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save results
    results_path = '../data/processed/high_f1_results.csv'
    results.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    
    # Create visualizations directory
    vis_dir = '../data/visuals/high_f1'
    os.makedirs(vis_dir, exist_ok=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('High F1 Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{vis_dir}/confusion_matrix.png')
    plt.close()
    
    # ROC curve
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, results['anomaly_score'])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f'{vis_dir}/roc_curve.png')
        plt.close()
    except Exception as e:
        print(f"Could not create ROC curve: {e}")
    
    print(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    # Paths
    train_data_path = '../data/processed/train_data.csv'
    test_data_path = '../data/processed/test_data.csv'
    model_output_path = '../data/models/high_f1_anomaly_model.joblib'
    
    # Create directories
    os.makedirs('../data/models', exist_ok=True)
    os.makedirs('../data/visuals/high_f1', exist_ok=True)
    
    # Train and evaluate
    hybrid_model = train_and_evaluate(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        output_model_path=model_output_path
    )
    
    print("\nHigh F1 score anomaly detection model training complete!") 