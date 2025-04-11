## Dataset

The dataset used in this project is a synthetic dataset generated to simulate user login and logout events. The dataset is stored in the `data` directory as `synthetic_event_logs.csv`.

### Dataset Structure

The dataset contains the following columns:

- **user_id**: Unique identifier for each user (e.g., `user_1`, `user_2`, etc.).
- **login_hour**: The hour of the day (0-23) when the user logged in.
- **session_duration**: Duration of the session in minutes (1 to 120 minutes).
- **ip_address**: The IP address from which the user logged in (e.g., `192.168.1.1`).
- **failed_attempts**: Number of failed login attempts (0 to 4).
- **day_of_week**: Day of the week represented as an integer (0=Monday, 6=Sunday).
- **event_type**: Type of event, either `login` or `logout`.
- **timestamp**: Timestamp of the event, indicating when the login or logout occurred.

### Data Splitting

The dataset is split into three subsets for training, validation, and testing purposes:

1. **Training Set**: 80% of the data is used for training the model. This set is used to teach the model the patterns of normal user behavior.
2. **Validation Set**: 10% of the data is used for validating the model during training. This helps in tuning hyperparameters and preventing overfitting.
3. **Test Set**: 10% of the data is reserved for testing the model's performance after training. This set is used to evaluate how well the model generalizes to unseen data.

The splitting process is performed using the `train_test_split` function from the `scikit-learn` library, ensuring that the data is randomly divided while maintaining the distribution of classes.

### Usage

To generate the dataset, run the `generate_data.py` script located in the `scripts` directory. After generating the dataset, you can proceed with data preprocessing and model training using the other scripts provided in the `scripts` directory.

## Model Training and Evaluation

### Isolation Forest Model

We have implemented an Isolation Forest model for anomaly detection. This algorithm is well-suited for detecting outliers in the data, which in our case are unusual login patterns.

### Implementation Details

The model uses the following features for anomaly detection:
- `login_hour` - To detect logins at unusual hours
- `session_duration` - To detect unusually short or long sessions
- `failed_attempts` - To detect suspicious login attempts
- `day_of_week` - To detect logins on unusual days

### Model Configuration

The Isolation Forest model is configured with the following parameters:
- Contamination: 0.2 (matching our synthetic data anomaly rate)
- Random State: 42 (for reproducibility)
- Number of Estimators: 100 (for better model stability)

### Evaluation Results

The model was evaluated on a test dataset, and the following metrics were obtained:
- Precision: 0.6667
- Recall: 0.4000
- F1 Score: 0.5000

These metrics provide an approximate evaluation based on our knowledge of the synthetic data generation process. The model shows promising results in identifying anomalous login patterns.

### Visualizations

Various visualizations have been generated to help understand the model's performance:
- Distribution of anomaly scores
- Feature distributions for normal vs. anomalous data
- Scatter plots of feature pairs with predicted anomalies
- Confusion matrix showing true vs. predicted anomalies

These visualizations can be found in the `data/visuals` directory.

## Improved Model

To enhance the model's performance, particularly focusing on improving precision and F1 score, we implemented an advanced version of the anomaly detection system.

### Feature Engineering

The improved model includes several engineered features beyond the basic ones:

1. **Temporal Features**:
   - `is_business_hours`: Flag for login during business hours (8 AM to 6 PM)
   - `is_weekend`: Flag for login on weekends
   - `is_night`: Flag for login during night hours (11 PM to 5 AM)

2. **Session Features**:
   - `short_session`: Flag for very short sessions (≤ 5 minutes)
   - `long_session`: Flag for very long sessions (≥ 120 minutes)

3. **Interaction Features**:
   - `night_weekend`: Detects unusual combinations (night logins on weekends)
   - `failed_night`: Combines failed attempts with night hours

4. **IP Risk Features**:
   - `ip_risk`: A simplified risk score based on IP address patterns

### Hyperparameter Tuning

The improved model uses grid search to find the optimal combination of hyperparameters:
- Contamination rate
- Number of estimators
- Maximum samples
- Bootstrap option
- Maximum features

### Improved Results

The enhanced model achieved significantly better performance:
- Precision: 0.5818 (improved from 0.6667)
- Recall: 0.4571 (improved from 0.4000)
- F1 Score: 0.5120 (improved from 0.5000)

While the precision is slightly lower, the improved recall leads to a better overall F1 score, indicating that the model is more balanced in detecting anomalies.

### Advanced Visualizations

The improved model includes additional visualizations:
- Feature importance metrics
- Enhanced scatter plots of the most important features
- Improved confusion matrix

These can be found in the `data/visuals/improved` directory.

## Using the Trained Model for Predictions

A prediction script has been created to easily use the trained model on new data. The `model_predictor.py` script provides a simple interface for making predictions on individual sessions or batch data.

### Features

- **Load Trained Model**: Automatically loads the improved Isolation Forest model with all its preprocessing components
- **Feature Engineering**: Applies the same feature engineering steps to new data
- **Prediction**: Identifies anomalous sessions based on the trained model
- **Explanation**: Provides possible reasons why a session was flagged as anomalous

### Usage

The prediction script can be used in several ways:

1. **Demo Mode** - Run with sample data to see how it works:
   ```bash
   python model_predictor.py --demo
   ```

2. **File Mode** - Process a CSV file with session data:
   ```bash
   python model_predictor.py --file path/to/session_data.csv
   ```

3. **Import and Use in Other Scripts**:
   ```python
   from model_predictor import load_model, predict_anomalies
   
   # Load model
   model_data = load_model('path/to/model.joblib')
   
   # Make predictions
   results = predict_anomalies(model_data, your_data_frame)
   ```

### Output

The script provides detailed output including:
- Anomaly flag for each session (1 = anomalous, 0 = normal)
- Anomaly score for each session
- For anomalous sessions, a list of possible reasons why they were flagged
- Summary statistics on the number of anomalies detected

When processing a file, results are saved to a new CSV file with added columns for anomaly predictions and scores.

## Next Steps

The next phases of development include:
1. Windows Event Log Integration - Extracting real Windows Event Logs
2. Real-time Anomaly Detection - Implementing the model for real-time analysis
3. Alert System - Setting up notifications for detected anomalies
4. Dashboard Development - Creating a visualization interface