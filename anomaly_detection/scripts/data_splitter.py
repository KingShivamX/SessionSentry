from sklearn.model_selection import train_test_split
import pandas as pd
import os

# Ensure output directory exists
os.makedirs('../data/processed', exist_ok=True)

# Load your dataset
data = pd.read_csv('../data/raw/synthetic_event_logs.csv')
print(f"Loaded data with shape: {data.shape}")

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)  # 80% train, 20% test

# Further split the training data into training and validation sets
train_data, val_data = train_test_split(train_data, test_size=0.15/(0.8), random_state=42)  # 15% of the original data for validation

# Save the split datasets
train_data.to_csv('../data/processed/train_data.csv', index=False)
val_data.to_csv('../data/processed/val_data.csv', index=False)
test_data.to_csv('../data/processed/test_data.csv', index=False)

print(f"Data split complete:")
print(f"Train data: {train_data.shape[0]} records")
print(f"Validation data: {val_data.shape[0]} records")
print(f"Test data: {test_data.shape[0]} records")
print("Data files saved to ../data/processed/")