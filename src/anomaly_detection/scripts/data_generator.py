import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(num_samples=1000):
    # Define parameters
    users = [f'user_{i}' for i in range(1, 11)]  # 10 users
    ip_addresses = [f'192.168.1.{i}' for i in range(1, 256)]  # 256 possible IPs
    anomalies = int(num_samples * 0.2)  # 20% anomalies

    # Create data
    data = {
        'user_id': [],
        'login_hour': [],
        'session_duration': [],
        'ip_address': [],
        'failed_attempts': [],
        'day_of_week': [],
        'event_type': [],
        'timestamp': []
    }

    for i in range(num_samples):
        user = np.random.choice(users)
        ip_address = np.random.choice(ip_addresses)

        # Generate login hour (0-23)
        login_hour = np.random.randint(0, 24)

        # Generate session duration (1 to 120 minutes)
        session_duration = np.random.randint(1, 120)

        # Generate failed attempts (0 to 4)
        failed_attempts = np.random.randint(0, 5)

        # Generate day of the week (0=Monday, 6=Sunday)
        day_of_week = np.random.randint(0, 7)

        # Generate event type (90% login, 10% logout)
        event_type = 'login' if np.random.rand() > 0.1 else 'logout'

        # Generate timestamp (current date with random time)
        timestamp = datetime.now() - timedelta(days=np.random.randint(0, 30))
        timestamp = timestamp.replace(hour=login_hour, minute=np.random.randint(0, 60))

        # Introduce anomalies
        if i < anomalies:
            if np.random.rand() > 0.5:  # 50% chance to create an anomaly
                # Anomaly: unusual login hour (e.g., 3 AM) or very short/long session
                login_hour = np.random.choice([0, 1, 2, 3])  # Unusual login hours
                session_duration = np.random.choice([0, 1, 180])  # Very short or long session
            else:
                # Anomaly: high number of failed attempts
                failed_attempts = np.random.randint(5, 10)  # 5 to 9 failed attempts

        # Append data
        data['user_id'].append(user)
        data['login_hour'].append(login_hour)
        data['session_duration'].append(session_duration)
        data['ip_address'].append(ip_address)
        data['failed_attempts'].append(failed_attempts)
        data['day_of_week'].append(day_of_week)
        data['event_type'].append(event_type)
        data['timestamp'].append(timestamp)

    # Create DataFrame
    df = pd.DataFrame(data)
    return df

# Generate dataset
num_samples = 1000  # Adjust the number of samples as needed
synthetic_data = generate_synthetic_data(num_samples)

# Save to CSV
synthetic_data.to_csv('synthetic_event_logs.csv', index=False)
print("Synthetic dataset generated and saved as 'synthetic_event_logs.csv'.")