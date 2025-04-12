# SessionSentry Client Service

A Windows service that monitors security events and sends them to a central server.

## Features

- Real-time monitoring of Windows security events
- Automatic event collection for:
  - User logins (Event ID 4624)
  - User logouts (Event ID 4634)
  - User initiated logouts (Event ID 4647)
  - Special privileges assignments (Event ID 4672)
- Offline event storage
- Automatic retry mechanism
- Local backup of events

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Configure the server settings in `main.py`:
```python
API_URL = "http://localhost:3000/api"  # Your server URL
API_KEY = "your-api-key"  # Your API key
```

## Usage

Run the service with administrative privileges:
```bash
python main.py
```

## Project Structure

```
client_service/
├── __init__.py
├── logger.py
├── main.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.6+
- pywin32
- requests

## Notes

- The service must be run with administrative privileges to access Windows event logs
- Events are stored locally in `login_events.json` when the server is unavailable
- The service automatically retries sending stored events when the server becomes available 