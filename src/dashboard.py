#!/usr/bin/env python
"""
Dashboard entry point for SessionSentry.
This script starts the Flask web dashboard.
"""

import os
import sys
from dashboard.app import app

if __name__ == "__main__":
    # Make sure database tables are created
    try:
        from data_collection.database import init_db
        init_db()
        print("Database tables verified.")
    except Exception as e:
        print(f"Warning: Could not verify database tables: {e}")
        
    # Start the Flask dashboard
    port = int(os.environ.get("DASHBOARD_PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True) 