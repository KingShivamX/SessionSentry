#!/usr/bin/env python
"""
Database initialization script for SessionSentry.
This script creates the necessary database tables.
"""

from data_collection.database import init_db

if __name__ == "__main__":
    print("Initializing SessionSentry database...")
    init_db()
    print("Database initialization complete.") 