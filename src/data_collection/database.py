"""
Database module for SessionSentry.
Handles database setup, models, and operations.
"""

import os
import logging
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger("SessionSentry.Database")

# Load environment variables
load_dotenv()

# Get database URI from environment or use SQLite as default
DB_URI = os.getenv("DATABASE_URI", "sqlite:///session_sentry.db")

# Create SQLAlchemy engine and session
engine = create_engine(DB_URI)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base for models
Base = declarative_base()

class SessionEvent(Base):
    """Model for session events (login/logout)."""
    
    __tablename__ = "session_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, nullable=False)
    event_type = Column(String(10), nullable=False)  # login or logout
    timestamp = Column(DateTime, nullable=False)
    user_sid = Column(String(100))
    username = Column(String(100))
    domain = Column(String(100))
    computer = Column(String(100))
    login_type = Column(String(20))  # For login events
    source_address = Column(String(50))  # IP address for network logins
    is_anomalous = Column(Boolean, default=False)
    anomaly_reason = Column(Text)

class AnomalyEvent(Base):
    """Model for detected anomalies."""
    
    __tablename__ = "anomaly_events"
    
    id = Column(Integer, primary_key=True, index=True)
    event_id = Column(Integer, nullable=False)  # References session_events.id
    detected_at = Column(DateTime, nullable=False)
    anomaly_type = Column(String(50), nullable=False)
    severity = Column(Integer, default=1)  # 1-10 scale
    description = Column(Text)
    is_reported = Column(Boolean, default=False)
    is_false_positive = Column(Boolean, default=False)
    analyst_notes = Column(Text)

def init_db():
    """Initialize the database, creating tables if they don't exist."""
    try:
        logger.info("Initializing database...")
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}", exc_info=True)
        raise

def get_db_session():
    """Get a new database session."""
    db = SessionLocal()
    try:
        return db
    except Exception as e:
        db.close()
        logger.error(f"Error getting database session: {e}", exc_info=True)
        raise 