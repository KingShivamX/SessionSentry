#!/usr/bin/env python
"""
SessionSentry - AI-Powered Windows Session Monitoring Tool
Main entry point for the application.
"""

import os
import time
import logging
from dotenv import load_dotenv

# Import components
from data_collection.windows_event_logger import WindowsEventLogger
from processing import anomaly_detector
from reporting import email_reporter
from data_collection.database import init_db, get_db_session

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sessionsentry.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SessionSentry")

def main():
    """Main function to orchestrate SessionSentry components."""
    logger.info("Starting SessionSentry...")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    try:
        # Initialize Windows event logger
        logger.info("Initializing Windows event logger...")
        event_logger = WindowsEventLogger()
        
        # Initialize anomaly detection
        logger.info("Initializing anomaly detection...")
        detector = anomaly_detector.AnomalyDetector()
        
        # Initialize email reporting
        logger.info("Initializing email reporting...")
        reporter = email_reporter.EmailReporter()
        
        # Main processing loop
        logger.info("SessionSentry is running. Press Ctrl+C to exit.")
        print("[*] Starting real-time security event monitoring...")
        print("[*] Press Ctrl+C to stop\n")
        
        while True:
            # Get database session
            db = get_db_session()
            
            try:
                # Collect recent events
                events = event_logger.get_new_events()
                
                # Print event information
                if events:
                    print(f"\n[+] {len(events)} new security events detected:")
                    for event in events:
                        print(f"\nTime: {event['timestamp']}")
                        print(f"Event Type: {event['event_type']}")
                        print(f"User: {event['username']}")
                        print(f"Computer: {event['computer']}")
                        print("-" * 50)
                
                # Save events to database
                if events:
                    event_logger.save_to_database(events, db)
                
                # Process events and detect anomalies
                anomalies = detector.detect_anomalies(events)
                
                # Report any detected anomalies
                if anomalies:
                    reporter.send_anomaly_report(anomalies)
            finally:
                # Always close the database session
                db.close()
            
            # Check for events more frequently (every second)
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("SessionSentry stopped by user.")
        print("\n[*] Stopping event monitoring...")
    except Exception as e:
        logger.error(f"Error in SessionSentry: {e}", exc_info=True)
    finally:
        if 'event_logger' in locals():
            event_logger.close()
        logger.info("SessionSentry shut down.")

if __name__ == "__main__":
    main() 