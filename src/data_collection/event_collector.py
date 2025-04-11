"""
EventCollector module for SessionSentry.
Extracts login and logout events from Windows event logs.
"""

import win32evtlog
import win32evtlogutil
import win32con
import win32security
import datetime
import logging
import pandas as pd
from typing import List, Dict, Any

# Configure logger
logger = logging.getLogger("SessionSentry.EventCollector")

class EventCollector:
    """Collects login/logout events from Windows event logs."""
    
    def __init__(self, server="localhost", log_type="Security", 
                 event_ids=(4624, 4634), max_events=100):
        """
        Initialize the event collector.
        
        Args:
            server: The server to collect logs from
            log_type: The type of log to collect (Security, System, etc.)
            event_ids: Tuple of event IDs to collect (4624=login, 4634=logout)
            max_events: Maximum number of events to retrieve per collection
        """
        self.server = server
        self.log_type = log_type
        self.event_ids = event_ids
        self.max_events = max_events
        self.last_collection_time = datetime.datetime.now() - datetime.timedelta(hours=1)
        logger.info(f"EventCollector initialized for {server}/{log_type}")
    
    def collect_events(self) -> List[Dict[str, Any]]:
        """
        Collect login/logout events from Windows event logs.
        
        Returns:
            List of event dictionaries containing event details
        """
        logger.info("Collecting events...")
        events = []
        
        try:
            # Open the event log
            hand = win32evtlog.OpenEventLog(self.server, self.log_type)
            
            # Get total number of records
            total = win32evtlog.GetNumberOfEventLogRecords(hand)
            logger.debug(f"Total event log records: {total}")
            
            flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
            
            # Read events
            events_raw = win32evtlog.ReadEventLog(hand, flags, 0, self.max_events)
            
            for event in events_raw:
                # Check if event ID matches what we're looking for
                if event.EventID in self.event_ids:
                    # Parse the event data
                    event_data = self._parse_event(event)
                    
                    # Only include events that occurred after our last collection
                    if event_data['timestamp'] > self.last_collection_time:
                        events.append(event_data)
            
            # Update the last collection time
            self.last_collection_time = datetime.datetime.now()
            
            logger.info(f"Collected {len(events)} login/logout events")
            return events
            
        except Exception as e:
            logger.error(f"Error collecting events: {e}", exc_info=True)
            return []
        finally:
            win32evtlog.CloseEventLog(hand)
    
    def _parse_event(self, event) -> Dict[str, Any]:
        """
        Parse a Windows event log entry into a structured dictionary.
        
        Args:
            event: The Windows event log entry
            
        Returns:
            Dictionary containing parsed event data
        """
        # Convert the time to a Python datetime
        timestamp = datetime.datetime.fromtimestamp(
            int(event.TimeGenerated)
        )
        
        # Determine event type (login or logout)
        event_type = "login" if event.EventID == 4624 else "logout"
        
        # Extract user information
        try:
            # The data is in the event's StringInserts
            sid = event.StringInserts[4] if event.StringInserts else None
            username = event.StringInserts[5] if event.StringInserts else None
            domain = event.StringInserts[6] if event.StringInserts else None
            
            # For login events, get login type
            login_type = None
            if event_type == "login" and event.StringInserts:
                login_type = event.StringInserts[8]
                
            # Source info for login events
            source_address = None
            if event_type == "login" and event.StringInserts and len(event.StringInserts) > 18:
                source_address = event.StringInserts[18]
        except Exception as e:
            logger.warning(f"Error parsing event details: {e}")
            sid = username = domain = login_type = source_address = None
        
        # Create a structured event dictionary
        event_data = {
            "event_id": event.EventID,
            "event_type": event_type,
            "timestamp": timestamp,
            "user_sid": sid,
            "username": username,
            "domain": domain,
            "computer": event.ComputerName,
            "login_type": login_type,
            "source_address": source_address
        }
        
        return event_data
    
    def save_to_database(self, events, db_connection):
        """
        Save collected events to a database.
        
        Args:
            events: List of event dictionaries
            db_connection: SQLAlchemy database connection
        """
        try:
            from data_collection.database import SessionEvent
            
            # If no events, do nothing
            if not events:
                return
                
            # Insert events into the database
            for event in events:
                db_event = SessionEvent(
                    event_id=event['event_id'],
                    event_type=event['event_type'],
                    timestamp=event['timestamp'],
                    user_sid=event['user_sid'],
                    username=event['username'],
                    domain=event['domain'],
                    computer=event['computer'],
                    login_type=event['login_type'],
                    source_address=event['source_address'],
                    is_anomalous=False  # Will be updated by anomaly detector
                )
                db_connection.add(db_event)
            
            # Commit the changes
            db_connection.commit()
            
            logger.info(f"Saved {len(events)} events to database")
        except Exception as e:
            logger.error(f"Error saving events to database: {e}", exc_info=True) 