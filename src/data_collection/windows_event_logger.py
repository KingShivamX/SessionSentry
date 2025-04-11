"""
Windows Event Logger for SessionSentry.
Extracts login/logout events from Windows event logs in real-time.
"""

import win32evtlog
import win32evtlogutil
import win32con
import datetime
import json
import logging
import time
from typing import List, Dict, Any

# Configure logger
logger = logging.getLogger("SessionSentry.WindowsEventLogger")

class WindowsEventLogger:
    def __init__(self, server="localhost", log_type="Security", 
                event_ids=(4624, 4634, 4647, 4672)):
        """
        Initialize the Windows event logger.
        
        Args:
            server: The server to collect logs from
            log_type: The type of log to collect (Security, System, etc.)
            event_ids: Tuple of event IDs to collect
        """
        self.server = server
        self.log_type = log_type
        self.event_ids = event_ids
        self.hand = win32evtlog.OpenEventLog(self.server, self.log_type)
        self.flags = win32evtlog.EVENTLOG_FORWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
        self.last_event_time = datetime.datetime.now()
        logger.info(f"WindowsEventLogger initialized for {server}/{log_type}")

    def get_new_events(self) -> List[Dict[str, Any]]:
        """
        Get new events since last check
        
        Returns:
            List of event dictionaries containing event details
        """
        events = []
        try:
            events_batch = win32evtlog.ReadEventLog(self.hand, self.flags, 0)
            if not events_batch:
                return events

            for event in events_batch:
                try:
                    # Convert the time to a Python datetime
                    event_time = datetime.datetime.fromtimestamp(
                        int(event.TimeGenerated)
                    )
                    
                    # Skip if event is older than our last check
                    if event_time <= self.last_event_time:
                        continue

                    # Only process events we're interested in
                    if event.EventID in self.event_ids:
                        # Extract username
                        username = self._get_username(event)
                        
                        # Skip system accounts
                        if username == "-" or not username:
                            continue
                            
                        # Extract other details
                        domain = self._get_domain(event)
                        computer = event.ComputerName
                        event_type = self._get_event_type(event.EventID)
                        source_address = self._get_source_address(event)
                        user_sid = self._get_sid(event)
                        login_type = self._get_login_type(event)
                        
                        # Create structured event data
                        event_data = {
                            "event_id": event.EventID,
                            "event_type": event_type.lower(),  # Login/logout in lowercase for db
                            "timestamp": event_time,
                            "user_sid": user_sid,
                            "username": username,
                            "domain": domain,
                            "computer": computer,
                            "login_type": login_type,
                            "source_address": source_address
                        }
                        
                        events.append(event_data)
                        self.last_event_time = event_time
                except Exception as e:
                    logger.error(f"Error parsing event: {e}", exc_info=True)
                    
        except Exception as e:
            logger.error(f"Error reading event log: {e}", exc_info=True)
            
        logger.info(f"Collected {len(events)} new login/logout events")
        return events
    
    def _get_event_type(self, event_id: int) -> str:
        """
        Get human-readable event type
        """
        event_types = {
            4624: "login",
            4634: "logout",
            4647: "logout",  # User initiated logout
            4672: "privileges"  # Special privileges assigned
        }
        return event_types.get(event_id, f"unknown({event_id})")
    
    def _get_username(self, event) -> str:
        """
        Extract username from event data
        """
        try:
            inserts = event.StringInserts
            if not inserts:
                return ""

            if event.EventID == 4624 and len(inserts) >= 6:
                return inserts[5]
            elif event.EventID in (4634, 4647, 4672) and len(inserts) >= 1:
                return inserts[1] if len(inserts) > 1 else inserts[0]
        except Exception as e:
            logger.warning(f"Error extracting username: {e}")

        return ""
    
    def _get_domain(self, event) -> str:
        """
        Extract domain from event data
        """
        try:
            inserts = event.StringInserts
            if not inserts:
                return ""

            if event.EventID == 4624 and len(inserts) >= 7:
                return inserts[6]
            elif event.EventID in (4634, 4647) and len(inserts) >= 2:
                return inserts[2]
        except Exception as e:
            logger.warning(f"Error extracting domain: {e}")

        return ""
    
    def _get_sid(self, event) -> str:
        """
        Extract user SID from event data
        """
        try:
            inserts = event.StringInserts
            if not inserts:
                return ""

            if event.EventID == 4624 and len(inserts) >= 5:
                return inserts[4]
            elif event.EventID in (4634, 4647) and len(inserts) >= 3:
                return inserts[3]
        except Exception as e:
            logger.warning(f"Error extracting SID: {e}")

        return ""
    
    def _get_login_type(self, event) -> str:
        """
        Extract login type from event data
        """
        try:
            inserts = event.StringInserts
            if event.EventID == 4624 and inserts and len(inserts) >= 9:
                login_type = inserts[8]
                login_types = {
                    "2": "Interactive",
                    "3": "Network",
                    "4": "Batch",
                    "5": "Service",
                    "7": "Unlock",
                    "8": "NetworkCleartext",
                    "9": "NewCredentials",
                    "10": "RemoteInteractive",
                    "11": "CachedInteractive"
                }
                return login_types.get(login_type, login_type)
        except Exception as e:
            logger.warning(f"Error extracting login type: {e}")

        return ""
    
    def _get_source_address(self, event) -> str:
        """
        Extract source address from event data
        """
        try:
            inserts = event.StringInserts
            if event.EventID == 4624 and inserts and len(inserts) > 18:
                return inserts[18]
        except Exception as e:
            logger.warning(f"Error extracting source address: {e}")

        return ""
    
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
                # Skip privilege events as they're not login/logout
                if event['event_type'] == 'privileges':
                    continue
                    
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
    
    def close(self):
        """Close the event log handle."""
        try:
            win32evtlog.CloseEventLog(self.hand)
            logger.info("Windows event log handle closed")
        except Exception as e:
            logger.error(f"Error closing event log: {e}", exc_info=True) 