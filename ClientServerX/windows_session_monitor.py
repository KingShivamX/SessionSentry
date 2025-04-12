import win32evtlog
import win32security
import win32con
import win32evtlogutil
import socket
import json
import os
import time
import datetime
import re
from pathlib import Path

class WindowsSessionMonitor:
    def __init__(self):
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "login_events.json")
        self.event_cache = {}  # Cache for deduplication
        self.cache_timeout = 60  # Seconds to keep events in cache
        self.last_record_number = 0
        
        # Ensure JSON file exists with valid content
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write('[]')
        elif os.path.getsize(self.log_file) == 0:
            with open(self.log_file, 'w') as f:
                f.write('[]')
                
        # Load existing events for deduplication check
        try:
            with open(self.log_file, 'r') as f:
                content = f.read().strip()
                self.events = json.loads(content) if content else []
        except (json.JSONDecodeError, FileNotFoundError):
            self.events = []

    def _get_local_ip(self):
        """Get the local IP address of the machine"""
        try:
            # Create a socket connection to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Doesn't need to be reachable
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return socket.gethostbyname(socket.gethostname())

    def _convert_sid_to_username(self, sid_string):
        """Convert a SID string to a username"""
        try:
            sid = win32security.ConvertStringSidToSid(sid_string)
            name, domain, _ = win32security.LookupAccountSid(None, sid)
            if domain and name:
                return f"{domain}\\{name}"
            return name or sid_string
        except Exception as e:
            return sid_string

    def _extract_username_from_event(self, event):
        """Extract username from event data"""
        event_data = event.StringInserts
        if not event_data:
            return "UNKNOWN"

        # Try to extract username from different event data positions
        # First check for Subject\Account Name
        for i, data in enumerate(event_data):
            if data == "Account Name:" and i+1 < len(event_data):
                return event_data[i+1]
            elif data == "Security ID:" and i+1 < len(event_data):
                sid = event_data[i+1]
                if sid.startswith('S-'):
                    return self._convert_sid_to_username(sid)
            # Look for patterns like 'Account Name:\t<username>'
            elif data and isinstance(data, str):
                account_match = re.search(r'Account Name:\s*(\S+)', data)
                if account_match:
                    return account_match.group(1)
                sid_match = re.search(r'Security ID:\s*(S-\S+)', data)
                if sid_match:
                    return self._convert_sid_to_username(sid_match.group(1))

        # Check if there's a SID in any field and convert it
        for data in event_data:
            if data and isinstance(data, str) and data.startswith('S-'):
                return self._convert_sid_to_username(data)
                
        return "UNKNOWN"

    def _create_event_signature(self, event_id, username, timestamp_minute):
        """Create a unique signature for deduplication"""
        return f"{event_id}_{username}_{timestamp_minute}"
        
    def _is_duplicate(self, event_id, username, timestamp):
        """Check if event is a duplicate based on cache"""
        # Create a signature that includes only the minute part of the timestamp
        timestamp_minute = timestamp.strftime("%Y-%m-%d %H:%M")
        signature = self._create_event_signature(event_id, username, timestamp_minute)
        
        # Check in memory cache
        if signature in self.event_cache:
            if time.time() - self.event_cache[signature] < self.cache_timeout:
                return True
                
        # Update cache with current time
        self.event_cache[signature] = time.time()
        
        # Clean old cache entries
        current_time = time.time()
        self.event_cache = {k: v for k, v in self.event_cache.items()
                          if current_time - v < self.cache_timeout}
        
        # Also check in existing events
        for event in self.events:
            if (event["event_id"] == event_id and 
                event["user_name"] == username and 
                event["time"] == timestamp_minute):
                return True
                
        return False

    def process_event(self, event):
        """Process a single Windows event"""
        # Extract event ID (mask qualifier bits)
        event_id = event.EventID & 0xFFFF
        
        # Only process login/logout events
        if event_id not in [4624, 4634, 4647]:
            return None
            
        # Extract timestamp - handling pywintypes.datetime correctly
        try:
            # Convert pywintypes.datetime to Python datetime
            if hasattr(event.TimeGenerated, 'timestamp'):
                # If it has timestamp method, use it
                timestamp = datetime.datetime.fromtimestamp(event.TimeGenerated.timestamp())
            else:
                # Otherwise try direct conversion
                timestamp = datetime.datetime(event.TimeGenerated.year, 
                                          event.TimeGenerated.month, 
                                          event.TimeGenerated.day,
                                          event.TimeGenerated.hour,
                                          event.TimeGenerated.minute,
                                          event.TimeGenerated.second)
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
        except Exception as e:
            print(f"Timestamp conversion error: {e}")
            # Fallback to current time if conversion fails
            timestamp = datetime.datetime.now()
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
        
        # Extract username
        username = self._extract_username_from_event(event)
        
        # Check for duplicates
        if self._is_duplicate(event_id, username, timestamp):
            return None
            
        # Get computer name
        computer_name = event.ComputerName.lower()
        
        # Determine event type
        event_type = "Login" if event_id == 4624 else "Logout"
        
        # Get IP address
        ip_address = self._get_local_ip()
        
        # Create event record
        event_record = {
            "event_id": event_id,
            "time": formatted_time,
            "computer_name": computer_name,
            "user_name": username,
            "event_type": event_type,
            "ip_address": ip_address
        }
        
        return event_record

    def monitor_events(self):
        """Monitor Windows security events in real-time"""
        print(f"Starting Windows session monitoring...")
        
        while True:
            h_evt_log = None
            try:
                # Open the event log with proper error handling
                h_evt_log = win32evtlog.OpenEventLog(None, "Security")
                if not h_evt_log:
                    print("Failed to open Security event log")
                    time.sleep(5)
                    continue
                    
                # Get total records to find starting point
                try:
                    total_records = win32evtlog.GetNumberOfEventLogRecords(h_evt_log)
                    oldest_record = win32evtlog.GetOldestEventLogRecord(h_evt_log)
                    
                    # If this is our first run, start from the end
                    if self.last_record_number == 0:
                        self.last_record_number = oldest_record + total_records - 1
                        
                    # Read events from the last seen record forward
                    flags = win32evtlog.EVENTLOG_FORWARDS_READ | win32evtlog.EVENTLOG_SEEK_READ
                    
                    # Read events
                    events = win32evtlog.ReadEventLog(
                        h_evt_log,
                        flags,
                        self.last_record_number
                    )
                except Exception as e:
                    print(f"Error reading event log: {e}")
                    # Make sure we close the handle in case of error
                    if h_evt_log:
                        try:
                            win32evtlog.CloseEventLog(h_evt_log)
                        except:
                            pass
                    time.sleep(2)
                    continue
                
                # Process each event
                for event in events:
                    # Update last record number
                    if event.RecordNumber > self.last_record_number:
                        self.last_record_number = event.RecordNumber
                    
                    # Process the event
                    event_record = self.process_event(event)
                    
                    # If we have a valid event, store it
                    if event_record:
                        self.events.append(event_record)
                        
                        # Save to JSON file
                        with open(self.log_file, 'w') as f:
                            json.dump(self.events, f, indent=4)
                        
                        print(f"Recorded {event_record['event_type']} event for {event_record['user_name']} at {event_record['time']}")
                
                # Properly close the handle after we're done with it
                if h_evt_log:
                    try:
                        win32evtlog.CloseEventLog(h_evt_log)
                    except Exception as e:
                        print(f"Error closing event log handle: {e}")
                
                # Wait for a short time before checking again (1 second)
                time.sleep(1)
                
            except Exception as e:
                print(f"Error monitoring events: {e}")
                # Always make sure to close the handle in case of exceptions
                if h_evt_log:
                    try:
                        win32evtlog.CloseEventLog(h_evt_log)
                    except:
                        pass
                time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    try:
        monitor = WindowsSessionMonitor()
        monitor.monitor_events()
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    except Exception as e:
        print(f"Error: {e}")
