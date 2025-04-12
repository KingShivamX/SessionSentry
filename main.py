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
import requests
from pathlib import Path

class WindowsSessionMonitor:
    def _init_(self):
        self.log_file = os.path.join(os.path.dirname(os.path.abspath(_file_)), "login_events.json")
        self.event_cache = {}  # Cache for deduplication
        self.cache_timeout = 60  # Seconds to keep events in cache
        self.last_record_number = 0
        
        # API Configuration
        self.api_url = "https://sessionsentryserver.onrender.com/api/events"  # Change this to your actual API endpoint
        # self.api_url = "http://localhost:3000/api/events"  # Change this to your actual API endpoint
        self.api_retry_count = 3
        self.api_retry_delay = 2  # seconds
        
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

        # Set up security privileges
        self._setup_security_privileges()

    def _setup_security_privileges(self):
        """Set up required security privileges for event log access"""
        try:
            # Get the current process token
            ph = win32security.GetCurrentProcess()
            th = win32security.OpenProcessToken(ph, win32con.TOKEN_ADJUST_PRIVILEGES | win32con.TOKEN_QUERY)
            
            # Enable all required privileges
            privileges = [
                win32security.LookupPrivilegeValue(None, win32security.SE_SECURITY_NAME),
                win32security.LookupPrivilegeValue(None, win32security.SE_SYSTEMTIME_NAME),
                win32security.LookupPrivilegeValue(None, win32security.SE_BACKUP_NAME),
                win32security.LookupPrivilegeValue(None, win32security.SE_RESTORE_NAME)
            ]
            
            for privilege in privileges:
                try:
                    win32security.AdjustTokenPrivileges(
                        th,
                        0,
                        [(privilege, win32security.SE_PRIVILEGE_ENABLED)]
                    )
                except Exception as e:
                    print(f"Warning: Could not enable privilege: {e}")
            
            # Verify privileges were set
            for privilege in privileges:
                try:
                    privs = win32security.GetTokenInformation(th, win32security.TokenPrivileges)
                    for priv in privs:
                        if priv[0] == privilege:
                            if priv[1] & win32security.SE_PRIVILEGE_ENABLED:
                                print(f"Successfully enabled privilege: {win32security.LookupPrivilegeName(None, privilege)}")
                            else:
                                print(f"Warning: Privilege not enabled: {win32security.LookupPrivilegeName(None, privilege)}")
                except Exception as e:
                    print(f"Warning: Could not verify privilege: {e}")
                    
        except Exception as e:
            print(f"Error setting up security privileges: {e}")
            print("The application might not have sufficient permissions to read the Security event log.")
            print("Please ensure the application is running with administrator privileges.")
            print("You may need to run this script from an elevated command prompt.")

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
        return f"{event_id}{username}{timestamp_minute}"
        
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

    def _send_event_to_api(self, event_data):
        """Send event data to the API endpoint with retry logic"""
        # Format the event data to match server expectations
        formatted_event = {
            "events": [{
                "event_id": event_data["event_id"],
                "time": event_data["time"],
                "computer_name": event_data["computer_name"],
                "user_name": event_data["user_name"],
                "event_type": event_data["event_type"],
                "ip_address": event_data["ip_address"],
                "status": "success" if event_data["event_type"] == "Login" else "logout"
            }]
        }

        for attempt in range(self.api_retry_count):
            try:
                response = requests.post(
                    self.api_url,
                    json=formatted_event,
                    headers={'Content-Type': 'application/json'},
                    timeout=5  # 5 second timeout
                )
                
                if response.status_code == 200:
                    print(f"Successfully sent event to API: {event_data['event_type']} for {event_data['user_name']}")
                    return True
                else:
                    print(f"API request failed with status code {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"API request failed (attempt {attempt + 1}/{self.api_retry_count}): {str(e)}")
                if attempt < self.api_retry_count - 1:
                    time.sleep(self.api_retry_delay)
                continue
                
        print("Failed to send event to API after all retry attempts")
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
            "ip_address": ip_address,
            "status": "success" if event_type == "Login" else "logout"
        }
        
        return event_record

    def monitor_events(self):
        """Monitor Windows security events in real-time"""
        print(f"Starting Windows session monitoring...")
        print("Checking event log access permissions...")
        
        # Test event log access before starting the main loop
        try:
            test_handle = win32evtlog.OpenEventLog(None, "Security")
            if test_handle:
                print("Successfully accessed Security event log")
                win32evtlog.CloseEventLog(test_handle)
            else:
                print("Failed to access Security event log")
        except Exception as e:
            print(f"Error accessing Security event log: {e}")
            print("Please ensure you have administrator privileges and the Security event log is accessible.")
            print("You may need to run this script from an elevated command prompt.")
            return
        
        while True:
            h_evt_log = None
            try:
                # Open the event log with proper error handling
                try:
                    h_evt_log = win32evtlog.OpenEventLog(None, "Security")
                    if not h_evt_log:
                        print("Failed to open Security event log. Please ensure you have administrator privileges.")
                        time.sleep(5)
                        continue
                except Exception as e:
                    print(f"Error opening Security event log: {e}")
                    print("Please ensure you have administrator privileges and the Security event log is accessible.")
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
                    print("Please ensure you have administrator privileges.")
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
                    
                    # If we have a valid event, store it and send to API
                    if event_record:
                        self.events.append(event_record)
                        
                        # Save to JSON file
                        with open(self.log_file, 'w') as f:
                            json.dump(self.events, f, indent=4)
                        
                        print(f"Recorded {event_record['event_type']} event for {event_record['user_name']} at {event_record['time']}")
                        
                        # Send event to API
                        self._send_event_to_api(event_record)
                
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

if __name__ == "_main_":
    try:
        monitor = WindowsSessionMonitor()
        monitor.monitor_events()
    except KeyboardInterrupt:
        print("Monitoring stopped by user")
    except Exception as e:
        print(f"Error: {e}")