import win32evtlog
import win32evtlogutil
import win32con
import win32security
import datetime
import json
import time
import requests
import socket
import os
import traceback
import re
from typing import List, Dict
from requests.exceptions import RequestException

class APIClient:
    """
    API client for sending events to the server
    """
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': api_key,
            'Accept': 'application/json'
        }
        self.local_server = "localhost" in api_url or "127.0.0.1" in api_url
        
    def check_server_availability(self) -> bool:
        """
        Check if the server is available
        """
        try:
            hostname = self.api_url.split('/')[2].split(':')[0]
            # For localhost, use a faster check
            if self.local_server:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                port = 80
                if ':' in self.api_url.split('/')[2]:
                    port = int(self.api_url.split('/')[2].split(':')[1])
                result = sock.connect_ex((hostname, port))
                sock.close()
                return result == 0
            else:
                # For remote servers, use a ping request
                response = requests.get(f"{self.api_url}/ping", headers=self.headers, timeout=3)
                return response.status_code == 200
        except (socket.error, RequestException) as e:
            return False
    
    def send_events(self, events: List[Dict]) -> bool:
        """
        Send events to the server
        """
        if not events:
            return True
            
        try:
            payload = json.dumps({'events': events})
            response = requests.post(f"{self.api_url}/events", headers=self.headers, data=payload, timeout=5)
            return response.status_code == 200
        except RequestException:
            return False

class WindowsEventLogger:
    """
    Windows Event Log reader for login events
    """
    def __init__(self, api_client=None):
        self.server = 'localhost'
        self.logtype = 'Security'
        self.hand = None
        self.last_event_record = 0  # Track the last event record we processed
        
        try:
            # Open the event log
            self.hand = win32evtlog.OpenEventLog(self.server, self.logtype)
            
            # Use newest read to get latest events first
            self.flags = win32evtlog.EVENTLOG_FORWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
            # Alternatively, you can use this to always get newest events:
            # self.flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
            
            # Current time to start monitoring from
            self.last_event_time = datetime.datetime.now() - datetime.timedelta(hours=1)  # Look back just 1 hour
            
            # Client settings
            self.api_client = api_client
            self.offline_events = []
            self.server_status = False
            self.check_interval = 30
            self.last_check = 0
            self.initialized = True
            self.local_mode = api_client is None
            
            # Get info about the event log
            oldest_record = win32evtlog.GetOldestEventLogRecord(self.hand)
            num_records = win32evtlog.GetNumberOfEventLogRecords(self.hand)
            
            # Calculate the newest record number
            if num_records > 0:
                self.last_event_record = oldest_record + num_records - 1
                # Start from near the end to avoid processing too much history
                self.last_event_record = max(oldest_record, self.last_event_record - 100)
            else:
                self.last_event_record = 0
            
            # Print startup message
            print("[*] Monitoring for login events (4624) and logout events (4634, 4647)")
            print(f"[*] Windows Event Log contains {num_records} total events")
            print(f"[*] Starting monitoring from most recent events")
            
        except win32evtlog.error as e:
            if e.winerror == 1314:  # ERROR_PRIVILEGE_NOT_HELD
                print("\n[!] ERROR: Administrator privileges required to access Windows Security logs")
                print("[!] Please run this script as Administrator (right-click Python/terminal, 'Run as administrator')")
                self.initialized = False
            else:
                print(f"\n[!] ERROR: Failed to open Windows Event Log: {str(e)}")
                self.initialized = False
            raise  # Re-raise to stop execution
    
    def check_server(self) -> bool:
        """
        Check server status periodically
        """
        if self.local_mode:
            return False  # No server in local mode
            
        current_time = time.time()
        if current_time - self.last_check > self.check_interval:
            self.server_status = self.api_client.check_server_availability()
            self.last_check = current_time
        return self.server_status
    
    def get_new_events(self) -> List[Dict]:
        """
        Get new events since last check
        """
        events = []
        processed = 0
        
        # Track event counts for debugging
        event_counts = {4624: 0, 4625: 0, 4634: 0, 4647: 0, 4648: 0, 4672: 0}
        
        try:
            # Only open the event log if handle is not valid
            if not self.hand:
                try:
                    self.hand = win32evtlog.OpenEventLog(self.server, self.logtype)
                    print("[*] Opened Windows Event Log")
                except Exception as e:
                    print(f"[!] Error opening event log: {e}")
                    return events
            
            # Get current event log state
            try:
                oldest_record = win32evtlog.GetOldestEventLogRecord(self.hand)
                num_records = win32evtlog.GetNumberOfEventLogRecords(self.hand)
                
                if num_records > 0:
                    newest_record = oldest_record + num_records - 1
                    print(f"[DEBUG] Event log state: oldest={oldest_record}, newest={newest_record}, count={num_records}")
                else:
                    newest_record = 0
            except Exception as e:
                print(f"[!] Error getting event log info: {e}")
                self.hand = None
                return events
            
            # Read event batches
            try:
                max_events_to_check = 500
                events_checked = 0
                
                # Read events in batches
                while events_checked < max_events_to_check:
                    try:
                        events_batch = win32evtlog.ReadEventLog(self.hand, self.flags, 0)
                        if not events_batch:
                            break
                            
                        # Process each event in the batch
                        for event in events_batch:
                            processed += 1
                            events_checked += 1
                            
                            # Get the actual event ID by masking off qualifiers
                            event_id = event.EventID & 0xFFFF
                            
                            try:
                                # Skip already processed events
                                if hasattr(event, 'RecordNumber') and event.RecordNumber <= self.last_event_record:
                                    continue
                                
                                # Update last event record number
                                if hasattr(event, 'RecordNumber') and event.RecordNumber > self.last_event_record:
                                    self.last_event_record = event.RecordNumber
                                
                                # Check if this is a login/logout event we care about
                                if event_id in [4624, 4625, 4634, 4647, 4648, 4672]:
                                    # Increment count for this event type
                                    event_counts[event_id] += 1
                                    
                                    # Get username from the event
                                    user_name = self._get_username(event)
                                    
                                    # Skip empty usernames or system accounts we don't care about
                                    if user_name in ["-", "Unknown"]:
                                        continue
                                    
                                    # Print raw event info for debugging
                                    record_num = event.RecordNumber if hasattr(event, 'RecordNumber') else 'N/A'
                                    print(f"[RAW EVENT] ID: {event_id}, Record: {record_num}, User: {user_name}")
                                    
                                    # Convert SID to username if needed
                                    if user_name.startswith("S-1-5-"):
                                        user_name = self._sid_to_username(user_name)
                                    
                                    # Only include login and logout events in our list
                                    if event_id in [4624, 4634, 4647]:
                                        # Format date/time in a readable format
                                        human_time = event.TimeGenerated.strftime("%Y-%m-%d %H:%M")
                                        
                                        # Get human-friendly event type
                                        event_type = self._get_event_type(event_id)
                                        
                                        # Extract IP address from the event
                                        ip_address = self._get_ip_address(event)
                                        
                                        # Create event data record with IP address
                                        event_data = {
                                            "event_id": event_id,
                                            "time": human_time,
                                            "computer_name": event.ComputerName.lower() if event.ComputerName else "-",
                                            "user_name": user_name,
                                            "event_type": event_type,
                                            "ip_address": ip_address
                                        }
                                        
                                        # Print detailed event info
                                        print(f"[EVENT] {event_type}: {user_name} on {event_data['computer_name']} at {human_time}")
                                        
                                        # Add to events list
                                        events.append(event_data)
                                        
                                        # Only update last event time for login events
                                        if event_id not in [4634, 4647]:
                                            self.last_event_time = event.TimeGenerated
                                    
                            except Exception as e:
                                print(f"[!] Error processing event: {e}")
                                traceback.print_exc()
                                
                    except Exception as batch_error:
                        print(f"[!] Error reading event batch: {batch_error}")
                        break
                        
            except Exception as e:
                print(f"[!] Error reading event log: {e}")
        
        except Exception as e:
            print(f"[!] Error in get_new_events: {e}")
        
        # Print debug summary
        print(f"[DEBUG] Processed {processed} events. Found: Login={event_counts[4624]}, Logout={event_counts[4634]}, User-Logout={event_counts[4647]}, Admin={event_counts[4672]}")
        
        return events
    
    def _sid_to_username(self, sid: str) -> str:
        """
        Convert a SID to a username
        """
        try:
            name, domain, acct_type = win32security.LookupAccountSid(None, win32security.ConvertStringSidToSid(sid))
            if name and domain:
                return f"{domain}\\{name}"
            return sid
        except Exception as e:
            print(f"[!] Error converting SID to username: {e}")
            return sid
    
    def _get_event_type(self, event_id: int) -> str:
        """
        Get human-readable event type
        """
        # Make sure we're using the actual event ID by masking off qualifiers
        event_id = event_id & 0xFFFF
        
        event_types = {
            4624: "Login",
            4625: "Failed login attempt",
            4634: "Logout",
            4647: "User initiated logout",
            4648: "Logon with explicit credentials",
            4672: "Special privileges assigned"
        }
        return event_types.get(event_id, f"Unknown Event ({event_id})")
    
    def _get_username(self, event) -> str:
        """
        Extract username from event data
        """
        try:
            # Get actual event ID by masking off qualifiers
            event_id = event.EventID & 0xFFFF
            
            # Get the string inserts from the event
            inserts = event.StringInserts
            if not inserts:
                return "Unknown"
            
            # Username position depends on event type
            if event_id == 4624 and len(inserts) >= 6:  # Login
                return inserts[5]  # Account Name target
            elif event_id == 4625 and len(inserts) >= 6:  # Failed login
                return inserts[5]  # Account Name target
            elif event_id == 4634 and len(inserts) >= 1:  # Logout
                return inserts[0]  # Account Name
            elif event_id == 4647 and len(inserts) >= 1:  # User initiated logout
                return inserts[0]  # Account Name
            elif event_id == 4648 and len(inserts) >= 1:  # Explicit creds
                return inserts[0]  # Account Name
            elif event_id == 4672 and len(inserts) >= 1:  # Special privileges
                return inserts[0]  # Account Name
            
            # Debug - print inserts for unknown events
            print(f"[DEBUG] Event {event_id} inserts: {inserts}")
            
        except Exception as e:
            print(f"[!] Error extracting username: {e}")
        
        return "Unknown"
    
    def _get_ip_address(self, event) -> str:
        """
        Extract IP address from event data
        """
        try:
            # Get the actual event ID
            event_id = event.EventID & 0xFFFF
            
            # Get string inserts
            inserts = event.StringInserts
            if not inserts:
                return "-"
                
            # For login events (4624), the IP is usually in position 18
            if event_id == 4624 and len(inserts) >= 19:
                ip = inserts[18]  # Network Address field
                if ip and ip != "-" and ip != "::1" and not ip.startswith("127."):
                    return ip
            
            # Try to extract IP addresses using regex
            event_data = str(inserts)
            ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'  # IPv4 pattern
            ips = re.findall(ip_pattern, event_data)
            if ips:
                # Filter out localhost IPs
                valid_ips = [ip for ip in ips if not ip.startswith("127.")]
                if valid_ips:
                    return valid_ips[0]
            
            # If no IP found in event, use the local machine's IP
            try:
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                return local_ip
            except Exception:
                pass
                
        except Exception as e:
            print(f"[!] Error extracting IP address: {e}")
            
        return "-"
            
    def _filter_event(self, event) -> bool:
        """
        Filter events based on user or other criteria
        For example, you might want to exclude system accounts
        """
        try:
            event_id = event.EventID & 0xFFFF
            user_name = self._get_username(event)
            
            # Allow logout events for all users
            if event_id in [4634, 4647]:
                return True
                
            # For login events, filter out some system accounts
            if event_id == 4624:
                # Skip system accounts like ANONYMOUS LOGON
                if user_name in ["ANONYMOUS LOGON", "LOCAL SERVICE", "NETWORK SERVICE"]:
                    return False
            
            return True
            
        except Exception as e:
            print(f"[!] Error in filter_event: {e}")
            return False
    
    def process_events(self, events: List[Dict]) -> None:
        """
        Process events and send to server
        """
        if not events:
            return
            
        # Create daily log directory if it doesn't exist
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        # Get today's date for the log file
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(logs_dir, f"events_{today}.log")
        
        # Read existing events from the JSON file
        json_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "login_events.json")
        existing_events = []
        
        try:
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    existing_events = json.load(f)
        except Exception as e:
            print(f"[!] Error reading events file: {e}")
            existing_events = []
        
        # Add new events to the list in chronological order
        # Sort events by time to ensure proper sequence
        events_with_time = []
        for event in events:
            # Convert time string to datetime object for sorting if needed
            if isinstance(event['time'], str):
                try:
                    if 'T' in event['time']:
                        dt = datetime.datetime.fromisoformat(event['time'])
                    else:
                        dt = datetime.datetime.strptime(event['time'], "%Y-%m-%d %H:%M")
                    events_with_time.append((dt, event))
                except (ValueError, TypeError):
                    # If time parsing fails, just use current time
                    events_with_time.append((datetime.datetime.now(), event))
            else:
                events_with_time.append((datetime.datetime.now(), event))
        
        # Sort events by time
        events_with_time.sort(key=lambda x: x[0])
        
        # Extend the existing list with sorted events
        for _, event in events_with_time:
            existing_events.append(event)
        
        # Write updated events to the JSON file immediately
        try:
            with open(json_file, 'w') as f:
                json.dump(existing_events, f, indent=4)
                f.flush()
                os.fsync(f.fileno())  # Force disk write for instant results
        except Exception as e:
            print(f"[!] Error writing events file: {e}")
        
        # Also write to the daily log file - write events in the same sorted order
        try:
            with open(log_file, 'a') as f:
                for _, event in events_with_time:
                    f.write(f"Time: {event['time']}\n")
                    f.write(f"Event Type: {event['event_type']}\n")
                    f.write(f"User: {event['user_name']}\n")
                    f.write(f"Computer: {event['computer_name']}\n")
                    f.write(f"IP Address: {event['ip_address']}\n")
                    f.write(f"Event ID: {event['event_id']}\n")
                    f.write("-" * 50 + "\n\n")
                    # Flush after each event for immediate writing
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as e:
            print(f"[!] Error writing log file: {e}")
        
        # If API client is available and server is reachable, send events
        if self.api_client and self.check_server():
            print("[*] Sending events to server")
            success = self.api_client.send_events(events)
            if success:
                print("[*] Events sent successfully")
            else:
                print("[!] Server not available, storing events offline")
                self.offline_events.extend(events)
    
    def close(self):
        """
        Close the event log handle
        """
        try:
            if hasattr(self, 'hand') and self.hand:
                try:
                    win32evtlog.CloseEventLog(self.hand)
                except Exception as e:
                    print(f"[!] Error closing event log: {e}")
                finally:
                    # Always set the handle to None to avoid repeated close attempts
                    self.hand = None
        except Exception as e:
            print(f"[!] Error in close method: {e}")
