import win32evtlog
import win32evtlogutil
import win32con
import datetime
import json
import time
import requests
import socket
from typing import List, Dict
from requests.exceptions import RequestException

class APIClient:
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
            if self.local_server:
                # Extract host and port from URL
                url_parts = self.api_url.split("://")[1].split("/")[0]
                host, port = url_parts.split(":")
                
                print(f"[*] Checking connection to {host}:{port}")
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, int(port)))
                sock.close()
                
                if result == 0:
                    print("[+] Local server is reachable")
                else:
                    print(f"[!] Local server connection failed with error code: {result}")
                return result == 0
            else:
                response = requests.get(self.api_url, timeout=2)
                return response.status_code == 200
        except Exception as e:
            print(f"[!] Server check failed: {str(e)}")
            return False

    def send_events(self, events: List[Dict]) -> bool:
        """
        Send events to the server
        """
        if not events:
            return True

        try:
            # For local server, use shorter timeout
            timeout = 2 if self.local_server else 5
            
            print(f"[*] Sending {len(events)} events to server...")
            print(f"[*] Server URL: {self.api_url}/events")
            
            response = requests.post(
                f"{self.api_url}/events",
                json={"events": events},
                headers=self.headers,
                timeout=timeout
            )
            
            print(f"[*] Server response status: {response.status_code}")
            print(f"[*] Server response: {response.text}")
            
            response.raise_for_status()
            return True
        except RequestException as e:
            print(f"[!] Failed to send events to server: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"[!] Server response: {e.response.text}")
            return False

class WindowsEventLogger:
    def __init__(self, api_client: APIClient):
        self.server = 'localhost'
        self.logtype = 'Security'
        self.hand = win32evtlog.OpenEventLog(self.server, self.logtype)
        self.flags = win32evtlog.EVENTLOG_FORWARDS_READ|win32evtlog.EVENTLOG_SEQUENTIAL_READ
        self.last_event_time = datetime.datetime.now()
        self.api_client = api_client
        self.offline_events = []
        self.server_status = False
        self.check_interval = 30
        self.last_check = 0

    def check_server(self) -> bool:
        """
        Check server status periodically
        """
        current_time = time.time()
        if current_time - self.last_check >= self.check_interval:
            self.server_status = self.api_client.check_server_availability()
            self.last_check = current_time
        return self.server_status

    def get_new_events(self) -> List[Dict]:
        """
        Get new events since last check
        """
        events = []
        try:
            while True:
                events_batch = win32evtlog.ReadEventLog(self.hand, self.flags, 0)
                if not events_batch:
                    break

                for event in events_batch:
                    try:
                        event_time = event.TimeGenerated
                        if event_time <= self.last_event_time:
                            continue

                        if event.EventID in [4624, 4634, 4647, 4672]:
                            user_name = self._get_username(event)
                            if user_name == "-":
                                continue

                            event_data = {
                                'event_id': event.EventID,
                                'time': event_time.isoformat(),
                                'computer_name': event.ComputerName,
                                'user_name': user_name,
                                'event_type': self._get_event_type(event.EventID)
                            }
                            events.append(event_data)
                            self.last_event_time = event_time
                    except Exception as e:
                        print(f"[!] Error parsing event: {e}")
        except Exception as e:
            print(f"[!] Error reading event log: {e}")

        return events

    def _get_event_type(self, event_id: int) -> str:
        """
        Get human-readable event type
        """
        event_types = {
            4624: "Login",
            4634: "Logout",
            4647: "User initiated logout",
            4672: "Special privileges assigned"
        }
        return event_types.get(event_id, f"Unknown Event ({event_id})")

    def _get_username(self, event) -> str:
        """
        Extract username from event data
        """
        try:
            inserts = event.StringInserts
            if not inserts:
                return "Unknown"

            if event.EventID == 4624 and len(inserts) >= 6:
                return inserts[5]
            elif event.EventID == 4634 and len(inserts) >= 1:
                return inserts[0]
            elif event.EventID == 4647 and len(inserts) >= 1:
                return inserts[0]
            elif event.EventID == 4672 and len(inserts) >= 1:
                return inserts[0]
        except Exception as e:
            print(f"[!] Error extracting username: {e}")

        return "Unknown"

    def process_events(self, events: List[Dict]):
        """
        Process events and send to server
        """
        if not events:
            return

        # Check server status
        server_available = self.check_server()

        # Try to send offline events first if any
        if self.offline_events and server_available:
            print(f"[*] Attempting to send {len(self.offline_events)} offline events...")
            if self.api_client.send_events(self.offline_events):
                print(f"[+] Successfully sent offline events")
                self.offline_events = []
            else:
                print("[!] Failed to send offline events, will retry later")

        # Send new events
        if server_available:
            if not self.api_client.send_events(events):
                print("[!] Failed to send new events, storing offline")
                self.offline_events.extend(events)
            else:
                print(f"[+] Successfully sent new events")
        else:
            print("[!] Server not available, storing events offline")
            self.offline_events.extend(events)

    def close(self):
        win32evtlog.CloseEventLog(self.hand) 