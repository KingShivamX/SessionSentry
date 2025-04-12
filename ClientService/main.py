from logger import WindowsEventLogger, APIClient
import json
import time
import sys
import traceback
import os
import datetime
import shutil

def main():
    # Configuration
    API_URL = "http://localhost:3000/api"  # Local server URL
    API_KEY = "2123456789"  # API key
    LOCAL_MODE = True  # Set to True to run without server dependency
    DEBUG_MODE = True  # Enable more detailed logging

    print("[*] Initializing SessionSentry...")
    
    # Create empty login_events.json if it doesn't exist or is corrupted
    try:
        with open('login_events.json', 'r') as f:
            json.load(f)  # Test if valid JSON
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[*] Creating new login_events.json file (Error: {str(e)})")
        with open('login_events.json', 'w') as f:
            json.dump([], f)
    
    # Ensure logs directory exists
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"[*] Created logs directory at {logs_dir}")
    
    # Only attempt server connection if not in local mode
    if not LOCAL_MODE:
        print(f"[*] Connecting to local server at {API_URL}")
        api_client = APIClient(API_URL, API_KEY)
        
        # Initial server check
        if not api_client.check_server_availability():
            print("[!] Warning: Could not connect to local server initially")
            print("[*] Will continue monitoring and retry sending when server is available")
    else:
        print("[*] Running in LOCAL MODE - no server connection required")
        api_client = None
    
    try:
        logger = WindowsEventLogger(api_client)
        
        print("\n[*] Starting real-time security event monitoring...")
        print("[*] Press Ctrl+C to stop\n")
        
        try:
            while True:
                try:
                    # Clear any existing handles and reconnect if needed
                    if not logger.hand:
                        logger.reconnect()
                    
                    # Get new events with better error handling
                    events = logger.get_new_events()
                    
                    if events:
                        print(f"\n[+] {len(events)} new security events detected:")
                except Exception as e:
                    print(f"[!] Error getting events: {e}")
                    time.sleep(5)  # Wait longer on error
                    continue
                
                if events:

                    
                    # Current date for organizing output
                    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                    print(f"\n=== SESSION EVENTS FOR {current_date} ===")
                    
                    # Display events with more details
                    for event in events:
                        time_obj = datetime.datetime.fromisoformat(event['time'])
                        formatted_time = time_obj.strftime("%Y-%m-%d %H:%M")
                        
                        print(f"\n[{formatted_time}] {event['event_type']}")
                        print(f"User: {event['user_name']}")
                        print(f"Computer: {event['computer_name']}")
                        print(f"Event ID: {event['event_id']}")
                        print("-" * 50)
                    
                    # Process and send events if api_client exists
                    if api_client:
                        logger.process_events(events)
                    
                    # Save to local JSON file with formatted timestamps
                    try:
                        with open('login_events.json', 'r') as f:
                            existing_events = json.load(f)
                    except (FileNotFoundError, json.JSONDecodeError):
                        existing_events = []
                    
                    # Format the timestamps before saving
                    for event in events:
                        # Create a copy of the event with formatted time
                        event_copy = event.copy()
                        
                        # Check if the time is already properly formatted
                        if 'T' in event['time']: # ISO format contains T
                            time_obj = datetime.datetime.fromisoformat(event['time'])
                            event_copy['time'] = time_obj.strftime("%Y-%m-%d %H:%M")
                        else:
                            # If already formatted, keep it as is
                            event_copy['time'] = event['time']
                            
                        existing_events.append(event_copy)
                    
                    # Write events to JSON immediately and flush to ensure instant writing
                    with open('login_events.json', 'w') as f:
                        json.dump(existing_events, f, indent=4)
                        f.flush()
                        os.fsync(f.fileno())
                        
                    # Also save to a daily log file for easier review
                    log_dir = "logs"
                    if not os.path.exists(log_dir):
                        os.makedirs(log_dir)
                        
                    log_file = os.path.join(log_dir, f"events_{current_date}.log")
                    with open(log_file, 'a') as f:
                        for event in events:
                            # Format time if needed
                            if 'T' in event['time']: # ISO format contains T
                                time_obj = datetime.datetime.fromisoformat(event['time'])
                                formatted_time = time_obj.strftime("%Y-%m-%d %H:%M")
                            else:
                                formatted_time = event['time']
                                
                            f.write(f"Time: {formatted_time}\n")
                            f.write(f"Event Type: {event['event_type']}\n")
                            f.write(f"User: {event['user_name']}\n")
                            f.write(f"Computer: {event['computer_name']}\n")
                            f.write(f"IP Address: {event.get('ip_address', '-')}\n")
                            f.write(f"Event ID: {event['event_id']}\n")
                            f.write("-" * 50 + "\n\n")
                            
                            # Flush the file to ensure it's written immediately
                            f.flush()
                            os.fsync(f.fileno())
                
                # Check for new events every second
                try:
                    # Force check for any missed events every 30 seconds
                    current_time = time.time()
                    if hasattr(logger, 'last_full_check') and (current_time - logger.last_full_check) > 30:
                        print("[*] Performing periodic full event check...")
                        logger.flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
                        logger.get_new_events(refresh_mode=True)  # Special refresh mode
                        logger.flags = win32evtlog.EVENTLOG_FORWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
                        logger.last_full_check = current_time
                except Exception as check_error:
                    print(f"[!] Error during periodic check: {check_error}")
                
                # Sleep between checks
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n[*] Stopping event monitoring...")
        finally:
            logger.close()
    
    except Exception as e:
        if "A required privilege is not held by the client" in str(e):
            print("\n[!] ERROR: Administrator privileges required")
            print("[!] SessionSentry needs to access Windows Security logs, which requires elevated privileges")
            print("[!] Please restart this program by:")
            print("    1. Close this window")
            print("    2. Right-click on PowerShell/Command Prompt")
            print("    3. Select 'Run as administrator'")
            print("    4. Navigate to the same directory")
            print("    5. Run 'python main.py' again\n")
        else:
            print(f"\n[!] An unexpected error occurred: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    main() 