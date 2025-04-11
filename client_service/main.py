from logger import WindowsEventLogger, APIClient
import json
import time

def main():
    # Configuration
    API_URL = "http://localhost:3000/api"  # Local server URL
    API_KEY = "2123456789"  # API key

    print("[*] Initializing SessionSentry...")
    print(f"[*] Connecting to local server at {API_URL}")
    
    api_client = APIClient(API_URL, API_KEY)
    logger = WindowsEventLogger(api_client)
    
    # Initial server check
    if not api_client.check_server_availability():
        print("[!] Warning: Could not connect to local server initially")
        print("[*] Will continue monitoring and retry sending when server is available")
    
    print("\n[*] Starting real-time security event monitoring...")
    print("[*] Press Ctrl+C to stop\n")
    
    try:
        while True:
            events = logger.get_new_events()
            
            if events:
                print(f"\n[+] {len(events)} new security events detected:")
                for event in events:
                    print(f"\nTime: {event['time']}")
                    print(f"Event Type: {event['event_type']}")
                    print(f"User: {event['user_name']}")
                    print(f"Computer: {event['computer_name']}")
                    print("-" * 50)
                
                # Process and send events
                logger.process_events(events)
                
                # Save to local JSON file as backup
                try:
                    with open('login_events.json', 'r') as f:
                        existing_events = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    existing_events = []
                
                existing_events.extend(events)
                
                with open('login_events.json', 'w') as f:
                    json.dump(existing_events, f, indent=4)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n[*] Stopping event monitoring...")
    finally:
        logger.close()

if __name__ == "__main__":
    main() 