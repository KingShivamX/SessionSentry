import json
import time
import os
import argparse
from logger import WindowsEventLogger, APIClient
from security_analytics import SecurityAnalytics

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from file or use defaults"""
    default_config = {
        "api_url": "http://localhost:3000/api",
        "api_key": "2123456789",
        "backup_file": "login_events.json",
        "analytics_file": "security_analytics.json",
        "check_interval": 1,
        "server_check_interval": 30
    }
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults to ensure all required keys exist
                return {**default_config, **config}
    except Exception as e:
        print(f"[!] Error loading config file: {e}")
        print("[*] Using default configuration")
    
    return default_config

def save_events_to_backup(events: list, backup_file: str):
    """Save events to backup file with error handling"""
    try:
        if os.path.exists(backup_file):
            with open(backup_file, 'r') as f:
                existing_events = json.load(f)
        else:
            existing_events = []
        
        existing_events.extend(events)
        
        with open(backup_file, 'w') as f:
            json.dump(existing_events, f, indent=4)
    except Exception as e:
        print(f"[!] Error saving to backup file: {e}")

def print_analytics_report(analytics: SecurityAnalytics):
    """Print a formatted analytics report"""
    report = analytics.get_analytics_report()
    
    print("\n=== Security Analytics Report ===")
    print(f"\nSummary:")
    print(f"Total Login Attempts: {report['summary']['total_attempts']}")
    print(f"Successful Logins: {report['summary']['successful_logins']}")
    print(f"Failed Logins: {report['summary']['failed_logins']}")
    print(f"Lockout Events: {report['summary']['lockout_events']}")
    print(f"Success Rate: {report['summary']['success_rate']:.2f}%")
    
    print(f"\nLast 24 Hours:")
    print(f"Attempts: {report['last_24_hours']['attempts']}")
    print(f"Successes: {report['last_24_hours']['successes']}")
    print(f"Failures: {report['last_24_hours']['failures']}")
    
    print("\nTop Users by Activity:")
    for user, stats in report['top_users']:
        print(f"\nUser: {user}")
        print(f"Total Attempts: {stats['total_attempts']}")
        print(f"Successful Logins: {stats['successful_logins']}")
        print(f"Failed Logins: {stats['failed_logins']}")
        if stats['ip_addresses']:
            print("IP Addresses Used:")
            for ip in stats['ip_addresses']:
                print(f"  - {ip}")
    
    print("\nSuspicious IP Addresses:")
    for ip, count in report['suspicious_ips']:
        ip_details = report['ip_details'].get(ip, {})
        print(f"\nIP: {ip}")
        print(f"Total Attempts: {count}")
        print(f"First Seen: {ip_details.get('first_seen', 'N/A')}")
        print(f"Last Activity: {ip_details.get('last_activity', 'N/A')}")
        print(f"Successful Logins: {ip_details.get('successful_logins', 0)}")
        print(f"Failed Logins: {ip_details.get('failed_logins', 0)}")
        
        if ip_details.get('associated_users'):
            print("\nAssociated Users:")
            for user in ip_details['associated_users']:
                print(f"  - {user}")
        
        if ip_details.get('associated_computers'):
            print("\nAssociated Computers:")
            for computer in ip_details['associated_computers']:
                print(f"  - {computer}")
        
        if ip_details.get('recent_events'):
            print("\nRecent Activity:")
            for event in ip_details['recent_events'][-5:]:  # Show last 5 events
                print(f"  - Time: {event['time']}")
                print(f"    Type: {event['type']}")
                print(f"    User: {event['user']}")
                print(f"    Computer: {event['computer']}")
                print(f"    Status: {event['status']}")
        
        print("-" * 50)
    
    print("\n" + "="*50)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='SessionSentry Security Monitor')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--api-url', type=str, help='Override API URL')
    parser.add_argument('--api-key', type=str, help='Override API key')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config if args.config else "config.json")
    
    # Override config with command line arguments if provided
    if args.api_url:
        config['api_url'] = args.api_url
    if args.api_key:
        config['api_key'] = args.api_key
    
    print("[*] Initializing SessionSentry...")
    print(f"[*] Connecting to server at {config['api_url']}")
    
    api_client = APIClient(config['api_url'], config['api_key'])
    logger = WindowsEventLogger(api_client)
    analytics = SecurityAnalytics(config['api_url'], config['api_key'])
    
    # Load existing analytics if available
    analytics.load_analytics(config['analytics_file'])
    
    # Initial server check
    if not api_client.check_server_availability():
        print("[!] Warning: Could not connect to server initially")
        print("[*] Will continue monitoring and retry sending when server is available")
    
    print("\n[*] Starting real-time security event monitoring...")
    print("[*] Press Ctrl+C to stop\n")
    
    last_analytics_save = time.time()
    last_analytics_send = time.time()
    analytics_save_interval = 300  # Save analytics every 5 minutes
    analytics_send_interval = 300  # Send analytics to API every 5 minutes
    
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
                    
                    # Update analytics
                    analytics.update_stats(event)
                
                # Process and send events
                logger.process_events(events)
                
                # Save to backup file
                save_events_to_backup(events, config['backup_file'])
                
                # Print analytics report
                print_analytics_report(analytics)
            
            current_time = time.time()
            
            # Periodically save analytics to file
            if current_time - last_analytics_save >= analytics_save_interval:
                analytics.save_analytics(config['analytics_file'])
                last_analytics_save = current_time
            
            # Periodically send analytics to API
            if current_time - last_analytics_send >= analytics_send_interval:
                if analytics.send_analytics_to_api():
                    last_analytics_send = current_time
            
            time.sleep(config['check_interval'])
            
    except KeyboardInterrupt:
        print("\n[*] Stopping event monitoring...")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
    finally:
        # Save final analytics before exiting
        analytics.save_analytics(config['analytics_file'])
        # Send final analytics to API
        analytics.send_analytics_to_api()
        logger.close()
        print("[*] Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main() 