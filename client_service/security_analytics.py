from typing import Dict, List
import json
from datetime import datetime, timedelta
import os
import requests

class SecurityAnalytics:
    def __init__(self, api_url: str = None, api_key: str = None):
        self.stats = {
            "total_attempts": 0,
            "successful_logins": 0,
            "failed_logins": 0,
            "lockout_events": 0,
            "user_activity": {},  # Track per-user activity
            "hourly_activity": {},  # Track activity by hour
            "ip_addresses": {},  # Track IP addresses
            "ip_details": {},  # Track detailed IP information
            "last_24_hours": {
                "attempts": 0,
                "successes": 0,
                "failures": 0
            }
        }
        self.events_history = []
        self.api_url = api_url
        self.api_key = api_key

    def update_stats(self, event: Dict):
        """Update statistics based on new event"""
        self.events_history.append(event)
        self.stats["total_attempts"] += 1
        
        # Update user activity
        username = event.get("user_name", "unknown")
        if username not in self.stats["user_activity"]:
            self.stats["user_activity"][username] = {
                "total_attempts": 0,
                "successful_logins": 0,
                "failed_logins": 0,
                "ip_addresses": set()  # Track IPs used by this user
            }
        
        user_stats = self.stats["user_activity"][username]
        user_stats["total_attempts"] += 1
        
        # Update event type specific stats
        event_type = event.get("event_type", "")
        if "success" in event_type.lower():
            self.stats["successful_logins"] += 1
            user_stats["successful_logins"] += 1
            self.stats["last_24_hours"]["successes"] += 1
        elif "failure" in event_type.lower():
            self.stats["failed_logins"] += 1
            user_stats["failed_logins"] += 1
            self.stats["last_24_hours"]["failures"] += 1
        elif "lockout" in event_type.lower():
            self.stats["lockout_events"] += 1
        
        # Update hourly activity
        event_time = datetime.fromisoformat(event.get("time", ""))
        hour = event_time.strftime("%Y-%m-%d %H:00")
        if hour not in self.stats["hourly_activity"]:
            self.stats["hourly_activity"][hour] = 0
        self.stats["hourly_activity"][hour] += 1
        
        # Update IP tracking with detailed information
        ip_address = event.get("ip_address", "unknown")
        if ip_address not in self.stats["ip_addresses"]:
            self.stats["ip_addresses"][ip_address] = 0
            self.stats["ip_details"][ip_address] = {
                "total_attempts": 0,
                "successful_logins": 0,
                "failed_logins": 0,
                "last_activity": event_time,
                "associated_users": set(),
                "associated_computers": set(),
                "first_seen": event_time,
                "recent_events": []  # Store recent events for this IP
            }
        
        self.stats["ip_addresses"][ip_address] += 1
        ip_details = self.stats["ip_details"][ip_address]
        ip_details["total_attempts"] += 1
        ip_details["last_activity"] = max(ip_details["last_activity"], event_time)
        ip_details["associated_users"].add(username)
        ip_details["associated_computers"].add(event.get("computer_name", "unknown"))
        user_stats["ip_addresses"].add(ip_address)  # Track IP for user
        
        # Store recent event
        ip_details["recent_events"].append({
            "time": event_time.isoformat(),
            "type": event_type,
            "user": username,
            "computer": event.get("computer_name", "unknown"),
            "status": "success" if "success" in event_type.lower() else "failure"
        })
        # Keep only last 10 events
        ip_details["recent_events"] = ip_details["recent_events"][-10:]
        
        if "success" in event_type.lower():
            ip_details["successful_logins"] += 1
        elif "failure" in event_type.lower():
            ip_details["failed_logins"] += 1
        
        # Update last 24 hours stats
        self.stats["last_24_hours"]["attempts"] += 1
        
        # Clean up old data
        self._cleanup_old_data()

    def _cleanup_old_data(self):
        """Remove data older than 24 hours"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Clean up hourly activity
        self.stats["hourly_activity"] = {
            hour: count for hour, count in self.stats["hourly_activity"].items()
            if datetime.strptime(hour, "%Y-%m-%d %H:00") > cutoff_time
        }
        
        # Reset last 24 hours stats
        self.stats["last_24_hours"] = {
            "attempts": 0,
            "successes": 0,
            "failures": 0
        }
        
        # Recalculate last 24 hours stats
        for event in self.events_history:
            event_time = datetime.fromisoformat(event.get("time", ""))
            if event_time > cutoff_time:
                self.stats["last_24_hours"]["attempts"] += 1
                if "success" in event.get("event_type", "").lower():
                    self.stats["last_24_hours"]["successes"] += 1
                elif "failure" in event.get("event_type", "").lower():
                    self.stats["last_24_hours"]["failures"] += 1

    def get_analytics_report(self) -> Dict:
        """Generate a comprehensive analytics report"""
        # Convert sets to lists for JSON serialization
        ip_details = {
            ip: {
                **details,
                "associated_users": list(details["associated_users"]),
                "associated_computers": list(details["associated_computers"]),
                "first_seen": details["first_seen"].isoformat(),
                "last_activity": details["last_activity"].isoformat()
            }
            for ip, details in self.stats["ip_details"].items()
        }
        
        # Convert user IP sets to lists
        user_activity = {
            user: {
                **stats,
                "ip_addresses": list(stats["ip_addresses"])
            }
            for user, stats in self.stats["user_activity"].items()
        }
        
        return {
            "summary": {
                "total_attempts": self.stats["total_attempts"],
                "successful_logins": self.stats["successful_logins"],
                "failed_logins": self.stats["failed_logins"],
                "lockout_events": self.stats["lockout_events"],
                "success_rate": (self.stats["successful_logins"] / self.stats["total_attempts"] * 100) 
                               if self.stats["total_attempts"] > 0 else 0
            },
            "last_24_hours": self.stats["last_24_hours"],
            "top_users": sorted(
                user_activity.items(),
                key=lambda x: x[1]["total_attempts"],
                reverse=True
            )[:5],
            "hourly_activity": self.stats["hourly_activity"],
            "suspicious_ips": sorted(
                self.stats["ip_addresses"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "ip_details": ip_details
        }

    def save_analytics(self, filename: str = "security_analytics.json"):
        """Save analytics data to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.get_analytics_report(), f, indent=4)
        except Exception as e:
            print(f"Error saving analytics: {e}")

    def load_analytics(self, filename: str = "security_analytics.json"):
        """Load analytics data from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    data = json.load(f)
                    # Update stats with loaded data
                    self.stats.update(data)
        except Exception as e:
            print(f"Error loading analytics: {e}")

    def send_analytics_to_api(self) -> bool:
        """Send analytics data to the API"""
        if not self.api_url or not self.api_key:
            print("[!] API URL or API key not configured for analytics")
            return False

        try:
            report = self.get_analytics_report()
            headers = {
                "Content-Type": "application/json",
                "X-API-Key": self.api_key
            }
            
            # Format data for /api/events/det endpoint
            analytics_data = {
                "summary": report["summary"],
                "last_24_hours": report["last_24_hours"],
                "ip_analytics": [
                    {
                        "ip": ip,
                        "total_attempts": details["total_attempts"],
                        "successful_logins": details["successful_logins"],
                        "failed_logins": details["failed_logins"],
                        "first_seen": details["first_seen"],
                        "last_activity": details["last_activity"],
                        "associated_users": details["associated_users"],
                        "associated_computers": details["associated_computers"],
                        "recent_events": details["recent_events"]
                    }
                    for ip, details in report["ip_details"].items()
                ],
                "user_analytics": [
                    {
                        "username": user,
                        "total_attempts": stats["total_attempts"],
                        "successful_logins": stats["successful_logins"],
                        "failed_logins": stats["failed_logins"],
                        "ip_addresses": stats["ip_addresses"]
                    }
                    for user, stats in report["top_users"]
                ],
                "hourly_activity": report["hourly_activity"]
            }
            
            # Send analytics data to the API
            response = requests.post(
                f"{self.api_url}/events/det",
                json=analytics_data,
                headers=headers
            )
            
            if response.status_code == 200:
                print("[+] Analytics data successfully sent to API")
                return True
            else:
                print(f"[!] Failed to send analytics data. Status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"[!] Error sending analytics to API: {e}")
            return False 