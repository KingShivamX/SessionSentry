"""
EmailReporter module for SessionSentry.
Generates and sends email reports for login events and detected anomalies.
"""

import os
import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Any
import jinja2
from dotenv import load_dotenv

# Configure logger
logger = logging.getLogger("SessionSentry.EmailReporter")

# Load environment variables
load_dotenv()

class EmailReporter:
    """
    Generates and sends email reports for login events and anomalies.
    Uses SMTP for sending emails and Jinja2 for templating.
    """
    
    def __init__(self):
        """Initialize the email reporter with settings from environment variables."""
        # Email server settings
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", 587))
        self.use_tls = os.getenv("SMTP_USE_TLS", "True").lower() in ("true", "1", "yes")
        
        # Authentication
        self.username = os.getenv("SMTP_USERNAME")
        self.password = os.getenv("SMTP_PASSWORD")
        
        # Email settings
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.recipient_emails = os.getenv("RECIPIENT_EMAILS", "").split(",")
        
        # Validate required settings
        if not all([self.username, self.password, self.sender_email]):
            logger.warning("Email settings incomplete. Email reporting will be disabled.")
        else:
            logger.info("EmailReporter initialized with settings from environment variables.")
            
        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("templates"),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Create templates directory if it doesn't exist
        os.makedirs("templates", exist_ok=True)
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self):
        """Create default email templates if they don't exist."""
        try:
            anomaly_template_path = "templates/anomaly_report.html"
            daily_template_path = "templates/daily_report.html"
            
            # Create anomaly report template
            if not os.path.exists(anomaly_template_path):
                with open(anomaly_template_path, "w", encoding="utf-8") as f:
                    f.write("""<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        h1 { color: #d9534f; }
        table { border-collapse: collapse; width: 100%; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .anomaly { background-color: #ffeeee; }
    </style>
</head>
<body>
    <h1>! SessionSentry: Anomalous Login Events Detected</h1>
    <p>The following login events were detected as potentially anomalous:</p>
    
    <table>
        <tr>
            <th>Time</th>
            <th>Username</th>
            <th>Event Type</th>
            <th>Computer</th>
            <th>Source</th>
            <th>Anomaly Reason</th>
        </tr>
        {% for event in anomalies %}
        <tr class="anomaly">
            <td>{{ event.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}</td>
            <td>{{ event.username }}</td>
            <td>{{ event.event_type }}</td>
            <td>{{ event.computer }}</td>
            <td>{{ event.source_address or 'N/A' }}</td>
            <td>{{ event.anomaly_reason }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <p>Please investigate these events to ensure they are legitimate.</p>
    <p>This report was generated automatically by SessionSentry.</p>
</body>
</html>""")
                    logger.info("Created default anomaly report template")
            
            # Create daily report template
            if not os.path.exists(daily_template_path):
                with open(daily_template_path, "w") as f:
                    f.write("""<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; }
        h1 { color: #5bc0de; }
        table { border-collapse: collapse; width: 100%; }
        th, td { text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .summary { background-color: #eeffee; }
    </style>
</head>
<body>
    <h1>SessionSentry: Daily Login Summary Report</h1>
    <p>Here is the daily summary report for {{ date }}:</p>
    
    <h2>Summary</h2>
    <table>
        <tr class="summary">
            <td>Total Login Events:</td>
            <td>{{ summary.total_logins }}</td>
        </tr>
        <tr class="summary">
            <td>Total Logout Events:</td>
            <td>{{ summary.total_logouts }}</td>
        </tr>
        <tr class="summary">
            <td>Anomalous Events:</td>
            <td>{{ summary.total_anomalies }}</td>
        </tr>
        <tr class="summary">
            <td>Unique Users:</td>
            <td>{{ summary.unique_users }}</td>
        </tr>
    </table>
    
    <h2>Login Events by Hour</h2>
    <table>
        <tr>
            <th>Hour</th>
            <th>Count</th>
        </tr>
        {% for hour, count in summary.logins_by_hour %}
        <tr>
            <td>{{ hour }}:00</td>
            <td>{{ count }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <h2>Top 10 Users by Login Count</h2>
    <table>
        <tr>
            <th>Username</th>
            <th>Login Count</th>
        </tr>
        {% for user, count in summary.top_users %}
        <tr>
            <td>{{ user }}</td>
            <td>{{ count }}</td>
        </tr>
        {% endfor %}
    </table>
    
    <p>This report was generated automatically by SessionSentry.</p>
</body>
</html>""")
                    logger.info("Created default daily report template")
                    
        except Exception as e:
            logger.error(f"Error creating default templates: {e}", exc_info=True)
    
    def send_anomaly_report(self, anomalies: List[Dict[str, Any]]) -> bool:
        """
        Generate and send an email report for detected anomalies.
        
        Args:
            anomalies: List of anomalous event dictionaries
            
        Returns:
            True if the email was sent successfully, False otherwise
        """
        if not anomalies:
            logger.info("No anomalies to report")
            return True
            
        if not all([self.username, self.password, self.sender_email, self.recipient_emails]):
            logger.warning("Email settings incomplete. Cannot send anomaly report.")
            return False
        
        try:
            logger.info(f"Generating anomaly report for {len(anomalies)} events")
            
            # Sort anomalies by timestamp (newest first)
            anomalies = sorted(anomalies, key=lambda x: x['timestamp'], reverse=True)
            
            # Generate report using template
            template = self.jinja_env.get_template("anomaly_report.html")
            html_content = template.render(anomalies=anomalies)
            
            # Create email message
            subject = f"WARNING: SessionSentry: {len(anomalies)} Anomalous Login Events Detected"
            message = self._create_message(subject, html_content)
            
            # Send email
            return self._send_email(message)
            
        except Exception as e:
            logger.error(f"Error sending anomaly report: {e}", exc_info=True)
            return False
    
    def send_daily_report(self, events: List[Dict[str, Any]], 
                         summary: Dict[str, Any]) -> bool:
        """
        Generate and send a daily summary report of login/logout events.
        
        Args:
            events: List of event dictionaries
            summary: Dictionary containing summary statistics
            
        Returns:
            True if the email was sent successfully, False otherwise
        """
        if not all([self.username, self.password, self.sender_email, self.recipient_emails]):
            logger.warning("Email settings incomplete. Cannot send daily report.")
            return False
        
        try:
            logger.info("Generating daily summary report")
            
            # Current date for the report
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Generate report using template
            template = self.jinja_env.get_template("daily_report.html")
            html_content = template.render(date=today, summary=summary)
            
            # Create email message
            subject = f"SessionSentry: Daily Login Summary Report for {today}"
            message = self._create_message(subject, html_content)
            
            # Send email
            return self._send_email(message)
            
        except Exception as e:
            logger.error(f"Error sending daily report: {e}", exc_info=True)
            return False
    
    def _create_message(self, subject: str, html_content: str) -> MIMEMultipart:
        """
        Create an email message with the given subject and HTML content.
        
        Args:
            subject: Email subject
            html_content: HTML content for the email
            
        Returns:
            MIMEMultipart message object
        """
        # Create message container
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = self.sender_email
        message["To"] = ", ".join(self.recipient_emails)
        
        # Create HTML part
        html_part = MIMEText(html_content, "html")
        
        # Attach parts to message
        message.attach(html_part)
        
        return message
    
    def _send_email(self, message: MIMEMultipart) -> bool:
        """
        Send an email using SMTP.
        
        Args:
            message: MIMEMultipart message to send
            
        Returns:
            True if the email was sent successfully, False otherwise
        """
        try:
            # Connect to server
            if self.use_tls:
                context = ssl.create_default_context()
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls(context=context)
            else:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            
            # Login to server
            server.login(self.username, self.password)
            
            # Send email
            server.sendmail(
                self.sender_email, 
                self.recipient_emails, 
                message.as_string()
            )
            
            # Close connection
            server.quit()
            
            logger.info(f"Email sent successfully to {', '.join(self.recipient_emails)}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}", exc_info=True)
            return False 