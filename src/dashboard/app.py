"""
Flask application for SessionSentry dashboard.
Provides a web interface to view login events and detected anomalies.
"""

import os
import logging
from datetime import datetime, timedelta
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from dotenv import load_dotenv
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database module
from data_collection.database import get_db_session, SessionEvent, AnomalyEvent

# Configure logger
logger = logging.getLogger("SessionSentry.Dashboard")

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24))

@app.route('/')
def index():
    """Render the dashboard home page."""
    return render_template('index.html')

@app.route('/events')
def events():
    """Display login/logout events."""
    # Get query parameters
    days = request.args.get('days', 1, type=int)
    username = request.args.get('username', '')
    event_type = request.args.get('event_type', '')
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get events from database
    db = get_db_session()
    try:
        query = db.query(SessionEvent).filter(SessionEvent.timestamp >= start_date)
        
        if username:
            query = query.filter(SessionEvent.username.like(f'%{username}%'))
        
        if event_type:
            query = query.filter(SessionEvent.event_type == event_type)
        
        # Order by timestamp (newest first)
        events = query.order_by(SessionEvent.timestamp.desc()).all()
        
        return render_template('events.html', events=events, days=days, 
                              username=username, event_type=event_type)
    finally:
        db.close()

@app.route('/anomalies')
def anomalies():
    """Display detected anomalies."""
    # Get query parameters
    days = request.args.get('days', 7, type=int)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get anomalies from database
    db = get_db_session()
    try:
        # Join AnomalyEvent with SessionEvent to get all information
        query = db.query(AnomalyEvent, SessionEvent) \
                 .join(SessionEvent, AnomalyEvent.event_id == SessionEvent.id) \
                 .filter(AnomalyEvent.detected_at >= start_date)
        
        # Order by detected_at (newest first)
        results = query.order_by(AnomalyEvent.detected_at.desc()).all()
        
        anomalies = []
        for anomaly, event in results:
            # Combine anomaly and event data
            anomaly_data = {
                'id': anomaly.id,
                'detected_at': anomaly.detected_at,
                'anomaly_type': anomaly.anomaly_type,
                'severity': anomaly.severity,
                'description': anomaly.description,
                'is_false_positive': anomaly.is_false_positive,
                'event_id': event.id,
                'username': event.username,
                'timestamp': event.timestamp,
                'event_type': event.event_type,
                'computer': event.computer,
                'source_address': event.source_address
            }
            anomalies.append(anomaly_data)
        
        return render_template('anomalies.html', anomalies=anomalies, days=days)
    finally:
        db.close()

@app.route('/current_events')
def current_events():
    """Display current events in real-time."""
    # Get query parameters
    minutes = request.args.get('minutes', 15, type=int)
    
    # Calculate time range
    now = datetime.now()
    start_time = now - timedelta(minutes=minutes)
    
    # Get recent events from database
    db = get_db_session()
    try:
        # Get recent events
        events = db.query(SessionEvent)\
            .filter(SessionEvent.timestamp >= start_time)\
            .order_by(SessionEvent.timestamp.desc())\
            .limit(50)\
            .all()
        
        return render_template('current_events.html', events=events, minutes=minutes, now=now)
    finally:
        db.close()

@app.route('/api/events')
def api_events():
    """API endpoint to get events as JSON."""
    # Get query parameters
    days = request.args.get('days', 1, type=int)
    username = request.args.get('username', '')
    event_type = request.args.get('event_type', '')
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get events from database
    db = get_db_session()
    try:
        query = db.query(SessionEvent).filter(SessionEvent.timestamp >= start_date)
        
        if username:
            query = query.filter(SessionEvent.username.like(f'%{username}%'))
        
        if event_type:
            query = query.filter(SessionEvent.event_type == event_type)
        
        # Order by timestamp (newest first)
        events = query.order_by(SessionEvent.timestamp.desc()).all()
        
        # Convert to list of dictionaries
        events_data = []
        for event in events:
            event_dict = {
                'id': event.id,
                'event_id': event.event_id,
                'event_type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'username': event.username,
                'domain': event.domain,
                'computer': event.computer,
                'login_type': event.login_type,
                'source_address': event.source_address,
                'is_anomalous': event.is_anomalous
            }
            events_data.append(event_dict)
        
        return jsonify(events_data)
    finally:
        db.close()

@app.route('/api/stats')
def api_stats():
    """API endpoint to get statistics for dashboard charts."""
    # Get query parameters
    days = request.args.get('days', 7, type=int)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get events from database
    db = get_db_session()
    try:
        # Get all events in range
        events = db.query(SessionEvent).filter(SessionEvent.timestamp >= start_date).all()
        
        # Convert to pandas DataFrame for easier analysis
        df = pd.DataFrame([{
            'id': e.id,
            'event_type': e.event_type,
            'timestamp': e.timestamp,
            'username': e.username,
            'is_anomalous': e.is_anomalous
        } for e in events])
        
        # If no events, return empty stats
        if df.empty:
            return jsonify({
                'total_events': 0,
                'total_logins': 0,
                'total_logouts': 0,
                'total_anomalies': 0,
                'events_by_day': [],
                'events_by_hour': [],
                'anomalies_by_day': []
            })
        
        # Calculate basic stats
        total_events = len(df)
        total_logins = len(df[df['event_type'] == 'login'])
        total_logouts = len(df[df['event_type'] == 'logout'])
        total_anomalies = len(df[df['is_anomalous'] == True])
        
        # Events by day
        df['date'] = df['timestamp'].dt.date
        events_by_day = df.groupby(['date', 'event_type']).size().unstack(fill_value=0).reset_index()
        events_by_day_data = [{
            'date': row['date'].isoformat(),
            'login': int(row.get('login', 0)),
            'logout': int(row.get('logout', 0))
        } for _, row in events_by_day.iterrows()]
        
        # Events by hour
        df['hour'] = df['timestamp'].dt.hour
        events_by_hour = df.groupby(['hour', 'event_type']).size().unstack(fill_value=0).reset_index()
        events_by_hour_data = [{
            'hour': int(row['hour']),
            'login': int(row.get('login', 0)),
            'logout': int(row.get('logout', 0))
        } for _, row in events_by_hour.iterrows()]
        
        # Anomalies by day
        anomalies_df = df[df['is_anomalous'] == True]
        if not anomalies_df.empty:
            anomalies_by_day = anomalies_df.groupby('date').size().reset_index()
            anomalies_by_day_data = [{
                'date': row['date'].isoformat(),
                'count': int(row[0])
            } for _, row in anomalies_by_day.iterrows()]
        else:
            anomalies_by_day_data = []
        
        # Return the stats
        return jsonify({
            'total_events': total_events,
            'total_logins': total_logins,
            'total_logouts': total_logouts,
            'total_anomalies': total_anomalies,
            'events_by_day': events_by_day_data,
            'events_by_hour': events_by_hour_data,
            'anomalies_by_day': anomalies_by_day_data
        })
    finally:
        db.close()

@app.route('/api/mark_false_positive/<int:anomaly_id>', methods=['POST'])
def mark_false_positive(anomaly_id):
    """Mark an anomaly as a false positive."""
    db = get_db_session()
    try:
        anomaly = db.query(AnomalyEvent).filter(AnomalyEvent.id == anomaly_id).first()
        if anomaly:
            anomaly.is_false_positive = True
            anomaly.analyst_notes = request.json.get('notes', '')
            db.commit()
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'error': 'Anomaly not found'}), 404
    except Exception as e:
        db.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        db.close()

def run_dashboard(host='127.0.0.1', port=5000, debug=False):
    """Run the Flask dashboard application."""
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the app
    run_dashboard(debug=True) 