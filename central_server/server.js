const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const app = express();
const port = 3000;

app.use(cors());
app.use(express.json());

// Constants
const DATA_DIR = path.join(__dirname, 'data');
const HISTORICAL_FILE = path.join(DATA_DIR, 'historical_data.json');
const ANALYTICS_FILE = path.join(DATA_DIR, 'current_analytics.json');

// Ensure data directory exists
if (!fs.existsSync(DATA_DIR)) {
    fs.mkdirSync(DATA_DIR, { recursive: true });
}

// Initialize analytics data structure
let analyticsData = {
    summary: {
        total_attempts: 0,
        successful_logins: 0,
        failed_logins: 0,
        lockout_events: 0,
        success_rate: 0
    },
    last_24_hours: {
        attempts: 0,
        successes: 0,
        failures: 0
    },
    ip_analytics: [],
    user_analytics: [],
    hourly_activity: {}
};

// Load existing data if available
function loadExistingData() {
    try {
        if (fs.existsSync(ANALYTICS_FILE)) {
            const data = JSON.parse(fs.readFileSync(ANALYTICS_FILE, 'utf8'));
            analyticsData = data;
            console.log('Loaded existing analytics data:', JSON.stringify(analyticsData.summary, null, 2));
        } else {
            console.log('No existing analytics data found, starting fresh');
        }
    } catch (error) {
        console.error('Error loading analytics data:', error);
    }
}

// Store historical data
let historicalData = [];

// Load historical data
function loadHistoricalData() {
    try {
        if (fs.existsSync(HISTORICAL_FILE)) {
            historicalData = JSON.parse(fs.readFileSync(HISTORICAL_FILE, 'utf8'));
            console.log(`Loaded ${historicalData.length} historical records`);
            if (historicalData.length > 0) {
                console.log('Latest historical record:', JSON.stringify(historicalData[historicalData.length - 1].timestamp, null, 2));
            }
        } else {
            console.log('No historical data found, starting fresh');
        }
    } catch (error) {
        console.error('Error loading historical data:', error);
    }
}

// Save current analytics data
function saveAnalyticsData() {
    try {
        fs.writeFileSync(ANALYTICS_FILE, JSON.stringify(analyticsData, null, 2));
        console.log('Analytics data saved successfully');
    } catch (error) {
        console.error('Error saving analytics data:', error);
    }
}

// Save historical data
function saveHistoricalData() {
    try {
        fs.writeFileSync(HISTORICAL_FILE, JSON.stringify(historicalData, null, 2));
        console.log(`Historical data saved successfully (${historicalData.length} records)`);
    } catch (error) {
        console.error('Error saving historical data:', error);
    }
}

// Function to print analytics data in a readable format
function printAnalyticsData() {
    const timestamp = new Date().toISOString();
    console.log('\n=== Analytics Data Report ===');
    console.log(`Timestamp: ${timestamp}`);
    
    // Print Summary
    console.log('\nSummary:');
    console.log(`Total Attempts: ${analyticsData.summary.total_attempts}`);
    console.log(`Successful Logins: ${analyticsData.summary.successful_logins}`);
    console.log(`Failed Logins: ${analyticsData.summary.failed_logins}`);
    console.log(`Lockout Events: ${analyticsData.summary.lockout_events}`);
    console.log(`Success Rate: ${analyticsData.summary.success_rate.toFixed(2)}%`);
    
    // Print Last 24 Hours
    console.log('\nLast 24 Hours:');
    console.log(`Attempts: ${analyticsData.last_24_hours.attempts}`);
    console.log(`Successes: ${analyticsData.last_24_hours.successes}`);
    console.log(`Failures: ${analyticsData.last_24_hours.failures}`);
    
    // Print Top Users
    console.log('\nTop Users:');
    if (analyticsData.user_analytics.length > 0) {
        analyticsData.user_analytics
            .sort((a, b) => b.total_attempts - a.total_attempts)
            .slice(0, 5)
            .forEach(user => {
                console.log(`\nUser: ${user.username}`);
                console.log(`Total Attempts: ${user.total_attempts}`);
                console.log(`Successful Logins: ${user.successful_logins}`);
                console.log(`Failed Logins: ${user.failed_logins}`);
                console.log(`IP Addresses Used: ${user.ip_addresses.join(', ')}`);
            });
    } else {
        console.log('No user data available');
    }
    
    // Print Suspicious IPs
    console.log('\nSuspicious IPs:');
    const suspiciousIPs = analyticsData.ip_analytics
        .filter(ip => ip.failed_logins > ip.successful_logins)
        .sort((a, b) => b.failed_logins - a.failed_logins)
        .slice(0, 5);
    
    if (suspiciousIPs.length > 0) {
        suspiciousIPs.forEach(ip => {
            console.log(`\nIP: ${ip.ip}`);
            console.log(`Total Attempts: ${ip.total_attempts}`);
            console.log(`Successful Logins: ${ip.successful_logins}`);
            console.log(`Failed Logins: ${ip.failed_logins}`);
            console.log(`First Seen: ${ip.first_seen}`);
            console.log(`Last Activity: ${ip.last_activity}`);
            console.log(`Associated Users: ${ip.associated_users.join(', ')}`);
        });
    } else {
        console.log('No suspicious IPs found');
    }
    
    console.log('\n' + '='.repeat(50));
}

// Set up interval to print analytics every minute
setInterval(() => {
    printAnalyticsData();
    saveAnalyticsData();
    saveHistoricalData();
}, 60000); // 60000 ms = 1 minute

// Load existing data on startup
loadExistingData();
loadHistoricalData();

app.post('/api/events', (req, res) => {
    console.log('Received events:', JSON.stringify(req.body.events, null, 2));
    res.json({ message: 'Events received successfully', count: req.body.events.length });
});

app.post('/api/events/det', (req, res) => {
    try {
        console.log('Received analytics data:', JSON.stringify(req.body, null, 2));
        const newAnalytics = req.body;
        const timestamp = new Date().toISOString();
        
        // Validate incoming data
        if (!newAnalytics || !newAnalytics.summary) {
            throw new Error('Invalid analytics data format');
        }
        
        // Update current analytics data
        analyticsData = {
            ...analyticsData,
            summary: newAnalytics.summary,
            last_24_hours: newAnalytics.last_24_hours,
            ip_analytics: newAnalytics.ip_analytics || [],
            user_analytics: newAnalytics.user_analytics || [],
            hourly_activity: newAnalytics.hourly_activity || {}
        };

        console.log('Updated analytics data:', JSON.stringify(analyticsData.summary, null, 2));

        // Add to historical data with timestamp
        historicalData.push({
            timestamp,
            data: newAnalytics
        });

        // Keep only last 24 hours of historical data
        const cutoffTime = new Date();
        cutoffTime.setHours(cutoffTime.getHours() - 24);
        historicalData = historicalData.filter(item => new Date(item.timestamp) > cutoffTime);

        // Save data immediately
        saveAnalyticsData();
        saveHistoricalData();

        // Print analytics data
        printAnalyticsData();

        res.json({ 
            message: 'Analytics data received successfully',
            summary: analyticsData.summary
        });
    } catch (error) {
        console.error('Error processing analytics data:', error);
        res.status(500).json({ error: 'Failed to process analytics data' });
    }
});

// Get current analytics data
app.get('/api/analytics/current', (req, res) => {
    res.json(analyticsData);
});

// Get historical analytics data
app.get('/api/analytics/history', (req, res) => {
    res.json(historicalData);
});

// Get suspicious IPs
app.get('/api/analytics/suspicious-ips', (req, res) => {
    const suspiciousIPs = analyticsData.ip_analytics
        .filter(ip => ip.failed_logins > ip.successful_logins)
        .sort((a, b) => b.failed_logins - a.failed_logins);
    
    res.json(suspiciousIPs);
});

// Get top users
app.get('/api/analytics/top-users', (req, res) => {
    const topUsers = analyticsData.user_analytics
        .sort((a, b) => b.total_attempts - a.total_attempts);
    
    res.json(topUsers);
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
    // Print initial analytics data
    printAnalyticsData();
}); 