import axios from 'axios';

// Base URL for the API - would be configured based on environment
const API_URL = 'http://localhost:3000/api';

// For development, since we don't have an actual backend server yet,
// we'll directly read from the JSON file using a relative path
export const fetchLoginEvents = async () => {
  try {
    // In a production environment, this would be an API call:
    // const response = await axios.get(`${API_URL}/login-events`);
    // return response.data;
    
    // For development, we're reading directly from the local JSON file
    const response = await axios.get('/login_events.json');
    return response.data;
  } catch (error) {
    console.error('Error fetching login events:', error);
    
    // Fallback to mock data if the API or file is not available
    return mockLoginEvents;
  }
};

// Mock data for development and fallback
const mockLoginEvents = [
  {
    "event_id": 4624,
    "time": "2025-04-12 03:14",
    "computer_name": "shivam",
    "user_name": "SHIVAM\\shiva",
    "event_type": "Login",
    "ip_address": "192.168.163.240"
  },
  {
    "event_id": 4634,
    "time": "2025-04-12 03:45",
    "computer_name": "shivam",
    "user_name": "SHIVAM\\shiva",
    "event_type": "Logout",
    "ip_address": "192.168.163.240"
  },
  {
    "event_id": 4624,
    "time": "2025-04-12 04:15",
    "computer_name": "shivam",
    "user_name": "SHIVAM\\shiva",
    "event_type": "Login",
    "ip_address": "192.168.163.240"
  },
  {
    "event_id": 4634,
    "time": "2025-04-12 05:30",
    "computer_name": "shivam",
    "user_name": "SHIVAM\\shiva",
    "event_type": "Logout",
    "ip_address": "192.168.163.240"
  }
];
