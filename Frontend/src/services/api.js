// import axios from 'axios';
import axios from "axios"

// Base URL for the API
const API_URL = "https://sessionsentryserver.onrender.com/api"

/**
 * Fetches all login events from the API
 * @returns {Promise<Array>} Array of login/logout events
 */
export const fetchLoginEvents = async () => {
    try {
        const response = await axios.get(`${API_URL}/events`)
        return response.data
    } catch (error) {
        console.error("Error fetching login events:", error)
        // Fallback to mock data if the API is not available
        return mockLoginEvents
    }
}

/**
 * Fetches all events for a specific user
 * @param {string} username - Username to fetch events for
 * @returns {Promise<Array>} Array of login/logout events for the user
 */
export const fetchUserEvents = async (username) => {
    try {
        const response = await axios.get(`${API_URL}/events/user/${username}`)
        return response.data
    } catch (error) {
        console.error(`Error fetching events for user ${username}:`, error)
        // Filter mock data for this user as fallback
        return mockLoginEvents.filter(
            (event) =>
                event.user_name === username ||
                event.user_name.includes(username)
        )
    }
}

/**
 * Fetches list of all unique users in the system
 * @returns {Promise<Array>} Array of usernames
 */
export const fetchUsers = async () => {
    try {
        // Get all events first
        const response = await axios.get(`${API_URL}/events`)
        const events = response.data

        // Extract unique usernames
        const userSet = new Set()
        events.forEach((event) => {
            if (
                event.user_name &&
                !event.user_name.includes("SYSTEM") &&
                !event.user_name.includes("LOCAL SERVICE") &&
                !event.user_name.includes("NETWORK SERVICE")
            ) {
                userSet.add(event.user_name)
            }
        })

        return Array.from(userSet)
    } catch (error) {
        console.error("Error fetching users:", error)

        // Return unique users from mock data as fallback
        const userSet = new Set()
        mockLoginEvents.forEach((event) => userSet.add(event.user_name))
        return Array.from(userSet)
    }
}

// Mock data for development and fallback - complete session data
const mockLoginEvents = [
    {
        event_id: 4624,
        time: "2025-04-12 03:14",
        computer_name: "shivam",
        user_name: "SHIVAM\\shiva",
        event_type: "Login",
        ip_address: "192.168.163.240",
    },
    {
        event_id: 4634,
        time: "2025-04-12 03:45",
        computer_name: "shivam",
        user_name: "SHIVAM\\shiva",
        event_type: "Logout",
        ip_address: "192.168.163.240",
    },
    {
        event_id: 4624,
        time: "2025-04-12 04:15",
        computer_name: "shivam",
        user_name: "SHIVAM\\shiva",
        event_type: "Login",
        ip_address: "192.168.163.240",
    },
    {
        event_id: 4634,
        time: "2025-04-12 05:30",
        computer_name: "shivam",
        user_name: "SHIVAM\\shiva",
        event_type: "Logout",
        ip_address: "192.168.163.240",
    },
    {
        event_id: 4624,
        time: "2025-04-12 06:15",
        computer_name: "shivam",
        user_name: "SHIVAM\\shiva",
        event_type: "Login",
        ip_address: "192.168.163.240",
    },
    {
        event_id: 4634,
        time: "2025-04-12 06:35",
        computer_name: "shivam",
        user_name: "SHIVAM\\shiva",
        event_type: "Logout",
        ip_address: "192.168.163.240",
    },
    {
        event_id: 4624,
        time: "2025-04-12 07:30",
        computer_name: "desktop-a123",
        user_name: "john.doe",
        event_type: "Login",
        ip_address: "192.168.1.100",
    },
    {
        event_id: 4634,
        time: "2025-04-12 09:45",
        computer_name: "desktop-a123",
        user_name: "john.doe",
        event_type: "Logout",
        ip_address: "192.168.1.100",
    },
]
