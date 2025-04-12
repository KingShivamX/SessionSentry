// import axios from 'axios';
import axios from "axios"

// Base URL for the API
const API_URL = "https://sessionsentryserver.onrender.com/api"

// Create a simple debounce function at the top of the file
let apiCalls = {}

const debounceApiCall = (key, fn, delay = 2000) => {
    if (apiCalls[key]) {
        console.log(`API call to ${key} debounced - already in progress`)
        return apiCalls[key]
    }

    console.log(`Making API call to ${key}`)
    const promise = fn()
    apiCalls[key] = promise

    // Clear the key after the promise resolves or rejects
    promise.finally(() => {
        setTimeout(() => {
            console.log(`API call to ${key} completed, debounce reset`)
            delete apiCalls[key]
        }, delay)
    })

    return promise
}

/**
 * Fetches all login events from the API
 * @returns {Promise<Array>} Array of login/logout events
 */
export const fetchLoginEvents = async () => {
    console.log(`Fetching login events from ${API_URL}/events`)

    try {
        const response = await axios.get(`${API_URL}/events`)
        console.log(
            "Login events API response:",
            response.status,
            response.statusText
        )

        if (!response.data) {
            console.warn("API returned empty data")
            return []
        }

        console.log(`Received ${response.data.length} login events`)
        return response.data
    } catch (error) {
        console.error("Error fetching login events:", error.message)
        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            console.error(
                "API error response:",
                error.response.status,
                error.response.statusText
            )
        } else if (error.request) {
            // The request was made but no response was received
            console.error("No response received from API")
        }

        // For now, let's return an empty array instead of mock data
        return []
    }
}

/**
 * Fetches all events for a specific computer
 * @param {string} computerName - Computer name to fetch events for
 * @returns {Promise<Array>} Array of login/logout events for the computer
 */
export const fetchUserEvents = async (computerName) => {
    console.log(`fetchUserEvents called for ${computerName}`)

    // Use debouncing to prevent rapid consecutive calls for the same computer
    return debounceApiCall(`events_${computerName}`, async () => {
        try {
            // Use computer_name instead of username for the events endpoint
            const response = await axios.get(
                `${API_URL}/events/user/${computerName}`
            )
            console.log(`Received data for ${computerName}`)
            return response.data
        } catch (error) {
            console.error(
                `Error fetching events for computer ${computerName}:`,
                error
            )
            return [] // Return empty array on error
        }
    })
}

/**
 * Fetches list of all users in the system
 * @returns {Promise<Array>} Array of user objects
 */
export const fetchUsers = async () => {
    try {
        // Use the /api/users endpoint exactly as specified in the docs
        const response = await axios.get(`${API_URL}/users`)
        return response.data
    } catch (error) {
        console.error("Error fetching users:", error)

        // For development fallback, create mock users
        const mockUsers = [
            {
                computer_name: "PC-001",
                user_name: "john_doe",
                ip_address: "192.168.1.100",
                first_seen: "2024-03-20T10:00:00.000Z",
                last_seen: "2024-03-20T14:30:00.000Z",
                total_events: 5,
                failed_attempts: 1,
                status: "active",
            },
            {
                computer_name: "PC-002",
                user_name: "jane_smith",
                ip_address: "192.168.1.101",
                first_seen: "2024-03-20T09:15:00.000Z",
                last_seen: "2024-03-20T17:30:00.000Z",
                total_events: 8,
                failed_attempts: 0,
                status: "active",
            },
            {
                computer_name: "LAPTOP-003",
                user_name: "admin",
                ip_address: "192.168.1.50",
                first_seen: "2024-03-19T08:00:00.000Z",
                last_seen: "2024-03-20T18:45:00.000Z",
                total_events: 12,
                failed_attempts: 0,
                status: "active",
            },
        ]

        console.log("Using mock user data due to API error")
        return mockUsers
    }
}

/**
 * Fetches details for a specific user/computer
 * @param {string} computerName - Computer name to fetch details for
 * @returns {Promise<Object|null>} User details or null on error
 */
export const fetchUserDetails = async (computerName) => {
    console.log(`fetchUserDetails called for ${computerName}`)

    // Use debouncing to prevent rapid consecutive calls for the same computer
    return debounceApiCall(`details_${computerName}`, async () => {
        try {
            const response = await axios.get(`${API_URL}/users/${computerName}`)
            return response.data
        } catch (error) {
            console.error(
                `Error fetching details for computer ${computerName}:`,
                error
            )
            return null // Return null on error
        }
    })
}
