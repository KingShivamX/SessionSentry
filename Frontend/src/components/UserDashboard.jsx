import { useState, useEffect } from "react"
import { useParams, Link } from "react-router-dom"
import { motion } from "framer-motion"
import { fetchUserEvents } from "../services/api"
import { calculateSessionMetrics } from "../utils/sessionCalculator"
import Dashboard from "./Dashboard"

const UserDashboard = () => {
    const { username } = useParams()
    const [sessions, setSessions] = useState([])
    const [metrics, setMetrics] = useState({
        todayHours: 0,
        weeklyHours: 0,
        monthlyHours: 0,
        currentStatus: "Unknown",
    })
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)

    // Decode the username from the URL
    const decodedUsername = decodeURIComponent(username)

    useEffect(() => {
        const loadUserData = async () => {
            try {
                setLoading(true)
                setError(null)

                // Fetch events for this specific user
                const events = await fetchUserEvents(decodedUsername)

                if (events.length === 0) {
                    setError(
                        `No session data found for user: ${decodedUsername}`
                    )
                    setSessions([])
                    setMetrics({
                        todayHours: 0,
                        weeklyHours: 0,
                        monthlyHours: 0,
                        currentStatus: "Unknown",
                    })
                } else {
                    setSessions(events)

                    // Calculate metrics from the session data
                    const calculatedMetrics = calculateSessionMetrics(events)
                    setMetrics(calculatedMetrics)
                }
            } catch (err) {
                console.error("Error loading user data:", err)
                setError(`Failed to load data for user: ${decodedUsername}`)
            } finally {
                setLoading(false)
            }
        }

        loadUserData()

        // Set up interval to refresh data every 5 minutes
        const intervalId = setInterval(() => loadUserData(), 5 * 60 * 1000)

        return () => clearInterval(intervalId)
    }, [decodedUsername])

    return (
        <div className="p-4 md:p-6">
            <div className="flex items-center mb-6">
                <Link
                    to="/"
                    className="flex items-center text-amber-600 hover:text-amber-800 mr-4"
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-5 w-5 mr-1"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                    >
                        <path
                            fillRule="evenodd"
                            d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z"
                            clipRule="evenodd"
                        />
                    </svg>
                    Back to Users
                </Link>
                <h1 className="text-2xl font-bold text-amber-800">
                    User Dashboard:{" "}
                    <span className="text-amber-600">{decodedUsername}</span>
                </h1>
            </div>

            {loading ? (
                <motion.div
                    className="flex justify-center items-center h-64"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                >
                    <div className="text-amber-500">
                        <svg
                            className="animate-spin h-12 w-12"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                        >
                            <circle
                                className="opacity-25"
                                cx="12"
                                cy="12"
                                r="10"
                                stroke="currentColor"
                                strokeWidth="4"
                            ></circle>
                            <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            ></path>
                        </svg>
                        <p className="mt-2 text-center font-medium">
                            Loading user data...
                        </p>
                    </div>
                </motion.div>
            ) : error ? (
                <motion.div
                    className="bg-red-50 border border-red-200 rounded-lg p-6 text-center"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <div className="text-red-500 mb-4">
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-12 w-12 mx-auto"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                            />
                        </svg>
                    </div>
                    <h2 className="text-lg font-medium text-red-800 mb-2">
                        Error
                    </h2>
                    <p className="text-red-600">{error}</p>
                    <div className="mt-6">
                        <Link
                            to="/"
                            className="inline-flex items-center px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-amber-500"
                        >
                            Return to User Selection
                        </Link>
                    </div>
                </motion.div>
            ) : (
                // Use the existing Dashboard component to display this user's data
                <Dashboard sessions={sessions} metrics={metrics} />
            )}
        </div>
    )
}

export default UserDashboard
