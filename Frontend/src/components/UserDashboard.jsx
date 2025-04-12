import { useState, useEffect, useContext } from "react"
import { useParams, Link } from "react-router-dom"
import { motion } from "framer-motion"
import { fetchUserEvents, fetchUserDetails } from "../services/api"
import { calculateSessionMetrics } from "../utils/sessionCalculator"
import Dashboard from "./Dashboard"
import { UserContext } from "../App"

const UserDashboard = () => {
    // Fix: We need to get computer_name from params instead of username
    const { computer_name: computerName } = useParams()
    const [sessions, setSessions] = useState([])
    const [metrics, setMetrics] = useState({
        todayHours: 0,
        weeklyHours: 0,
        monthlyHours: 0,
        currentStatus: "Unknown",
    })
    const [userDetails, setUserDetails] = useState(null)
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [userName, setUserName] = useState("Unknown User")
    // Add a flag to prevent multiple fetches and a stable key to prevent rerenders
    const [hasLoadedData, setHasLoadedData] = useState(false)
    const [dashboardKey] = useState(Math.random().toString(36).substring(7))

    // Get context for updating active user
    const { setActiveUser, setActiveUserStatus } = useContext(UserContext)

    // Decode the computer name from the URL
    const decodedComputerName = decodeURIComponent(computerName)

    // Update the active user in the header when this component mounts
    useEffect(() => {
        // Will set a default name until we get the real user name
        setActiveUser(`Computer: ${decodedComputerName}`)

        // Clean up when component unmounts
        return () => {
            setActiveUser("SessionSentry Admin")
            setActiveUserStatus("Online")
        }
    }, [decodedComputerName, setActiveUser, setActiveUserStatus])

    // Main data loading effect - separate from the one that updates the header
    useEffect(() => {
        // Prevent multiple fetches
        if (hasLoadedData) {
            console.log(
                `Dashboard [${dashboardKey}]: Data already loaded, skipping fetch`
            )
            return
        }

        console.log(
            `Dashboard [${dashboardKey}]: Loading data for ${decodedComputerName}`
        )

        const loadUserData = async () => {
            try {
                setLoading(true)
                setError(null)

                // 1. Try to get user details by computer name first
                try {
                    console.log(
                        `Fetching user details for ${decodedComputerName}`
                    )
                    const details = await fetchUserDetails(decodedComputerName)
                    if (details) {
                        setUserDetails(details)
                        const detailsUserName =
                            details.user_name || "Unknown User"
                        setUserName(detailsUserName)

                        // If we have user details, determine online status
                        if (details.status) {
                            const isOnline = details.status === "active"
                            setActiveUserStatus(isOnline ? "Online" : "Offline")
                        }

                        // Update the active user name with the real user name
                        setActiveUser(
                            details.user_name ||
                                `Computer: ${decodedComputerName}`
                        )
                    }
                } catch (detailsError) {
                    console.error("Error fetching user details:", detailsError)
                    // This is non-critical, we continue
                }

                // 2. Fetch events for this specific computer
                console.log(`Fetching events for ${decodedComputerName}`)
                const events = await fetchUserEvents(decodedComputerName)
                console.log(
                    `Received ${
                        events?.length || 0
                    } events for ${decodedComputerName}`
                )

                if (!events || events.length === 0) {
                    console.log(`No events found for ${decodedComputerName}`)
                    setError(
                        `No session data found for computer: ${decodedComputerName}`
                    )
                    setSessions([])
                    setMetrics({
                        todayHours: 0,
                        weeklyHours: 0,
                        monthlyHours: 0,
                        currentStatus: "Unknown",
                    })
                    setActiveUserStatus("Unknown")
                } else {
                    // If we didn't get a username from user details, try to get it from events
                    if (
                        userName === "Unknown User" &&
                        events.length > 0 &&
                        events[0].user_name
                    ) {
                        const eventUserName = events[0].user_name
                        console.log(
                            `Setting username from events: ${eventUserName}`
                        )
                        setUserName(eventUserName)
                        setActiveUser(eventUserName)
                    }

                    // Sort events by time (oldest first)
                    const sortedEvents = [...events].sort(
                        (a, b) => new Date(a.time) - new Date(b.time)
                    )

                    // Determine if user is currently online from the events
                    // Check if last event is a login
                    const lastEvent = sortedEvents[sortedEvents.length - 1]
                    const isOnline =
                        lastEvent &&
                        (lastEvent.event_type === "Login" ||
                            lastEvent.event_type === "login" ||
                            (lastEvent.event_type === "login_attempt" &&
                                lastEvent.status === "success"))
                    const userStatus = isOnline ? "Online" : "Offline"

                    console.log(
                        `Setting sessions (${sortedEvents.length} events)`
                    )
                    setSessions(sortedEvents)

                    // Calculate metrics from the session data
                    console.log(
                        `Calculating session metrics for ${decodedComputerName}`
                    )
                    try {
                        const calculatedMetrics =
                            calculateSessionMetrics(sortedEvents)

                        // Override the calculated status with our direct check
                        calculatedMetrics.currentStatus = userStatus

                        console.log(
                            `Setting metrics for ${decodedComputerName}`
                        )
                        setMetrics(calculatedMetrics)
                    } catch (metricsError) {
                        console.error(
                            "Error calculating metrics:",
                            metricsError
                        )
                        setError(
                            `Error calculating session metrics: ${metricsError.message}`
                        )
                    }

                    // Update header status if we don't have user details
                    if (!userDetails) {
                        setActiveUserStatus(userStatus)
                    }
                }
            } catch (err) {
                console.error("Error loading user data:", err)
                setError(
                    `Failed to load data for computer: ${decodedComputerName}`
                )
                setActiveUserStatus("Unknown")
            } finally {
                setLoading(false)
                setHasLoadedData(true)
                console.log(`Data loading complete for ${decodedComputerName}`)
            }
        }

        loadUserData()

        // No interval means no clean-up needed
        return () => {
            // Clean up when component unmounts
            setActiveUser("SessionSentry Admin")
            setActiveUserStatus("Online")
        }
        // Only depend on computerName and the stable key, not on the userName which could cause rerenders
    }, [
        decodedComputerName,
        dashboardKey,
        hasLoadedData,
        setActiveUser,
        setActiveUserStatus,
    ])

    // Add a manual refresh button function
    const handleRefresh = () => {
        console.log("Manual refresh requested")
        setHasLoadedData(false)
        setLoading(true)
        setError(null)
    }

    // Helper to safely format date
    const formatDate = (dateString) => {
        try {
            return new Date(dateString).toLocaleString()
        } catch (e) {
            return "Unknown"
        }
    }

    return (
        <div className="p-4 md:p-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center mb-6 gap-3">
                <Link
                    to="/"
                    className="flex items-center text-amber-600 hover:text-amber-800 mr-2"
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
                    Back to Computers
                </Link>
                <h1 className="text-xl sm:text-2xl font-bold text-amber-800 break-words">
                    <span className="block sm:inline sm:mr-2">
                        Computer Dashboard:
                    </span>
                    <span className="text-amber-600 break-all max-w-full inline-block">
                        {decodedComputerName}
                    </span>
                    {(metrics.currentStatus === "Online" ||
                        (userDetails && userDetails.status === "active")) && (
                        <span className="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                            <div className="h-2 w-2 rounded-full bg-green-500 mr-1.5"></div>
                            Online
                        </span>
                    )}
                </h1>
            </div>

            {/* User details section (only if available) */}
            {userDetails && !loading && !error && (
                <motion.div
                    className="bg-white rounded-lg shadow-sm p-4 mb-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    key={`details-${dashboardKey}`}
                >
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="border-r border-gray-100 pr-4">
                            <p className="text-sm text-gray-500">User</p>
                            <p className="font-medium break-all" title={userDetails.user_name || userName || "Unknown"}>
                                {userDetails.user_name || userName || "Unknown"}
                            </p>
                        </div>
                        <div className="border-r border-gray-100 pr-4">
                            <p className="text-sm text-gray-500">IP Address</p>
                            <p className="font-medium break-all">
                                {userDetails.ip_address || "Unknown"}
                            </p>
                        </div>
                        <div className="border-r border-gray-100 pr-4">
                            <p className="text-sm text-gray-500">Last Seen</p>
                            <p className="font-medium">
                                {userDetails.last_seen
                                    ? formatDate(userDetails.last_seen)
                                    : "Unknown"}
                            </p>
                        </div>
                        <div>
                            <p className="text-sm text-gray-500">
                                Total Events
                            </p>
                            <p className="font-medium">
                                {userDetails.total_events || 0}
                                {userDetails.failed_attempts > 0 && (
                                    <span className="text-xs text-red-500 ml-2">
                                        ({userDetails.failed_attempts} failed
                                        attempts)
                                    </span>
                                )}
                            </p>
                        </div>
                    </div>
                </motion.div>
            )}

            {/* Add refresh button below heading */}
            <div className="mb-4">
                <button
                    onClick={handleRefresh}
                    className="px-3 py-1 bg-amber-500 text-white rounded hover:bg-amber-600 transition-colors text-sm"
                >
                    Refresh Data
                </button>
            </div>

            {loading ? (
                <motion.div
                    className="flex justify-center items-center h-64"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    key={`loading-${dashboardKey}`}
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
                            Loading session data...
                        </p>
                    </div>
                </motion.div>
            ) : error ? (
                <motion.div
                    className="bg-red-50 border border-red-200 rounded-lg p-6 text-center"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    key={`error-${dashboardKey}`}
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
                    <p className="text-red-700">{error}</p>
                </motion.div>
            ) : (
                <Dashboard
                    sessions={sessions}
                    metrics={metrics}
                    key={`dashboard-${dashboardKey}`}
                />
            )}
        </div>
    )
}

export default UserDashboard
