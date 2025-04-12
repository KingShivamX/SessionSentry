import { useState, useEffect } from "react"
import { fetchLoginEvents } from "../services/api"
import { calculateSessionMetrics } from "../utils/sessionCalculator"

const Sessions = () => {
    const [sessions, setSessions] = useState([])
    const [rawEvents, setRawEvents] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [hasAttemptedFetch, setHasAttemptedFetch] = useState(false)
    const [sessionKey] = useState(Math.random().toString(36).substring(7))

    // Only fetch data once when the component mounts
    useEffect(() => {
        // Prevent multiple fetch attempts that could cause infinite reloads
        if (hasAttemptedFetch) {
            console.log("Already attempted fetch, not fetching again")
            return
        }

        console.log(`Sessions [${sessionKey}]: Fetching data...`)

        const fetchSessions = async () => {
            try {
                setLoading(true)
                setHasAttemptedFetch(true)
                console.log(
                    `Making API call to fetch events... [${sessionKey}]`
                )

                const events = await fetchLoginEvents()
                console.log(
                    `API response received [${sessionKey}]:`,
                    events ? events.length : 0,
                    "events"
                )

                if (!events || events.length === 0) {
                    console.log("No events received from API")
                    setError("No session data available")
                    setRawEvents([])
                    setSessions([])
                } else {
                    console.log("Setting raw events and calculating metrics...")
                    setRawEvents(events)

                    try {
                        // Calculate session metrics which includes session history with durations
                        const metrics = calculateSessionMetrics(events)
                        setSessions(metrics.sessionHistory || [])
                        console.log(
                            `Session metrics calculated successfully [${sessionKey}]:`,
                            metrics.sessionHistory.length,
                            "sessions found"
                        )
                    } catch (metricsError) {
                        console.error(
                            "Error calculating session metrics:",
                            metricsError
                        )
                        setError(
                            "Error processing session data: " +
                                metricsError.message
                        )
                        setSessions([])
                    }
                }
            } catch (err) {
                console.error("Error fetching sessions:", err)
                setError(
                    "Failed to load session data: " +
                        (err.message || "Unknown error")
                )
            } finally {
                setLoading(false)
                console.log(
                    `Data fetch complete [${sessionKey}], loading set to false`
                )
            }
        }

        fetchSessions()

        // No return cleanup needed as we're not setting up any intervals
    }, [hasAttemptedFetch, sessionKey])

    // Manual refresh function - only called when the user clicks the Refresh button
    const handleRefresh = () => {
        console.log("Manual refresh requested")
        setHasAttemptedFetch(false)
        setLoading(true)
        setError(null)
    }

    // Display a loading spinner while fetching data
    if (loading) {
        return (
            <div className="p-4 flex justify-center items-center h-64">
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
                        Loading sessions...
                    </p>
                </div>
            </div>
        )
    }

    // Display an error message if there's an error
    if (error) {
        return (
            <div className="p-4">
                <h1 className="text-2xl font-bold mb-4">Session Data</h1>
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800 mb-4">
                    {error}
                </div>

                <button
                    onClick={handleRefresh}
                    className="mb-6 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                >
                    Retry
                </button>

                {/* Show raw events if available */}
                {rawEvents.length > 0 && (
                    <div className="mt-6">
                        <h2 className="text-xl font-semibold mb-3">
                            Raw Events (Not Paired)
                        </h2>
                        <div className="overflow-x-auto">
                            <table className="min-w-full">
                                <thead>
                                    <tr>
                                        <th className="px-4 py-2">Computer</th>
                                        <th className="px-4 py-2">User</th>
                                        <th className="px-4 py-2">Time</th>
                                        <th className="px-4 py-2">Event</th>
                                        <th className="px-4 py-2">Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {rawEvents.map((event, index) => (
                                        <tr key={index} className="border-t">
                                            <td className="px-4 py-2">
                                                {event.computer_name}
                                            </td>
                                            <td className="px-4 py-2">
                                                {event.user_name}
                                            </td>
                                            <td className="px-4 py-2">
                                                {new Date(
                                                    event.time
                                                ).toLocaleString()}
                                            </td>
                                            <td className="px-4 py-2">
                                                {event.event_type}
                                            </td>
                                            <td className="px-4 py-2">
                                                <span
                                                    className={`px-2 py-1 rounded ${
                                                        event.status ===
                                                        "success"
                                                            ? "bg-green-100 text-green-800"
                                                            : "bg-red-100 text-red-800"
                                                    }`}
                                                >
                                                    {event.status}
                                                </span>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        )
    }

    return (
        <div className="p-4">
            <h1 className="text-2xl font-bold mb-4">Session Data</h1>

            <button
                onClick={handleRefresh}
                className="mb-6 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
            >
                Refresh Data
            </button>

            {sessions.length === 0 ? (
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-amber-800">
                    No session data available
                </div>
            ) : (
                <div className="overflow-x-auto">
                    <table className="min-w-full">
                        <thead>
                            <tr>
                                <th className="px-4 py-2">User</th>
                                <th className="px-4 py-2">Computer</th>
                                <th className="px-4 py-2">Start Time</th>
                                <th className="px-4 py-2">End Time</th>
                                <th className="px-4 py-2">Duration</th>
                                <th className="px-4 py-2">IP</th>
                            </tr>
                        </thead>
                        <tbody>
                            {sessions.map((session, index) => {
                                // Extract additional data from raw events
                                const startEvent = rawEvents.find(
                                    (e) => e.time === session.start
                                )

                                // Note: Using session.user instead of finding user from events when possible
                                return (
                                    <tr key={index} className="border-t">
                                        <td className="px-4 py-2">
                                            {session.user ||
                                                startEvent?.user_name ||
                                                "Unknown"}
                                        </td>
                                        <td className="px-4 py-2">
                                            {startEvent?.computer_name ||
                                                "Unknown"}
                                        </td>
                                        <td className="px-4 py-2">
                                            {new Date(
                                                session.start
                                            ).toLocaleString()}
                                            {session.artificialLogin && (
                                                <span className="ml-1 text-xs text-orange-500">
                                                    (est.)
                                                </span>
                                            )}
                                        </td>
                                        <td className="px-4 py-2">
                                            {session.end === "Current" ? (
                                                <span className="text-green-600 font-medium">
                                                    Current Session
                                                </span>
                                            ) : (
                                                new Date(
                                                    session.end
                                                ).toLocaleString()
                                            )}
                                        </td>
                                        <td className="px-4 py-2 font-medium">
                                            {session.durationText}
                                            {session.fallbackSession && (
                                                <span className="ml-1 text-xs text-blue-500">
                                                    (est.)
                                                </span>
                                            )}
                                        </td>
                                        <td className="px-4 py-2">
                                            {session.ip || "Unknown"}
                                        </td>
                                    </tr>
                                )
                            })}
                        </tbody>
                    </table>
                </div>
            )}

            {/* Show raw events below the sessions */}
            {rawEvents.length > 0 && (
                <div className="mt-8">
                    <h2 className="text-xl font-semibold mb-3">Raw Events</h2>
                    <div className="overflow-x-auto">
                        <table className="min-w-full">
                            <thead>
                                <tr>
                                    <th className="px-4 py-2">Computer</th>
                                    <th className="px-4 py-2">User</th>
                                    <th className="px-4 py-2">Time</th>
                                    <th className="px-4 py-2">Event</th>
                                    <th className="px-4 py-2">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {rawEvents.map((event, index) => (
                                    <tr key={index} className="border-t">
                                        <td className="px-4 py-2">
                                            {event.computer_name}
                                        </td>
                                        <td className="px-4 py-2">
                                            {event.user_name}
                                        </td>
                                        <td className="px-4 py-2">
                                            {new Date(
                                                event.time
                                            ).toLocaleString()}
                                        </td>
                                        <td className="px-4 py-2">
                                            {event.event_type}
                                        </td>
                                        <td className="px-4 py-2">
                                            <span
                                                className={`px-2 py-1 rounded ${
                                                    event.status === "success"
                                                        ? "bg-green-100 text-green-800"
                                                        : "bg-red-100 text-red-800"
                                                }`}
                                            >
                                                {event.status}
                                            </span>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    )
}

export default Sessions
