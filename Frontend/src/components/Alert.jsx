import { useState, useEffect } from "react"
import { motion } from "framer-motion"

const Alert = () => {
    const [alerts, setAlerts] = useState([
        {
            id: 1,
            type: "security",
            message: "Multiple failed login attempts detected",
            computer: "PC-001",
            user: "john_doe",
            timestamp: new Date(Date.now() - 3600000).toISOString(),
            severity: "high",
        },
        {
            id: 2,
            type: "system",
            message: "System update pending",
            computer: "PC-002",
            user: "jane_smith",
            timestamp: new Date(Date.now() - 7200000).toISOString(),
            severity: "medium",
        },
        {
            id: 3,
            type: "security",
            message: "Unusual login time detected",
            computer: "LAPTOP-003",
            user: "admin",
            timestamp: new Date(Date.now() - 10800000).toISOString(),
            severity: "medium",
        },
    ])

    // Placeholder for future API integration
    useEffect(() => {
        // This would fetch alerts from an API in a real implementation
    }, [])

    // Helper to format date
    const formatDate = (dateString) => {
        try {
            return new Date(dateString).toLocaleString()
        } catch (e) {
            return "Unknown"
        }
    }

    // Get severity badge style
    const getSeverityStyle = (severity) => {
        switch (severity) {
            case "high":
                return "bg-red-100 text-red-800"
            case "medium":
                return "bg-amber-100 text-amber-800"
            case "low":
                return "bg-blue-100 text-blue-800"
            default:
                return "bg-gray-100 text-gray-800"
        }
    }

    return (
        <div className="p-4 md:p-6">
            <h1 className="text-2xl font-bold text-amber-800 mb-6">
                Security Alerts
            </h1>

            <div className="bg-white rounded-lg shadow-md overflow-hidden">
                {alerts.length === 0 ? (
                    <div className="p-6 text-center text-gray-500">
                        No alerts found
                    </div>
                ) : (
                    <motion.div
                        className="divide-y divide-gray-200"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        {alerts.map((alert) => (
                            <motion.div
                                key={alert.id}
                                className="p-4 hover:bg-gray-50"
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                            >
                                <div className="flex flex-wrap items-start gap-2">
                                    <span
                                        className={`px-2.5 py-0.5 rounded-full text-xs font-medium ${getSeverityStyle(
                                            alert.severity
                                        )}`}
                                    >
                                        {alert.severity.toUpperCase()}
                                    </span>
                                    <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                                        {alert.type}
                                    </span>
                                </div>

                                <h3 className="text-lg font-medium text-gray-900 mt-2">
                                    {alert.message}
                                </h3>

                                <div className="mt-2 text-sm text-gray-500">
                                    <p>
                                        Computer:{" "}
                                        <span className="font-medium">
                                            {alert.computer}
                                        </span>
                                    </p>
                                    <p>
                                        User:{" "}
                                        <span className="font-medium">
                                            {alert.user}
                                        </span>
                                    </p>
                                    <p>
                                        Time:{" "}
                                        <span className="font-medium">
                                            {formatDate(alert.timestamp)}
                                        </span>
                                    </p>
                                </div>

                                <div className="mt-3">
                                    <button className="text-sm text-amber-600 hover:text-amber-800">
                                        View Details
                                    </button>
                                </div>
                            </motion.div>
                        ))}
                    </motion.div>
                )}
            </div>
        </div>
    )
}

export default Alert
