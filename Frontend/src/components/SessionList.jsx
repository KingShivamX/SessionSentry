import { motion } from "framer-motion"

const SessionList = ({ sessions }) => {
    // Helper to format date
    const formatDate = (dateString) => {
        try {
            if (dateString === "Current") return "Current Session"
            return new Date(dateString).toLocaleString()
        } catch (e) {
            return "Unknown"
        }
    }

    return (
        <div className="overflow-x-auto">
            {sessions.length === 0 ? (
                <div className="text-center text-gray-500 py-6">
                    No session data available
                </div>
            ) : (
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Login Time
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Logout Time
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                Duration
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                User
                            </th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                                IP Address
                            </th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {sessions.map((session, index) => (
                            <tr
                                key={index}
                                className={`hover:bg-gray-50 ${
                                    session.active ? "bg-green-50" : ""
                                }`}
                            >
                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-900">
                                    {formatDate(session.start)}
                                    {session.artificialLogin && (
                                        <span className="ml-1 text-xs text-orange-500">
                                            (est.)
                                        </span>
                                    )}
                                </td>
                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                                    {formatDate(session.end)}
                                </td>
                                <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                                    <span className="font-medium">
                                        {session.durationText || "Unknown"}
                                    </span>
                                    {session.fallbackSession && (
                                        <span className="ml-1 text-xs text-blue-500">
                                            (est.)
                                        </span>
                                    )}
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-500 max-w-[150px]">
                                    <div
                                        className="truncate"
                                        title={session.user || "Unknown"}
                                    >
                                        {session.user || "Unknown"}
                                    </div>
                                </td>
                                <td className="px-4 py-3 text-sm text-gray-500 max-w-[140px]">
                                    <div
                                        className="truncate"
                                        title={session.ip || "Unknown"}
                                    >
                                        {session.ip || "Unknown"}
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
        </div>
    )
}

export default SessionList
