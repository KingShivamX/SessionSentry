import { useState, useEffect, useContext } from "react"
import { motion } from "framer-motion"
import { Link } from "react-router-dom"
import { fetchUsers } from "../services/api"
import { UserContext } from "../App"

const Users = () => {
    const [users, setUsers] = useState([])
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(null)
    const [searchTerm, setSearchTerm] = useState("")

    // Get context for updating active user
    const { setActiveUser, setActiveUserStatus } = useContext(UserContext)

    // Reset active user to admin when viewing users list
    useEffect(() => {
        setActiveUser("SessionSentry Admin")
        setActiveUserStatus("Online")
    }, [setActiveUser, setActiveUserStatus])

    const loadUsers = async () => {
        try {
            setLoading(true)
            setError(null)
            const usersList = await fetchUsers()

            if (usersList.length === 0) {
                setError(
                    "No users found in the system. There might be an issue with the API connection."
                )
            } else {
                setError(null)
            }

            setUsers(usersList)
        } catch (error) {
            console.error("Error loading users:", error)
            setError(
                "Failed to load users from the API. Please check your connection or try again later."
            )
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        loadUsers()

        // Set up refresh interval (every 2 minutes)
        const intervalId = setInterval(loadUsers, 2 * 60 * 1000)

        return () => clearInterval(intervalId)
    }, [])

    // Filter users based on search term (can search by computer_name or user_name)
    const filteredUsers = users.filter(
        (user) =>
            user.computer_name
                .toLowerCase()
                .includes(searchTerm.toLowerCase()) ||
            user.user_name.toLowerCase().includes(searchTerm.toLowerCase())
    )

    // Animation variants
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.05,
            },
        },
    }

    const itemVariants = {
        hidden: { y: 20, opacity: 0 },
        visible: { y: 0, opacity: 1 },
    }

    const handleRetry = () => {
        loadUsers()
    }

    // Helper function to format the date
    const formatDate = (dateString) => {
        try {
            return new Date(dateString).toLocaleString()
        } catch (e) {
            return "Unknown"
        }
    }

    return (
        <div className="p-4 md:p-6">
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
                <h1 className="text-2xl font-bold text-amber-800">
                    User Selection
                </h1>

                {!loading && (
                    <button
                        onClick={handleRetry}
                        className="px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-md flex items-center w-full sm:w-auto justify-center"
                    >
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-4 w-4 mr-1"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                            />
                        </svg>
                        Refresh
                    </button>
                )}
            </div>

            {/* Search */}
            <div className="mb-6">
                <div className="relative">
                    <input
                        type="text"
                        className="w-full p-3 pl-10 rounded-lg border border-amber-200 focus:outline-none focus:ring-2 focus:ring-amber-500"
                        placeholder="Search by computer name or username..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                    <div className="absolute left-3 top-3 text-amber-400">
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-5 w-5"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                            />
                        </svg>
                    </div>
                </div>
            </div>

            {/* Error message */}
            {error && (
                <motion.div
                    className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6 text-amber-800 flex items-start"
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <svg
                        xmlns="http://www.w3.org/2000/svg"
                        className="h-5 w-5 mr-2 mt-0.5 text-amber-500 flex-shrink-0"
                        viewBox="0 0 20 20"
                        fill="currentColor"
                    >
                        <path
                            fillRule="evenodd"
                            d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                            clipRule="evenodd"
                        />
                    </svg>
                    <div>
                        <p className="font-medium">{error}</p>
                        <p className="text-sm mt-1">
                            Using mock data or cached results for now.
                        </p>
                    </div>
                </motion.div>
            )}

            {loading ? (
                <motion.div
                    className="flex justify-center items-center h-64"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
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
                            Loading users...
                        </p>
                    </div>
                </motion.div>
            ) : (
                <>
                    {filteredUsers.length === 0 ? (
                        <div className="text-center py-8 text-gray-500">
                            <p>No users found matching "{searchTerm}"</p>
                        </div>
                    ) : (
                        <motion.div
                            className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
                            variants={containerVariants}
                            initial="hidden"
                            animate="visible"
                        >
                            {filteredUsers.map((user, index) => (
                                <motion.div key={index} variants={itemVariants}>
                                    <Link
                                        to={`/user/${encodeURIComponent(
                                            user.computer_name
                                        )}`}
                                        className="block h-full"
                                    >
                                        <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow p-4 border-l-4 border-amber-500 h-full flex flex-col">
                                            <div className="flex flex-col sm:flex-row justify-between gap-2 mb-3">
                                                <div className="flex items-center min-w-0 max-w-full">
                                                    <div className="bg-amber-100 rounded-full p-2 mr-3 flex-shrink-0">
                                                        <svg
                                                            xmlns="http://www.w3.org/2000/svg"
                                                            className="h-6 w-6 text-amber-500"
                                                            fill="none"
                                                            viewBox="0 0 24 24"
                                                            stroke="currentColor"
                                                        >
                                                            <path
                                                                strokeLinecap="round"
                                                                strokeLinejoin="round"
                                                                strokeWidth={2}
                                                                d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                                                            />
                                                        </svg>
                                                    </div>
                                                    <div className="min-w-0 flex-1">
                                                        <h3
                                                            className="font-bold text-gray-800 truncate"
                                                            title={
                                                                user.computer_name
                                                            }
                                                        >
                                                            {user.computer_name}
                                                        </h3>
                                                    </div>
                                                </div>
                                                <div
                                                    className={`mt-1 sm:mt-0 px-2 py-1 rounded-full text-xs font-medium self-start sm:self-center ${
                                                        user.status === "active"
                                                            ? "bg-green-100 text-green-800"
                                                            : "bg-gray-100 text-gray-800"
                                                    }`}
                                                >
                                                    <div className="flex items-center whitespace-nowrap">
                                                        <div
                                                            className={`h-2 w-2 rounded-full mr-1 ${
                                                                user.status ===
                                                                "active"
                                                                    ? "bg-green-500"
                                                                    : "bg-gray-500"
                                                            }`}
                                                        ></div>
                                                        {user.status ===
                                                        "active"
                                                            ? "Online"
                                                            : "Offline"}
                                                    </div>
                                                </div>
                                            </div>

                                            <div className="mt-2 space-y-2 flex-grow">
                                                <div>
                                                    <p className="text-xs text-gray-500">
                                                        User:
                                                    </p>
                                                    <p
                                                        className="text-sm font-medium text-gray-800 break-all"
                                                        title={user.user_name}
                                                    >
                                                        {user.user_name}
                                                    </p>
                                                </div>
                                                <div>
                                                    <p className="text-xs text-gray-500">
                                                        Last seen:
                                                    </p>
                                                    <p className="text-sm text-gray-600">
                                                        {formatDate(
                                                            user.last_seen
                                                        )}
                                                    </p>
                                                </div>
                                            </div>

                                            <div className="mt-3 pt-3 border-t border-gray-100 flex justify-between items-center">
                                                <div className="min-w-0 mr-2">
                                                    <span className="text-xs font-medium text-gray-500 whitespace-nowrap">
                                                        {user.total_events || 0}{" "}
                                                        events
                                                    </span>
                                                    {user.failed_attempts >
                                                        0 && (
                                                        <span className="ml-2 text-xs font-medium text-red-500">
                                                            (
                                                            {
                                                                user.failed_attempts
                                                            }{" "}
                                                            failed)
                                                        </span>
                                                    )}
                                                </div>
                                                <div className="text-amber-500">
                                                    <svg
                                                        xmlns="http://www.w3.org/2000/svg"
                                                        className="h-5 w-5"
                                                        viewBox="0 0 20 20"
                                                        fill="currentColor"
                                                    >
                                                        <path
                                                            fillRule="evenodd"
                                                            d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                                                            clipRule="evenodd"
                                                        />
                                                    </svg>
                                                </div>
                                            </div>
                                        </div>
                                    </Link>
                                </motion.div>
                            ))}
                        </motion.div>
                    )}
                </>
            )}
        </div>
    )
}

export default Users
