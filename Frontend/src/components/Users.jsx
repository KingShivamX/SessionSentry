import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Link } from "react-router-dom"
import { fetchUsers } from "../services/api"

const Users = () => {
    const [users, setUsers] = useState([])
    const [loading, setLoading] = useState(true)
    const [searchTerm, setSearchTerm] = useState("")

    useEffect(() => {
        const loadUsers = async () => {
            try {
                setLoading(true)
                const usersList = await fetchUsers()
                setUsers(usersList)
            } catch (error) {
                console.error("Error loading users:", error)
            } finally {
                setLoading(false)
            }
        }

        loadUsers()
    }, [])

    // Filter users based on search term
    const filteredUsers = users.filter((user) =>
        user.toLowerCase().includes(searchTerm.toLowerCase())
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

    return (
        <div className="p-4 md:p-6">
            <h1 className="text-2xl font-bold text-amber-800 mb-6">
                User Selection
            </h1>

            {/* Search */}
            <div className="mb-6">
                <div className="relative">
                    <input
                        type="text"
                        className="w-full p-3 pl-10 rounded-lg border border-amber-200 focus:outline-none focus:ring-2 focus:ring-amber-500"
                        placeholder="Search users..."
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
                            className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"
                            variants={containerVariants}
                            initial="hidden"
                            animate="visible"
                        >
                            {filteredUsers.map((user, index) => (
                                <motion.div key={index} variants={itemVariants}>
                                    <Link
                                        to={`/user/${encodeURIComponent(user)}`}
                                        className="block"
                                    >
                                        <div className="bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow p-4 border-l-4 border-amber-500">
                                            <div className="flex items-center">
                                                <div className="bg-amber-100 rounded-full p-2 mr-3">
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
                                                            d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                                                        />
                                                    </svg>
                                                </div>
                                                <div className="truncate">
                                                    <p className="font-medium text-gray-800 hover:text-amber-600 truncate">
                                                        {user}
                                                    </p>
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
