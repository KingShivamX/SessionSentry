import { useState } from "react"
import { motion } from "framer-motion"
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    LineChart,
    Line,
} from "recharts"
import { parseEventDate } from "../utils/sessionCalculator"

// Dashboard components
import MetricCard from "./MetricCard"
import SessionList from "./SessionList"

const Dashboard = ({ sessions, metrics }) => {
    const [activeTab, setActiveTab] = useState("overview")

    // Animation variants for staggered animations
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.1,
            },
        },
    }

    const itemVariants = {
        hidden: { y: 20, opacity: 0 },
        visible: { y: 0, opacity: 1 },
    }

    // Helper to format date
    const formatDate = (dateString) => {
        try {
            return new Date(dateString).toLocaleString()
        } catch (e) {
            return "Unknown"
        }
    }

    return (
        <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="space-y-6"
        >
            {/* Tabs */}
            <div className="flex space-x-2 border-b border-amber-200 overflow-x-auto pb-1">
                <button
                    onClick={() => setActiveTab("overview")}
                    className={`px-4 py-2 text-sm font-medium whitespace-nowrap ${
                        activeTab === "overview"
                            ? "text-amber-600 border-b-2 border-amber-500"
                            : "text-amber-700 hover:text-amber-900"
                    }`}
                >
                    Overview
                </button>
                <button
                    onClick={() => setActiveTab("sessions")}
                    className={`px-4 py-2 text-sm font-medium whitespace-nowrap ${
                        activeTab === "sessions"
                            ? "text-amber-600 border-b-2 border-amber-500"
                            : "text-amber-700 hover:text-amber-900"
                    }`}
                >
                    Sessions
                </button>
            </div>

            {/* Overview Tab */}
            {activeTab === "overview" && (
                <div className="space-y-6">
                    {/* Metrics */}
                    <motion.div
                        className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 mb-6"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                    >
                        {/* Today's Usage */}
                        <div className="bg-white rounded-lg shadow-sm p-4 border-l-4 border-blue-500">
                            <h3 className="text-sm text-gray-500 mb-1">
                                Today's Usage
                            </h3>
                            <p className="text-2xl font-bold text-gray-800">
                                {metrics.todayHours} hrs
                            </p>
                        </div>

                        {/* Weekly Usage */}
                        <div className="bg-white rounded-lg shadow-sm p-4 border-l-4 border-amber-500">
                            <h3 className="text-sm text-gray-500 mb-1">
                                Weekly Usage
                            </h3>
                            <p className="text-2xl font-bold text-gray-800">
                                {metrics.weeklyHours} hrs
                            </p>
                        </div>

                        {/* Monthly Usage */}
                        <div className="bg-white rounded-lg shadow-sm p-4 border-l-4 border-green-500">
                            <h3 className="text-sm text-gray-500 mb-1">
                                Monthly Usage
                            </h3>
                            <p className="text-2xl font-bold text-gray-800">
                                {metrics.monthlyHours} hrs
                            </p>
                        </div>

                        {/* Current Status */}
                        <div className="bg-white rounded-lg shadow-sm p-4 border-l-4 border-purple-500">
                            <h3 className="text-sm text-gray-500 mb-1">
                                Current Status
                            </h3>
                            <div className="flex items-center">
                                <div
                                    className={`h-3 w-3 rounded-full mr-2 ${
                                        metrics.currentStatus === "Online"
                                            ? "bg-green-500"
                                            : "bg-gray-400"
                                    }`}
                                ></div>
                                <p className="text-lg font-bold text-gray-800 truncate">
                                    {metrics.currentStatus}
                                </p>
                            </div>
                        </div>
                    </motion.div>

                    {/* Weekly Activity Chart */}
                    <motion.div
                        variants={itemVariants}
                        className="dashboard-card p-4 bg-white rounded-lg shadow-sm"
                    >
                        <h2 className="text-lg font-semibold text-amber-800 mb-4">
                            Weekly Activity
                        </h2>
                        <div className="h-64">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart
                                    data={metrics.dailySessions || []}
                                    margin={{
                                        top: 5,
                                        right: 30,
                                        left: 0,
                                        bottom: 5,
                                    }}
                                >
                                    <CartesianGrid strokeDasharray="3 3" />
                                    <XAxis dataKey="day" />
                                    <YAxis
                                        label={{
                                            value: "Hours",
                                            angle: -90,
                                            position: "insideLeft",
                                        }}
                                    />
                                    <Tooltip
                                        formatter={(value) => [
                                            `${value} hours`,
                                            "Duration",
                                        ]}
                                    />
                                    <Bar
                                        dataKey="hours"
                                        fill="#f59e0b"
                                        radius={[4, 4, 0, 0]}
                                        animationDuration={1500}
                                    />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </motion.div>

                    {/* Recent Sessions */}
                    <motion.div
                        variants={itemVariants}
                        className="dashboard-card p-4 bg-white rounded-lg shadow-sm"
                    >
                        <h2 className="text-lg font-semibold text-amber-800 mb-4">
                            Recent Sessions
                        </h2>
                        <SessionList
                            sessions={metrics.sessionHistory?.slice(0, 5) || []}
                        />
                    </motion.div>
                </div>
            )}

            {/* Sessions Tab */}
            {activeTab === "sessions" && (
                <div className="space-y-6">
                    <motion.div
                        variants={itemVariants}
                        className="dashboard-card p-4 bg-white rounded-lg shadow-sm"
                    >
                        <h2 className="text-lg font-semibold text-amber-800 mb-4">
                            All Sessions
                        </h2>
                        <SessionList sessions={metrics.sessionHistory || []} />
                    </motion.div>
                </div>
            )}
        </motion.div>
    )
}

export default Dashboard
