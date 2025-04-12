import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  LineChart,
  Line
} from 'recharts';

// Dashboard components
import MetricCard from './MetricCard';
import SessionList from './SessionList';

const Dashboard = ({ sessions, metrics }) => {
  const [activeTab, setActiveTab] = useState('overview');
  
  // Animation variants for staggered animations
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { 
        staggerChildren: 0.1
      }
    }
  };
  
  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: { y: 0, opacity: 1 }
  };

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-6"
    >
      {/* Tabs */}
      <div className="flex space-x-2 border-b border-amber-200">
        <button
          onClick={() => setActiveTab('overview')}
          className={`px-4 py-2 text-sm font-medium ${
            activeTab === 'overview'
              ? 'text-amber-600 border-b-2 border-amber-500'
              : 'text-amber-700 hover:text-amber-900'
          }`}
        >
          Overview
        </button>
        <button
          onClick={() => setActiveTab('sessions')}
          className={`px-4 py-2 text-sm font-medium ${
            activeTab === 'sessions'
              ? 'text-amber-600 border-b-2 border-amber-500'
              : 'text-amber-700 hover:text-amber-900'
          }`}
        >
          Sessions
        </button>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <motion.div variants={itemVariants}>
              <MetricCard
                title="Today's Hours"
                value={metrics.todayHours}
                unit="hours"
                icon={
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                }
                color="bg-blue-500"
              />
            </motion.div>
            
            <motion.div variants={itemVariants}>
              <MetricCard
                title="This Week"
                value={metrics.weeklyHours}
                unit="hours"
                icon={
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                }
                color="bg-purple-500"
              />
            </motion.div>
            
            <motion.div variants={itemVariants}>
              <MetricCard
                title="This Month"
                value={metrics.monthlyHours}
                unit="hours"
                icon={
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                }
                color="bg-green-500"
              />
            </motion.div>
          </div>

          {/* Weekly Activity Chart */}
          <motion.div
            variants={itemVariants}
            className="dashboard-card p-4"
          >
            <h2 className="text-lg font-semibold text-amber-800 mb-4">Weekly Activity</h2>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={metrics.dailySessions || []}
                  margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis label={{ value: 'Hours', angle: -90, position: 'insideLeft' }} />
                  <Tooltip formatter={(value) => [`${value} hours`, 'Duration']} />
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
            className="dashboard-card p-4"
          >
            <h2 className="text-lg font-semibold text-amber-800 mb-4">Recent Sessions</h2>
            <SessionList sessions={metrics.sessionHistory?.slice(0, 5) || []} />
          </motion.div>
        </div>
      )}

      {/* Sessions Tab */}
      {activeTab === 'sessions' && (
        <div className="space-y-6">
          <motion.div
            variants={itemVariants}
            className="dashboard-card p-4"
          >
            <h2 className="text-lg font-semibold text-amber-800 mb-4">All Sessions</h2>
            <SessionList sessions={metrics.sessionHistory || []} />
          </motion.div>
        </div>
      )}
    </motion.div>
  );
};

export default Dashboard;
