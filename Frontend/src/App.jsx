import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import { fetchLoginEvents } from './services/api';
import { calculateSessionMetrics } from './utils/sessionCalculator';

function App() {
  const [userSessions, setUserSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [metrics, setMetrics] = useState({
    todayHours: 0,
    weeklyHours: 0,
    monthlyHours: 0,
    currentStatus: 'Offline'
  });
  
  // Current user - in future versions this would be selectable
  const currentUser = 'SHIVAM\\shiva'; 

  useEffect(() => {
    const loadSessionData = async () => {
      try {
        setLoading(true);
        const events = await fetchLoginEvents();
        
        // Filter events for the current user
        const userEvents = events.filter(event => event.user_name === currentUser);
        setUserSessions(userEvents);
        
        // Calculate metrics from the session data
        const calculatedMetrics = calculateSessionMetrics(userEvents);
        setMetrics(calculatedMetrics);
      } catch (error) {
        console.error('Error loading session data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadSessionData();
    
    // Set up interval to refresh data every 5 minutes
    const intervalId = setInterval(loadSessionData, 5 * 60 * 1000);
    
    return () => clearInterval(intervalId);
  }, [currentUser]);

  return (
    <div className="flex h-screen bg-gray-50">
      <Sidebar />
      
      <div className="flex flex-col flex-1 overflow-hidden">
        <Header username={currentUser} status={metrics.currentStatus} />
        
        <main className="flex-1 overflow-y-auto p-4 md:p-6">
          {loading ? (
            <motion.div 
              className="flex justify-center items-center h-full"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
            >
              <div className="text-blue-500">
                <svg className="animate-spin h-12 w-12" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <p className="mt-2 text-center font-medium">Loading session data...</p>
              </div>
            </motion.div>
          ) : (
            <Dashboard sessions={userSessions} metrics={metrics} />
          )}
        </main>
      </div>
    </div>
  );
}

export default App;