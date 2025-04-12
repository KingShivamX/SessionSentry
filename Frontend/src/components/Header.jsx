import { motion } from 'framer-motion';

const Header = ({ username, status }) => {
  // Format the username for display (remove domain prefix if present)
  const displayName = username.includes('\\') 
    ? username.split('\\')[1] 
    : username;
  
  // Get current time
  const now = new Date();
  const formattedDate = now.toLocaleDateString('en-US', { 
    weekday: 'long', 
    year: 'numeric', 
    month: 'long', 
    day: 'numeric' 
  });
  
  return (
    <motion.header 
      className="bg-white shadow-sm px-6 py-4 flex items-center justify-between border-b border-amber-200"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <div>
        <h1 className="text-xl font-semibold text-amber-800">Session Dashboard</h1>
        <p className="text-sm text-amber-600">{formattedDate}</p>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="text-right hidden md:block">
          <p className="text-sm font-medium text-amber-700">{displayName}</p>
          <div className="flex items-center">
            <div className={`h-2 w-2 rounded-full mr-1 ${status === 'Online' ? 'bg-green-500' : 'bg-amber-300'}`}></div>
            <p className="text-xs text-amber-600">{status}</p>
          </div>
        </div>
        
        <div className="h-10 w-10 rounded-full bg-amber-100 flex items-center justify-center text-amber-800 font-bold">
          {displayName.charAt(0).toUpperCase()}
        </div>
      </div>
    </motion.header>
  );
};

export default Header;
