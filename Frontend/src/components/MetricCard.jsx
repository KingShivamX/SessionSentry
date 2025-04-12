import { motion } from 'framer-motion';

const MetricCard = ({ title, value, unit, icon, color }) => {
  return (
    <div className="dashboard-card overflow-hidden">
      <div className="p-5">
        <div className="flex items-center">
          <div className={`rounded-full ${color} text-white p-3 mr-4`}>
            {icon}
          </div>
          
          <div>
            <h3 className="text-sm font-medium text-amber-600">{title}</h3>
            <div className="flex items-baseline">
              <p className="text-2xl font-bold text-amber-800">{value}</p>
              <p className="ml-1 text-sm text-amber-600">{unit}</p>
            </div>
          </div>
        </div>
      </div>
      
      <div className={`h-1 ${color}`}></div>
    </div>
  );
};

export default MetricCard;
