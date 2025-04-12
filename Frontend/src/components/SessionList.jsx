import { motion } from 'framer-motion';

const SessionList = ({ sessions }) => {
  if (!sessions || sessions.length === 0) {
    return (
      <div className="text-center py-8 text-amber-600">
        No session data available
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-amber-200">
        <thead className="bg-amber-50">
          <tr>
            <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-amber-700 uppercase tracking-wider">
              Start Time
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-amber-700 uppercase tracking-wider">
              End Time
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-amber-700 uppercase tracking-wider">
              Duration
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-amber-700 uppercase tracking-wider">
              IP Address
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-amber-700 uppercase tracking-wider">
              Status
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-amber-100">
          {sessions.map((session, index) => (
            <motion.tr 
              key={`${session.start}-${index}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className="hover:bg-amber-50"
            >
              <td className="px-4 py-3 whitespace-nowrap text-sm text-amber-900">
                {session.start}
              </td>
              <td className="px-4 py-3 whitespace-nowrap text-sm text-amber-900">
                {session.end}
              </td>
              <td className="px-4 py-3 whitespace-nowrap text-sm text-amber-900">
                {session.durationText}
              </td>
              <td className="px-4 py-3 whitespace-nowrap text-sm text-amber-900">
                {session.ip}
              </td>
              <td className="px-4 py-3 whitespace-nowrap">
                {session.ongoing ? (
                  <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                    Active
                  </span>
                ) : (
                  <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-amber-100 text-amber-800">
                    Completed
                  </span>
                )}
              </td>
            </motion.tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default SessionList;
