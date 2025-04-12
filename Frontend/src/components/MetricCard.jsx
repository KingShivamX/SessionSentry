import { motion } from "framer-motion"

const MetricCard = ({ title, value, unit, icon, color }) => {
    return (
        <div className="bg-white rounded-lg shadow-sm p-4 h-full">
            <div className="flex items-start">
                <div className={`rounded-full p-2 ${color} text-white mr-3`}>
                    {icon}
                </div>
                <div>
                    <h3 className="text-sm text-gray-500">{title}</h3>
                    <div className="flex items-baseline">
                        <span className="text-2xl font-bold text-gray-800 mr-1">
                            {value}
                        </span>
                        {unit && (
                            <span className="text-sm text-gray-500">
                                {unit}
                            </span>
                        )}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default MetricCard
