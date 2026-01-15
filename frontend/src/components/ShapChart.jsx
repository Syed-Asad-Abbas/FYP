import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

function ShapChart({ shapValues, maxFeatures = 5 }) {
    if (!shapValues || Object.keys(shapValues).length === 0) {
        return null;
    }

    // Convert SHAP values object to sorted array
    const data = Object.entries(shapValues)
        .map(([name, value]) => ({
            name: formatFeatureName(name),
            value: Math.abs(value),
            rawValue: value,
            isPositive: value > 0
        }))
        .sort((a, b) => b.value - a.value)
        .slice(0, maxFeatures);

    // Custom tooltip
    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            const item = payload[0].payload;
            return (
                <div className="glass rounded-lg px-3 py-2 text-sm">
                    <p className="text-white font-medium">{item.name}</p>
                    <p className="text-slate-400">
                        Impact: <span className={item.isPositive ? 'text-rose-400' : 'text-emerald-400'}>
                            {item.rawValue.toFixed(3)}
                        </span>
                    </p>
                </div>
            );
        }
        return null;
    };

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.4 }}
            className="w-full"
        >
            <h4 className="text-sm font-medium text-slate-400 mb-4">
                Top Feature Contributions
            </h4>

            <div className="h-[200px] sm:h-[240px]">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                        data={data}
                        layout="vertical"
                        margin={{ top: 0, right: 20, left: 0, bottom: 0 }}
                        barCategoryGap={8}
                    >
                        <XAxis
                            type="number"
                            hide
                            domain={[0, 'dataMax']}
                        />
                        <YAxis
                            type="category"
                            dataKey="name"
                            axisLine={false}
                            tickLine={false}
                            tick={{ fill: '#94A3B8', fontSize: 12 }}
                            width={100}
                        />
                        <Tooltip
                            content={<CustomTooltip />}
                            cursor={{ fill: 'rgba(255,255,255,0.03)' }}
                        />
                        <Bar
                            dataKey="value"
                            radius={[0, 6, 6, 0]}
                            animationDuration={600}
                            animationEasing="ease-out"
                        >
                            {data.map((entry, index) => (
                                <Cell
                                    key={`cell-${index}`}
                                    fill={entry.isPositive ? 'rgba(248, 113, 113, 0.6)' : 'rgba(74, 222, 128, 0.6)'}
                                />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>

            {/* Legend */}
            <div className="flex items-center justify-center gap-6 mt-4 text-xs text-slate-500">
                <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded bg-rose-400/60" />
                    <span>Increases risk</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded bg-emerald-400/60" />
                    <span>Decreases risk</span>
                </div>
            </div>
        </motion.div>
    );
}

// Helper to format feature names
function formatFeatureName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase())
        .substring(0, 15) + (name.length > 15 ? '...' : '');
}

export default ShapChart;
