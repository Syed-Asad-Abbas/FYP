import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { adminService } from '../services/adminService';
import PageTransition from '../components/PageTransition';
import GlassCard from '../components/GlassCard';
import HealthIndicator from '../components/HealthIndicator';
import { LoadingCard } from '../components/LoadingShimmer';
import { PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts';

function AdminDashboardPage() {
    const [stats, setStats] = useState(null);
    const [health, setHealth] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            const [statsData, healthData] = await Promise.all([
                adminService.getStats(),
                adminService.getMLHealth()
            ]);
            setStats(statsData);
            setHealth(healthData);
        } catch (error) {
            console.error('Failed to load admin data:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <PageTransition>
                <div className="max-w-7xl mx-auto px-4 py-8">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                        <LoadingCard />
                        <LoadingCard />
                        <LoadingCard />
                        <LoadingCard />
                    </div>
                </div>
            </PageTransition>
        );
    }

    const pieData = [
        { name: 'Phishing', value: stats?.stats?.phishingCount || 0, color: '#F87171' },
        { name: 'Benign', value: stats?.stats?.benignCount || 0, color: '#4ADE80' },
    ];

    const statCards = [
        {
            label: 'Total Scans',
            value: stats?.stats?.totalScans || 0,
            color: 'cyan'
        },
        {
            label: 'Total Users',
            value: stats?.stats?.totalUsers || 0,
            color: 'cyan'
        },
        {
            label: 'Threats Detected',
            value: stats?.stats?.phishingCount || 0,
            color: 'rose'
        },
        {
            label: 'Phishing Rate',
            value: `${stats?.stats?.phishingRatio || 0}%`,
            color: 'amber'
        },
    ];

    // Custom tooltip for charts
    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div className="glass rounded-lg px-3 py-2 text-sm">
                    <p className="text-white">{label || payload[0].name}</p>
                    <p className="text-slate-400">{payload[0].value} scans</p>
                </div>
            );
        }
        return null;
    };

    return (
        <PageTransition>
            <div className="max-w-7xl mx-auto px-4 py-8">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8"
                >
                    <div>
                        <h1 className="text-2xl sm:text-3xl font-light text-white">Admin Dashboard</h1>
                        <p className="text-slate-400 mt-1">System overview and analytics</p>
                    </div>
                    <HealthIndicator status={health?.status} />
                </motion.div>

                {/* Stat Cards */}
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
                    {statCards.map((stat, index) => (
                        <motion.div
                            key={stat.label}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.1 + index * 0.05 }}
                        >
                            <GlassCard>
                                <p className="text-sm text-slate-400 mb-1">{stat.label}</p>
                                <p className={`text-2xl sm:text-3xl font-medium ${stat.color === 'cyan' ? 'text-cyan-400' : ''
                                    }${stat.color === 'rose' ? 'text-rose-400' : ''
                                    }${stat.color === 'amber' ? 'text-amber-400' : ''
                                    }`}>
                                    {stat.value}
                                </p>
                            </GlassCard>
                        </motion.div>
                    ))}
                </div>

                {/* Charts Row */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                    {/* Pie Chart */}
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                    >
                        <GlassCard>
                            <h3 className="text-lg font-medium text-white mb-4">Detection Breakdown</h3>
                            <div className="h-[200px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <PieChart>
                                        <Pie
                                            data={pieData}
                                            cx="50%"
                                            cy="50%"
                                            innerRadius={50}
                                            outerRadius={80}
                                            paddingAngle={2}
                                            dataKey="value"
                                        >
                                            {pieData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Pie>
                                        <Tooltip content={<CustomTooltip />} />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>
                            <div className="flex justify-center gap-6 mt-4">
                                {pieData.map((item) => (
                                    <div key={item.name} className="flex items-center gap-2">
                                        <span
                                            className="w-3 h-3 rounded-full"
                                            style={{ backgroundColor: item.color }}
                                        />
                                        <span className="text-sm text-slate-400">{item.name}</span>
                                    </div>
                                ))}
                            </div>
                        </GlassCard>
                    </motion.div>

                    {/* Line Chart - Daily Activity */}
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.35 }}
                    >
                        <GlassCard>
                            <h3 className="text-lg font-medium text-white mb-4">Daily Activity</h3>
                            <div className="h-[200px]">
                                <ResponsiveContainer width="100%" height="100%">
                                    <LineChart data={stats?.dailyStats || []}>
                                        <XAxis
                                            dataKey="_id"
                                            axisLine={false}
                                            tickLine={false}
                                            tick={{ fill: '#64748B', fontSize: 11 }}
                                            tickFormatter={(value) => {
                                                const date = new Date(value);
                                                return `${date.getMonth() + 1}/${date.getDate()}`;
                                            }}
                                        />
                                        <YAxis
                                            axisLine={false}
                                            tickLine={false}
                                            tick={{ fill: '#64748B', fontSize: 11 }}
                                            width={30}
                                        />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Line
                                            type="monotone"
                                            dataKey="total"
                                            stroke="#38BDF8"
                                            strokeWidth={2}
                                            dot={false}
                                            activeDot={{ r: 4, fill: '#38BDF8' }}
                                        />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </GlassCard>
                    </motion.div>
                </div>

                {/* Top Domains */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                >
                    <GlassCard>
                        <h3 className="text-lg font-medium text-white mb-4">Most Scanned Domains</h3>
                        {stats?.topDomains?.length > 0 ? (
                            <div className="space-y-3">
                                {stats.topDomains.slice(0, 5).map((domain, index) => (
                                    <div
                                        key={domain.domain}
                                        className="flex items-center gap-4"
                                    >
                                        <span className="text-sm text-slate-500 w-6">{index + 1}.</span>
                                        <div className="flex-1 min-w-0">
                                            <p className="text-white truncate">{domain.domain}</p>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <span className="text-sm text-slate-400">{domain.count} scans</span>
                                            {domain.phishingCount > 0 && (
                                                <span className="badge-danger text-xs">
                                                    {domain.phishingCount} threats
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-slate-400 text-center py-4">No domains scanned yet</p>
                        )}
                    </GlassCard>
                </motion.div>
            </div>
        </PageTransition>
    );
}

export default AdminDashboardPage;
