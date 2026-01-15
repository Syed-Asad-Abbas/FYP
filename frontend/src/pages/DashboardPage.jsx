import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../hooks/useAuth';
import { scanService } from '../services/scanService';
import PageTransition from '../components/PageTransition';
import GlassCard from '../components/GlassCard';
import ResultBadge from '../components/ResultBadge';
import { LoadingCard } from '../components/LoadingShimmer';
import { NoHistoryState } from '../components/EmptyState';

function DashboardPage() {
    const { user } = useAuth();
    const navigate = useNavigate();
    const [recentScans, setRecentScans] = useState([]);
    const [loading, setLoading] = useState(true);
    const [stats, setStats] = useState({ total: 0, phishing: 0, benign: 0 });

    useEffect(() => {
        loadDashboardData();
    }, []);

    const loadDashboardData = async () => {
        try {
            const { results, pagination } = await scanService.getHistory(1, 5);
            setRecentScans(results);

            // Calculate stats from total history
            const allData = await scanService.getHistory(1, 1000);
            const phishing = allData.results.filter(r => r.prediction === 'phishing').length;
            setStats({
                total: allData.pagination.total,
                phishing,
                benign: allData.pagination.total - phishing
            });
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        } finally {
            setLoading(false);
        }
    };

    const statCards = [
        {
            label: 'Total Scans',
            value: stats.total,
            icon: (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
            ),
            color: 'cyan'
        },
        {
            label: 'Threats Found',
            value: stats.phishing,
            icon: (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
            ),
            color: 'rose'
        },
        {
            label: 'Safe URLs',
            value: stats.benign,
            icon: (
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            ),
            color: 'emerald'
        },
    ];

    return (
        <PageTransition>
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Header */}
                <div className="mb-8">
                    <motion.h1
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-2xl sm:text-3xl font-light text-white"
                    >
                        Welcome back
                    </motion.h1>
                    <p className="text-slate-400 mt-1">{user?.email}</p>
                </div>

                {/* Quick Scan CTA */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="mb-8"
                >
                    <GlassCard className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                        <div>
                            <h2 className="text-lg font-medium text-white">Ready to scan a URL?</h2>
                            <p className="text-slate-400 text-sm mt-1">
                                Analyze any website for phishing threats in seconds.
                            </p>
                        </div>
                        <Link to="/scan" className="btn-primary shrink-0">
                            Scan URL
                        </Link>
                    </GlassCard>
                </motion.div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-8">
                    {statCards.map((stat, index) => (
                        <motion.div
                            key={stat.label}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.15 + index * 0.05 }}
                        >
                            <GlassCard className="flex items-center gap-4">
                                <div className={`
                  w-12 h-12 rounded-xl flex items-center justify-center
                  ${stat.color === 'cyan' ? 'bg-cyan-400/10 text-cyan-400' : ''}
                  ${stat.color === 'rose' ? 'bg-rose-400/10 text-rose-400' : ''}
                  ${stat.color === 'emerald' ? 'bg-emerald-400/10 text-emerald-400' : ''}
                `}>
                                    {stat.icon}
                                </div>
                                <div>
                                    <p className="text-2xl font-medium text-white">{stat.value}</p>
                                    <p className="text-sm text-slate-400">{stat.label}</p>
                                </div>
                            </GlassCard>
                        </motion.div>
                    ))}
                </div>

                {/* Recent Activity */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <div className="flex items-center justify-between mb-4">
                        <h2 className="text-lg font-medium text-white">Recent Activity</h2>
                        {recentScans.length > 0 && (
                            <Link to="/history" className="text-sm text-cyan-400 hover:text-cyan-300 transition-colors">
                                View all
                            </Link>
                        )}
                    </div>

                    {loading ? (
                        <div className="space-y-4">
                            <LoadingCard />
                            <LoadingCard />
                        </div>
                    ) : recentScans.length === 0 ? (
                        <GlassCard>
                            <NoHistoryState onScan={() => navigate('/scan')} />
                        </GlassCard>
                    ) : (
                        <div className="space-y-3">
                            {recentScans.map((scan, index) => (
                                <motion.div
                                    key={scan._id}
                                    initial={{ opacity: 0, x: -10 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.35 + index * 0.05 }}
                                >
                                    <Link to={`/result/${scan._id}`}>
                                        <GlassCard hover className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                            <div className="flex-1 min-w-0">
                                                <p className="text-white font-medium truncate">{scan.url}</p>
                                                <p className="text-sm text-slate-500">
                                                    {new Date(scan.timestamp).toLocaleDateString('en-US', {
                                                        month: 'short',
                                                        day: 'numeric',
                                                        hour: '2-digit',
                                                        minute: '2-digit'
                                                    })}
                                                </p>
                                            </div>
                                            <ResultBadge
                                                prediction={scan.prediction}
                                                confidence={scan.confidence}
                                                size="sm"
                                            />
                                        </GlassCard>
                                    </Link>
                                </motion.div>
                            ))}
                        </div>
                    )}
                </motion.div>
            </div>
        </PageTransition>
    );
}

export default DashboardPage;
