import { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { scanService } from '../services/scanService';
import PageTransition from '../components/PageTransition';
import GlassCard from '../components/GlassCard';
import ResultBadge from '../components/ResultBadge';
import { LoadingCard } from '../components/LoadingShimmer';
import { NoHistoryState } from '../components/EmptyState';

function HistoryPage() {
    const [scans, setScans] = useState([]);
    const [loading, setLoading] = useState(true);
    const [pagination, setPagination] = useState({ page: 1, pages: 1, total: 0 });
    const [filter, setFilter] = useState('all'); // 'all' | 'phishing' | 'benign'

    const navigate = useNavigate();

    useEffect(() => {
        loadHistory(1);
    }, []);

    const loadHistory = async (page) => {
        setLoading(true);
        try {
            const data = await scanService.getHistory(page, 10);
            setScans(data.results);
            setPagination(data.pagination);
        } catch (error) {
            console.error('Failed to load history:', error);
        } finally {
            setLoading(false);
        }
    };

    const filteredScans = scans.filter(scan => {
        if (filter === 'all') return true;
        return scan.prediction === filter;
    });

    const handlePageChange = (newPage) => {
        if (newPage >= 1 && newPage <= pagination.pages) {
            loadHistory(newPage);
        }
    };

    return (
        <PageTransition>
            <div className="max-w-4xl mx-auto px-4 py-8">
                {/* Header */}
                <motion.div
                    initial={{ opacity: 0, y: -10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6"
                >
                    <div>
                        <h1 className="text-2xl sm:text-3xl font-light text-white">Scan History</h1>
                        <p className="text-slate-400 mt-1">{pagination.total} total scans</p>
                    </div>

                    {/* Filter Tabs */}
                    {scans.length > 0 && (
                        <div className="flex gap-1 p-1 glass rounded-xl">
                            {['all', 'phishing', 'benign'].map((option) => (
                                <button
                                    key={option}
                                    onClick={() => setFilter(option)}
                                    className={`
                    px-4 py-2 rounded-lg text-sm font-medium transition-all capitalize
                    ${filter === option
                                            ? 'bg-white/10 text-white'
                                            : 'text-slate-400 hover:text-white'
                                        }
                  `}
                                >
                                    {option}
                                </button>
                            ))}
                        </div>
                    )}
                </motion.div>

                {/* Content */}
                {loading ? (
                    <div className="space-y-4">
                        <LoadingCard />
                        <LoadingCard />
                        <LoadingCard />
                    </div>
                ) : scans.length === 0 ? (
                    <GlassCard>
                        <NoHistoryState onScan={() => navigate('/scan')} />
                    </GlassCard>
                ) : (
                    <>
                        {/* Scan List */}
                        <div className="space-y-3 mb-6">
                            {filteredScans.map((scan, index) => (
                                <motion.div
                                    key={scan._id}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: index * 0.03 }}
                                >
                                    <Link to={`/result/${scan._id}`}>
                                        <GlassCard hover className="flex flex-col sm:flex-row sm:items-center gap-4">
                                            {/* URL and Time */}
                                            <div className="flex-1 min-w-0">
                                                <p className="text-white font-medium truncate mb-1">
                                                    {scan.url}
                                                </p>
                                                <p className="text-sm text-slate-500">
                                                    {new Date(scan.timestamp).toLocaleString('en-US', {
                                                        month: 'short',
                                                        day: 'numeric',
                                                        year: 'numeric',
                                                        hour: '2-digit',
                                                        minute: '2-digit'
                                                    })}
                                                </p>
                                            </div>

                                            {/* Result */}
                                            <div className="flex items-center gap-3">
                                                <ResultBadge
                                                    prediction={scan.prediction}
                                                    confidence={scan.confidence}
                                                    size="sm"
                                                />
                                                <svg className="w-5 h-5 text-slate-500 hidden sm:block" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                                </svg>
                                            </div>
                                        </GlassCard>
                                    </Link>
                                </motion.div>
                            ))}
                        </div>

                        {/* Empty Filter State */}
                        {filteredScans.length === 0 && (
                            <GlassCard className="text-center py-8">
                                <p className="text-slate-400">No {filter} URLs found</p>
                            </GlassCard>
                        )}

                        {/* Pagination */}
                        {pagination.pages > 1 && (
                            <motion.div
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: 0.2 }}
                                className="flex items-center justify-center gap-2"
                            >
                                <button
                                    onClick={() => handlePageChange(pagination.page - 1)}
                                    disabled={pagination.page === 1}
                                    className="btn-ghost disabled:opacity-30"
                                    aria-label="Previous page"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                                    </svg>
                                </button>

                                <span className="text-sm text-slate-400 px-4">
                                    Page {pagination.page} of {pagination.pages}
                                </span>

                                <button
                                    onClick={() => handlePageChange(pagination.page + 1)}
                                    disabled={pagination.page === pagination.pages}
                                    className="btn-ghost disabled:opacity-30"
                                    aria-label="Next page"
                                >
                                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                    </svg>
                                </button>
                            </motion.div>
                        )}
                    </>
                )}
            </div>
        </PageTransition>
    );
}

export default HistoryPage;
