import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { scanService } from '../services/scanService';
import PageTransition from '../components/PageTransition';
import GlassCard from '../components/GlassCard';
import ResultBadge from '../components/ResultBadge';
import ShapChart from '../components/ShapChart';
import { LoadingCard } from '../components/LoadingShimmer';

function ResultPage() {
    const { id } = useParams();
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState('');

    useEffect(() => {
        loadResult();
    }, [id]);

    const loadResult = async () => {
        try {
            const data = await scanService.getScanById(id);
            setResult(data.result);
        } catch (err) {
            setError('Failed to load scan result');
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <PageTransition>
                <div className="max-w-3xl mx-auto px-4 py-8">
                    <LoadingCard />
                </div>
            </PageTransition>
        );
    }

    if (error || !result) {
        return (
            <PageTransition>
                <div className="max-w-3xl mx-auto px-4 py-8">
                    <GlassCard className="text-center py-12">
                        <p className="text-rose-400 mb-4">{error || 'Result not found'}</p>
                        <Link to="/scan" className="btn-primary">
                            Scan New URL
                        </Link>
                    </GlassCard>
                </div>
            </PageTransition>
        );
    }

    const isPhishing = result.prediction === 'phishing';
    const confidencePercent = (result.confidence * 100).toFixed(1);

    return (
        <PageTransition>
            <div className="max-w-3xl mx-auto px-4 py-8">
                {/* Back Link */}
                <motion.div
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    className="mb-6"
                >
                    <Link
                        to="/dashboard"
                        className="inline-flex items-center gap-2 text-slate-400 hover:text-white transition-colors"
                    >
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                        </svg>
                        Back to Dashboard
                    </Link>
                </motion.div>

                {/* Main Result Card */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                >
                    <GlassCard className="text-center mb-6">
                        {/* URL */}
                        <p className="text-slate-400 text-sm mb-6 break-all">
                            {result.url}
                        </p>

                        {/* Verdict */}
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            transition={{ delay: 0.2, type: 'spring', bounce: 0.4 }}
                            className="mb-6"
                        >
                            <div className={`
                inline-flex items-center justify-center w-20 h-20 rounded-full mb-4
                ${isPhishing ? 'bg-rose-400/10' : 'bg-emerald-400/10'}
              `}>
                                {isPhishing ? (
                                    <svg className="w-10 h-10 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                                    </svg>
                                ) : (
                                    <svg className="w-10 h-10 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                                    </svg>
                                )}
                            </div>

                            <h1 className={`text-3xl font-light mb-2 ${isPhishing ? 'text-rose-400' : 'text-emerald-400'}`}>
                                {isPhishing ? 'Phishing Detected' : 'URL Appears Safe'}
                            </h1>

                            <ResultBadge
                                prediction={result.prediction}
                                confidence={result.confidence}
                                size="lg"
                            />
                        </motion.div>

                        {/* Confidence Bar */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            transition={{ delay: 0.3 }}
                            className="mb-6"
                        >
                            <div className="flex items-center justify-between text-sm mb-2">
                                <span className="text-slate-400">Confidence</span>
                                <span className="text-white font-medium">{confidencePercent}%</span>
                            </div>
                            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${confidencePercent}%` }}
                                    transition={{ delay: 0.4, duration: 0.6, ease: 'easeOut' }}
                                    className={`h-full rounded-full ${isPhishing ? 'bg-rose-400' : 'bg-emerald-400'}`}
                                />
                            </div>
                        </motion.div>

                        {/* Timestamp */}
                        <p className="text-sm text-slate-500">
                            Scanned {new Date(result.timestamp).toLocaleString()}
                        </p>
                    </GlassCard>
                </motion.div>

                {/* Explanation */}
                {result.explanation && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 }}
                        className="mb-6"
                    >
                        <GlassCard>
                            <h3 className="text-lg font-medium text-white mb-3">Analysis Summary</h3>
                            <p className="text-slate-300 leading-relaxed">
                                {result.explanation}
                            </p>
                        </GlassCard>
                    </motion.div>
                )}

                {/* Features */}
                {result.features && Object.keys(result.features).length > 0 && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4 }}
                        className="mb-6"
                    >
                        <GlassCard>
                            <h3 className="text-lg font-medium text-white mb-4">Detected Features</h3>
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                {Object.entries(result.features).slice(0, 6).map(([key, value]) => (
                                    <div
                                        key={key}
                                        className="flex items-center justify-between p-3 bg-white/[0.02] rounded-xl"
                                    >
                                        <span className="text-slate-400 text-sm capitalize">
                                            {key.replace(/_/g, ' ')}
                                        </span>
                                        <span className={`
                      text-sm font-medium px-2 py-1 rounded-lg
                      ${value === 'high' || value === 'suspicious' ? 'bg-rose-400/10 text-rose-400' : ''}
                      ${value === 'low' || value === 'normal' ? 'bg-emerald-400/10 text-emerald-400' : ''}
                      ${value !== 'high' && value !== 'suspicious' && value !== 'low' && value !== 'normal' ? 'bg-slate-700 text-slate-300' : ''}
                    `}>
                                            {String(value)}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </GlassCard>
                    </motion.div>
                )}

                {/* SHAP Chart */}
                {result.shap_values && Object.keys(result.shap_values).length > 0 && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.5 }}
                        className="mb-6"
                    >
                        <GlassCard>
                            <h3 className="text-lg font-medium text-white mb-4">Feature Importance</h3>
                            <ShapChart shapValues={result.shap_values} maxFeatures={5} />
                        </GlassCard>
                    </motion.div>
                )}

                {/* Actions */}
                <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.6 }}
                    className="flex flex-col sm:flex-row gap-3"
                >
                    <Link to="/scan" className="btn-primary flex-1 text-center">
                        Scan Another URL
                    </Link>
                    <Link to="/history" className="btn-secondary flex-1 text-center">
                        View History
                    </Link>
                </motion.div>
            </div>
        </PageTransition>
    );
}

export default ResultPage;
