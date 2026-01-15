import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { scanService } from '../services/scanService';
import PageTransition from '../components/PageTransition';
import GlassCard from '../components/GlassCard';
import { MLApiErrorState } from '../components/EmptyState';

function ScanPage() {
    const [url, setUrl] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [mlError, setMlError] = useState(false);

    const navigate = useNavigate();

    const validateUrl = (urlString) => {
        try {
            new URL(urlString);
            return true;
        } catch {
            return false;
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');
        setMlError(false);

        // Add protocol if missing
        let urlToScan = url.trim();
        if (!urlToScan.startsWith('http://') && !urlToScan.startsWith('https://')) {
            urlToScan = 'https://' + urlToScan;
        }

        if (!validateUrl(urlToScan)) {
            setError('Please enter a valid URL');
            return;
        }

        setLoading(true);

        try {
            const { result } = await scanService.scanUrl(urlToScan);
            navigate(`/result/${result.id}`);
        } catch (err) {
            if (err.response?.data?.code === 'ML_SERVICE_DOWN') {
                setMlError(true);
            } else {
                setError(err.response?.data?.error || 'Scan failed. Please try again.');
            }
        } finally {
            setLoading(false);
        }
    };

    const handleRetry = () => {
        setMlError(false);
        setError('');
    };

    if (mlError) {
        return (
            <PageTransition>
                <div className="max-w-2xl mx-auto px-4 py-16">
                    <GlassCard>
                        <MLApiErrorState onRetry={handleRetry} />
                    </GlassCard>
                </div>
            </PageTransition>
        );
    }

    return (
        <PageTransition>
            <div className="min-h-[calc(100vh-64px)] flex items-center justify-center px-4 py-12">
                <div className="w-full max-w-2xl">
                    {/* Header */}
                    <motion.div
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-center mb-8"
                    >
                        <h1 className="text-3xl sm:text-4xl font-light text-white mb-3">
                            Scan a URL
                        </h1>
                        <p className="text-slate-400">
                            Enter any website URL to analyze for phishing threats
                        </p>
                    </motion.div>

                    {/* Scan Form */}
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 }}
                    >
                        <GlassCard className="p-8">
                            <form onSubmit={handleSubmit} className="space-y-6">
                                {error && (
                                    <motion.div
                                        initial={{ opacity: 0, y: -10 }}
                                        animate={{ opacity: 1, y: 0 }}
                                        className="p-4 rounded-xl bg-rose-400/10 border border-rose-400/20 text-rose-400 text-sm"
                                    >
                                        {error}
                                    </motion.div>
                                )}

                                <div>
                                    <label htmlFor="url" className="sr-only">URL to scan</label>
                                    <div className="relative">
                                        <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                                            <svg className="w-5 h-5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
                                            </svg>
                                        </div>
                                        <input
                                            id="url"
                                            type="text"
                                            value={url}
                                            onChange={(e) => setUrl(e.target.value)}
                                            className="input pl-12 text-lg py-4"
                                            placeholder="example.com"
                                            required
                                            autoFocus
                                            autoComplete="off"
                                        />
                                    </div>
                                    <p className="mt-2 text-sm text-slate-500">
                                        We'll add https:// automatically if needed
                                    </p>
                                </div>

                                <button
                                    type="submit"
                                    disabled={loading || !url.trim()}
                                    className="btn-primary w-full py-4 text-lg flex items-center justify-center gap-3"
                                >
                                    {loading ? (
                                        <>
                                            <div className="w-5 h-5 border-2 border-slate-900 border-t-transparent rounded-full animate-spin" />
                                            Analyzing...
                                        </>
                                    ) : (
                                        <>
                                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                                            </svg>
                                            Analyze URL
                                        </>
                                    )}
                                </button>
                            </form>
                        </GlassCard>
                    </motion.div>

                    {/* Tips */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.3 }}
                        className="mt-8 text-center"
                    >
                        <p className="text-sm text-slate-500">
                            Our ML model analyzes URLs for phishing indicators including domain age,
                            URL patterns, and more.
                        </p>
                    </motion.div>
                </div>
            </div>
        </PageTransition>
    );
}

export default ScanPage;
