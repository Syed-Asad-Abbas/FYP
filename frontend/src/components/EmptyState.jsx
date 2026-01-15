import { motion } from 'framer-motion';

function EmptyState({
    icon,
    title,
    description,
    action,
    variant = 'default' // 'default' | 'error'
}) {
    const variants = {
        default: {
            iconBg: 'bg-slate-800',
            iconColor: 'text-slate-500'
        },
        error: {
            iconBg: 'bg-rose-400/10',
            iconColor: 'text-rose-400'
        }
    };

    const style = variants[variant];

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-col items-center justify-center py-12 px-4 text-center"
        >
            {/* Icon */}
            <div className={`w-16 h-16 rounded-2xl ${style.iconBg} flex items-center justify-center mb-4`}>
                {icon || (
                    <svg className={`w-8 h-8 ${style.iconColor}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                    </svg>
                )}
            </div>

            {/* Title */}
            <h3 className="text-lg font-medium text-white mb-2">
                {title}
            </h3>

            {/* Description */}
            <p className="text-sm text-slate-400 max-w-sm mb-6">
                {description}
            </p>

            {/* Action */}
            {action}
        </motion.div>
    );
}

// Pre-built empty states
export function NoHistoryState({ onScan }) {
    return (
        <EmptyState
            icon={
                <svg className="w-8 h-8 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
            }
            title="No scan history yet"
            description="Start scanning URLs to build your security analysis history."
            action={
                onScan && (
                    <button onClick={onScan} className="btn-primary">
                        Scan Your First URL
                    </button>
                )
            }
        />
    );
}

export function MLApiErrorState({ onRetry }) {
    return (
        <EmptyState
            variant="error"
            icon={
                <svg className="w-8 h-8 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
            }
            title="ML Service Unavailable"
            description="The phishing detection service is currently offline. Please try again later."
            action={
                onRetry && (
                    <button onClick={onRetry} className="btn-secondary">
                        Try Again
                    </button>
                )
            }
        />
    );
}

export default EmptyState;
