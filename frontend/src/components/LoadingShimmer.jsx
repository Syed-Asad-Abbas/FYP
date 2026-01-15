function LoadingShimmer({ className = '', lines = 1 }) {
    return (
        <div className={`space-y-3 ${className}`}>
            {Array.from({ length: lines }).map((_, i) => (
                <div
                    key={i}
                    className="h-4 rounded-lg shimmer"
                    style={{ width: `${100 - i * 15}%` }}
                />
            ))}
        </div>
    );
}

function LoadingCard() {
    return (
        <div className="glass rounded-2xl p-6 space-y-4">
            <div className="h-6 w-32 rounded-lg shimmer" />
            <LoadingShimmer lines={3} />
        </div>
    );
}

function LoadingSpinner({ size = 'md', className = '' }) {
    const sizes = {
        sm: 'w-4 h-4 border',
        md: 'w-8 h-8 border-2',
        lg: 'w-12 h-12 border-2'
    };

    return (
        <div
            className={`${sizes[size]} border-cyan-400 border-t-transparent rounded-full animate-spin ${className}`}
            role="status"
            aria-label="Loading"
        />
    );
}

export { LoadingShimmer, LoadingCard, LoadingSpinner };
export default LoadingShimmer;
