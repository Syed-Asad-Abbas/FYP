function HealthIndicator({ status, label = 'ML API' }) {
    const isUp = status === 'up';

    return (
        <div className="flex items-center gap-2">
            <div className="relative">
                <span
                    className={`block w-2.5 h-2.5 rounded-full ${isUp ? 'bg-emerald-400' : 'bg-rose-400'
                        }`}
                />
                {isUp && (
                    <span className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping opacity-50" />
                )}
            </div>
            <span className="text-sm text-slate-400">
                {label}: <span className={isUp ? 'text-emerald-400' : 'text-rose-400'}>
                    {isUp ? 'Online' : 'Offline'}
                </span>
            </span>
        </div>
    );
}

export default HealthIndicator;
