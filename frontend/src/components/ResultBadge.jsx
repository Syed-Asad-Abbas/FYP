import { motion } from 'framer-motion';

function ResultBadge({ prediction, confidence, size = 'md' }) {
    const isPhishing = prediction === 'phishing';

    const sizes = {
        sm: 'px-3 py-1 text-sm',
        md: 'px-4 py-2 text-base',
        lg: 'px-6 py-3 text-lg'
    };

    const confidencePercent = (confidence * 100).toFixed(1);

    return (
        <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
            className={`
        inline-flex items-center gap-2 rounded-full font-medium
        ${sizes[size]}
        ${isPhishing
                    ? 'bg-rose-400/10 text-rose-400 border border-rose-400/20'
                    : 'bg-emerald-400/10 text-emerald-400 border border-emerald-400/20'
                }
      `}
        >
            {/* Icon */}
            <span className={`w-2 h-2 rounded-full ${isPhishing ? 'bg-rose-400' : 'bg-emerald-400'}`} />

            {/* Label */}
            <span className="capitalize">{prediction}</span>

            {/* Confidence */}
            <span className="opacity-70">
                {confidencePercent}%
            </span>
        </motion.div>
    );
}

export default ResultBadge;
