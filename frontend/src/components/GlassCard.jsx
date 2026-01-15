import { motion } from 'framer-motion';

function GlassCard({ children, className = '', hover = false, ...props }) {
    const Component = hover ? motion.div : 'div';

    const hoverProps = hover ? {
        whileHover: { y: -2, transition: { duration: 0.15 } },
        whileTap: { scale: 0.98 }
    } : {};

    return (
        <Component
            className={`glass rounded-2xl p-6 ${className}`}
            {...hoverProps}
            {...props}
        >
            {children}
        </Component>
    );
}

export default GlassCard;
