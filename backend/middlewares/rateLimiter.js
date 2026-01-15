const rateLimit = require('express-rate-limit');

// General API rate limiter
const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // 100 requests per window
    message: {
        error: 'Too many requests',
        message: 'Please try again later'
    },
    standardHeaders: true,
    legacyHeaders: false
});

// Stricter limiter for scan endpoints (20 per minute per user)
const scanLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 20, // 20 scans per minute
    message: {
        error: 'Rate limit exceeded',
        message: 'You can only perform 20 scans per minute. Please wait and try again.'
    },
    standardHeaders: true,
    legacyHeaders: false,
    keyGenerator: (req) => {
        // Use user ID if authenticated, otherwise IP
        return req.user?.id || req.ip;
    }
});

// Auth rate limiter to prevent brute force
const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // 10 attempts per window
    message: {
        error: 'Too many login attempts',
        message: 'Please try again in 15 minutes'
    },
    standardHeaders: true,
    legacyHeaders: false
});

module.exports = { apiLimiter, scanLimiter, authLimiter };
