const express = require('express');
const router = express.Router();
const { authLimiter } = require('../middlewares/rateLimiter');
const auth = require('../middlewares/auth');
const {
    register,
    login,
    getMe,
    registerValidation,
    loginValidation
} = require('../controllers/authController');

// Public routes
router.post('/register', authLimiter, registerValidation, register);
router.post('/login', authLimiter, loginValidation, login);

// Protected routes
router.get('/me', auth, getMe);

module.exports = router;
