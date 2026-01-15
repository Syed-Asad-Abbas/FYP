const express = require('express');
const router = express.Router();
const auth = require('../middlewares/auth');
const { scanLimiter } = require('../middlewares/rateLimiter');
const {
    scanUrl,
    getHistory,
    getScanById,
    scanValidation
} = require('../controllers/scanController');

// All routes require authentication
router.use(auth);

// Scan URL (rate limited)
router.post('/url', scanLimiter, scanValidation, scanUrl);

// Get scan history
router.get('/history', getHistory);

// Get single scan by ID
router.get('/:id', getScanById);

module.exports = router;
