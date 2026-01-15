const express = require('express');
const router = express.Router();
const auth = require('../middlewares/auth');
const roleCheck = require('../middlewares/roleCheck');
const { getStats, getMLHealth, getUsers } = require('../controllers/adminController');

// All routes require authentication + admin role
router.use(auth);
router.use(roleCheck('admin'));

// Get statistics
router.get('/stats', getStats);

// Check ML API health
router.get('/ml-health', getMLHealth);

// Get all users
router.get('/users', getUsers);

module.exports = router;
