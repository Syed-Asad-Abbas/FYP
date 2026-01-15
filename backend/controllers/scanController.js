const { body, validationResult } = require('express-validator');
const ScanResult = require('../models/ScanResult');
const { predictUrl } = require('../utils/mlApiService');

// URL validation
const scanValidation = [
    body('url')
        .trim()
        .notEmpty().withMessage('URL is required')
        .isURL({ require_protocol: true }).withMessage('Please enter a valid URL with http:// or https://')
];

// Scan a URL
const scanUrl = async (req, res) => {
    try {
        // Check validation errors
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({
                error: 'Validation failed',
                details: errors.array()
            });
        }

        const { url } = req.body;

        // Call Flask ML API
        const mlResult = await predictUrl(url);

        if (!mlResult.success) {
            return res.status(503).json({
                error: mlResult.error,
                code: mlResult.code
            });
        }

        // Save result to database
        const scanResult = new ScanResult({
            userId: req.user._id,
            url,
            prediction: mlResult.data.prediction,
            confidence: mlResult.data.confidence,
            explanation: mlResult.data.explanation,
            features: mlResult.data.features,
            shap_values: mlResult.data.shap_values
        });

        await scanResult.save();

        res.json({
            message: 'Scan completed',
            result: {
                id: scanResult._id,
                url: scanResult.url,
                prediction: scanResult.prediction,
                confidence: scanResult.confidence,
                explanation: scanResult.explanation,
                features: scanResult.features,
                shap_values: scanResult.shap_values,
                timestamp: scanResult.timestamp
            }
        });
    } catch (error) {
        console.error('Scan error:', error);
        res.status(500).json({ error: 'Failed to scan URL' });
    }
};

// Get scan history for current user
const getHistory = async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10;
        const skip = (page - 1) * limit;

        const [results, total] = await Promise.all([
            ScanResult.find({ userId: req.user._id })
                .sort({ timestamp: -1 })
                .skip(skip)
                .limit(limit)
                .lean(),
            ScanResult.countDocuments({ userId: req.user._id })
        ]);

        res.json({
            results,
            pagination: {
                page,
                limit,
                total,
                pages: Math.ceil(total / limit)
            }
        });
    } catch (error) {
        console.error('History error:', error);
        res.status(500).json({ error: 'Failed to fetch scan history' });
    }
};

// Get a single scan result by ID
const getScanById = async (req, res) => {
    try {
        const result = await ScanResult.findOne({
            _id: req.params.id,
            userId: req.user._id
        });

        if (!result) {
            return res.status(404).json({ error: 'Scan result not found' });
        }

        res.json({ result });
    } catch (error) {
        console.error('Get scan error:', error);
        res.status(500).json({ error: 'Failed to fetch scan result' });
    }
};

module.exports = {
    scanUrl,
    getHistory,
    getScanById,
    scanValidation
};
