const axios = require('axios');

const ML_API_BASE_URL = process.env.ML_API_BASE_URL || 'http://localhost:5000';

/**
 * Call Flask ML API to predict if URL is phishing
 * @param {string} url - The URL to analyze
 * @returns {Promise<Object>} Prediction result with confidence, features, and SHAP values
 */
const predictUrl = async (url) => {
    try {
        const response = await axios.post(
            `${ML_API_BASE_URL}/predict`,
            { url },
            {
                timeout: 30000, // 30 second timeout
                headers: {
                    'Content-Type': 'application/json'
                }
            }
        );

        return {
            success: true,
            data: {
                prediction: response.data.prediction,
                confidence: response.data.confidence,
                explanation: response.data.explanation || '',
                features: response.data.features || {},
                shap_values: response.data.shap_values || {}
            }
        };
    } catch (error) {
        console.error('ML API Error:', error.message);

        if (error.code === 'ECONNREFUSED') {
            return {
                success: false,
                error: 'ML service is unavailable',
                code: 'ML_SERVICE_DOWN'
            };
        }

        if (error.response) {
            return {
                success: false,
                error: error.response.data?.error || 'ML prediction failed',
                code: 'ML_PREDICTION_ERROR'
            };
        }

        return {
            success: false,
            error: 'Failed to connect to ML service',
            code: 'ML_CONNECTION_ERROR'
        };
    }
};

/**
 * Check if Flask ML API is healthy
 * @returns {Promise<Object>} Health status
 */
const checkHealth = async () => {
    try {
        const response = await axios.get(`${ML_API_BASE_URL}/health`, {
            timeout: 5000
        });

        return {
            status: 'up',
            message: 'ML API is operational',
            details: response.data
        };
    } catch (error) {
        // Try the predict endpoint as fallback health check
        try {
            await axios.post(
                `${ML_API_BASE_URL}/predict`,
                { url: 'https://example.com' },
                { timeout: 10000 }
            );

            return {
                status: 'up',
                message: 'ML API is operational (predict endpoint)'
            };
        } catch {
            return {
                status: 'down',
                message: 'ML API is unavailable',
                error: error.message
            };
        }
    }
};

module.exports = { predictUrl, checkHealth };
