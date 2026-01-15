const mongoose = require('mongoose');

const scanResultSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    url: {
        type: String,
        required: [true, 'URL is required'],
        trim: true
    },
    prediction: {
        type: String,
        enum: ['phishing', 'benign'],
        required: true
    },
    confidence: {
        type: Number,
        required: true,
        min: 0,
        max: 1
    },
    explanation: {
        type: String,
        default: ''
    },
    features: {
        type: mongoose.Schema.Types.Mixed,
        default: {}
    },
    shap_values: {
        type: mongoose.Schema.Types.Mixed,
        default: {}
    },
    timestamp: {
        type: Date,
        default: Date.now
    }
});

// Index for faster queries
scanResultSchema.index({ userId: 1, timestamp: -1 });
scanResultSchema.index({ prediction: 1 });

// Virtual to extract domain from URL
scanResultSchema.virtual('domain').get(function () {
    try {
        const url = new URL(this.url);
        return url.hostname;
    } catch {
        return this.url;
    }
});

module.exports = mongoose.model('ScanResult', scanResultSchema);
