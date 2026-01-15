const ScanResult = require('../models/ScanResult');
const User = require('../models/User');
const { checkHealth } = require('../utils/mlApiService');

// Get admin statistics
const getStats = async (req, res) => {
    try {
        // Get total counts
        const [totalScans, totalUsers, phishingCount, benignCount] = await Promise.all([
            ScanResult.countDocuments(),
            User.countDocuments(),
            ScanResult.countDocuments({ prediction: 'phishing' }),
            ScanResult.countDocuments({ prediction: 'benign' })
        ]);

        // Get most scanned domains (top 10)
        const topDomains = await ScanResult.aggregate([
            {
                $addFields: {
                    domain: {
                        $regexFind: {
                            input: '$url',
                            regex: /^(?:https?:\/\/)?(?:www\.)?([^\/]+)/
                        }
                    }
                }
            },
            {
                $group: {
                    _id: '$domain.match',
                    count: { $sum: 1 },
                    phishingCount: {
                        $sum: { $cond: [{ $eq: ['$prediction', 'phishing'] }, 1, 0] }
                    }
                }
            },
            { $sort: { count: -1 } },
            { $limit: 10 },
            {
                $project: {
                    domain: '$_id',
                    count: 1,
                    phishingCount: 1,
                    _id: 0
                }
            }
        ]);

        // Get recent scan activity (last 7 days)
        const sevenDaysAgo = new Date();
        sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7);

        const dailyStats = await ScanResult.aggregate([
            { $match: { timestamp: { $gte: sevenDaysAgo } } },
            {
                $group: {
                    _id: { $dateToString: { format: '%Y-%m-%d', date: '$timestamp' } },
                    total: { $sum: 1 },
                    phishing: { $sum: { $cond: [{ $eq: ['$prediction', 'phishing'] }, 1, 0] } },
                    benign: { $sum: { $cond: [{ $eq: ['$prediction', 'benign'] }, 1, 0] } }
                }
            },
            { $sort: { _id: 1 } }
        ]);

        res.json({
            stats: {
                totalScans,
                totalUsers,
                phishingCount,
                benignCount,
                phishingRatio: totalScans > 0 ? (phishingCount / totalScans * 100).toFixed(1) : 0
            },
            topDomains,
            dailyStats
        });
    } catch (error) {
        console.error('Stats error:', error);
        res.status(500).json({ error: 'Failed to fetch statistics' });
    }
};

// Check ML API health
const getMLHealth = async (req, res) => {
    try {
        const health = await checkHealth();
        res.json(health);
    } catch (error) {
        console.error('Health check error:', error);
        res.status(500).json({
            status: 'error',
            message: 'Failed to check ML API health'
        });
    }
};

// Get all users (admin only)
const getUsers = async (req, res) => {
    try {
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 20;
        const skip = (page - 1) * limit;

        const [users, total] = await Promise.all([
            User.find()
                .select('-password')
                .sort({ createdAt: -1 })
                .skip(skip)
                .limit(limit)
                .lean(),
            User.countDocuments()
        ]);

        // Get scan count for each user
        const userIds = users.map(u => u._id);
        const scanCounts = await ScanResult.aggregate([
            { $match: { userId: { $in: userIds } } },
            { $group: { _id: '$userId', count: { $sum: 1 } } }
        ]);

        const scanCountMap = scanCounts.reduce((acc, { _id, count }) => {
            acc[_id.toString()] = count;
            return acc;
        }, {});

        const usersWithScans = users.map(user => ({
            ...user,
            scanCount: scanCountMap[user._id.toString()] || 0
        }));

        res.json({
            users: usersWithScans,
            pagination: {
                page,
                limit,
                total,
                pages: Math.ceil(total / limit)
            }
        });
    } catch (error) {
        console.error('Get users error:', error);
        res.status(500).json({ error: 'Failed to fetch users' });
    }
};

module.exports = { getStats, getMLHealth, getUsers };
