import api from './api';

export const adminService = {
    async getStats() {
        const response = await api.get('/admin/stats');
        return response.data;
    },

    async getMLHealth() {
        const response = await api.get('/admin/ml-health');
        return response.data;
    },

    async getUsers(page = 1, limit = 20) {
        const response = await api.get('/admin/users', {
            params: { page, limit }
        });
        return response.data;
    }
};

export default adminService;
