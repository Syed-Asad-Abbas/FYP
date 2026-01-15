import api from './api';

export const scanService = {
    async scanUrl(url) {
        const response = await api.post('/scan/url', { url });
        return response.data;
    },

    async getHistory(page = 1, limit = 10) {
        const response = await api.get('/scan/history', {
            params: { page, limit }
        });
        return response.data;
    },

    async getScanById(id) {
        const response = await api.get(`/scan/${id}`);
        return response.data;
    }
};

export default scanService;
