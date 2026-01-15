import api from './api';

export const authService = {
    async login(email, password) {
        const response = await api.post('/auth/login', { email, password });
        return response.data;
    },

    async register(email, password) {
        const response = await api.post('/auth/register', { email, password });
        return response.data;
    },

    async getMe() {
        const response = await api.get('/auth/me');
        return response.data;
    },

    logout() {
        localStorage.removeItem('token');
        localStorage.removeItem('user');
    }
};

export default authService;
