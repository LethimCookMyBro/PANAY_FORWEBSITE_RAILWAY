/**
 * Centralized API module for frontend
 * Provides consistent axios instance with auth interceptors
 */

import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// Create axios instance with default config
const api = axios.create({
    baseURL: API_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Request interceptor - add auth token
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// Response interceptor - handle auth errors
api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            localStorage.removeItem('access_token');
            window.location.reload(); // Force re-auth
        }
        return Promise.reject(error);
    }
);

// Auth API
export const authAPI = {
    login: (email, password) =>
        api.post('/api/auth/login', { email, password }),

    register: (fullName, email, password) =>
        api.post('/api/auth/register', { full_name: fullName, email, password }),

    me: () =>
        api.get('/api/auth/me'),
};

// Chat API
export const chatAPI = {
    sendMessage: (message, sessionId = null, collection = 'plc_docs') =>
        api.post('/api/chat', { message, session_id: sessionId, collection }),

    getSessions: () =>
        api.get('/api/chat/sessions'),

    getMessages: (sessionId) =>
        api.get(`/api/chat/sessions/${sessionId}`),

    deleteSession: (sessionId) =>
        api.delete(`/api/chat/sessions/${sessionId}`),

    transcribe: (audioBlob) => {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recording.webm');
        return api.post('/api/transcribe', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
    },
};

export default api;
