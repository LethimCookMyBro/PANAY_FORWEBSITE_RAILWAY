/**
 * Centralized API module for frontend
 * Provides consistent axios instance with auth interceptors
 */

import axios from "axios";

const API_URL = import.meta.env.VITE_API_URL || "";
const API_TIMEOUT_MS = Number(import.meta.env.VITE_API_TIMEOUT_MS || 180000);

export const getApiErrorMessage = (error, fallback = "Request failed") => {
  const data = error?.response?.data;

  if (typeof data === "string" && data.trim()) {
    return data;
  }

  if (data && typeof data === "object") {
    if (typeof data.message === "string" && data.message.trim()) {
      return data.message;
    }
    if (typeof data.detail === "string" && data.detail.trim()) {
      return data.detail;
    }
    if (Array.isArray(data.detail) && data.detail.length > 0) {
      const first = data.detail[0];
      if (typeof first?.msg === "string" && first.msg.trim()) {
        return first.msg;
      }
    }
    if (typeof data.error === "string" && data.error.trim()) {
      return data.error;
    }
  }

  if (typeof error?.message === "string" && error.message.trim()) {
    return error.message;
  }

  return fallback;
};

// Create axios instance with default config
const api = axios.create({
  baseURL: API_URL,
  withCredentials: true,
  timeout: API_TIMEOUT_MS,
  headers: {
    "Content-Type": "application/json",
  },
});

const refreshClient = axios.create({
  baseURL: API_URL,
  withCredentials: true,
  timeout: API_TIMEOUT_MS,
  headers: {
    "Content-Type": "application/json",
  },
});

let refreshPromise = null;

// Request interceptor - add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem("access_token");
    if (token) {
      config.headers = config.headers || {};
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error),
);

// Response interceptor - handle auth errors
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config || {};
    const url = originalRequest.url || "";
    const isAuthEndpoint =
      url.includes("/auth/login") ||
      url.includes("/auth/register") ||
      url.includes("/auth/refresh");

    if (
      error.response?.status === 401 &&
      !isAuthEndpoint &&
      !originalRequest._retry
    ) {
      originalRequest._retry = true;

      try {
        if (!refreshPromise) {
          refreshPromise = refreshClient
            .post("/api/auth/refresh")
            .then((res) => {
              const newToken = res.data?.access_token;
              if (!newToken) {
                throw new Error("Refresh token response missing access_token");
              }
              localStorage.setItem("access_token", newToken);
              return newToken;
            })
            .finally(() => {
              refreshPromise = null;
            });
        }

        const newToken = await refreshPromise;
        originalRequest.headers = originalRequest.headers || {};
        originalRequest.headers.Authorization = `Bearer ${newToken}`;
        return api(originalRequest);
      } catch (refreshError) {
        localStorage.removeItem("access_token");
        if (typeof window !== "undefined") {
          window.location.reload();
        }
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  },
);

// Auth API
export const authAPI = {
  login: (email, password) => api.post("/api/auth/login", { email, password }),

  register: (fullName, email, password) =>
    api.post("/api/auth/register", { full_name: fullName, email, password }),

  me: () => api.get("/api/auth/me"),
};

// Chat API
export const chatAPI = {
  sendMessage: (message, sessionId = null, collection = "plcnext") =>
    api.post("/api/chat", { message, session_id: sessionId, collection }),

  getSessions: () => api.get("/api/chat/sessions"),

  getMessages: (sessionId) => api.get(`/api/chat/sessions/${sessionId}`),

  deleteSession: (sessionId) => api.delete(`/api/chat/sessions/${sessionId}`),

  transcribe: (audioBlob, signal) => {
    const formData = new FormData();
    formData.append("file", audioBlob, "recording.webm");
    return api.post("/api/transcribe", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      signal,
    });
  },
};

export default api;
