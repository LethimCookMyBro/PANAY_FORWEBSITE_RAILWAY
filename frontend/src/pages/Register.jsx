import { useState } from "react";
import { Bot, LoaderCircle } from "lucide-react";
import { authAPI } from "../utils/api";

export default function Register({ onRegisterSuccess, onBackToLogin }) {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await authAPI.register(fullName, email, password);
      onRegisterSuccess();
    } catch (err) {
      setError(err.response?.data?.detail || err.message || "Register failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 p-4">
      <div className="w-full max-w-sm">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="bg-gradient-to-br from-blue-600 to-blue-700 p-3 rounded-2xl shadow-lg shadow-blue-200 mb-4">
            <Bot className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-gray-800">PLC Assistant</h1>
          <p className="text-sm text-gray-500 mt-1">Create your account</p>
        </div>

        {/* Card */}
        <form
          onSubmit={handleSubmit}
          className="bg-white p-6 sm:p-8 rounded-2xl shadow-xl border border-gray-100"
        >
          <h2 className="text-xl font-semibold mb-6 text-center text-gray-800">
            Get started
          </h2>

          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-xl text-red-600 text-sm text-center">
              {error}
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">
                Username
              </label>
              <input
                type="text"
                placeholder="Your Username"
                className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400 transition-all bg-gray-50 focus:bg-white"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">
                Email
              </label>
              <input
                type="email"
                placeholder="Your Email"
                className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400 transition-all bg-gray-50 focus:bg-white"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1.5">
                Password
              </label>
              <input
                type="password"
                placeholder="••••••••"
                className="w-full px-4 py-2.5 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-400 transition-all bg-gray-50 focus:bg-white"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-6 bg-gradient-to-r from-blue-500 to-blue-600 text-white py-2.5 rounded-xl hover:from-blue-600 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-medium shadow-md hover:shadow-lg flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <LoaderCircle size={18} className="animate-spin" />
                Creating account...
              </>
            ) : (
              "Create account"
            )}
          </button>

          <p className="mt-6 text-sm text-center text-gray-600">
            Already have an account?{" "}
            <button
              type="button"
              className="text-blue-600 hover:text-blue-700 font-medium hover:underline"
              onClick={onBackToLogin}
            >
              Sign in
            </button>
          </p>
        </form>
      </div>
    </div>
  );
}
