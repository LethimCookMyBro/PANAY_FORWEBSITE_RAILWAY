import { useState } from "react";
import { Bot, LoaderCircle, Mail, Lock, Sparkles } from "lucide-react";
import { authAPI, getApiErrorMessage } from "../utils/api";

function Login({ onLogin, onGoRegister }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const res = await authAPI.login(email, password);
      onLogin(res.data.access_token);
    } catch (err) {
      setError(getApiErrorMessage(err, "Invalid credentials"));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen auth-shell flex items-center justify-center p-4 relative overflow-hidden">
      {/* Animated gradient background */}
      <div className="fixed inset-0 animated-gradient" />

      {/* Decorative orbs */}
      <div className="fixed top-1/4 left-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl" />
      <div className="fixed bottom-1/4 right-1/4 w-80 h-80 bg-cyan-500/15 rounded-full blur-3xl" />
      <div className="fixed top-1/2 right-1/3 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl" />

      <div className="w-full max-w-sm relative z-10 fade-in-up">
        {/* Logo */}
        <div className="flex flex-col items-center mb-8">
          <div className="relative mb-4">
            <div className="bg-gradient-to-br from-blue-500 to-cyan-400 p-4 rounded-2xl shadow-2xl shadow-blue-500/25">
              <Bot className="w-10 h-10 text-white" />
            </div>
            <div className="absolute -top-1 -right-1">
              <Sparkles className="w-5 h-5 text-cyan-400" />
            </div>
          </div>
          <h1 className="text-3xl font-bold text-white tracking-tight">
            Panya
          </h1>
          <p className="text-sm text-blue-300/70 mt-1 font-medium">
            PLC & Automation AI Assistant
          </p>
        </div>

        {/* Card */}
        <form
          className="auth-card rounded-2xl p-6 sm:p-8"
          onSubmit={handleSubmit}
        >
          <h2 className="text-xl font-semibold mb-6 text-center text-white">
            Welcome back
          </h2>

          {error && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-xl text-red-400 text-sm text-center">
              {error}
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-blue-200/80 mb-1.5">
                Email
              </label>
              <div className="relative">
                <Mail
                  size={16}
                  className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500"
                />
                <input
                  type="email"
                  placeholder="you@company.com"
                  className="auth-input w-full pl-10 pr-4 py-2.5 rounded-xl focus:outline-none text-white placeholder-slate-500"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-blue-200/80 mb-1.5">
                Password
              </label>
              <div className="relative">
                <Lock
                  size={16}
                  className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-500"
                />
                <input
                  type="password"
                  placeholder="••••••••"
                  className="auth-input w-full pl-10 pr-4 py-2.5 rounded-xl focus:outline-none text-white placeholder-slate-500"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                />
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-6 bg-gradient-to-r from-blue-500 to-cyan-500 text-white py-2.5 rounded-xl hover:from-blue-600 hover:to-cyan-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all font-semibold shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <LoaderCircle size={18} className="animate-spin" />
                Signing in...
              </>
            ) : (
              "Sign in"
            )}
          </button>

          <p className="mt-6 text-sm text-center text-slate-400">
            Don't have an account?{" "}
            <button
              type="button"
              onClick={onGoRegister}
              className="text-cyan-400 hover:text-cyan-300 font-medium hover:underline transition-colors"
            >
              Create one
            </button>
          </p>
        </form>

        <p className="text-center text-[11px] text-slate-600 mt-6">
          Powered by Panya AI • Industrial Automation
        </p>
      </div>
    </div>
  );
}

export default Login;
