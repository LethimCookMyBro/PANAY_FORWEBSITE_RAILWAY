import { useState, useEffect } from "react";
import Chat from "./pages/Chat";
import Login from "./pages/Login";
import Register from "./pages/Register";
import { LoaderCircle } from "lucide-react";

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState("login"); // login | register

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (token) {
      setIsAuthenticated(true);
    }
    setLoading(false);
  }, []);

  const handleLogin = (token) => {
    localStorage.setItem("access_token", token);
    setIsAuthenticated(true);
  };

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    setIsAuthenticated(false);
    setPage("login");
  };

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100 gap-4">
        <img
          src="/panya-logo.png"
          alt="Panya logo"
          className="w-14 h-14 object-contain"
        />
        <div className="flex items-center gap-2 text-gray-600">
          <LoaderCircle size={20} className="animate-spin text-blue-500" />
          <span className="font-medium">Loading...</span>
        </div>
      </div>
    );
  }

  if (isAuthenticated) {
    return <Chat onLogout={handleLogout} />;
  }

  return (
    <>
      {page === "login" && (
        <Login
          onLogin={handleLogin}
          onGoRegister={() => setPage("register")}
        />
      )}

      {page === "register" && (
        <Register
          onRegisterSuccess={() => setPage("login")}
          onBackToLogin={() => setPage("login")}
        />
      )}
    </>
  );
}

export default App;
