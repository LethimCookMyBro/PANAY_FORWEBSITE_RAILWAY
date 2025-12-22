import { useState, useEffect } from "react";
import Chat from "./pages/Chat";
import Login from "./pages/Login";
import Register from "./pages/Register";

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
      <div className="h-screen flex items-center justify-center">
        Loading...
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
