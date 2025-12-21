import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import {
  Plus,
  Send,
  MessageSquareText,
  LoaderCircle,
  LogOut,
  Bot,
  MoreVertical,
  Pencil,
  Trash2,
  Check,
  X
} from "lucide-react";

/* ================= CONFIG ================= */
const CONFIG = {
  API_URL: import.meta.env.VITE_API_URL || "http://localhost:5000",
};

function Chat({ onLogout }) {
  /* ================= API ================= */
  const api = axios.create({
    baseURL: CONFIG.API_URL,
  });

  api.interceptors.request.use((config) => {
    const token = localStorage.getItem("access_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  });

  api.interceptors.response.use(
    (res) => res,
    (err) => {
      if (err.response?.status === 401) {
        localStorage.removeItem("access_token");
        onLogout?.();
      }
      return Promise.reject(err);
    }
  );

  /* ================= STATE ================= */
  const [user, setUser] = useState({ full_name: "User", name: "User" }); // เพิ่มรองรับ full_name
  const [chatHistory, setChatHistory] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [isNewChat, setIsNewChat] = useState(false);

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // State สำหรับจัดการ Menu (Rename/Delete)
  const [menuOpenId, setMenuOpenId] = useState(null); // ID ของ chat ที่เปิดเมนูอยู่
  const [editingChatId, setEditingChatId] = useState(null); // ID ของ chat ที่กำลังเปลี่ยนชื่อ
  const [editTitle, setEditTitle] = useState(""); // ข้อความชื่อใหม่ที่กำลังพิมพ์

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  /* ================= LOAD USER ================= */
  useEffect(() => {
    api.get("/api/auth/me")
      .then(res => setUser(res.data))
      .catch(() => { });
  }, []);

  /* ================= LOAD SESSIONS ================= */
  useEffect(() => {
    api.get("/api/chat/sessions")
      .then(res => {
        const sessions = res.data.items.map(s => ({
          id: s.id,
          title: s.title,
          messages: [],
        }));
        setChatHistory(sessions);
        // ถ้าไม่มี activeChat และมี session ให้เลือกอันแรก
        if (sessions.length > 0 && !activeChatId) {
            setActiveChatId(sessions[0].id);
        }
      })
      .catch(console.error);
  }, []);

  /* ================= LOAD MESSAGES ================= */
  useEffect(() => {
    if (!activeChatId) return;

    const currentChat = chatHistory.find(c => c.id === activeChatId);
    if (currentChat && currentChat.messages.length > 0) return;

    api.get(`/api/chat/sessions/${activeChatId}`)
      .then(res => {
        const messages = res.data.items.map(m => ({
          text: m.content,
          sender: m.role === "user" ? "user" : "bot",
        }));

        setChatHistory(prev =>
          prev.map(c =>
            c.id === activeChatId ? { ...c, messages } : c
          )
        );
      })
      .catch(() => {
        // กรณีหา session ไม่เจอ หรือ error อาจจะ clear activeChatId
      });
  }, [activeChatId]);

  /* ================= HANDLERS ================= */

  const handleNewChat = () => {
    setActiveChatId(null);
    setIsNewChat(true);
    setInput("");
    setEditingChatId(null);
    setMenuOpenId(null);
  };

  const handleSendMessage = useCallback(async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { text: input, sender: "user" };
    setInput("");
    setIsLoading(true);

    if (activeChatId) {
      setChatHistory(prev =>
        prev.map(c =>
          c.id === activeChatId
            ? { ...c, messages: [...c.messages, userMessage] }
            : c
        )
      );
    }

    try {
      const res = await api.post("/api/chat", {
        message: userMessage.text,
        session_id: isNewChat ? null : activeChatId,
      });

      const sessionId = res.data.session_id;

      if (isNewChat) {
        const newSession = {
          id: sessionId,
          title: userMessage.text.substring(0, 50),
          messages: [userMessage],
        };
        setChatHistory(prev => [newSession, ...prev]);
        setActiveChatId(sessionId);
        setIsNewChat(false);
      }

      const botMessage = { text: res.data.reply, sender: "bot" };

      setChatHistory(prev =>
        prev.map(c =>
          c.id === sessionId
            ? { ...c, messages: [...c.messages, botMessage] }
            : c
        )
      );
    } catch (error) {
        console.error("Chat error:", error);
    } finally {
      setIsLoading(false);
      setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    }
  }, [input, isLoading, activeChatId, isNewChat]);

  /* ================= SESSION MANAGEMENT (RENAME / DELETE) ================= */

  // 1. Delete Session
  const handleDeleteSession = async (e, sessionId) => {
    e.stopPropagation();
    if (!window.confirm("Are you sure you want to delete this chat?")) return;

    try {
      await api.delete(`/api/chat/sessions/${sessionId}`);
      setChatHistory(prev => prev.filter(c => c.id !== sessionId));
      
      if (activeChatId === sessionId) {
        setActiveChatId(null);
        setIsNewChat(true);
      }
    } catch (error) {
      console.error("Delete failed", error);
    }
    setMenuOpenId(null);
  };

  const startRename = (e, chat) => {
    e.stopPropagation();
    setEditingChatId(chat.id);
    setEditTitle(chat.title);
    setMenuOpenId(null);
  };

  const saveRename = async (e) => {
    e.stopPropagation();
    if (!editTitle.trim()) return;

    try {
      await api.patch(`/api/chat/sessions/${editingChatId}`, { title: editTitle });
      setChatHistory(prev =>
        prev.map(c => c.id === editingChatId ? { ...c, title: editTitle } : c)
      );
      setEditingChatId(null);
    } catch (error) {
      console.error("Rename failed", error);
    }
  };

  const cancelRename = (e) => {
    e.stopPropagation();
    setEditingChatId(null);
  };

  const activeChat = chatHistory.find(c => c.id === activeChatId);

  /* ================= UI ================= */
  return (
    <div className="flex h-screen bg-gray-100 font-sans">

      {/* ===== SIDEBAR ===== */}
      <aside className="w-72 p-4 bg-white border-r flex flex-col">
        {/* Logo Area */}
        <div className="flex-shrink-0 mb-6 flex items-center gap-3 px-2">
          <div className="bg-gradient-to-br from-blue-600 to-blue-700 p-2 rounded-xl shadow-lg shadow-blue-200">
            <Bot className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-gray-800 tracking-tight">PLC Assistant</h1>
            <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Industrial AI</p>
          </div>
        </div>

        {/* New Chat Button */}
        <button
          onClick={handleNewChat}
          className="flex items-center justify-center gap-2 w-full p-3 mb-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white hover:from-blue-600 hover:to-blue-700 rounded-xl transition-all text-sm font-semibold shadow-md hover:shadow-lg"
        >
          <Plus size={18} />
          New Chat
        </button>

        <div className="px-2 mb-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">Recent</div>

        {/* Chat List */}
        <div className="flex-1 overflow-y-auto space-y-1 pr-1 custom-scrollbar">
          {chatHistory.map(chat => (
            <div
              key={chat.id}
              onClick={() => setActiveChatId(chat.id)}
              className={`group relative w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-3 cursor-pointer transition-all
                ${chat.id === activeChatId ? "bg-blue-50/80 text-blue-700" : "text-gray-600 hover:bg-gray-50"}`}
            >
              <MessageSquareText size={18} className={`flex-shrink-0 ${chat.id === activeChatId ? "text-blue-600" : "text-gray-400"}`} />
              
              {/* === RENAME INPUT MODE === */}
              {editingChatId === chat.id ? (
                <div className="flex items-center flex-1 gap-1 min-w-0" onClick={e => e.stopPropagation()}>
                    <input 
                        className="flex-1 bg-white border border-blue-300 rounded px-1 py-0.5 text-sm focus:outline-none min-w-0"
                        value={editTitle}
                        onChange={(e) => setEditTitle(e.target.value)}
                        autoFocus
                        onKeyDown={(e) => {
                            if(e.key === 'Enter') saveRename(e);
                            if(e.key === 'Escape') cancelRename(e);
                        }}
                    />
                    <button onClick={saveRename} className="text-green-600 hover:bg-green-100 p-1 rounded"><Check size={14}/></button>
                    <button onClick={cancelRename} className="text-red-500 hover:bg-red-100 p-1 rounded"><X size={14}/></button>
                </div>
              ) : (
                /* === NORMAL TITLE MODE === */
                <span className="truncate text-sm font-medium flex-1">
                    {chat.title || "New Chat"}
                </span>
              )}

              {/* === HAMBURGER MENU === */}
              {!editingChatId && (
                  <div className={`absolute right-2 opacity-0 group-hover:opacity-100 transition-opacity ${menuOpenId === chat.id ? 'opacity-100' : ''}`}>
                    <button
                        onClick={(e) => {
                            e.stopPropagation();
                            setMenuOpenId(menuOpenId === chat.id ? null : chat.id);
                        }}
                        className="p-1 hover:bg-gray-200 rounded text-gray-500"
                    >
                        <MoreVertical size={16} />
                    </button>

                    {/* Dropdown Menu */}
                    {menuOpenId === chat.id && (
                        <div className="absolute right-0 top-8 w-32 bg-white shadow-xl border border-gray-100 rounded-lg z-50 overflow-hidden py-1 animate-in fade-in zoom-in-95 duration-100">
                            <button 
                                onClick={(e) => startRename(e, chat)}
                                className="w-full text-left px-3 py-2 text-xs text-gray-700 hover:bg-gray-50 flex items-center gap-2"
                            >
                                <Pencil size={12} /> Rename
                            </button>
                            <button 
                                onClick={(e) => handleDeleteSession(e, chat.id)}
                                className="w-full text-left px-3 py-2 text-xs text-red-600 hover:bg-red-50 flex items-center gap-2"
                            >
                                <Trash2 size={12} /> Delete
                            </button>
                        </div>
                    )}
                  </div>
              )}
            </div>
          ))}
        </div>
        
        {/* User Profile*/}
        <div className="mt-4 pt-4 border-t flex items-center gap-3">
             <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 text-xs font-bold">
                {user.full_name ? user.full_name.charAt(0) : "U"}
             </div>
             <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-700 truncate">{user.full_name || user.name}</p>
             </div>
        </div>
      </aside>

      {/* ===== MAIN ===== */}
      <div className="flex-1 flex flex-col h-screen relative">

        {/* HEADER */}
        <header className="h-16 bg-white/80 backdrop-blur-md border-b flex items-center justify-between px-6 shrink-0 z-10 sticky top-0">
          
          {/* 2. Active Session Title ก*/}
          <div className="flex flex-col">
            <span className="font-semibold text-gray-800 text-lg">
                {activeChat ? activeChat.title : "New Chat"}
            </span>
            <span className="text-xs text-gray-400">
                {activeChatId ? "History enabled" : "Start a new conversation"}
            </span>
          </div>

          <div className="flex items-center gap-4">
             {/* 3. User Full Name (พื้นที่สีเหลือง) */}
             <div className="hidden md:flex flex-col items-end mr-2">
                <span className="text-sm font-semibold text-gray-700">
                    {user.full_name || user.name}
                </span>
             </div>

             <button
                onClick={onLogout}
                className="flex items-center gap-2 text-gray-500 hover:text-red-600 hover:bg-red-50 px-3 py-2 rounded-lg transition-all text-sm font-medium"
             >
                <LogOut size={18} />
             </button>
          </div>
        </header>

        {/* CHAT AREA */}
        <div 
            className="flex-1 overflow-y-auto p-4 sm:p-6 scroll-smooth" 
            onClick={() => setMenuOpenId(null)} // คลิกพื้นที่ว่างเพื่อปิดเมนู
        >
          <div className="max-w-3xl mx-auto flex flex-col gap-6 pb-4">

            {activeChat?.messages.length === 0 && !isLoading && (
                 <div className="flex flex-col items-center justify-center h-[50vh] text-gray-300 gap-4">
                    <Bot size={48} className="text-gray-200" />
                    <p>Ask anything about Industrial Automation...</p>
                 </div>
            )}

            {activeChat?.messages.map((m, i) => (
              <div
                key={i}
                className={`flex ${m.sender === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] px-5 py-3.5 rounded-2xl shadow-sm text-[15px] leading-relaxed
                    ${m.sender === "user"
                      ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-br-sm"
                      : "bg-white text-gray-800 border border-gray-100 rounded-bl-sm"
                    }`}
                >
                  {m.text}
                </div>
              </div>
            ))}

            {isLoading && (
              <div className="flex justify-start animate-pulse">
                <div className="bg-white px-4 py-3 rounded-2xl border border-gray-100 flex items-center gap-3 text-gray-500 text-sm shadow-sm">
                  <LoaderCircle size={18} className="animate-spin text-blue-500" />
                  <span className="font-medium">Thinking...</span>
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        </div>

        {/* INPUT AREA */}
        <div className="p-4 bg-white border-t shrink-0">
          <form
            onSubmit={handleSendMessage}
            className="max-w-3xl mx-auto flex items-end gap-2"
          >
            {/* Input Wrapper */}
            <div className="flex-1 flex items-center bg-gray-50 border border-gray-200 rounded-3xl px-2 py-2 focus-within:ring-2 focus-within:ring-blue-100 transition-all focus-within:border-blue-300 focus-within:bg-white shadow-sm">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about PLC, automation, troubleshooting..."
                className="flex-1 bg-transparent focus:outline-none text-gray-800 placeholder-gray-400 px-3 py-1.5"
                disabled={isLoading}
                aria-label="Message input"
              />

              {/* Send Button */}
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="bg-blue-600 text-white p-2.5 rounded-full hover:bg-blue-700 disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed transition-all shadow-sm flex-shrink-0"
                aria-label="Send message"
              >
                {isLoading ? (
                  <LoaderCircle size={20} className="animate-spin" />
                ) : (
                  <Send size={20} className={input.trim() ? "translate-x-0.5" : ""} />
                )}
              </button>
            </div>
          </form>

          <div className="text-center text-[10px] text-gray-400 mt-3 font-medium">
            PLC Assistant can make mistakes. Check important info.
          </div>
        </div>
      </div>
    </div>
  );
}

export default Chat;