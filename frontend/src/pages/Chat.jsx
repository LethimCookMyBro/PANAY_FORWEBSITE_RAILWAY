import { useState, useEffect, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import {
  Plus,
  Send,
  MessageSquareText,
  LoaderCircle,
  LogOut,
  Bot,
  Trash2,
  PanelLeftClose,
  PanelLeft,
  Menu,
  Mic,
  MicOff,
  Pin,
  Search,
  X,
  Copy,
  Check,
  CornerDownLeft
} from "lucide-react";

// Centralized API and hooks
import api, { authAPI, chatAPI } from "../utils/api";
import { useVoiceRecording } from "../hooks/useVoiceRecording";

/* ================= HELPER FUNCTIONS ================= */
const formatTimeAgo = (timestamp) => {
  if (!timestamp) return '';
  const now = new Date();
  const time = new Date(timestamp);
  const diffMs = now - time;
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffSec < 60) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  if (diffHour < 24) return `${diffHour}h ago`;
  if (diffDay === 1) return 'yesterday';
  if (diffDay < 7) return `${diffDay}d ago`;
  return time.toLocaleDateString();
};

const formatTime = (timestamp) => {
  if (!timestamp) return '';
  const time = new Date(timestamp);
  return time.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

/**
 * Convert markdown tables (including inline/malformed ones) to bullet points.
 * This directly transforms table content to bullet format, bypassing
 * ReactMarkdown table rendering for more reliable output.
 */
const fixMarkdownTable = (text) => {
  if (!text || !text.includes('|')) return text;

  const lines = text.split('\n');
  const fixedLines = [];

  for (const line of lines) {
    // Check if line contains table-like content (multiple | chars)
    const pipeCount = (line.match(/\|/g) || []).length;

    if (pipeCount >= 4) {
      // Split by | and get non-empty, non-separator parts
      const parts = line.split('|')
        .map(p => p.trim())
        .filter(p => p !== '' && !/^-+$/.test(p)); // Remove empty and separator cells

      if (parts.length >= 2) {
        // Check if this looks like a header row (Term, Description, etc)
        const isHeader = parts.every(p =>
          p.length < 30 && /^[A-Z][a-zA-Z\s()]*$/.test(p)
        );

        if (isHeader) {
          // Skip header row, we'll use content as bullet labels
          continue;
        }

        // Convert data cells to bullet points
        // Each non-empty cell becomes a bullet point
        const bullets = [];
        for (const part of parts) {
          if (part.length > 5) { // Skip very short cells
            // Check if it's "Term: Description" format
            const colonPos = part.indexOf(':');
            if (colonPos > 0 && colonPos < 50) {
              const term = part.substring(0, colonPos).trim();
              const desc = part.substring(colonPos + 1).trim();
              bullets.push(`• **${term}**: ${desc}`);
            } else {
              bullets.push(`• ${part}`);
            }
          }
        }

        if (bullets.length > 0) {
          fixedLines.push(bullets.join('\n'));
          continue;
        }
      }
    }
    fixedLines.push(line);
  }

  return fixedLines.join('\n');
};



function Chat({ onLogout }) {
  /* ================= STATE ================= */
  const [user, setUser] = useState({ full_name: "User", name: "User" });
  const [chatHistory, setChatHistory] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [isNewChat, setIsNewChat] = useState(false);

  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  // Pinned chats (stored in localStorage)
  const [pinnedChats, setPinnedChats] = useState(() => {
    const saved = localStorage.getItem('pinnedChats');
    return saved ? JSON.parse(saved) : [];
  });

  // Search state
  const [searchQuery, setSearchQuery] = useState("");

  // Copy state
  const [copiedMessageId, setCopiedMessageId] = useState(null);

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  // Voice recording hook
  const {
    isRecording,
    isTranscribing,
    startRecording,
    stopRecording,
    cancelTranscription
  } = useVoiceRecording((text) => {
    setInput(prev => prev + (prev ? ' ' : '') + text);
    inputRef.current?.focus();
  });

  // Copy to clipboard function
  const copyToClipboard = async (text, messageId) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedMessageId(messageId);
      setTimeout(() => setCopiedMessageId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  /* ================= MOBILE DETECTION & AUTO-COLLAPSE ================= */
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (mobile) {
        setSidebarCollapsed(true);
      }
    };

    checkMobile(); // Check on mount
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

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
          created_at: s.created_at,
          updated_at: s.updated_at || s.created_at,
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
  };

  // Temporary state to show first message in new chat before session is created
  const [pendingMessage, setPendingMessage] = useState(null);

  const handleSendMessage = useCallback(async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = { text: input, sender: "user", timestamp: new Date().toISOString() };
    setInput("");
    setIsLoading(true);

    if (activeChatId) {
      // Existing chat - add message immediately
      setChatHistory(prev =>
        prev.map(c =>
          c.id === activeChatId
            ? { ...c, messages: [...c.messages, userMessage] }
            : c
        )
      );
    } else {
      // New chat - show message immediately via pending state
      setPendingMessage(userMessage);
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
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        };
        setChatHistory(prev => [newSession, ...prev]);
        setActiveChatId(sessionId);
        setIsNewChat(false);
        setPendingMessage(null); // Clear pending since message is now in session
      }

      const botMessage = { text: res.data.reply, sender: "bot", timestamp: new Date().toISOString() };

      setChatHistory(prev =>
        prev.map(c =>
          c.id === sessionId
            ? { ...c, messages: [...c.messages, botMessage], updated_at: new Date().toISOString() }
            : c
        )
      );
    } catch (error) {
      console.error("Chat error:", error);
      setPendingMessage(null); // Clear on error too
    } finally {
      setIsLoading(false);
      setTimeout(() => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    }
  }, [input, isLoading, activeChatId, isNewChat]);

  /* ================= CHAT PINNING ================= */

  const togglePin = (e, chatId) => {
    e.stopPropagation();
    setPinnedChats(prev => {
      const newPinned = prev.includes(chatId)
        ? prev.filter(id => id !== chatId)
        : [...prev, chatId];
      localStorage.setItem('pinnedChats', JSON.stringify(newPinned));
      return newPinned;
    });
  };

  // Filter and sort chats: filter by search, then pinned first
  const sortedChatHistory = [...chatHistory]
    .filter(chat => {
      if (!searchQuery.trim()) return true;
      return (chat.title || '').toLowerCase().includes(searchQuery.toLowerCase());
    })
    .sort((a, b) => {
      const aPinned = pinnedChats.includes(a.id);
      const bPinned = pinnedChats.includes(b.id);
      if (aPinned && !bPinned) return -1;
      if (!aPinned && bPinned) return 1;
      return 0; // Keep original order within groups
    });

  /* ================= SESSION MANAGEMENT (RENAME / DELETE) ================= */

  // 1. Delete Session
  const handleDeleteSession = async (e, sessionId) => {
    e.stopPropagation();

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
  };



  const activeChat = chatHistory.find(c => c.id === activeChatId);

  /* ================= UI ================= */
  return (
    <div className="flex h-screen bg-gray-100 font-sans relative">

      {/* ===== MOBILE BACKDROP ===== */}
      {isMobile && !sidebarCollapsed && (
        <div
          className="fixed inset-0 bg-black/50 z-40"
          onClick={() => setSidebarCollapsed(true)}
        />
      )}

      {/* ===== SIDEBAR ===== */}
      <aside className={`
        ${isMobile
          ? `fixed top-0 left-0 h-full z-50 transform transition-transform duration-200 ease-in-out ${sidebarCollapsed ? '-translate-x-full' : 'translate-x-0'} w-72`
          : `${sidebarCollapsed ? 'w-16' : 'w-72'} transition-[width] duration-200 ease-in-out`
        } 
        p-4 bg-white border-r flex flex-col overflow-hidden
      `}>
        {/* Header with Logo and Collapse Button */}
        <div className={`flex-shrink-0 mb-6 flex ${!isMobile && sidebarCollapsed ? 'flex-col items-center gap-2' : 'items-center justify-between'}`}>
          {/* Logo Area - Expanded (and always on mobile when open) */}
          {(isMobile || !sidebarCollapsed) && (
            <div className="flex items-center gap-3 px-2">
              <div className="bg-gradient-to-br from-blue-600 to-blue-700 p-2 rounded-xl shadow-lg shadow-gray-200">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-800 tracking-tight">PLC Assistant</h1>
                <p className="text-[11px] font-medium text-gray-400 uppercase tracking-wider">Industrial AI</p>
              </div>
            </div>
          )}
          {/* Collapsed Logo - Desktop only */}
          {!isMobile && sidebarCollapsed && (
            <div className="bg-gradient-to-br from-blue-600 to-blue-700 p-2 rounded-xl shadow-lg shadow-gray-200">
              <Bot className="w-6 h-6 text-white" />
            </div>
          )}
          {/* Collapse/Close Button */}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-2 hover:bg-gray-100 rounded-lg transition-all text-gray-500 hover:text-gray-700"
            title={isMobile ? 'Close menu' : (sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar')}
          >
            {isMobile ? (
              <X size={20} />
            ) : (
              sidebarCollapsed ? <PanelLeft size={20} /> : <PanelLeftClose size={20} />
            )}
          </button>
        </div>

        {/* New Chat Button */}
        <button
          onClick={handleNewChat}
          className={`flex items-center justify-center gap-2 w-full p-3 mb-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white hover:from-blue-600 hover:to-blue-700 rounded-xl transition-all text-sm font-semibold shadow-md hover:shadow-lg ${!isMobile && sidebarCollapsed ? 'px-0' : ''}`}
          title="New Chat"
        >
          <Plus size={18} />
          {(isMobile || !sidebarCollapsed) && 'New Chat'}
        </button>

        {/* Search Box */}
        {(isMobile || !sidebarCollapsed) && (
          <div className="relative mb-3">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search chats..."
              className="w-full pl-9 pr-8 py-2 text-sm bg-gray-50 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-100 focus:border-blue-300"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-gray-200 rounded text-gray-400 hover:text-gray-600"
              >
                <X size={14} />
              </button>
            )}
          </div>
        )}

        {/* Recent Label */}
        {(isMobile || !sidebarCollapsed) && (
          <div className="px-2 mb-2 text-xs font-semibold text-gray-400 uppercase tracking-wider">
            {searchQuery ? `Results (${sortedChatHistory.length})` : 'Recent'}
          </div>
        )}
        {/* Chat List */}
        {(isMobile || !sidebarCollapsed) && (
          <div className="flex-1 overflow-y-auto space-y-1 pr-1 custom-scrollbar">
            {sortedChatHistory.map(chat => (
              <div
                key={chat.id}
                onClick={() => setActiveChatId(chat.id)}
                className={`group relative w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-3 cursor-pointer transition-colors
                ${chat.id === activeChatId ? "bg-blue-50/80 text-blue-700" : "text-gray-600 hover:bg-gray-50"}
                ${sidebarCollapsed ? 'justify-center px-0' : ''}`}
                title={sidebarCollapsed ? (chat.title || "New Chat") : undefined}
              >
                <div className="relative flex-shrink-0">
                  <MessageSquareText size={18} className={`${chat.id === activeChatId ? "text-blue-600" : "text-gray-400"}`} />
                  {pinnedChats.includes(chat.id) && (
                    <Pin size={10} className="absolute -top-1 -right-1 text-amber-500 fill-amber-500" />
                  )}
                </div>

                {/* === TITLE & TIMESTAMP === */}
                {!sidebarCollapsed && (
                  <div className="flex-1 min-w-0">
                    <span className="truncate text-sm font-medium block max-w-[120px]">
                      {chat.title || "New Chat"}
                    </span>
                    {chat.updated_at && (
                      <span className="text-[10px] text-gray-400">
                        {formatTimeAgo(chat.updated_at)}
                      </span>
                    )}
                  </div>
                )}

                {/* === ACTION BUTTONS on Hover === */}
                {!sidebarCollapsed && (
                  <div className="absolute right-2 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => togglePin(e, chat.id)}
                      className={`p-1.5 rounded ${pinnedChats.includes(chat.id) ? 'text-amber-500' : 'text-gray-400 hover:text-amber-500 hover:bg-amber-50'}`}
                      title={pinnedChats.includes(chat.id) ? "Unpin" : "Pin"}
                    >
                      <Pin size={14} className={pinnedChats.includes(chat.id) ? 'fill-amber-500' : ''} />
                    </button>
                    <button
                      onClick={(e) => handleDeleteSession(e, chat.id)}
                      className="p-1.5 hover:bg-red-100 rounded text-gray-400 hover:text-red-500"
                      title="Delete"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* User Profile with Logout */}
        <div className={`mt-4 pt-4 border-t flex items-center gap-3 ${!isMobile && sidebarCollapsed ? 'justify-center flex-col' : ''}`}>
          <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 text-xs font-bold flex-shrink-0">
            {user.full_name ? user.full_name.charAt(0) : "U"}
          </div>
          {(isMobile || !sidebarCollapsed) && (
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-700 truncate">{user.full_name || user.name}</p>
            </div>
          )}
          <button
            onClick={onLogout}
            className={`flex items-center gap-1.5 text-gray-500 hover:text-red-600 hover:bg-red-50 px-2 py-1.5 rounded-lg transition-all text-sm font-medium ${!isMobile && sidebarCollapsed ? 'mt-2' : ''}`}
            title="Logout"
          >
            <LogOut size={16} />
            {(isMobile || !sidebarCollapsed) && <span>Logout</span>}
          </button>
        </div>
      </aside>

      {/* ===== MAIN ===== */}
      <div className="flex-1 flex flex-col h-screen relative">

        {/* HEADER */}
        <header className="h-16 bg-white/80 backdrop-blur-md border-b flex items-center justify-between px-6 shrink-0 z-10 sticky top-0">

          {/* Mobile menu button + Title */}
          <div className="flex items-center gap-3">
            {/* Hamburger menu for mobile */}
            {isMobile && (
              <button
                onClick={() => setSidebarCollapsed(false)}
                className="p-2 hover:bg-gray-100 rounded-lg text-gray-600"
                title="Open menu"
              >
                <Menu size={20} />
              </button>
            )}
            <div className="flex flex-col">
              <span
                className="font-semibold text-gray-800 text-lg"
                title={activeChat ? activeChat.title : "New Chat"}
              >
                {activeChat
                  ? (activeChat.title && activeChat.title.length > 45
                    ? activeChat.title.substring(0, 45) + "..."
                    : activeChat.title || "New Chat")
                  : "New Chat"}
              </span>
            </div>
          </div>


        </header>

        {/* CHAT AREA */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 scroll-smooth">
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
                className={`flex flex-col ${m.sender === "user" ? "items-end" : "items-start"}`}
              >
                <div
                  className={`max-w-[85%] px-5 py-3.5 rounded-2xl shadow-sm text-[15px] leading-relaxed break-words overflow-hidden
                    ${m.sender === "user"
                      ? "bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-br-sm"
                      : "bg-white text-gray-800 border border-gray-100 rounded-bl-sm prose prose-sm max-w-none"
                    }`}
                  style={{ overflowWrap: 'anywhere', wordBreak: 'break-word' }}
                >
                  {m.sender === "bot" ? (
                    <ReactMarkdown
                      components={{
                        code({ node, inline, className, children, ...props }) {
                          const match = /language-(\w+)/.exec(className || '');
                          return !inline && match ? (
                            <SyntaxHighlighter
                              style={oneDark}
                              language={match[1]}
                              PreTag="div"
                              className="rounded-lg text-sm my-2"
                              {...props}
                            >
                              {String(children).replace(/\n$/, '')}
                            </SyntaxHighlighter>
                          ) : (
                            <code className="bg-gray-100 px-1.5 py-0.5 rounded text-sm font-mono" {...props}>
                              {children}
                            </code>
                          );
                        },
                        p({ children }) {
                          return <p className="mb-2 last:mb-0">{children}</p>;
                        },
                        ul({ children }) {
                          return <ul className="list-disc ml-4 mb-2">{children}</ul>;
                        },
                        ol({ children }) {
                          return <ol className="list-decimal ml-4 mb-2">{children}</ol>;
                        },
                        li({ children }) {
                          return <li className="mb-1">{children}</li>;
                        },
                        strong({ children }) {
                          return <strong className="font-semibold">{children}</strong>;
                        },
                        a({ href, children }) {
                          return <a href={href} className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">{children}</a>;
                        },
                        table({ children }) {
                          // Convert table to bullet point list
                          return <div className="my-3 space-y-1">{children}</div>;
                        },
                        thead() {
                          // Hide thead - we'll use header values inline
                          return null;
                        },
                        tbody({ children }) {
                          return <ul className="list-disc ml-4 space-y-2">{children}</ul>;
                        },
                        tr({ children }) {
                          // Convert row to a bullet item with all cells
                          const cells = [];
                          if (Array.isArray(children)) {
                            children.forEach((child, i) => {
                              if (child?.props?.children) {
                                cells.push(child.props.children);
                              }
                            });
                          }
                          if (cells.length === 0) return null;
                          return (
                            <li className="text-sm">
                              <span className="font-semibold">{cells[0]}</span>
                              {cells.length > 1 && `: ${cells.slice(1).join(' | ')}`}
                            </li>
                          );
                        },
                        th({ children }) {
                          return <span className="font-semibold">{children}</span>;
                        },
                        td({ children }) {
                          return <span>{children}</span>;
                        }
                      }}
                    >
                      {fixMarkdownTable(m.text)}
                    </ReactMarkdown>
                  ) : (
                    m.text
                  )}
                </div>
                <div className={`flex items-center gap-2 mt-1 ${m.sender === "user" ? "flex-row-reverse" : ""}`}>
                  {m.timestamp && (
                    <span className="text-[10px] text-gray-400 px-1">
                      {formatTime(m.timestamp)}
                    </span>
                  )}
                  <button
                    onClick={() => copyToClipboard(m.text, i)}
                    className="p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-all"
                    title={copiedMessageId === i ? "Copied!" : "Copy message"}
                  >
                    {copiedMessageId === i ? (
                      <Check size={14} className="text-green-500" />
                    ) : (
                      <Copy size={14} />
                    )}
                  </button>
                  <button
                    onClick={() => {
                      setInput(m.text);
                      inputRef.current?.focus();
                    }}
                    className="p-1 rounded hover:bg-blue-100 text-gray-400 hover:text-blue-600 transition-all"
                    title="Copy to chatbar"
                  >
                    <CornerDownLeft size={14} />
                  </button>
                </div>
              </div>
            ))}

            {/* Show pending message for new chat before session is created */}
            {pendingMessage && !activeChat && (
              <div className="flex flex-col items-end">
                <div
                  className="max-w-[85%] px-5 py-3.5 rounded-2xl shadow-sm text-[15px] leading-relaxed bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-br-sm break-words overflow-hidden"
                  style={{ overflowWrap: 'anywhere', wordBreak: 'break-word' }}
                >
                  {pendingMessage.text}
                </div>
                {pendingMessage.timestamp && (
                  <span className="text-[10px] text-gray-400 mt-1 px-1">
                    {formatTime(pendingMessage.timestamp)}
                  </span>
                )}
              </div>
            )}

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
                placeholder={isRecording ? "Listening..." : isTranscribing ? "Transcribing..." : "Ask about PLC, automation, troubleshooting..."}
                className="flex-1 bg-transparent focus:outline-none text-gray-800 placeholder-gray-400 px-3 py-1.5"
                disabled={isLoading || isRecording || isTranscribing}
                aria-label="Message input"
              />

              {/* Microphone Button */}
              <button
                type="button"
                onClick={isTranscribing ? cancelTranscription : (isRecording ? stopRecording : startRecording)}
                disabled={isLoading}
                className={`p-2.5 rounded-full transition-all flex-shrink-0 mr-1 group ${isRecording
                  ? 'bg-red-500 text-white animate-pulse'
                  : isTranscribing
                    ? 'bg-orange-100 text-orange-500 hover:bg-red-100 hover:text-red-500 cursor-pointer'
                    : 'hover:bg-gray-200 text-gray-500 hover:text-gray-700'
                  }`}
                aria-label={isTranscribing ? "Cancel transcription" : (isRecording ? "Stop recording" : "Start voice input")}
                title={isTranscribing ? "Click to cancel" : (isRecording ? "Click to stop" : "Voice input")}
              >
                {isTranscribing ? (
                  <LoaderCircle size={20} className="animate-spin group-hover:hidden" />
                ) : isRecording ? (
                  <MicOff size={20} />
                ) : (
                  <Mic size={20} />
                )}
                {isTranscribing && <MicOff size={20} className="hidden group-hover:block" />}
              </button>

              {/* Send Button */}
              <button
                type="submit"
                disabled={isLoading || !input.trim() || isRecording || isTranscribing}
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