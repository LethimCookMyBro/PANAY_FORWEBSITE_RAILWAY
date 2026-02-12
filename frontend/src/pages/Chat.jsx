import { useState, useEffect, useRef, useCallback, useMemo } from "react";
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
  CornerDownLeft,
  Sparkles,
} from "lucide-react";
import api, { getApiErrorMessage } from "../utils/api";
import { useVoiceRecording } from "../hooks/useVoiceRecording";

/* ============ HELPERS ============ */
const formatTimeAgo = (ts) => {
  if (!ts) return "";
  const diff = Math.floor((Date.now() - new Date(ts)) / 1000);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 172800) return "yesterday";
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(ts).toLocaleDateString();
};

const formatTime = (ts) =>
  ts
    ? new Date(ts).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";

const toArray = (value) => (Array.isArray(value) ? value : []);

const normalizeChatId = (value) => {
  if (value == null || value === "") return null;
  if (typeof value === "number") return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : value;
};

const pickListPayload = (payload, keys) => {
  if (Array.isArray(payload)) return payload;
  if (!payload || typeof payload !== "object") return [];
  for (const key of keys) {
    if (Array.isArray(payload[key])) return payload[key];
  }
  return [];
};

const unwrapResponsePayload = (payload) => {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return {};
  const nested = payload.data;
  if (nested && typeof nested === "object" && !Array.isArray(nested)) {
    return nested;
  }
  return payload;
};

const getReplyText = (payload) => {
  const normalizedPayload = unwrapResponsePayload(payload);
  const candidates = [
    normalizedPayload?.reply,
    normalizedPayload?.answer,
    normalizedPayload?.message,
    normalizedPayload?.response,
    normalizedPayload?.content,
    payload?.reply,
    payload?.answer,
    payload?.message,
    payload?.response,
    payload?.content,
  ];
  const text = candidates.find(
    (candidate) => typeof candidate === "string" && candidate.trim(),
  );
  return text?.trim() || "I couldn't generate a response right now. Please try again.";
};

const getResponseSessionId = (payload) => {
  const normalizedPayload = unwrapResponsePayload(payload);
  const candidates = [
    normalizedPayload?.session_id,
    normalizedPayload?.sessionId,
    normalizedPayload?.id,
    normalizedPayload?.chat_id,
    normalizedPayload?.chatId,
    normalizedPayload?.session?.id,
    normalizedPayload?.session?.session_id,
    normalizedPayload?.chat?.id,
    normalizedPayload?.chat?.session_id,
    normalizedPayload?.meta?.session_id,
    payload?.session_id,
    payload?.sessionId,
    payload?.id,
    payload?.chat_id,
    payload?.chatId,
    payload?.session?.id,
    payload?.session?.session_id,
    payload?.chat?.id,
    payload?.chat?.session_id,
    payload?.meta?.session_id,
  ];

  for (const candidate of candidates) {
    const id = normalizeChatId(candidate);
    if (id != null) return id;
  }
  return null;
};

const findFallbackSessionId = (payload, userText) => {
  const sessions = pickListPayload(payload, ["items", "sessions"])
    .map((s) => ({
      id: normalizeChatId(s?.id ?? s?.session_id ?? s?.sessionId),
      title: typeof s?.title === "string" ? s.title : "",
      updated_at: s?.updated_at || s?.created_at,
    }))
    .filter((s) => s.id != null);
  if (!sessions.length) return null;

  const targetTitle = userText.slice(0, 50).trim().toLowerCase();
  const now = Date.now();
  const freshSessions = sessions.filter((s) => {
    const timestamp = Date.parse(s.updated_at || "");
    if (Number.isNaN(timestamp)) return false;
    return Math.abs(now - timestamp) <= 5 * 60 * 1000;
  });
  const titleMatched = freshSessions.find(
    (s) => s.title.trim().toLowerCase() === targetTitle,
  );
  if (titleMatched) return titleMatched.id;

  return freshSessions[0]?.id ?? sessions[0]?.id ?? null;
};

const fixMarkdownTable = (text) => {
  if (!text?.includes("|")) return text;
  return text
    .split("\n")
    .map((line) => {
      const pipes = (line.match(/\|/g) || []).length;
      if (pipes < 4) return line;
      const parts = line
        .split("|")
        .map((p) => p.trim())
        .filter((p) => p && !/^-+$/.test(p));
      if (parts.length < 2) return line;
      if (parts.every((p) => p.length < 30 && /^[A-Z][a-zA-Z\s()]*$/.test(p)))
        return null;
      const bullets = parts
        .filter((p) => p.length > 5)
        .map((p) => {
          const c = p.indexOf(":");
          return c > 0 && c < 50
            ? `• **${p.slice(0, c).trim()}**: ${p.slice(c + 1).trim()}`
            : `• ${p}`;
        });
      return bullets.length ? bullets.join("\n") : line;
    })
    .filter(Boolean)
    .join("\n");
};

/* ============ MARKDOWN COMPONENTS (stable ref) ============ */
const mdComponents = {
  code({ inline, className, children, ...props }) {
    const match = /language-(\w+)/.exec(className || "");
    return !inline && match ? (
      <SyntaxHighlighter
        style={oneDark}
        language={match[1]}
        PreTag="div"
        className="rounded-lg text-sm my-2"
        {...props}
      >
        {String(children).replace(/\n$/, "")}
      </SyntaxHighlighter>
    ) : (
      <code
        className="bg-slate-100 px-1.5 py-0.5 rounded text-sm font-mono"
        {...props}
      >
        {children}
      </code>
    );
  },
  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="list-disc ml-4 mb-2">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal ml-4 mb-2">{children}</ol>,
  li: ({ children }) => <li className="mb-1">{children}</li>,
  strong: ({ children }) => (
    <strong className="font-semibold">{children}</strong>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      className="text-blue-500 hover:underline"
      target="_blank"
      rel="noopener noreferrer"
    >
      {children}
    </a>
  ),
  table: ({ children }) => <div className="my-3 space-y-1">{children}</div>,
  thead: () => null,
  tbody: ({ children }) => (
    <ul className="list-disc ml-4 space-y-2">{children}</ul>
  ),
  tr: ({ children }) => {
    const cells = [];
    toArray(children).forEach((c) => {
      if (c?.props?.children) cells.push(c.props.children);
    });
    if (!cells.length) return null;
    return (
      <li className="text-sm">
        <span className="font-semibold">{cells[0]}</span>
        {cells.length > 1 && `: ${cells.slice(1).join(" | ")}`}
      </li>
    );
  },
  th: ({ children }) => <span className="font-semibold">{children}</span>,
  td: ({ children }) => <span>{children}</span>,
};

/* ============ MAIN COMPONENT ============ */
export default function Chat({ onLogout }) {
  const [user, setUser] = useState({ full_name: "User" });
  const [chatHistory, setChatHistory] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [isNewChat, setIsNewChat] = useState(true);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const [pinnedChats, setPinnedChats] = useState(() => {
    try {
      return toArray(JSON.parse(localStorage.getItem("pinnedChats") || "[]"))
        .map((id) => normalizeChatId(id))
        .filter((id) => id != null);
    } catch {
      return [];
    }
  });
  const [searchQuery, setSearchQuery] = useState("");
  const [copiedId, setCopiedId] = useState(null);
  const [pendingMessage, setPendingMessage] = useState(null);
  const [apiError, setApiError] = useState("");

  const chatEndRef = useRef(null);
  const inputRef = useRef(null);

  const {
    isRecording,
    isTranscribing,
    startRecording,
    stopRecording,
    cancelTranscription,
  } = useVoiceRecording((text) => {
    setInput((p) => p + (p ? " " : "") + text);
    inputRef.current?.focus();
  });

  /* ---- derived ---- */
  const activeChat = chatHistory.find((c) => c.id === activeChatId);
  const activeMessages = useMemo(() => toArray(activeChat?.messages), [activeChat]);
  const hasMessages = activeMessages.length > 0 || !!pendingMessage;

  const sortedChats = useMemo(
    () =>
      [...chatHistory]
        .filter(
          (c) =>
            !searchQuery.trim() ||
            (c.title || "").toLowerCase().includes(searchQuery.toLowerCase()),
        )
        .sort((a, b) => {
          const ap = pinnedChats.includes(a.id),
            bp = pinnedChats.includes(b.id);
          return ap === bp ? 0 : ap ? -1 : 1;
        }),
    [chatHistory, searchQuery, pinnedChats],
  );

  /* ---- effects ---- */
  useEffect(() => {
    const check = () => {
      const m = window.innerWidth < 768;
      setIsMobile(m);
      if (m) setSidebarCollapsed(true);
    };
    check();
    window.addEventListener("resize", check);
    return () => window.removeEventListener("resize", check);
  }, []);

  useEffect(() => {
    api
      .get("/api/auth/me")
      .then((r) => {
        setUser(r.data);
      })
      .catch((err) => {
        console.error(err);
        setApiError(getApiErrorMessage(err, "Failed to load user profile"));
      });
  }, []);

  useEffect(() => {
    api
      .get("/api/chat/sessions")
      .then((r) => {
        setApiError("");
        const sessions = pickListPayload(r.data, ["items", "sessions"])
          .map((s) => ({
            id: normalizeChatId(s?.id),
            title: s?.title,
            messages: [],
            created_at: s?.created_at,
            updated_at: s?.updated_at || s?.created_at,
          }))
          .filter((s) => s.id != null);
        setChatHistory(sessions);
        if (sessions.length > 0) {
          setActiveChatId(sessions[0].id);
          setIsNewChat(false);
        } else {
          setActiveChatId(null);
          setIsNewChat(true);
        }
      })
      .catch((err) => {
        console.error(err);
        setApiError(getApiErrorMessage(err, "Failed to load chat sessions"));
      });
  }, []);

  useEffect(() => {
    if (activeChatId == null) return;
    const chat = chatHistory.find((c) => c.id === activeChatId);
    if (toArray(chat?.messages).length > 0) return;
    api
      .get(`/api/chat/sessions/${activeChatId}`)
      .then((r) => {
        setApiError("");
        const msgs = pickListPayload(r.data, ["items", "messages"]).map((m) => ({
          text: m?.content || "",
          sender: m?.role === "user" ? "user" : "bot",
          timestamp: m?.created_at,
          processingTime: m?.metadata?.processing_time,
          ragas: m?.metadata?.ragas,
        }));
        setChatHistory((p) =>
          p.map((c) => (c.id === activeChatId ? { ...c, messages: msgs } : c)),
        );
      })
      .catch((err) => {
        console.error(err);
        setApiError(getApiErrorMessage(err, "Failed to load chat messages"));
      });
  }, [activeChatId]);

  // Auto-focus input
  useEffect(() => {
    inputRef.current?.focus();
  }, [activeChatId, isNewChat]);

  const resizeComposer = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "0px";
    const next = Math.min(180, Math.max(44, el.scrollHeight));
    el.style.height = `${next}px`;
  }, []);

  useEffect(() => {
    resizeComposer();
  }, [input, resizeComposer, hasMessages]);

  /* ---- handlers ---- */
  const handleNewChat = () => {
    setActiveChatId(null);
    setIsNewChat(true);
    setInput("");
    setPendingMessage(null);
    setApiError("");
    setTimeout(() => inputRef.current?.focus(), 50);
  };

  const handleSend = useCallback(
    async (e) => {
      e?.preventDefault?.();
      const trimmedInput = input.trim();
      if (!trimmedInput || isLoading) return;

      const userMsg = {
        text: trimmedInput,
        sender: "user",
        timestamp: new Date().toISOString(),
      };
      const requestStartsNewChat = activeChatId == null || isNewChat;
      setApiError("");
      setInput("");
      setIsLoading(true);

      if (activeChatId != null) {
        setChatHistory((p) =>
          p.map((c) =>
            c.id === activeChatId
              ? { ...c, messages: [...toArray(c.messages), userMsg] }
              : c,
          ),
        );
      } else {
        setPendingMessage(userMsg);
      }

      try {
        const res = await api.post("/api/chat", {
          message: userMsg.text,
          session_id: requestStartsNewChat ? null : activeChatId,
        });
        const payload = res?.data || {};
        let sid =
          getResponseSessionId(payload) ??
          (requestStartsNewChat ? null : normalizeChatId(activeChatId));
        if (sid == null && requestStartsNewChat) {
          try {
            const sessionsRes = await api.get("/api/chat/sessions");
            sid = findFallbackSessionId(sessionsRes?.data, userMsg.text);
          } catch (lookupErr) {
            console.warn("Session fallback lookup failed:", lookupErr);
          }
        }
        if (sid == null) {
          throw new Error(
            "Chat response is missing session_id (check API response format/config)",
          );
        }
        const replyText = getReplyText(payload);
        const normalizedPayload = unwrapResponsePayload(payload);
        const nowIso = new Date().toISOString();

        if (requestStartsNewChat) {
          setChatHistory((p) => {
            if (p.some((c) => c.id === sid)) {
              return p.map((c) =>
                c.id === sid
                  ? {
                      ...c,
                      title: c.title || userMsg.text.slice(0, 50),
                      messages: [...toArray(c.messages), userMsg],
                      updated_at: nowIso,
                    }
                  : c,
              );
            }
            const newSession = {
              id: sid,
              title: userMsg.text.slice(0, 50),
              messages: [userMsg],
              created_at: nowIso,
              updated_at: nowIso,
            };
            return [newSession, ...p];
          });
          setActiveChatId(sid);
          setIsNewChat(false);
          setPendingMessage(null);
        }

        const botMsg = {
          text: replyText,
          sender: "bot",
          timestamp: new Date().toISOString(),
          processingTime:
            normalizedPayload.processing_time ?? payload.processing_time,
          ragas: normalizedPayload.ragas ?? payload.ragas,
        };

        setChatHistory((p) =>
          p.map((c) =>
            c.id === sid
              ? {
                  ...c,
                  messages: [...toArray(c.messages), botMsg],
                  updated_at: new Date().toISOString(),
                }
              : c,
          ),
        );
      } catch (err) {
        console.error("Chat error:", err);
        setApiError(getApiErrorMessage(err, "Failed to send message"));
        setInput((current) => current || userMsg.text);
        if (requestStartsNewChat) {
          setPendingMessage(userMsg);
        }
      } finally {
        setIsLoading(false);
        setTimeout(
          () => chatEndRef.current?.scrollIntoView({ behavior: "smooth" }),
          100,
        );
      }
    },
    [input, isLoading, activeChatId, isNewChat],
  );

  const handleComposerKeyDown = useCallback(
    (e) => {
      if (e.key !== "Enter" || e.shiftKey) return;
      e.preventDefault();
      if (isLoading || isRecording || isTranscribing || !input.trim()) return;
      handleSend();
    },
    [handleSend, input, isLoading, isRecording, isTranscribing],
  );

  const togglePin = (e, id) => {
    e.stopPropagation();
    const normalizedId = normalizeChatId(id);
    if (normalizedId == null) return;
    setPinnedChats((p) => {
      const next = p.includes(normalizedId)
        ? p.filter((x) => x !== normalizedId)
        : [...p, normalizedId];
      localStorage.setItem("pinnedChats", JSON.stringify(next));
      return next;
    });
  };

  const handleDelete = async (e, id) => {
    e.stopPropagation();
    const normalizedId = normalizeChatId(id);
    if (normalizedId == null) return;
    try {
      await api.delete(`/api/chat/sessions/${normalizedId}`);
      setChatHistory((p) => p.filter((c) => c.id !== normalizedId));
      if (activeChatId === normalizedId) handleNewChat();
    } catch (err) {
      console.error("Delete failed", err);
      setApiError(getApiErrorMessage(err, "Failed to delete chat"));
    }
  };

  const copyMsg = async (text, id) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error("Copy failed:", err);
    }
  };

  /* ---- input bar (shared between centered & bottom) ---- */
  const renderInputBar = (centered = false) => (
    <form
      onSubmit={handleSend}
      className={`w-full ${centered ? "max-w-2xl" : "max-w-3xl"} mx-auto flex items-end gap-2`}
    >
      <div className="liquid-input-wrap glass-input flex-1 rounded-[26px] px-3 py-2.5 transition-all shadow-lg shadow-black/5">
        <textarea
          ref={inputRef}
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleComposerKeyDown}
          placeholder={
            isRecording
              ? "Listening..."
              : isTranscribing
                ? "Transcribing..."
                : "Ask about PLC, automation, troubleshooting..."
          }
          className="composer-textarea w-full bg-transparent focus:outline-none text-slate-800 placeholder-slate-400 px-2 py-1.5 resize-none"
          disabled={isLoading || isRecording || isTranscribing}
        />
        <div className="flex items-center justify-end gap-1 pr-1">
          <button
            type="button"
            onClick={
              isTranscribing
                ? cancelTranscription
                : isRecording
                  ? stopRecording
                  : startRecording
            }
            disabled={isLoading}
            className={`p-2 rounded-full transition-all flex-shrink-0 ${
              isRecording
                ? "bg-red-500 text-white animate-pulse"
                : isTranscribing
                  ? "bg-orange-100 text-orange-500"
                  : "hover:bg-slate-100/80 text-slate-400 hover:text-slate-600"
            }`}
          >
            {isTranscribing ? (
              <LoaderCircle size={18} className="animate-spin" />
            ) : isRecording ? (
              <MicOff size={18} />
            ) : (
              <Mic size={18} />
            )}
          </button>
          <button
            type="submit"
            disabled={isLoading || !input.trim() || isRecording || isTranscribing}
            className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white p-2 rounded-full hover:from-blue-600 hover:to-cyan-600 disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-md shadow-blue-500/20 flex-shrink-0"
          >
            {isLoading ? (
              <LoaderCircle size={18} className="animate-spin" />
            ) : (
              <Send size={18} className={input.trim() ? "translate-x-0.5" : ""} />
            )}
          </button>
        </div>
      </div>
    </form>
  );

  /* ================= RENDER ================= */
  return (
    <div className="liquid-shell flex h-screen font-sans relative overflow-hidden">
      <div className="liquid-orb liquid-orb-a" />
      <div className="liquid-orb liquid-orb-b" />
      <div className="liquid-orb liquid-orb-c" />
      {/* Mobile backdrop */}
      {isMobile && !sidebarCollapsed && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          onClick={() => setSidebarCollapsed(true)}
        />
      )}

      {/* ===== SIDEBAR ===== */}
      <aside
        className={`
        ${
          isMobile
            ? `fixed top-0 left-0 h-full z-50 transform transition-transform duration-200 ease-out ${sidebarCollapsed ? "-translate-x-full" : "translate-x-0"} w-72`
            : `${sidebarCollapsed ? "w-16" : "w-72"} transition-[width] duration-200 ease-out`
        }
        p-4 glass-dark border-r border-slate-800/50 flex flex-col overflow-hidden shadow-xl shadow-slate-950/30
      `}
      >
        {/* Logo */}
        <div
          className={`flex-shrink-0 mb-6 flex ${!isMobile && sidebarCollapsed ? "flex-col items-center gap-2" : "items-center justify-between"}`}
        >
          {(isMobile || !sidebarCollapsed) && (
            <div className="flex items-center gap-3 px-1">
              <div className="bg-gradient-to-br from-blue-500 to-cyan-400 p-2 rounded-xl shadow-lg shadow-blue-500/25">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-white tracking-tight">
                  Panya
                </h1>
                <p className="text-[11px] font-medium text-cyan-400/50 uppercase tracking-wider">
                  PLC Assistant
                </p>
              </div>
            </div>
          )}
          {!isMobile && sidebarCollapsed && (
            <div className="bg-gradient-to-br from-blue-500 to-cyan-400 p-2 rounded-xl shadow-lg shadow-blue-500/25">
              <Bot className="w-5 h-5 text-white" />
            </div>
          )}
          <button
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-2 hover:bg-white/5 rounded-lg transition-all text-slate-400 hover:text-white"
          >
            {isMobile ? (
              <X size={20} />
            ) : sidebarCollapsed ? (
              <PanelLeft size={20} />
            ) : (
              <PanelLeftClose size={20} />
            )}
          </button>
        </div>

        {/* New Chat */}
        <button
          onClick={handleNewChat}
          className={`flex items-center justify-center gap-2 w-full p-3 mb-4 bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:from-blue-600 hover:to-cyan-600 rounded-xl transition-all text-sm font-semibold shadow-lg shadow-blue-500/20 ${!isMobile && sidebarCollapsed ? "px-0" : ""}`}
        >
          <Plus size={18} />
          {(isMobile || !sidebarCollapsed) && "New Chat"}
        </button>

        {/* Search */}
        {(isMobile || !sidebarCollapsed) && (
          <div className="relative mb-3">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
            />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search chats..."
              className="w-full pl-9 pr-8 py-2 text-sm bg-white/5 border border-white/10 rounded-lg focus:outline-none focus:ring-1 focus:ring-cyan-500/30 text-slate-200 placeholder-slate-500"
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-white/10 rounded text-slate-500"
              >
                <X size={14} />
              </button>
            )}
          </div>
        )}

        {/* Label */}
        {(isMobile || !sidebarCollapsed) && (
          <div className="px-2 mb-2 text-[10px] font-semibold text-slate-500 uppercase tracking-widest">
            {searchQuery ? `Results (${sortedChats.length})` : "Recent"}
          </div>
        )}

        {/* Chat list */}
        {(isMobile || !sidebarCollapsed) && (
          <div className="flex-1 overflow-y-auto space-y-0.5 pr-1 sidebar-scroll">
            {sortedChats.map((chat) => (
              <div
                key={chat.id}
                onClick={() => {
                  setActiveChatId(chat.id);
                  setIsNewChat(false);
                  setPendingMessage(null);
                  if (isMobile) setSidebarCollapsed(true);
                }}
                className={`group relative w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-3 cursor-pointer transition-all
                  ${chat.id === activeChatId ? "bg-white/10 text-white" : "text-slate-400 hover:bg-white/5 hover:text-slate-200"}`}
              >
                <div className="relative flex-shrink-0">
                  <MessageSquareText
                    size={16}
                    className={
                      chat.id === activeChatId
                        ? "text-cyan-400"
                        : "text-slate-500"
                    }
                  />
                  {pinnedChats.includes(chat.id) && (
                    <Pin
                      size={8}
                      className="absolute -top-1 -right-1 text-amber-400 fill-amber-400"
                    />
                  )}
                </div>
                {!sidebarCollapsed && (
                  <div className="flex-1 min-w-0">
                    <span className="truncate text-sm font-medium block">
                      {chat.title || "New Chat"}
                    </span>
                    {chat.updated_at && (
                      <span className="text-[10px] text-slate-500">
                        {formatTimeAgo(chat.updated_at)}
                      </span>
                    )}
                  </div>
                )}
                {!sidebarCollapsed && (
                  <div className="absolute right-1 flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => togglePin(e, chat.id)}
                      className={`p-1 rounded ${pinnedChats.includes(chat.id) ? "text-amber-400" : "text-slate-500 hover:text-amber-400"}`}
                    >
                      <Pin
                        size={13}
                        className={
                          pinnedChats.includes(chat.id) ? "fill-amber-400" : ""
                        }
                      />
                    </button>
                    <button
                      onClick={(e) => handleDelete(e, chat.id)}
                      className="p-1 rounded text-slate-500 hover:text-red-400"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* User profile */}
        <div
          className={`mt-4 pt-4 border-t border-white/5 flex items-center gap-3 ${!isMobile && sidebarCollapsed ? "justify-center flex-col" : ""}`}
        >
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500/30 to-cyan-400/30 flex items-center justify-center text-cyan-300 text-xs font-bold flex-shrink-0">
            {user.full_name ? user.full_name.charAt(0) : "U"}
          </div>
          {(isMobile || !sidebarCollapsed) && (
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-slate-200 truncate">
                {user.full_name || user.name}
              </p>
            </div>
          )}
          <button
            onClick={onLogout}
            className={`flex items-center gap-1.5 text-slate-500 hover:text-red-400 px-2 py-1.5 rounded-lg transition-all text-sm ${!isMobile && sidebarCollapsed ? "mt-2" : ""}`}
          >
            <LogOut size={16} />
            {(isMobile || !sidebarCollapsed) && <span>Logout</span>}
          </button>
        </div>
      </aside>

      {/* ===== MAIN AREA ===== */}
      <div className="flex-1 flex flex-col h-screen relative z-10">
        {/* Header */}
        <header className="h-12 glass border-b border-slate-200/40 flex items-center justify-between px-5 shrink-0 z-10">
          <div className="flex items-center gap-3">
            {isMobile && (
              <button
                onClick={() => setSidebarCollapsed(false)}
                className="p-1.5 hover:bg-slate-100 rounded-lg text-slate-500"
              >
                <Menu size={20} />
              </button>
            )}
            <span className="font-semibold text-slate-700 text-sm truncate max-w-[300px]">
              {activeChat
                ? activeChat.title?.length > 45
                  ? activeChat.title.slice(0, 45) + "..."
                  : activeChat.title || "New Chat"
                : "New Chat"}
            </span>
          </div>
        </header>

        {apiError && (
          <div className="px-5 pt-3">
            <div className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {apiError}
            </div>
          </div>
        )}

        {/* ===== CHAT CONTENT ===== */}
        {!hasMessages ? (
          /* ---- CENTERED WELCOME (ChatGPT-style) ---- */
          <div className="flex-1 flex flex-col items-center justify-center px-4 pb-8">
            {/* Hero */}
            <div className="mb-8 text-center fade-in-up">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-400 shadow-xl shadow-blue-500/20 mb-5">
                <Sparkles size={28} className="text-white" />
              </div>
              <h2 className="text-2xl font-bold text-slate-800 mb-2">
                Hey{user.full_name ? `, ${user.full_name.split(" ")[0]}` : ""}!
              </h2>
              <p className="text-slate-400 max-w-md text-sm">
                Your PLC & Industrial Automation expert. Ask me anything about
                troubleshooting, error codes, or technical docs.
              </p>
            </div>

            {/* Centered Input */}
            <div
              className="w-full max-w-2xl mb-6 fade-in-up"
              style={{ animationDelay: "0.1s" }}
            >
              {renderInputBar(true)}
            </div>

            {/* Quick prompts */}
            <div
              className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-2xl w-full fade-in-up"
              style={{ animationDelay: "0.2s" }}
            >
              {[
                "What does error code F800H mean?",
                "How to configure CC-Link IE Field?",
                "FX3 timer instructions",
                "Data Collector troubleshooting",
              ].map((q, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setInput(q);
                    inputRef.current?.focus();
                  }}
                  className="text-left px-4 py-3 glass-prompt rounded-xl text-sm text-slate-500 hover:text-blue-600 hover:border-blue-200 transition-all"
                >
                  {q}
                </button>
              ))}
            </div>

            <p className="text-[10px] text-slate-400 mt-6">
              Panya may make mistakes. Verify important information.
            </p>
          </div>
        ) : (
          /* ---- MESSAGES + BOTTOM INPUT ---- */
          <>
            <div className="flex-1 overflow-y-auto p-4 sm:p-6 scroll-smooth">
              <div className="max-w-3xl mx-auto flex flex-col gap-5 pb-4">
                {activeMessages.map((m, i) => {
                  const processingTime = Number(m?.processingTime);
                  const hasProcessingTime =
                    Number.isFinite(processingTime) && processingTime > 0;
                  return (
                    <div
                      key={i}
                      className={`flex ${m.sender === "user" ? "justify-end" : "items-start gap-3"} fade-in-up`}
                    >
                    {/* Bot avatar */}
                    {m.sender === "bot" && (
                      <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center flex-shrink-0 mt-1 shadow-sm">
                        <Bot size={14} className="text-white" />
                      </div>
                    )}
                    <div
                      className={`flex flex-col ${m.sender === "user" ? "items-end" : "items-start flex-1 min-w-0"}`}
                    >
                      <div
                        className={`max-w-[85%] px-4 py-3 rounded-2xl text-[14px] leading-relaxed break-words overflow-hidden
                          ${
                            m.sender === "user"
                              ? "bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-br-md shadow-md shadow-blue-500/15"
                              : "bg-white text-slate-700 border border-slate-100 rounded-bl-md shadow-sm prose prose-sm max-w-none"
                          }`}
                        style={{ overflowWrap: "anywhere" }}
                      >
                        {m.sender === "bot" ? (
                          <ReactMarkdown components={mdComponents}>
                            {fixMarkdownTable(m.text)}
                          </ReactMarkdown>
                        ) : (
                          m.text
                        )}
                      </div>

                      {/* Metrics */}
                      {m.sender === "bot" && hasProcessingTime && (
                        <div className="flex items-center gap-2 mt-1.5 text-[10px] text-slate-400">
                          <span>⏱ {processingTime.toFixed(2)}s</span>
                          {m.ragas?.scores?.faithfulness != null && (
                            <span>
                              • Faithfulness:{" "}
                              {(m.ragas.scores.faithfulness * 100).toFixed(0)}%
                            </span>
                          )}
                        </div>
                      )}

                      {/* Actions */}
                      <div
                        className={`flex items-center gap-1 mt-1 ${m.sender === "user" ? "flex-row-reverse" : ""}`}
                      >
                        {m.timestamp && (
                          <span className="text-[10px] text-slate-300 px-1">
                            {formatTime(m.timestamp)}
                          </span>
                        )}
                        <button
                          onClick={() => copyMsg(m.text, i)}
                          className="p-1 rounded hover:bg-slate-100 text-slate-300 hover:text-slate-500 transition-all"
                        >
                          {copiedId === i ? (
                            <Check size={12} className="text-green-500" />
                          ) : (
                            <Copy size={12} />
                          )}
                        </button>
                        <button
                          onClick={() => {
                            setInput(m.text);
                            inputRef.current?.focus();
                          }}
                          className="p-1 rounded hover:bg-blue-50 text-slate-300 hover:text-blue-500 transition-all"
                        >
                          <CornerDownLeft size={12} />
                        </button>
                      </div>
                    </div>
                  </div>
                  );
                })}

                {/* Pending message */}
                {pendingMessage && !activeChat && (
                  <div className="flex justify-end fade-in-up">
                    <div
                      className="max-w-[85%] px-4 py-3 rounded-2xl rounded-br-md bg-gradient-to-r from-blue-500 to-blue-600 text-white text-[14px] shadow-md shadow-blue-500/15 break-words"
                      style={{ overflowWrap: "anywhere" }}
                    >
                      {pendingMessage.text}
                    </div>
                  </div>
                )}

                {/* Typing indicator */}
                {isLoading && (
                  <div className="flex items-start gap-3 fade-in-up">
                    <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-400 flex items-center justify-center flex-shrink-0 shadow-sm pulse-glow">
                      <Bot size={14} className="text-white" />
                    </div>
                    <div className="bg-white px-5 py-3.5 rounded-2xl border border-slate-100 shadow-sm">
                      <div className="flex items-center gap-1.5">
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                      </div>
                    </div>
                  </div>
                )}

                <div ref={chatEndRef} />
              </div>
            </div>

            {/* Bottom Input */}
            <div className="p-3 glass border-t border-slate-200/40 shrink-0">
              {renderInputBar(false)}
              <p className="text-center text-[10px] text-slate-400 mt-2">
                Panya may make mistakes. Verify important information.
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
