import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Menu, RefreshCw } from "lucide-react";
import api, { getApiErrorMessage, retryApiRequest } from "../utils/api";
import { useVoiceRecording } from "../hooks/useVoiceRecording";
import ChatComposer from "./chat/ChatComposer";
import ChatMessages from "./chat/ChatMessages";
import ChatSidebar from "./chat/ChatSidebar";
import ChatWelcome from "./chat/ChatWelcome";
import {
  findFallbackSessionId,
  getReplyText,
  getResponseSessionId,
  makeLocalMessageId,
  mapMessagesFromPayload,
  mapSessionsFromPayload,
  normalizeChatId,
  normalizePageNumber,
  normalizeSourceItems,
  toArray,
  unwrapResponsePayload,
} from "./chat/utils";

export default function Chat({ onLogout }) {
  const [user, setUser] = useState({ full_name: "User" });
  const [chatHistory, setChatHistory] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [isNewChat, setIsNewChat] = useState(true);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isCompactLayout, setIsCompactLayout] = useState(false);
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
  const [isRecovering, setIsRecovering] = useState(false);

  const messagesContainerRef = useRef(null);
  const inputRef = useRef(null);
  const compactModeRef = useRef(null);

  const {
    isRecording,
    isTranscribing,
    startRecording,
    stopRecording,
    cancelTranscription,
  } = useVoiceRecording((text) => {
    setInput((prev) => prev + (prev ? " " : "") + text);
    inputRef.current?.focus();
  });

  const activeChat = useMemo(
    () => chatHistory.find((chat) => chat.id === activeChatId),
    [activeChatId, chatHistory],
  );

  const activeMessages = useMemo(() => toArray(activeChat?.messages), [activeChat]);
  const hasMessages = activeMessages.length > 0 || !!pendingMessage;

  const sortedChats = useMemo(
    () =>
      [...chatHistory]
        .filter(
          (chat) =>
            !searchQuery.trim() ||
            (chat.title || "").toLowerCase().includes(searchQuery.toLowerCase()),
        )
        .sort((left, right) => {
          const leftPinned = pinnedChats.includes(left.id);
          const rightPinned = pinnedChats.includes(right.id);
          return leftPinned === rightPinned ? 0 : leftPinned ? -1 : 1;
        }),
    [chatHistory, pinnedChats, searchQuery],
  );

  const loadBootstrapData = useCallback(async () => {
    const [profileRes, sessionsRes] = await Promise.all([
      retryApiRequest(() => api.get("/api/auth/me"), {
        retries: 2,
        baseDelayMs: 550,
      }),
      retryApiRequest(() => api.get("/api/chat/sessions"), {
        retries: 2,
        baseDelayMs: 550,
      }),
    ]);

    setUser(profileRes?.data || { full_name: "User" });

    const sessions = mapSessionsFromPayload(sessionsRes?.data);
    setChatHistory(sessions);
    setActiveChatId((currentId) => {
      if (currentId != null && sessions.some((session) => session.id === currentId)) {
        return currentId;
      }
      return sessions[0]?.id ?? null;
    });
    setIsNewChat(sessions.length === 0);
    setApiError("");

    return sessions;
  }, []);

  const loadMessagesForSession = useCallback(async (sessionId) => {
    const response = await retryApiRequest(
      () => api.get(`/api/chat/sessions/${sessionId}`),
      {
        retries: 1,
        baseDelayMs: 500,
      },
    );

    const messages = mapMessagesFromPayload(response?.data);
    setChatHistory((prev) =>
      prev.map((chat) => (chat.id === sessionId ? { ...chat, messages } : chat)),
    );
    setApiError("");

    return messages;
  }, []);

  useEffect(() => {
    const updateResponsiveMode = () => {
      const nextCompact = window.innerWidth <= 1024;
      setIsCompactLayout(nextCompact);

      if (compactModeRef.current == null || compactModeRef.current !== nextCompact) {
        setSidebarCollapsed(nextCompact);
        compactModeRef.current = nextCompact;
      }
    };

    updateResponsiveMode();
    window.addEventListener("resize", updateResponsiveMode);
    return () => window.removeEventListener("resize", updateResponsiveMode);
  }, []);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        await loadBootstrapData();
      } catch (error) {
        if (cancelled) return;
        console.error(error);
        setApiError(getApiErrorMessage(error, "Failed to load chat data"));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [loadBootstrapData]);

  useEffect(() => {
    if (activeChatId == null) return;

    const chat = chatHistory.find((item) => item.id === activeChatId);
    if (toArray(chat?.messages).length > 0) return;

    let cancelled = false;
    loadMessagesForSession(activeChatId).catch((error) => {
      if (cancelled) return;
      console.error(error);
      setApiError(getApiErrorMessage(error, "Failed to load chat messages"));
    });

    return () => {
      cancelled = true;
    };
  }, [activeChatId, chatHistory, loadMessagesForSession]);

  useEffect(() => {
    inputRef.current?.focus();
  }, [activeChatId, isNewChat]);

  const resizeComposer = useCallback(() => {
    const element = inputRef.current;
    if (!element) return;

    element.style.height = "0px";
    const nextHeight = Math.min(180, Math.max(44, element.scrollHeight));
    element.style.height = `${nextHeight}px`;
  }, []);

  useEffect(() => {
    resizeComposer();
  }, [input, resizeComposer, hasMessages]);

  const scrollMessagesToBottom = useCallback((behavior = "smooth") => {
    const container = messagesContainerRef.current;
    if (!container) return;

    if (typeof container.scrollTo === "function") {
      container.scrollTo({ top: container.scrollHeight, behavior });
      return;
    }

    container.scrollTop = container.scrollHeight;
  }, []);

  useEffect(() => {
    if (!hasMessages) return;

    const rafId = window.requestAnimationFrame(() => {
      scrollMessagesToBottom(isLoading ? "smooth" : "auto");
    });

    return () => window.cancelAnimationFrame(rafId);
  }, [
    activeChatId,
    activeMessages.length,
    hasMessages,
    isLoading,
    pendingMessage,
    scrollMessagesToBottom,
  ]);

  const handleNewChat = useCallback(() => {
    setActiveChatId(null);
    setIsNewChat(true);
    setInput("");
    setPendingMessage(null);
    setApiError("");

    if (isCompactLayout) {
      setSidebarCollapsed(true);
    }

    setTimeout(() => inputRef.current?.focus(), 50);
  }, [isCompactLayout]);

  const handleRetryConnection = useCallback(async () => {
    if (isRecovering) return;

    setIsRecovering(true);
    setApiError("");

    try {
      const sessions = await loadBootstrapData();
      const hasCurrentSession =
        activeChatId != null && sessions.some((session) => session.id === activeChatId);
      const targetSessionId = hasCurrentSession ? activeChatId : sessions[0]?.id ?? null;

      if (targetSessionId != null) {
        await loadMessagesForSession(targetSessionId);
      }
    } catch (error) {
      console.error(error);
      setApiError(getApiErrorMessage(error, "Failed to reconnect backend"));
    } finally {
      setIsRecovering(false);
    }
  }, [activeChatId, isRecovering, loadBootstrapData, loadMessagesForSession]);

  const handleSend = useCallback(
    async (event) => {
      event?.preventDefault?.();

      const trimmedInput = input.trim();
      if (!trimmedInput || isLoading) return;

      const userMessage = {
        id: makeLocalMessageId(),
        text: trimmedInput,
        sender: "user",
        timestamp: new Date().toISOString(),
        status: "sent",
      };

      const requestStartsNewChat = activeChatId == null || isNewChat;

      setApiError("");
      setInput("");
      setIsLoading(true);

      if (activeChatId != null) {
        setChatHistory((prev) =>
          prev.map((chat) =>
            chat.id === activeChatId
              ? { ...chat, messages: [...toArray(chat.messages), userMessage] }
              : chat,
          ),
        );
      } else {
        setPendingMessage(userMessage);
      }

      try {
        const response = await api.post("/api/chat", {
          message: userMessage.text,
          session_id: requestStartsNewChat ? null : activeChatId,
        });

        const payload = response?.data || {};
        let sessionId =
          getResponseSessionId(payload) ??
          (requestStartsNewChat ? null : normalizeChatId(activeChatId));

        if (sessionId == null && requestStartsNewChat) {
          try {
            const sessionsRes = await retryApiRequest(() => api.get("/api/chat/sessions"), {
              retries: 1,
              baseDelayMs: 500,
            });
            sessionId = findFallbackSessionId(sessionsRes?.data, userMessage.text);
          } catch (lookupError) {
            console.warn("Session fallback lookup failed:", lookupError);
          }
        }

        if (sessionId == null) {
          throw new Error(
            "Chat response is missing session_id (check API response format/config)",
          );
        }

        const normalizedPayload = unwrapResponsePayload(payload);
        const botMessage = {
          id: makeLocalMessageId(),
          text: getReplyText(payload),
          sender: "bot",
          timestamp: new Date().toISOString(),
          processingTime:
            normalizedPayload.processing_time ?? payload.processing_time,
          ragas: normalizedPayload.ragas ?? payload.ragas,
          sources: normalizeSourceItems(
            normalizedPayload?.sources ?? payload?.sources,
          ),
          status: "sent",
        };

        const nowIso = new Date().toISOString();

        if (requestStartsNewChat) {
          setChatHistory((prev) => {
            if (prev.some((chat) => chat.id === sessionId)) {
              return prev.map((chat) =>
                chat.id === sessionId
                  ? {
                      ...chat,
                      title: chat.title || userMessage.text.slice(0, 50),
                      messages: [...toArray(chat.messages), userMessage],
                      updated_at: nowIso,
                    }
                  : chat,
              );
            }

            const newSession = {
              id: sessionId,
              title: userMessage.text.slice(0, 50),
              messages: [userMessage],
              created_at: nowIso,
              updated_at: nowIso,
            };

            return [newSession, ...prev];
          });

          setActiveChatId(sessionId);
          setIsNewChat(false);
          setPendingMessage(null);
        }

        setChatHistory((prev) =>
          prev.map((chat) =>
            chat.id === sessionId
              ? {
                  ...chat,
                  messages: [...toArray(chat.messages), botMessage],
                  updated_at: new Date().toISOString(),
                }
              : chat,
          ),
        );
      } catch (error) {
        console.error("Chat error:", error);
        setApiError(getApiErrorMessage(error, "Failed to send message"));

        if (!requestStartsNewChat && activeChatId != null) {
          setChatHistory((prev) =>
            prev.map((chat) =>
              chat.id === activeChatId
                ? {
                    ...chat,
                    messages: toArray(chat.messages).map((message) =>
                      message?.id === userMessage.id
                        ? { ...message, status: "failed" }
                        : message,
                    ),
                  }
                : chat,
            ),
          );
        }

        setInput((current) => current || userMessage.text);
        if (requestStartsNewChat) {
          setPendingMessage({ ...userMessage, status: "failed" });
        }
      } finally {
        setIsLoading(false);
        setTimeout(() => scrollMessagesToBottom("smooth"), 100);
      }
    },
    [activeChatId, input, isLoading, isNewChat, scrollMessagesToBottom],
  );

  const handleComposerKeyDown = useCallback(
    (event) => {
      if (event.key !== "Enter" || event.shiftKey) return;
      event.preventDefault();
      if (isLoading || isRecording || isTranscribing || !input.trim()) return;
      handleSend();
    },
    [handleSend, input, isLoading, isRecording, isTranscribing],
  );

  const handleSelectChat = useCallback(
    (chatId) => {
      setActiveChatId(chatId);
      setIsNewChat(false);
      setPendingMessage(null);
      if (isCompactLayout) {
        setSidebarCollapsed(true);
      }
    },
    [isCompactLayout],
  );

  const togglePin = useCallback((event, id) => {
    event.stopPropagation();

    const normalizedId = normalizeChatId(id);
    if (normalizedId == null) return;

    setPinnedChats((prev) => {
      const next = prev.includes(normalizedId)
        ? prev.filter((value) => value !== normalizedId)
        : [...prev, normalizedId];
      localStorage.setItem("pinnedChats", JSON.stringify(next));
      return next;
    });
  }, []);

  const handleDelete = useCallback(
    async (event, id) => {
      event.stopPropagation();

      const normalizedId = normalizeChatId(id);
      if (normalizedId == null) return;

      try {
        await api.delete(`/api/chat/sessions/${normalizedId}`);
        setChatHistory((prev) => prev.filter((chat) => chat.id !== normalizedId));

        if (normalizeChatId(activeChatId) === normalizedId) {
          handleNewChat();
        }
      } catch (error) {
        if (error?.response?.status === 404) {
          setChatHistory((prev) => prev.filter((chat) => chat.id !== normalizedId));
          if (normalizeChatId(activeChatId) === normalizedId) {
            handleNewChat();
          }
          return;
        }

        console.error("Delete failed", error);
        setApiError(getApiErrorMessage(error, "Failed to delete chat"));
      }
    },
    [activeChatId, handleNewChat],
  );

  const copyMessage = useCallback(async (text, id) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (error) {
      console.error("Copy failed:", error);
    }
  }, []);

  const openSourceDocument = useCallback(async (sourceItem) => {
    const source = String(sourceItem?.source || "").trim();
    if (!source) return;

    const page = normalizePageNumber(sourceItem?.page);
    const isPdf = source.toLowerCase().endsWith(".pdf");
    const endpoint = `/api/chat/sources/${encodeURIComponent(source)}`;

    try {
      const response = await api.get(endpoint, { responseType: "blob" });
      const blob = response?.data;
      if (!(blob instanceof Blob)) {
        throw new Error("Invalid source response");
      }

      const objectUrl = URL.createObjectURL(blob);
      const targetUrl = isPdf && page > 0 ? `${objectUrl}#page=${page}` : objectUrl;
      const opened = window.open(targetUrl, "_blank", "noopener,noreferrer");
      if (!opened) {
        window.location.href = targetUrl;
      }

      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 10 * 60 * 1000);
    } catch (error) {
      console.error("Open source failed", error);
      setApiError(getApiErrorMessage(error, "Failed to open source document"));
    }
  }, []);

  const handlePromptSelect = useCallback((prompt) => {
    setInput(prompt);
    inputRef.current?.focus();
  }, []);

  const handleReuseMessage = useCallback((messageText) => {
    setInput(messageText || "");
    inputRef.current?.focus();
  }, []);

  const composerProps = {
    input,
    inputRef,
    onInputChange: setInput,
    onKeyDown: handleComposerKeyDown,
    onSubmit: handleSend,
    isLoading,
    isRecording,
    isTranscribing,
    startRecording,
    stopRecording,
    cancelTranscription,
  };

  return (
    <div className="liquid-shell chat-shell-height flex font-sans relative overflow-hidden">
      <div className="liquid-orb liquid-orb-a" />
      <div className="liquid-orb liquid-orb-b" />
      <div className="liquid-orb liquid-orb-c" />

      <ChatSidebar
        isCompactLayout={isCompactLayout}
        sidebarCollapsed={sidebarCollapsed}
        setSidebarCollapsed={setSidebarCollapsed}
        sortedChats={sortedChats}
        activeChatId={activeChatId}
        onSelectChat={handleSelectChat}
        pinnedChats={pinnedChats}
        onTogglePin={togglePin}
        onDeleteChat={handleDelete}
        searchQuery={searchQuery}
        onSearchQueryChange={setSearchQuery}
        user={user}
        onLogout={onLogout}
        onNewChat={handleNewChat}
      />

      <div className="flex-1 flex flex-col h-full relative z-10 chat-main-surface">
        <header className="h-14 glass border-b border-slate-200/40 flex items-center justify-between px-4 sm:px-6 shrink-0 z-10">
          <div className="flex items-center gap-3 min-w-0">
            {isCompactLayout && (
              <button
                type="button"
                onClick={() => setSidebarCollapsed(false)}
                className="p-1.5 hover:bg-slate-100 rounded-lg text-slate-500"
              >
                <Menu size={20} />
              </button>
            )}
            <div className="min-w-0">
              <span className="font-semibold text-slate-800 text-sm truncate block max-w-[320px] sm:max-w-[420px]">
                {activeChat
                  ? activeChat.title?.length > 45
                    ? `${activeChat.title.slice(0, 45)}...`
                    : activeChat.title || "New Chat"
                  : "New Chat"}
              </span>
              <span className="text-[11px] text-slate-600">Panya Assistant</span>
            </div>
          </div>

          <div className="hidden sm:flex items-center gap-2 text-[11px] font-semibold text-slate-600">
            <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_0_3px_rgba(16,185,129,0.15)]" />
            Ready
          </div>
        </header>

        {apiError && (
          <div className="absolute left-0 right-0 top-14 z-30 px-3 sm:px-5 pointer-events-none">
            <div className="max-w-3xl mx-auto rounded-xl border border-red-200 bg-red-50/95 px-3 py-2.5 shadow-lg pointer-events-auto">
              <div className="flex items-start gap-2.5">
                <AlertTriangle size={16} className="text-red-600 mt-0.5 shrink-0" />
                <p className="text-sm text-red-700 flex-1">{apiError}</p>
                <button
                  type="button"
                  onClick={handleRetryConnection}
                  disabled={isRecovering}
                  className="inline-flex items-center gap-1 rounded-md border border-red-200 bg-white px-2 py-1 text-xs font-semibold text-red-700 hover:bg-red-100 disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  <RefreshCw size={12} className={isRecovering ? "animate-spin" : ""} />
                  Retry
                </button>
              </div>
            </div>
          </div>
        )}

        {!hasMessages ? (
          <ChatWelcome
            user={user}
            onPromptSelect={handlePromptSelect}
            composer={<ChatComposer centered {...composerProps} />}
          />
        ) : (
          <>
            <ChatMessages
              messagesContainerRef={messagesContainerRef}
              activeMessages={activeMessages}
              pendingMessage={pendingMessage}
              activeChat={activeChat}
              isLoading={isLoading}
              copiedId={copiedId}
              onCopyMessage={copyMessage}
              onReuseMessage={handleReuseMessage}
              onOpenSourceDocument={openSourceDocument}
              apiError={apiError}
            />

            <div className="p-3 glass border-t border-slate-200/40 shrink-0 chat-composer-safe chat-dock">
              <ChatComposer centered={false} {...composerProps} />
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
