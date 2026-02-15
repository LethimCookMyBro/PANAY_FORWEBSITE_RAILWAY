import {
  LogOut,
  MessageSquareText,
  PanelLeft,
  PanelLeftClose,
  Pin,
  Plus,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { formatTimeAgo } from "./utils";

export default function ChatSidebar({
  isCompactLayout,
  sidebarCollapsed,
  setSidebarCollapsed,
  sortedChats,
  activeChatId,
  onSelectChat,
  pinnedChats,
  onTogglePin,
  onDeleteChat,
  searchQuery,
  onSearchQueryChange,
  user,
  onLogout,
  onNewChat,
}) {
  return (
    <>
      {isCompactLayout && !sidebarCollapsed && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
          onClick={() => setSidebarCollapsed(true)}
        />
      )}

      <aside
        className={`
          ${
            isCompactLayout
              ? `fixed top-0 left-0 h-full z-50 transform transition-transform duration-200 ease-out ${sidebarCollapsed ? "-translate-x-full" : "translate-x-0"} w-[85vw] max-w-[20rem] sm:w-72`
              : `${sidebarCollapsed ? "w-16" : "w-72"} transition-[width] duration-200 ease-out`
          }
          p-4 glass-dark chat-sidebar-surface border-r border-slate-800/50 flex flex-col overflow-hidden shadow-xl shadow-slate-950/30
        `}
      >
        <div
          className={`flex-shrink-0 mb-6 flex ${!isCompactLayout && sidebarCollapsed ? "flex-col items-center gap-2" : "items-center justify-between"}`}
        >
          {(isCompactLayout || !sidebarCollapsed) && (
            <div className="flex items-center gap-3 px-1">
              <img src="/panya-logo.png" alt="Panya logo" className="w-8 h-8 object-contain" />
              <div>
                <h1 className="text-lg font-bold text-white tracking-tight">Panya</h1>
                <p className="text-[11px] font-semibold text-cyan-300/85 uppercase tracking-wider">
                  PLC Assistant
                </p>
              </div>
            </div>
          )}

          {!isCompactLayout && sidebarCollapsed && (
            <img src="/panya-logo.png" alt="Panya logo" className="w-8 h-8 object-contain" />
          )}

          <button
            type="button"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            className="p-2 hover:bg-white/10 rounded-lg transition-all text-slate-300 hover:text-white"
          >
            {isCompactLayout ? (
              <X size={20} />
            ) : sidebarCollapsed ? (
              <PanelLeft size={20} />
            ) : (
              <PanelLeftClose size={20} />
            )}
          </button>
        </div>

        <button
          type="button"
          onClick={onNewChat}
          className={`flex items-center justify-center gap-2 w-full p-3 mb-4 bg-gradient-to-r from-blue-500 to-cyan-500 text-white hover:from-blue-600 hover:to-cyan-600 rounded-xl transition-all text-sm font-semibold shadow-lg shadow-blue-500/20 border border-cyan-300/30 ${!isCompactLayout && sidebarCollapsed ? "px-0" : ""}`}
        >
          <Plus size={18} />
          {(isCompactLayout || !sidebarCollapsed) && "New Chat"}
        </button>

        {(isCompactLayout || !sidebarCollapsed) && (
          <div className="relative mb-3">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(event) => onSearchQueryChange(event.target.value)}
              placeholder="Search chats..."
              className="w-full pl-9 pr-8 py-2.5 text-sm bg-slate-950/45 border border-slate-600/55 rounded-lg focus:outline-none focus:ring-1 focus:ring-cyan-400/55 text-slate-100 placeholder-slate-400"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={() => onSearchQueryChange("")}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 hover:bg-white/10 rounded text-slate-400"
              >
                <X size={14} />
              </button>
            )}
          </div>
        )}

        {(isCompactLayout || !sidebarCollapsed) && (
          <div className="px-2 mb-2 text-[11px] font-semibold text-slate-300/85 uppercase tracking-widest">
            {searchQuery ? `Results (${sortedChats.length})` : "Recent"}
          </div>
        )}

        {(isCompactLayout || !sidebarCollapsed) && (
          <div className="flex-1 overflow-y-auto space-y-1 pr-1 sidebar-scroll">
            {sortedChats.map((chat) => (
              <div
                key={chat.id}
                onClick={() => onSelectChat(chat.id)}
                className={`chat-session-item group relative w-full text-left px-3 py-2.5 rounded-lg flex items-center gap-3 cursor-pointer transition-all ${
                  chat.id === activeChatId
                    ? "chat-session-item-active text-white"
                    : "chat-session-item-idle text-slate-300 hover:text-white"
                }`}
              >
                <div className="relative flex-shrink-0">
                  <MessageSquareText
                    size={16}
                    className={chat.id === activeChatId ? "text-cyan-400" : "text-slate-400"}
                  />
                  {pinnedChats.includes(chat.id) && (
                    <Pin size={8} className="absolute -top-1 -right-1 text-amber-400 fill-amber-400" />
                  )}
                </div>

                {!sidebarCollapsed && (
                  <div className="flex-1 min-w-0">
                    <span className="truncate text-sm font-medium block">
                      {chat.title || "New Chat"}
                    </span>
                    {chat.updated_at && (
                      <span className="text-[11px] text-slate-400">{formatTimeAgo(chat.updated_at)}</span>
                    )}
                  </div>
                )}

                {!sidebarCollapsed && (
                  <div className="absolute right-1 flex items-center gap-0.5 opacity-100 md:opacity-0 md:group-hover:opacity-100 transition-opacity">
                    <button
                      type="button"
                      onClick={(event) => onTogglePin(event, chat.id)}
                      className={`p-1 rounded ${pinnedChats.includes(chat.id) ? "text-amber-400" : "text-slate-400 hover:text-amber-300"}`}
                    >
                      <Pin size={13} className={pinnedChats.includes(chat.id) ? "fill-amber-400" : ""} />
                    </button>
                    <button
                      type="button"
                      onClick={(event) => onDeleteChat(event, chat.id)}
                      className="p-1 rounded text-slate-400 hover:text-red-300"
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        <div
          className={`mt-4 pt-4 border-t border-white/5 flex items-center gap-3 ${!isCompactLayout && sidebarCollapsed ? "justify-center flex-col" : ""}`}
        >
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500/30 to-cyan-400/30 flex items-center justify-center text-cyan-300 text-xs font-bold flex-shrink-0">
            {user.full_name ? user.full_name.charAt(0) : "U"}
          </div>

          {(isCompactLayout || !sidebarCollapsed) && (
            <div className="flex-1 min-w-0">
              <p className="text-sm font-semibold text-slate-100 truncate">
                {user.full_name || user.name}
              </p>
            </div>
          )}

          <button
            type="button"
            onClick={onLogout}
            className={`flex items-center gap-1.5 text-slate-300 hover:text-red-300 px-2 py-1.5 rounded-lg transition-all text-sm ${!isCompactLayout && sidebarCollapsed ? "mt-2" : ""}`}
          >
            <LogOut size={16} />
            {(isCompactLayout || !sidebarCollapsed) && <span>Logout</span>}
          </button>
        </div>
      </aside>
    </>
  );
}
