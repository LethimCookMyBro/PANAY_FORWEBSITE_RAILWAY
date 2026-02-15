import ReactMarkdown from "react-markdown";
import { Check, Copy, CornerDownLeft, FileText } from "lucide-react";
import { fixMarkdownTable, markdownComponents } from "./markdown";
import ThinkingIndicator from "./ThinkingIndicator";
import { formatSourceItemLabel, formatTime, toArray } from "./utils";

export default function ChatMessages({
  messagesContainerRef,
  activeMessages,
  pendingMessage,
  activeChat,
  isLoading,
  copiedId,
  onCopyMessage,
  onReuseMessage,
  onOpenSourceDocument,
  apiError,
}) {
  return (
    <div
      ref={messagesContainerRef}
      className={`flex-1 overflow-y-auto p-4 sm:p-6 scroll-smooth ${apiError ? "pt-16 sm:pt-14" : ""}`}
    >
      <div className="max-w-4xl mx-auto min-h-full flex flex-col justify-end gap-5 pb-3 sm:pb-4">
          {activeMessages.map((message, index) => {
            const processingTime = Number(message?.processingTime);
            const hasProcessingTime = Number.isFinite(processingTime) && processingTime > 0;

            return (
              <div
                key={message?.id ?? index}
                className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"} fade-in-up`}
              >
                <div
                  className={`flex flex-col ${message.sender === "user" ? "items-end" : "items-start max-w-full"}`}
                >
                  <div
                    className={`max-w-[95%] sm:max-w-[86%] px-4 py-3 rounded-2xl text-[14px] leading-relaxed break-words overflow-hidden ${
                      message.sender === "user"
                        ? message.status === "failed"
                          ? "bg-red-500 text-white rounded-br-md shadow-md shadow-red-500/20 border border-red-400/60"
                          : "bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-br-md shadow-md shadow-blue-500/20 border border-blue-300/20"
                        : "bg-white text-slate-800 border border-slate-200 rounded-bl-md shadow-sm"
                    }`}
                    style={{ overflowWrap: "anywhere" }}
                  >
                    {message.sender === "bot" ? (
                      <ReactMarkdown components={markdownComponents}>
                        {fixMarkdownTable(message.text)}
                      </ReactMarkdown>
                    ) : (
                      message.text
                    )}
                  </div>

                  {message.sender === "bot" && hasProcessingTime && (
                    <div className="flex items-center gap-2 mt-1.5 text-[10px] text-slate-400">
                      <span>⏱ {processingTime.toFixed(2)}s</span>
                      {message.ragas?.scores?.faithfulness != null && (
                        <span>
                          • Faithfulness: {(message.ragas.scores.faithfulness * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                  )}

                  {message.sender === "bot" && toArray(message.sources).length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {toArray(message.sources).map((sourceItem, sourceIndex) => {
                        const label = formatSourceItemLabel(sourceItem);
                        if (!label) return null;

                        return (
                          <button
                            type="button"
                            key={`${message.id || index}-src-${sourceIndex}`}
                            onClick={() => onOpenSourceDocument(sourceItem)}
                            className="inline-flex items-center gap-1.5 rounded-full bg-blue-50 text-blue-700 border border-blue-100 px-2.5 py-1 text-[11px] font-medium hover:bg-blue-100 hover:border-blue-200 transition-colors cursor-pointer"
                            title="Open source document"
                          >
                            <FileText size={12} />
                            {label}
                          </button>
                        );
                      })}
                    </div>
                  )}

                  <div
                    className={`flex items-center gap-1 mt-1 ${message.sender === "user" ? "flex-row-reverse" : ""}`}
                  >
                    {message.timestamp && (
                      <span className="text-[10px] text-slate-400 px-1">{formatTime(message.timestamp)}</span>
                    )}
                    <button
                      type="button"
                      onClick={() => onCopyMessage(message.text, index)}
                      className="p-1 rounded hover:bg-slate-100 text-slate-400 hover:text-slate-600 transition-all"
                    >
                      {copiedId === index ? (
                        <Check size={12} className="text-green-500" />
                      ) : (
                        <Copy size={12} />
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() => onReuseMessage(message.text)}
                      className="p-1 rounded hover:bg-blue-50 text-slate-400 hover:text-blue-500 transition-all"
                    >
                      <CornerDownLeft size={12} />
                    </button>
                  </div>

                  {message.sender === "user" && message.status === "failed" && (
                    <span className="mt-1 text-[10px] text-red-500">
                      Message failed to send. Edit and resend.
                    </span>
                  )}
                </div>
              </div>
            );
          })}

          {pendingMessage && !activeChat && (
            <div className="flex justify-end fade-in-up">
              <div
                className={`max-w-[92%] sm:max-w-[85%] px-4 py-3 rounded-2xl rounded-br-md text-white text-[14px] shadow-md break-words ${
                  pendingMessage.status === "failed"
                    ? "bg-red-500 border border-red-400/60 shadow-red-500/20"
                    : "bg-gradient-to-r from-blue-500 to-blue-600 shadow-blue-500/15"
                }`}
                style={{ overflowWrap: "anywhere" }}
              >
                {pendingMessage.text}
              </div>
            </div>
          )}

          {pendingMessage?.status === "failed" && !activeChat && (
            <div className="flex justify-end -mt-3">
              <span className="text-[10px] text-red-500">Message failed to send. Edit and resend.</span>
            </div>
          )}

          {isLoading && (
            <ThinkingIndicator />
          )}
      </div>
    </div>
  );
}
