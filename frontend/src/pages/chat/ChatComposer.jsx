import { LoaderCircle, Mic, MicOff, Send } from "lucide-react";

export default function ChatComposer({
  centered = false,
  input,
  inputRef,
  onInputChange,
  onKeyDown,
  onSubmit,
  isLoading,
  isRecording,
  isTranscribing,
  startRecording,
  stopRecording,
  cancelTranscription,
}) {
  return (
    <form
      onSubmit={onSubmit}
      className={`chat-composer-form w-full ${centered ? "max-w-3xl" : "max-w-4xl"} mx-auto flex items-end gap-2`}
    >
      <div className="liquid-input-wrap glass-input chat-composer-shell flex-1 rounded-[26px] px-3 py-2.5 transition-all shadow-lg shadow-black/5">
        <textarea
          ref={inputRef}
          rows={1}
          value={input}
          onChange={(event) => onInputChange(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            isRecording
              ? "Listening..."
              : isTranscribing
                ? "Transcribing..."
                : "Ask about PLC, automation, troubleshooting..."
          }
          className="composer-textarea w-full bg-transparent focus:outline-none text-slate-900 placeholder-slate-500 px-2 py-1.5 resize-none text-[15px] leading-6"
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
            className={`p-2.5 rounded-full transition-all flex-shrink-0 ${
              isRecording
                ? "bg-red-500 text-white animate-pulse"
                : isTranscribing
                  ? "bg-orange-100 text-orange-500"
                  : "hover:bg-white/90 text-slate-400 hover:text-slate-700 border border-transparent hover:border-slate-200/80"
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
            className="bg-gradient-to-r from-blue-500 to-cyan-500 text-white p-2.5 rounded-full hover:from-blue-600 hover:to-cyan-600 disabled:opacity-30 disabled:cursor-not-allowed transition-all shadow-md shadow-blue-500/25 flex-shrink-0 border border-blue-300/30"
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
}
