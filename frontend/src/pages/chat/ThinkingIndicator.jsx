export default function ThinkingIndicator() {
  return (
    <div className="fade-in-up flex justify-start">
      <div className="bg-white/95 backdrop-blur px-4 py-3 rounded-2xl border border-slate-200 shadow-sm min-w-[180px]">
        <div className="flex items-center justify-between gap-3 mb-2">
          <span className="text-xs font-semibold tracking-wide text-slate-500 uppercase">
            Thinking
          </span>
          <span className="text-[11px] text-slate-400">Generating</span>
        </div>

        <div className="flex items-center gap-1.5">
          {[0, 1, 2, 3].map((idx) => (
            <span
              key={`thinking-dot-${idx}`}
              className="h-1.5 w-1.5 rounded-full bg-slate-400 animate-bounce"
              style={{ animationDelay: `${idx * 0.08}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
