export default function ThinkingIndicator() {
  return (
    <div className="fade-in-up flex justify-start">
      <div className="thinking-shell inline-flex items-center gap-2.5 rounded-2xl border border-white/90 bg-white/95 px-2.5 py-2 shadow-sm backdrop-blur">
        <div className="thinking-pill relative overflow-hidden rounded-xl bg-gradient-to-r from-sky-500 to-blue-600 px-3 py-1.5 text-white shadow-sm">
          <span className="text-[11px] font-semibold tracking-wide uppercase">
            Thinking
          </span>
          <span className="thinking-pill-shimmer" />
        </div>

        <div className="flex items-center gap-1">
          {[0, 1, 2, 3].map((idx) => (
            <span
              key={`thinking-dot-${idx}`}
              className="thinking-dot-modern h-1.5 w-1.5 rounded-full bg-sky-500/80"
              style={{ animationDelay: `${idx * 0.12}s` }}
            />
          ))}
        </div>

        <span className="pr-1 text-[11px] font-medium text-slate-500">Generating</span>
      </div>
    </div>
  );
}
