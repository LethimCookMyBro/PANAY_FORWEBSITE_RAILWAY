export default function ThinkingIndicator() {
  return (
    <div className="fade-in-up flex justify-start">
      <div className="thinking-shell inline-flex items-center rounded-2xl border px-4 py-3 backdrop-blur">
        <div className="flex items-center gap-2">
          {[0, 1, 2, 3].map((idx) => (
            <span
              key={`thinking-dot-${idx}`}
              className="thinking-dot-modern h-2 w-2 rounded-full bg-sky-500/80"
              style={{ animationDelay: `${idx * 0.12}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
