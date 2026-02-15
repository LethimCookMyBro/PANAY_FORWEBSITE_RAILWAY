import { Sparkles } from "lucide-react";

export default function ChatWelcome({ user, onPromptSelect, composer }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4 pb-8">
      <div className="mb-8 text-center fade-in-up">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-cyan-400 shadow-xl shadow-blue-500/20 mb-5">
          <Sparkles size={28} className="text-white" />
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold text-slate-800 mb-2">
          Hey{user.full_name ? `, ${user.full_name.split(" ")[0]}` : ""}!
        </h2>
        <p className="text-slate-600 max-w-xl text-sm sm:text-[15px] leading-relaxed">
          Your PLC & Industrial Automation expert. Ask me anything about troubleshooting,
          error codes, or technical docs.
        </p>
      </div>

      <div className="w-full max-w-2xl mb-6 fade-in-up" style={{ animationDelay: "0.1s" }}>
        {composer}
      </div>

      <div
        className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-3xl w-full fade-in-up"
        style={{ animationDelay: "0.2s" }}
      >
        {[
          "What does error code F800H mean?",
          "How to configure CC-Link IE Field?",
          "FX3 timer instructions",
          "Data Collector troubleshooting",
        ].map((prompt, index) => (
          <button
            type="button"
            key={index}
            onClick={() => onPromptSelect(prompt)}
            className="text-left px-4 py-3.5 glass-prompt rounded-2xl text-sm font-medium text-slate-700 hover:text-blue-800 hover:border-blue-200 transition-all shadow-sm"
          >
            {prompt}
          </button>
        ))}
      </div>

      <p className="text-[10px] text-slate-400 mt-6">
        Panya may make mistakes. Verify important information.
      </p>
    </div>
  );
}
