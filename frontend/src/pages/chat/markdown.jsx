import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism";
import { toArray } from "./utils";

export const fixMarkdownTable = (text) => {
  if (!text?.includes("|")) return text;
  return text
    .split("\n")
    .map((line) => {
      const pipes = (line.match(/\|/g) || []).length;
      if (pipes < 4) return line;

      const parts = line
        .split("|")
        .map((part) => part.trim())
        .filter((part) => part && !/^-+$/.test(part));

      if (parts.length < 2) return line;
      if (parts.every((part) => part.length < 30 && /^[A-Z][a-zA-Z\s()]*$/.test(part))) {
        return null;
      }

      const bullets = parts
        .filter((part) => part.length > 5)
        .map((part) => {
          const colonIndex = part.indexOf(":");
          return colonIndex > 0 && colonIndex < 50
            ? `• **${part.slice(0, colonIndex).trim()}**: ${part.slice(colonIndex + 1).trim()}`
            : `• ${part}`;
        });

      return bullets.length ? bullets.join("\n") : line;
    })
    .filter(Boolean)
    .join("\n");
};

export const markdownComponents = {
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
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
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
  tbody: ({ children }) => <ul className="list-disc ml-4 space-y-2">{children}</ul>,
  tr: ({ children }) => {
    const cells = [];
    toArray(children).forEach((cell) => {
      if (cell?.props?.children) cells.push(cell.props.children);
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
