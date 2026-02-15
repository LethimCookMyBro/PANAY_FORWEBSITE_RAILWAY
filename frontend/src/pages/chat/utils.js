const BULLET_LINE_RE = /^\s*(?:[-*•]|\d+[.)])\s+/;
const SECTION_HEADER_RE = /^\s*\[[A-Z][A-Z0-9 _/.-]{2,}\]\s*$/;
const KEY_VALUE_LINE_RE = /^([A-Za-z][^:]{1,60}):\s+(.+)$/;
const HAS_CODE_BLOCK_RE = /```/;

export const formatTimeAgo = (ts) => {
  if (!ts) return "";
  const diff = Math.floor((Date.now() - new Date(ts)) / 1000);
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 172800) return "yesterday";
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(ts).toLocaleDateString();
};

export const formatTime = (ts) =>
  ts
    ? new Date(ts).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";

export const toArray = (value) => (Array.isArray(value) ? value : []);

export const normalizeChatId = (value) => {
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

export const unwrapResponsePayload = (payload) => {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return {};
  }
  const nested = payload.data;
  if (nested && typeof nested === "object" && !Array.isArray(nested)) {
    return nested;
  }
  return payload;
};

export const stripTrailingSourcesBlock = (text) => {
  if (typeof text !== "string") return "";
  return text
    .replace(
      /\n{2,}(?:Sources|Source citations|References)\s*:\s*(?:\n-\s.*)+\s*$/i,
      "",
    )
    .trim();
};

export const normalizePageNumber = (value) => {
  if (value == null || value === "") return 0;
  const n = Number(value);
  if (Number.isFinite(n) && n > 0) return Math.trunc(n);
  const match = String(value).match(/\d+/);
  if (!match) return 0;
  const parsed = Number(match[0]);
  return Number.isFinite(parsed) && parsed > 0 ? Math.trunc(parsed) : 0;
};

export const normalizeSourceItems = (value) =>
  toArray(value)
    .map((item) => {
      if (!item || typeof item !== "object") return null;
      const source = String(item.source || item.source_key || "").trim();
      if (!source) return null;
      return {
        source,
        page: normalizePageNumber(item.page),
      };
    })
    .filter(Boolean);

export const formatSourceItemLabel = (item) => {
  if (!item || typeof item !== "object") return "";
  const source = String(item.source || "").trim();
  if (!source) return "";
  return item.page > 0 ? `${source} • p.${item.page}` : source;
};

const normalizePatternLinesToBullets = (text) => {
  const lines = String(text || "").split("\n");
  const result = [];
  let converted = 0;

  for (const raw of lines) {
    const line = raw.trim();
    if (!line) {
      if (result[result.length - 1] !== "") result.push("");
      continue;
    }

    if (SECTION_HEADER_RE.test(line)) {
      const label = line.replace(/^\[/, "").replace(/\]$/, "").trim();
      result.push(`- **${label}**`);
      converted += 1;
      continue;
    }

    if (/^\s*•\s+/.test(line)) {
      result.push(line.replace(/^\s*•\s+/, "- "));
      converted += 1;
      continue;
    }

    if (BULLET_LINE_RE.test(line)) {
      result.push(line);
      continue;
    }

    const keyValue = line.match(KEY_VALUE_LINE_RE);
    if (keyValue) {
      result.push(`- **${keyValue[1].trim()}**: ${keyValue[2].trim()}`);
      converted += 1;
      continue;
    }

    result.push(line);
  }

  return converted > 0 ? result.join("\n").trim() : text;
};

const convertParagraphToBullets = (paragraph) => {
  const normalized = paragraph.replace(/\s+/g, " ").trim();
  if (!normalized) return [];

  const sentenceParts = normalized
    .split(/(?<=[.!?])\s+/)
    .map((part) => part.trim())
    .filter(Boolean);

  if (sentenceParts.length >= 2 && sentenceParts.length <= 8) {
    return sentenceParts.map((part) => `- ${part}`);
  }

  const clauseParts = normalized
    .split(/\s*[;•]\s*/)
    .map((part) => part.trim())
    .filter(Boolean);

  if (clauseParts.length >= 2 && clauseParts.length <= 8) {
    return clauseParts.map((part) => `- ${part}`);
  }

  return [`- ${normalized}`];
};

const ensureBulletPointResponse = (text) => {
  const normalized = String(text || "").trim();
  if (!normalized) return "";
  if (HAS_CODE_BLOCK_RE.test(normalized)) return normalized;

  const lines = normalized.split("\n");
  const bulletCount = lines.filter((line) => BULLET_LINE_RE.test(line.trim())).length;
  if (bulletCount >= 2) return normalized;

  const paragraphs = normalized
    .split(/\n{2,}/)
    .map((part) => part.trim())
    .filter(Boolean);

  if (paragraphs.length === 0) return normalized;

  return paragraphs
    .flatMap((paragraph) => convertParagraphToBullets(paragraph))
    .join("\n")
    .trim();
};

const appendTroubleshootingHints = (text) => {
  const normalized = String(text || "").trim();
  if (!normalized) return normalized;

  const needsTroubleshootingHint =
    /(error|fault|alarm|problem|issue|cannot|unable|missing|not\s+found|insufficient|timeout|not\s+detected|disconnected)/i.test(
      normalized,
    );

  if (!needsTroubleshootingHint) return normalized;
  if (/troubleshooting/i.test(normalized)) return normalized;

  return `${normalized}\n\n**Troubleshooting**\n- Confirm PLC model/module and software version (for example, GX Works version).\n- Record the exact error code, LED status (RUN/ERR/LINK), and observed symptom.\n- Verify network/protocol settings and list what has already been tested.\n- Check common faults: station not detected, link scan timeout, or remote disconnect.`;
};

export const formatAssistantText = (rawText) => {
  const cleaned = stripTrailingSourcesBlock(rawText || "");
  if (!cleaned) {
    return "- I couldn't generate a response right now. Please try again.";
  }

  const normalizedPattern = normalizePatternLinesToBullets(cleaned);
  const bulletText = ensureBulletPointResponse(normalizedPattern);
  const withHints = appendTroubleshootingHints(bulletText);

  return withHints || "- I couldn't generate a response right now. Please try again.";
};

export const getReplyText = (payload) => {
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
  return formatAssistantText(text || "");
};

export const getResponseSessionId = (payload) => {
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

export const findFallbackSessionId = (payload, userText) => {
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

export const mapSessionsFromPayload = (payload) =>
  pickListPayload(payload, ["items", "sessions"])
    .map((s) => ({
      id: normalizeChatId(s?.id ?? s?.session_id ?? s?.sessionId),
      title: s?.title,
      messages: [],
      created_at: s?.created_at,
      updated_at: s?.updated_at || s?.created_at,
    }))
    .filter((s) => s.id != null);

export const mapMessagesFromPayload = (payload) =>
  pickListPayload(payload, ["items", "messages"]).map((m) => ({
    id:
      m?.id ??
      m?.message_id ??
      `${m?.created_at || Date.now()}-${m?.role || "msg"}-${Math.random().toString(36).slice(2, 8)}`,
    text:
      m?.role === "assistant"
        ? formatAssistantText(m?.content || "")
        : m?.content || "",
    sender: m?.role === "user" ? "user" : "bot",
    timestamp: m?.created_at,
    processingTime: m?.metadata?.processing_time,
    ragas: m?.metadata?.ragas,
    sources: normalizeSourceItems(m?.metadata?.sources),
    status: "sent",
  }));

export const makeLocalMessageId = () =>
  `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
