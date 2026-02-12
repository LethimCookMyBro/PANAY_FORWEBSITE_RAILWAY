const express = require("express");
const path = require("path");
const { createProxyMiddleware } = require("http-proxy-middleware");

const app = express();
const port = Number(process.env.PORT || 5173);

const apiTarget =
  process.env.API_PROXY_TARGET ||
  process.env.BACKEND_URL ||
  process.env.BACKEND_API_URL ||
  "http://localhost:5000";

const targetHost = (() => {
  try {
    return new URL(apiTarget).host.toLowerCase();
  } catch {
    return "";
  }
})();

const selfHost = (process.env.RAILWAY_PUBLIC_DOMAIN || "").toLowerCase();
if (targetHost && selfHost && targetHost === selfHost) {
  console.error(
    `[frontend] API proxy loop: API_PROXY_TARGET points to this frontend domain (${selfHost}). Set it to the backend service domain.`,
  );
  process.exit(1);
}

const distDir = path.join(__dirname, "dist");
const indexFile = path.join(distDir, "index.html");

app.use(
  createProxyMiddleware({
    pathFilter: "/api",
    target: apiTarget,
    changeOrigin: true,
    secure: false,
    xfwd: true,
    logLevel: "warn",
    onError(err, req, res) {
      if (res.headersSent) return;
      res.status(502).json({
        message: "API proxy target is unreachable",
        detail: String(err?.message || err),
        target: apiTarget,
        path: req.originalUrl,
      });
    },
  }),
);

app.use(express.static(distDir));
app.get("*", (_req, res) => res.sendFile(indexFile));

app.listen(port, "0.0.0.0", () => {
  console.log(`[frontend] listening on 0.0.0.0:${port}`);
  console.log(`[frontend] serving: ${distDir}`);
  console.log(`[frontend] proxy /api -> ${apiTarget}`);
});
