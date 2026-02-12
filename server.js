const express = require("express");
const path = require("path");
const { createProxyMiddleware } = require("http-proxy-middleware");

const app = express();
const port = Number(process.env.PORT || 3000);

const apiTarget =
  process.env.API_PROXY_TARGET ||
  process.env.BACKEND_URL ||
  process.env.BACKEND_API_URL ||
  process.env.VITE_API_URL ||
  "http://localhost:5000";

const distDir = path.join(__dirname, "frontend", "dist");
const indexFile = path.join(distDir, "index.html");

app.use(
  "/api",
  createProxyMiddleware({
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

app.get("*", (_req, res) => {
  res.sendFile(indexFile);
});

app.listen(port, "0.0.0.0", () => {
  console.log(`[web] listening on 0.0.0.0:${port}`);
  console.log(`[web] serving: ${distDir}`);
  console.log(`[web] proxy /api -> ${apiTarget}`);
});
