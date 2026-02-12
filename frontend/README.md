# Frontend Deploy Notes

If you want a single Railway service for both frontend and backend, use the root `Dockerfile` in repo root (`Panya/`) instead of this `frontend/` service setup.

## Railway Frontend Service (Production)

Use these settings in the `frontend` service:

- Root Directory: `frontend`
- Build Command: `npm run build`
- Start Command: `npm start`

`npm start` runs `server.cjs`, which serves `dist` and proxies `/api/*` at runtime.

Do not use these as production start commands:

- `npm run preview`
- `serve -s dist`

Those modes serve static files only and can break `/api` routing.

## Required Environment Variables

```env
API_PROXY_TARGET=https://<your-backend-service>.up.railway.app
```

Optional:

```env
PORT=5173
```

Important: unset `VITE_API_URL` on the frontend production service. The runtime proxy should handle `/api/*`.

## Proxy Behavior

`server.cjs` uses `http-proxy-middleware` with `pathFilter: "/api"` and forwards the full path unchanged:

- `/api/auth/me` -> `${API_PROXY_TARGET}/api/auth/me`
- `/api/chat` -> `${API_PROXY_TARGET}/api/chat`

This is required because backend routes are prefixed with `/api`.

## Quick Verification

```bash
curl -i https://<frontend-service>.up.railway.app/api/auth/me
curl -i -X POST https://<frontend-service>.up.railway.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{}'
```

Expected: JSON responses from backend (`401`/`422` is fine), not HTML and not frontend `404`.

If you see `API returned HTML instead of JSON`, frontend is still not proxying to backend correctly.

If you see `431 Request Header Fields Too Large` on `/api/*`, your proxy target is likely pointing back to the frontend domain (self-loop). Set `API_PROXY_TARGET` to the backend Railway domain.

Invalid examples (do not use):

```env
DATABASE_URL=${DATABASE_URL}
API_PROXY_TARGET=http://127.0.0.1:$PORT
```
