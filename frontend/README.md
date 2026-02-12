# Frontend Deploy Notes

## Production start command

Use:

```bash
npm run build
npm start
```

`npm start` runs `server.cjs`, which serves `dist` and proxies `/api/*` to backend.

## Required env for Railway frontend service

```env
API_PROXY_TARGET=https://<your-backend-service>.up.railway.app
```

If you see `API returned HTML instead of JSON`, your frontend is not proxying to backend correctly.
