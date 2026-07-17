# vite.config.ts

## Purpose

Configures the local viewer server and makes backend HTTP/WebSocket routes appear
same-origin during development.

## Components

### Default Vite config
- **Does**: Serves on localhost:5173 and proxies `/api` and `/ws` to port 8000.
- **Interacts with**: FastAPI routes in `server.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `socket.ts` | `/ws` upgrades to the backend | Proxy route or target |
| README | Viewer starts on port 5173 | Host or port |
