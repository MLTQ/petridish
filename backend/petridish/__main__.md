# __main__.py

## Purpose

Provides a zero-configuration module entry point for the production-style local
server.

## Components

### `main`
- **Does**: Runs the FastAPI application on localhost port 8000 through Uvicorn.
- **Interacts with**: `server.app`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| CLI users | `python -m petridish` starts the server | Module path or port |
