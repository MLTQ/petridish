# package.json

## Purpose

Defines the lightweight TypeScript/PixiJS viewer and its development, checking,
and production build commands.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Contributors | `npm run dev`, `check`, and `build` are available | Script names |
| Renderer | PixiJS 8 Graphics and Application APIs | Pixi major version |
| Build | Vite emits static output to `dist/` | Build tool or output directory |
