# tsconfig.json

## Purpose

Enforces browser-focused strict TypeScript checking without emitting duplicate
build artifacts outside Vite.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Source modules | DOM and ES2022 APIs are typed | Library or target changes |
| CI/build | `tsc --noEmit` fails on unsafe indexed access | Strictness flags |
