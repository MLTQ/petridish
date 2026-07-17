"""FastAPI entry point for the live Petri Dish experiment."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .runtime import ExperimentRuntime


runtime = ExperimentRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()


app = FastAPI(title="Neural Petri Dish", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, object]:
    experiment = runtime.simulation
    return {
        "status": "ok",
        "device": str(experiment.device),
        "experiment": runtime.experiment_name,
        "tick": experiment.state.tick if hasattr(experiment, "state") else experiment.tick,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await runtime.connect(websocket)
    try:
        while True:
            message = await websocket.receive_json()
            try:
                await runtime.handle_command(message)
            except (TypeError, ValueError) as error:
                await websocket.send_json({"type": "error", "message": str(error)})
    except WebSocketDisconnect:
        runtime.disconnect(websocket)


frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=frontend_dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def frontend(path: str) -> FileResponse:
        requested = frontend_dist / path
        return FileResponse(requested if requested.is_file() else frontend_dist / "index.html")
