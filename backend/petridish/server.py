"""FastAPI entry point for the live Petri Dish experiment."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .laboratory import Laboratory, LaunchSpec
from .runtime import ExperimentRuntime


repository_root = Path(__file__).resolve().parents[2]
runtime = ExperimentRuntime(device=os.getenv("PETRIDISH_DEVICE", "auto"))
if os.getenv("PETRIDISH_AUTOPLAY", "1") == "0":
    runtime.running = False
configured_run_root = os.getenv("PETRIDISH_RUN_ROOT")
laboratory = Laboratory(
    repository_root,
    run_root=Path(configured_run_root) if configured_run_root else None,
    control_enabled=os.getenv("PETRIDISH_LAB_CONTROL") == "1",
)


class LabLaunchRequest(BaseModel):
    """Browser-safe schema for one unattended corpus experiment."""

    runId: str
    gpuUuid: str
    architecture: str = "gru"
    fieldSize: int = 68
    batchSize: int = Field(default=16, ge=1, le=256)
    contextLength: int = Field(default=64, ge=8, le=256)
    messageSteps: int = Field(default=2, ge=1, le=16)
    updates: int = Field(default=100_000, ge=1)
    seed: int = 1
    amp: str = "bfloat16"
    lifecycle: bool = False


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    await runtime.start()
    try:
        yield
    finally:
        await runtime.stop()
        laboratory.close()


app = FastAPI(title="Neural Petri Dish", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
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


@app.get("/api/lab")
async def lab_status() -> dict[str, object]:
    """Return measured GPU state and persisted experiment summaries."""

    return await asyncio.to_thread(laboratory.snapshot)


@app.get("/api/lab/runs/{run_id}/metrics")
async def lab_metrics(run_id: str, limit: int = 600) -> dict[str, object]:
    """Return one bounded metric history without loading its checkpoint."""

    try:
        records = await asyncio.to_thread(laboratory.metrics, run_id, limit=limit)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"runId": run_id, "records": records}


@app.post("/api/lab/runs", status_code=201)
async def launch_lab_run(request: LabLaunchRequest) -> dict[str, object]:
    """Launch an explicitly enabled, UUID-pinned trainer process."""

    spec = LaunchSpec(
        run_id=request.runId,
        gpu_uuid=request.gpuUuid,
        architecture=request.architecture,
        field_size=request.fieldSize,
        batch_size=request.batchSize,
        context_length=request.contextLength,
        message_steps=request.messageSteps,
        updates=request.updates,
        seed=request.seed,
        amp=request.amp,
        lifecycle=request.lifecycle,
    )
    try:
        return await asyncio.to_thread(laboratory.launch, spec)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/lab/runs/{run_id}/stop")
async def stop_lab_run(run_id: str) -> dict[str, object]:
    """Ask one trainer to checkpoint and stop after its current update."""

    try:
        return await asyncio.to_thread(laboratory.stop_run, run_id)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


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
