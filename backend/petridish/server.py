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

from .laboratory import (
    ContinueSpec, EvaluateSpec, ForkSpec, Laboratory, LaunchSpec, ResumeSpec,
    RetrySpec,
)
from .sequence_experiment import MAX_STATE_LANES
from .runtime import ExperimentRuntime


repository_root = Path(__file__).resolve().parents[2]
runtime = ExperimentRuntime(device=os.getenv("PETRIDISH_DEVICE", "auto"))
if os.getenv("PETRIDISH_AUTOPLAY", "1") == "0":
    runtime.running = False
configured_run_root = os.getenv("PETRIDISH_RUN_ROOT")
configured_benchmark_root = os.getenv("PETRIDISH_BENCHMARK_ROOT")
laboratory = Laboratory(
    repository_root,
    run_root=Path(configured_run_root) if configured_run_root else None,
    benchmark_root=Path(configured_benchmark_root) if configured_benchmark_root else None,
    control_enabled=os.getenv("PETRIDISH_LAB_CONTROL") == "1",
)


class LabLaunchRequest(BaseModel):
    """Browser-safe schema for one unattended corpus experiment."""

    runId: str
    gpuUuid: str
    task: str = "tiny_shakespeare"
    architecture: str = "gru"
    fieldSize: int = 68
    batchSize: int = Field(default=16, ge=1, le=256)
    contextLength: int = Field(default=64, ge=8, le=256)
    vocabularySize: int = Field(default=2_048, ge=64, le=2_048)
    tokenizerProfile: str = "wordpiece"
    streamMode: str = "continuous"
    stateRetention: float = Field(default=0.9, ge=0, le=1)
    stateLanes: int = Field(default=1, ge=1, le=MAX_STATE_LANES)
    randomOffsetAuxiliaryWeight: float = Field(default=0, ge=0, le=10)
    randomOffsetAuxiliaryScope: str = Field(
        default="active_shard", pattern="^(active_shard|full_corpus)$"
    )
    messageSteps: int = Field(default=2, ge=1, le=16)
    broadcastGain: float = Field(default=0.3, ge=0, le=2)
    updates: int = Field(default=100_000, ge=1)
    seed: int = 1
    learningRateScale: float = Field(default=1.0, ge=0.01, le=1.0)
    amp: str = "bfloat16"
    lifecycle: bool = False
    lifecycleProfile: str = "off"
    structure: bool = True
    topologyProfile: str | None = None


class LabContinueRequest(BaseModel):
    """Browser-safe plasticity transition for one checkpointed organism."""

    gpuUuid: str
    additionalUpdates: int = Field(default=1_000, ge=1)
    lifecycle: bool = False
    lifecycleProfile: str = "off"
    structure: bool = True
    topologyProfile: str | None = None
    phaseName: str | None = Field(default=None, max_length=120)
    trainingShardTokens: int | None = Field(default=None, ge=0)
    stateLanes: int | None = Field(default=None, ge=1, le=MAX_STATE_LANES)
    gradientClip: float | None = Field(default=None, ge=0.01, le=100)
    randomOffsetAuxiliaryWeight: float | None = Field(default=None, ge=0, le=10)
    randomOffsetAuxiliaryScope: str | None = Field(
        default=None, pattern="^(active_shard|full_corpus)$"
    )


class LabForkRequest(BaseModel):
    """Browser-safe name for an exact stopped-checkpoint branch."""

    forkRunId: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")


class LabEvaluateRequest(BaseModel):
    """Browser-safe read-only checkpoint evaluation request."""

    gpuUuid: str
    stateHorizons: bool = False
    evaluationSplit: str = Field(
        default="validation",
        pattern=(
            "^(validation|training|trajectory|random_context|"
            "full_corpus_context)$"
        ),
    )
    trajectoryLane: int | None = Field(
        default=None, ge=0, le=MAX_STATE_LANES - 1
    )


class LabRetryRequest(BaseModel):
    """Browser-safe GPU selection for an exact failed-phase retry."""

    gpuUuid: str


class LabResumeRequest(BaseModel):
    """Browser-safe GPU selection for an unchanged stopped phase."""

    gpuUuid: str


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
        task=request.task,
        architecture=request.architecture,
        field_size=request.fieldSize,
        batch_size=request.batchSize,
        context_length=request.contextLength,
        vocabulary_size=request.vocabularySize,
        tokenizer_profile=request.tokenizerProfile,
        stream_mode=request.streamMode,
        state_retention=request.stateRetention,
        state_lanes=request.stateLanes,
        random_offset_auxiliary_weight=request.randomOffsetAuxiliaryWeight,
        random_offset_auxiliary_scope=request.randomOffsetAuxiliaryScope,
        message_steps=request.messageSteps,
        broadcast_gain=request.broadcastGain,
        updates=request.updates,
        seed=request.seed,
        learning_rate_scale=request.learningRateScale,
        amp=request.amp,
        lifecycle=request.lifecycle,
        lifecycle_profile=request.lifecycleProfile,
        structure=request.structure,
        topology_profile=request.topologyProfile,
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


@app.post("/api/lab/runs/{run_id}/continue")
async def continue_lab_run(
    run_id: str, request: LabContinueRequest
) -> dict[str, object]:
    """Advance one checkpoint lineage into a new plasticity phase."""

    spec = ContinueSpec(
        run_id=run_id,
        gpu_uuid=request.gpuUuid,
        additional_updates=request.additionalUpdates,
        lifecycle=request.lifecycle,
        lifecycle_profile=request.lifecycleProfile,
        structure=request.structure,
        topology_profile=request.topologyProfile,
        phase_name=request.phaseName,
        training_shard_tokens=request.trainingShardTokens,
        state_lanes=request.stateLanes,
        gradient_clip=request.gradientClip,
        random_offset_auxiliary_weight=request.randomOffsetAuxiliaryWeight,
        random_offset_auxiliary_scope=request.randomOffsetAuxiliaryScope,
    )
    try:
        return await asyncio.to_thread(laboratory.continue_run, spec)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/lab/runs/{run_id}/evaluate")
async def evaluate_lab_run(
    run_id: str, request: LabEvaluateRequest
) -> dict[str, object]:
    """Measure a checkpoint without an optimizer or plasticity step."""

    try:
        return await asyncio.to_thread(
            laboratory.evaluate_run,
            EvaluateSpec(
                run_id=run_id,
                gpu_uuid=request.gpuUuid,
                state_horizons=request.stateHorizons,
                evaluation_split=request.evaluationSplit,
                trajectory_lane=request.trajectoryLane,
            ),
        )
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/lab/runs/{run_id}/retry")
async def retry_lab_run(
    run_id: str, request: LabRetryRequest
) -> dict[str, object]:
    """Restart a failed phase from that run's unchanged atomic checkpoint."""

    try:
        return await asyncio.to_thread(
            laboratory.retry_run,
            RetrySpec(run_id=run_id, gpu_uuid=request.gpuUuid),
        )
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/lab/runs/{run_id}/resume")
async def resume_lab_run(
    run_id: str, request: LabResumeRequest
) -> dict[str, object]:
    """Restart a deliberately stopped checkpoint in its existing phase."""

    try:
        return await asyncio.to_thread(
            laboratory.resume_run,
            ResumeSpec(run_id=run_id, gpu_uuid=request.gpuUuid),
        )
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/lab/runs/{run_id}/fork", status_code=201)
async def fork_lab_run(run_id: str, request: LabForkRequest) -> dict[str, object]:
    """Branch an exact stopped checkpoint without constructing a new organism."""

    try:
        return await asyncio.to_thread(
            laboratory.fork_run,
            ForkSpec(source_run_id=run_id, fork_run_id=request.forkRunId),
        )
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
