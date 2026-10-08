"""Offline local API. Run with ``python -m server.app``."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from server.bundles import BundleStore, confined
from server.measurements import MeasurementRequest, measure
from server.preview import preview
from server.settings import Settings

logger = logging.getLogger(__name__)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str = Field(min_length=1, max_length=4096)
    profile: Literal["fast", "balanced", "max"] = "balanced"


def create_app(settings: Settings | None = None, run_service=None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = BundleStore(settings.runs_root, settings.max_json_bytes)
    jobs: dict[str, dict] = {}
    lock = threading.Lock()
    pool = ThreadPoolExecutor(max_workers=settings.workers, thread_name_prefix="drishti-run")

    @asynccontextmanager
    async def lifespan(app):
        yield
        pool.shutdown(wait=True)

    app = FastAPI(title="DRISHTI local survey API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Nonstandard JSON NaN/Infinity must be rejected without reflecting those
        # values in a JSONResponse (which would itself fail to serialize).
        return JSONResponse(status_code=422, content={"detail": [
            {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
            for error in exc.errors()
        ]})

    def service():
        nonlocal run_service
        if run_service is None:
            from server.runner import LocalRunService
            run_service = LocalRunService(settings)
        return run_service

    @app.get("/api/health")
    def health():
        return {"status": "ok", "application": "DRISHTI", "mode": "offline_ground",
                "runs_root": str(store.root)}

    @app.get("/api/doctor")
    def doctor():
        return service().doctor()

    @app.get("/api/runs")
    def runs():
        result = []
        if store.root.is_dir():
            for candidate in sorted(store.root.iterdir(), key=lambda p: p.name):
                if not candidate.is_dir():
                    continue
                try:
                    manifest = store.manifest(candidate.name)
                    result.append({"id": candidate.name, "status": manifest.get("status", "unknown"),
                                   "created_utc": manifest.get("created_utc"),
                                   "dataset": (manifest.get("config_snapshot") or
                                               manifest.get("config") or {}).get("dataset"),
                                   "error": manifest.get("error")})
                except HTTPException as exc:
                    if exc.status_code == 409:
                        result.append({"id": candidate.name, "status": "unreadable", "error": exc.detail})
        with lock:
            by_id = {row["id"]: row for row in result}
            for run_id, job in jobs.items():
                by_id.setdefault(run_id, {"id": run_id}).update({"job": dict(job)})
            return {"runs": list(by_id.values())}

    @app.get("/api/runs/{run_id}/manifest")
    def manifest(run_id: str):
        return store.manifest(run_id)

    @app.get("/api/runs/{run_id}/report")
    def report(run_id: str):
        return store.report(run_id)

    @app.get("/api/runs/{run_id}/scene")
    def scene(run_id: str):
        return store.scene(run_id)

    @app.get("/api/runs/{run_id}/preview")
    def cloud_preview(run_id: str, limit: int | None = Query(default=None, ge=1)):
        if limit is not None and limit > settings.max_points:
            raise HTTPException(413, "Requested preview exceeds configured point limit")
        return preview(store, run_id, limit or settings.max_points, settings.max_npz_bytes)

    @app.get("/api/runs/{run_id}/files/{path:path}")
    def file(run_id: str, path: str, download: bool = False):
        resolved = confined(store.bundle(run_id), path)
        if not resolved.is_file():
            raise HTTPException(404, "Artifact is not a file")
        media = {".glb": "model/gltf-binary", ".gltf": "model/gltf+json",
                 ".tif": "image/tiff", ".tiff": "image/tiff", ".png": "image/png"}.get(
                     resolved.suffix.lower())
        return FileResponse(resolved, media_type=media,
                            filename=Path(path).name if download else None,
                            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-cache"})

    @app.post("/api/runs/{run_id}/measure")
    def measurement(run_id: str, request: MeasurementRequest):
        return measure(store, run_id, request, settings.max_measure_points, settings.max_volume_cells)

    @app.post("/api/runs", status_code=202)
    def start_run(request: RunRequest):
        dataset = Path(request.dataset)
        root = settings.dataset_root.resolve()
        if dataset.is_absolute():
            dataset = dataset.resolve()
            if not dataset.is_relative_to(root):
                raise HTTPException(403, "Dataset descriptor is outside dataset_root")
        else:
            dataset = confined(root, request.dataset)
        if not dataset.is_file() or dataset.suffix.lower() not in (".yaml", ".yml"):
            raise HTTPException(422, "dataset must name an existing YAML descriptor")
        prepared = service().prepare(dataset, request.profile)
        run_id = prepared.run_id
        with lock:
            if run_id in jobs and jobs[run_id]["status"] in ("queued", "running"):
                raise HTTPException(409, "This run is already queued or running")
            jobs[run_id] = {"status": "queued", "error": None}

        def execute():
            with lock:
                jobs[run_id]["status"] = "running"
            try:
                outcome = prepared.execute()
            except Exception as exc:
                logger.exception("Run %s failed", run_id)
                with lock:
                    jobs[run_id].update(status="failed", error=str(exc))
            else:
                with lock:
                    jobs[run_id]["status"] = outcome if isinstance(outcome, str) else "finished"

        pool.submit(execute)
        return {"id": run_id, "status": "queued", "manifest_url": f"/api/runs/{run_id}/manifest"}

    @app.get("/{path:path}", include_in_schema=False)
    def viewer(path: str):
        if path.startswith("api/") or path == "api":
            raise HTTPException(404, "API endpoint not found")
        base = settings.viewer_dist.resolve()
        requested = confined(base, path or "index.html", exists=False)
        if requested.is_file():
            return FileResponse(requested)
        if path and Path(path).suffix:
            raise HTTPException(404, "Viewer asset not found")
        index = confined(base, "index.html")
        return FileResponse(index)

    return app


app = create_app()


def main():
    import uvicorn

    settings = Settings.from_env()
    uvicorn.run("server.app:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
