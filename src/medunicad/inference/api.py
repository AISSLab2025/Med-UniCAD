"""FastAPI service for the Med-UniCAD inference engine.

Run from the repository root:
    python scripts/serve_api.py --config configs/inference.example.json

The model is loaded once when the API starts. GPU inference is serialized with
a process-local lock so concurrent HTTP requests do not execute the same model
simultaneously.
"""

from __future__ import annotations

import argparse
import json
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from medunicad.inference.engine import (
    InferenceConfig,
    InferenceRequest,
    UnifiedInferenceEngine,
    _json_ready,
    _safe_id,
    _write_json,
)


class PathInferencePayload(BaseModel):
    """JSON request when the image already exists on the API server."""

    image_path: str = Field(
        ...,
        description="Absolute path, or a path resolvable on the API server.",
    )
    modality: str = Field(
        ...,
        description="xray, ct, mri, or ultrasound",
    )
    tasks: Union[str, List[str]] = Field(
        ...,
        description="Comma-separated string or JSON list of tasks.",
    )
    request_id: Optional[str] = None
    save_input_copy: bool = True


def _parse_tasks(tasks: Union[str, List[str]]) -> List[str]:
    if isinstance(tasks, str):
        values = [part.strip() for part in tasks.split(",")]
    else:
        values = [str(part).strip() for part in tasks]

    values = [value for value in values if value]
    if not values:
        raise ValueError("At least one task must be provided.")
    return values


def _safe_client_filename(filename: Optional[str]) -> str:
    name = Path(filename or "uploaded_image.png").name
    cleaned = "".join(
        char if char.isalnum() or char in "._-" else "_"
        for char in name
    ).strip("._")
    return cleaned or "uploaded_image.png"


def _rewrite_result_json(result: Dict[str, Any]) -> None:
    result_path = result.get("result_json")
    if not result_path:
        return
    _write_json(Path(result_path), result)


def _attach_artifact_urls(
    result: Dict[str, Any],
    request: Request,
) -> None:
    """Add HTTP URLs next to absolute server-local artifact paths."""
    request_id = str(result.get("request_id", ""))
    request_root_value = result.get("request_root")
    if not request_id or not request_root_value:
        return

    request_root = Path(request_root_value).resolve(strict=False)
    base_url = str(request.base_url).rstrip("/")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            additions: Dict[str, str] = {}
            for key, item in list(value.items()):
                if isinstance(item, str):
                    try:
                        target = Path(item).resolve(strict=False)
                    except Exception:
                        target = None

                    if target is not None and (
                        target == request_root
                        or request_root in target.parents
                    ) and target != request_root:
                        relative = target.relative_to(
                            request_root
                        ).as_posix()
                        artifact_url = (
                            f"{base_url}/artifacts/"
                            f"{quote(request_id, safe='')}/"
                            f"{quote(relative, safe='/')}"
                        )

                        if key.endswith("_path"):
                            url_key = key[:-5] + "_url"
                        elif key in {"raw", "heatmap", "overlay"}:
                            url_key = key + "_url"
                        else:
                            url_key = key + "_url"
                        additions[url_key] = artifact_url
                else:
                    visit(item)
            value.update(additions)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(result.get("outputs", {}))


def create_app(config_path: str | Path) -> FastAPI:
    resolved_config = Path(config_path).expanduser().resolve(strict=False)
    inference_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        if not resolved_config.is_file():
            raise RuntimeError(
                f"Inference configuration was not found: {resolved_config}"
            )

        config = InferenceConfig.from_json(resolved_config)
        engine = UnifiedInferenceEngine(config)

        # Compatibility safeguard:
        # older inference-core files may not define engine.output_root.
        # Normalize it here so every API endpoint can use absolute paths.
        engine.output_root = Path(
            getattr(engine, "output_root", engine.config.output_root)
        ).expanduser().resolve(strict=False)
        engine.output_root.mkdir(parents=True, exist_ok=True)
        engine.config.output_root = str(engine.output_root)

        app_instance.state.engine = engine
        app_instance.state.config_path = resolved_config
        app_instance.state.inference_lock = inference_lock
        yield

        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    api = FastAPI(
        title="Unified MedUniCAD Inference API",
        version="1.0.0",
        description=(
            "Local-network API for X-ray, CT, MRI, and ultrasound "
            "multitask inference."
        ),
        lifespan=lifespan,
    )

    cors_value = os.environ.get("MEDUNICAD_CORS_ORIGINS", "*")
    cors_origins = [
        value.strip()
        for value in cors_value.split(",")
        if value.strip()
    ] or ["*"]

    api.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def get_engine() -> UnifiedInferenceEngine:
        engine = getattr(api.state, "engine", None)
        if engine is None:
            raise HTTPException(
                status_code=503,
                detail="The inference model is not ready.",
            )
        return engine

    def run_locked(request: InferenceRequest) -> Dict[str, Any]:
        engine = get_engine()
        with api.state.inference_lock:
            return engine.infer(request)

    def translate_exception(error: Exception) -> HTTPException:
        if isinstance(error, FileNotFoundError):
            return HTTPException(status_code=404, detail=str(error))
        if isinstance(error, FileExistsError):
            return HTTPException(
                status_code=409,
                detail=(
                    "The request_id already exists. Use a different "
                    "request_id or omit it to generate one automatically."
                ),
            )
        if isinstance(error, (ValueError, KeyError, TypeError)):
            return HTTPException(status_code=400, detail=str(error))
        if isinstance(error, torch.cuda.OutOfMemoryError):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return HTTPException(
                status_code=503,
                detail="CUDA ran out of memory while processing the request.",
            )
        return HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        )

    @api.get("/")
    async def root() -> Dict[str, Any]:
        return {
            "service": "Unified MedUniCAD Inference API",
            "status": "ready",
            "docs": "/docs",
            "health": "/health",
            "path_inference": "/infer/path",
            "upload_inference": "/infer/upload",
        }

    @api.get("/health")
    async def health() -> Dict[str, Any]:
        engine = get_engine()
        return {
            "status": "ready",
            "device": str(engine.device),
            "cuda_available": bool(torch.cuda.is_available()),
            "config_path": str(api.state.config_path),
            "checkpoint_path": str(
                Path(engine.config.checkpoint_path).resolve(strict=False)
            ),
            "output_root": str(engine.output_root),
            "modality_map": dict(engine.config.modality_map),
        }

    @api.post("/infer/path")
    async def infer_from_server_path(
        payload: PathInferencePayload,
        http_request: Request,
    ) -> Dict[str, Any]:
        """Run inference for an image path visible to the API server."""
        try:
            request = InferenceRequest(
                image_path=payload.image_path,
                modality=payload.modality,
                tasks=_parse_tasks(payload.tasks),
                request_id=payload.request_id,
                save_input_copy=payload.save_input_copy,
            )
            result = await run_in_threadpool(run_locked, request)
            _attach_artifact_urls(result, http_request)
            _rewrite_result_json(result)
            return _json_ready(result)
        except Exception as error:
            raise translate_exception(error) from error

    @api.post("/infer/upload")
    async def infer_from_upload(
        http_request: Request,
        image: UploadFile = File(...),
        modality: str = Form(...),
        tasks: str = Form(...),
        request_id: Optional[str] = Form(None),
    ) -> Dict[str, Any]:
        """
        Upload an image and run inference.

        `tasks` is a comma-separated value such as:
        segmentation,classification,report
        """
        engine = get_engine()
        upload_root = (
            engine.output_root / "_api_uploads"
        ).resolve(strict=False)
        upload_root.mkdir(parents=True, exist_ok=True)

        client_filename = _safe_client_filename(image.filename)
        temporary_path = (
            upload_root
            / f"{uuid.uuid4().hex}_{client_filename}"
        ).resolve(strict=False)

        try:
            with open(temporary_path, "wb") as handle:
                shutil.copyfileobj(image.file, handle)

            request = InferenceRequest(
                image_path=str(temporary_path),
                modality=modality,
                tasks=_parse_tasks(tasks),
                request_id=request_id,
                save_input_copy=True,
            )

            result = await run_in_threadpool(run_locked, request)

            saved_image = result.get("input", {}).get("saved_image")
            result.setdefault("input", {})
            result["input"]["client_filename"] = client_filename
            result["input"]["uploaded_via_api"] = True
            if saved_image:
                result["input"]["source_image"] = saved_image

            _attach_artifact_urls(result, http_request)
            _rewrite_result_json(result)
            return _json_ready(result)

        except Exception as error:
            raise translate_exception(error) from error
        finally:
            try:
                await image.close()
            except Exception:
                pass
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                pass

    @api.get("/requests/{request_id}")
    async def get_request_result(request_id: str) -> Dict[str, Any]:
        """Return the stored result.json for a previous request."""
        engine = get_engine()
        safe_request_id = _safe_id(request_id)

        if safe_request_id != request_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid request_id.",
            )

        result_path = (
            engine.output_root / request_id / "result.json"
        ).resolve(strict=False)

        if not result_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Result not found for request_id={request_id}",
            )

        with open(result_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @api.get("/artifacts/{request_id}/{artifact_path:path}")
    async def get_artifact(
        request_id: str,
        artifact_path: str,
    ) -> FileResponse:
        """
        Download or view one saved artifact.

        Example:
            /artifacts/CT_001/ct_gradcam/gradcam_overlay.png
        """
        engine = get_engine()
        safe_request_id = _safe_id(request_id)
        if safe_request_id != request_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid request_id.",
            )

        request_root = (
            engine.output_root / request_id
        ).resolve(strict=False)
        target = (
            request_root / artifact_path
        ).resolve(strict=False)

        if target != request_root and request_root not in target.parents:
            raise HTTPException(
                status_code=403,
                detail="Artifact path is outside the request directory.",
            )

        if not target.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Artifact was not found: {artifact_path}",
            )

        return FileResponse(
            path=str(target),
            filename=target.name,
        )

    return api


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Unified MedUniCAD FastAPI server."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the inference JSON configuration.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address. Use 0.0.0.0 for local-network access.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8100,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Keep this at 1 for a single GPU/model instance. "
            "Multiple workers load multiple model copies."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help=(
            "Development only. Do not use reload for normal GPU inference."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.workers != 1:
        raise ValueError(
            "Use --workers 1. Multiple workers would load multiple copies "
            "of the full model into memory."
        )

    server_app = create_app(args.config)
    uvicorn.run(
        server_app,
        host=args.host,
        port=args.port,
        workers=1,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
