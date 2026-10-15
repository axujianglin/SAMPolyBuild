"""FastAPI entry point for the Dockerized SAMPoly Auto inference service."""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from services.auto_inference_service import (
    AutoInferenceService,
    AutoServiceConfig,
    AutoServiceError,
)

from .schemas import AutoInferenceResponse, ErrorResponse, HealthResponse


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
UPLOAD_CHUNK_SIZE = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    """Parse one integer environment variable with an explicit error."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer.") from exc


def _env_float(name: str, default: float) -> float:
    """Parse one floating-point environment variable with an explicit error."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be numeric.") from exc


def service_config_from_env() -> AutoServiceConfig:
    """Build the service configuration without hard-coded host paths."""
    max_upload_mb = _env_int("SAMPOLY_MAX_UPLOAD_MB", 100)
    return AutoServiceConfig(
        model_config=Path(
            os.getenv("SAMPOLY_AUTO_CONFIG", "configs/auto_whumix.py")
        ),
        checkpoint=Path(
            os.getenv("SAMPOLY_AUTO_CHECKPOINT", "checkpoints/auto_whumix.pth")
        ),
        runtime_root=Path(os.getenv("SAMPOLY_RUNTIME_ROOT", "runtime")),
        device=os.getenv("SAMPOLY_DEVICE", "cuda"),
        gpu=_env_int("SAMPOLY_GPU", 0),
        batch_size=_env_int("SAMPOLY_BATCH_SIZE", 1),
        num_workers=_env_int("SAMPOLY_NUM_WORKERS", 1),
        default_score_threshold=_env_float(
            "SAMPOLY_DEFAULT_SCORE_THRESHOLD", 0.1
        ),
        max_upload_bytes=max_upload_mb * 1024 * 1024,
    )


def _safe_upload_name(file_name: str | None) -> tuple[str, str]:
    """Return a basename and normalized supported suffix for one upload."""
    normalized = str(file_name or "").strip().replace("\\", "/")
    safe_name = normalized.rsplit("/", 1)[-1]
    if not safe_name or safe_name in (".", ".."):
        raise AutoServiceError("missing_file_name", "Uploaded image needs a file name.", 400)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_IMAGE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_IMAGE_SUFFIXES))
        raise AutoServiceError(
            "unsupported_image_format",
            f"Unsupported image suffix {suffix or '<none>'}. Supported suffixes: {supported}",
            400,
        )
    # Preserve the original stem for PredictDataset image_id matching while
    # normalizing the suffix used by its case-sensitive file scan.
    return f"{Path(safe_name).stem}{suffix}", suffix


async def _save_upload(
    upload: UploadFile,
    destination: Path,
    max_upload_bytes: int,
) -> int:
    """Stream an upload to disk while enforcing the configured byte limit."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    try:
        with destination.open("wb") as stream:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_upload_bytes:
                    raise AutoServiceError(
                        "upload_too_large",
                        f"Uploaded image exceeds the {max_upload_bytes} byte limit.",
                        413,
                    )
                stream.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()
    if total_bytes == 0:
        destination.unlink(missing_ok=True)
        raise AutoServiceError("empty_upload", "Uploaded image is empty.", 400)
    return total_bytes


def _error_response(error: AutoServiceError) -> JSONResponse:
    """Map a structured service error to the stable HTTP failure envelope."""
    return JSONResponse(
        status_code=error.status_code,
        content={
            "success": False,
            "error": {"code": error.code, "message": error.message},
        },
    )


def create_app(
    service: AutoInferenceService | None = None,
    api_key: str | None = None,
) -> FastAPI:
    """Create the API with an injectable service for model-free tests."""
    inference_service = service or AutoInferenceService(service_config_from_env())
    configured_api_key = (
        os.getenv("SAMPOLY_API_KEY", "") if api_key is None else str(api_key)
    ).strip()
    application = FastAPI(
        title="SAMPoly Auto Inference Service",
        version="1.0.0",
        description="Single-image Auto polygon inference for the SHORTER desktop GUI.",
    )
    application.state.auto_service = inference_service

    @application.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        """Return FastAPI parsing failures through the same stable envelope."""
        return _error_response(
            AutoServiceError(
                "request_validation_error",
                f"Invalid request: {error.errors()}",
                422,
            )
        )

    @application.get("/health", response_model=HealthResponse)
    async def health() -> dict:
        """Report configuration and CUDA readiness without loading the model."""
        return inference_service.health()

    @application.post(
        "/v1/auto/infer",
        response_model=AutoInferenceResponse,
        responses={
            400: {"model": ErrorResponse},
            401: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def infer_auto(
        image: UploadFile = File(...),
        score_threshold: float | None = Form(default=None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ):
        """Run the unchanged Auto pipeline on one uploaded image."""
        request_id = uuid.uuid4().hex
        temporary_root: Path | None = None
        try:
            if configured_api_key and not secrets.compare_digest(
                str(x_api_key or ""), configured_api_key
            ):
                raise AutoServiceError(
                    "unauthorized",
                    "A valid X-API-Key header is required.",
                    401,
                )
            safe_name, _ = _safe_upload_name(image.filename)
            inference_service.config.runtime_root.mkdir(parents=True, exist_ok=True)
            temporary_root = Path(
                tempfile.mkdtemp(
                    prefix=f"request_{request_id}_",
                    dir=inference_service.config.runtime_root,
                )
            )
            image_path = temporary_root / "input" / safe_name
            await _save_upload(
                image,
                image_path,
                inference_service.config.max_upload_bytes,
            )
            return await run_in_threadpool(
                inference_service.infer,
                image_path,
                request_id=request_id,
                original_file_name=safe_name,
                score_threshold=score_threshold,
            )
        except AutoServiceError as exc:
            return _error_response(exc)
        except Exception as exc:
            return _error_response(
                AutoServiceError(
                    "internal_error", f"{type(exc).__name__}: {exc}", 500
                )
            )
        finally:
            if temporary_root is not None:
                shutil.rmtree(temporary_root, ignore_errors=True)

    return application


app = create_app()
