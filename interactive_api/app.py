"""FastAPI implementation of SHORTER interactive refinement protocol 2.0."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import math
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
from fastapi import Body, FastAPI, File, Form, Header, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from .config import InteractiveServiceConfig
from .errors import InteractiveAPIError, error_payload
from .predictor import InteractivePredictorAdapter
from .schemas import RefineRequest, SessionCreateRequest
from .session_store import InteractionState, SessionEntry, SessionStore


LOGGER = logging.getLogger(__name__)
PROTOCOL_VERSION = "2.0"
COORDINATE_SPACE = "original_image_pixel"
SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SUPPORTED_PIL_FORMATS = {"PNG", "JPEG", "TIFF"}
UPLOAD_CHUNK_SIZE = 1024 * 1024


def create_app(
    *,
    config: InteractiveServiceConfig | None = None,
    predictor: Any | None = None,
    session_store: SessionStore | None = None,
) -> FastAPI:
    """Build an injectable application; production dependencies load at startup."""

    service_config = config or InteractiveServiceConfig.from_env()
    model_adapter = predictor or InteractivePredictorAdapter(service_config)
    store = session_store or SessionStore(
        service_config.session_ttl_seconds,
        service_config.max_sessions,
    )
    application = FastAPI(
        title="SAMPoly Interactive Refinement Service",
        version=PROTOCOL_VERSION,
        description="Stateful positive/negative point refinement for SHORTER.",
    )
    application.state.interactive_config = service_config
    application.state.interactive_predictor = model_adapter
    application.state.interactive_sessions = store

    @application.on_event("startup")
    def startup() -> None:
        if not service_config.api_key:
            raise RuntimeError("SAMPOLY_API_KEY must be configured and non-empty.")
        model_adapter.initialize()

    @application.on_event("shutdown")
    def shutdown() -> None:
        store.clear()
        model_adapter.close()

    @application.exception_handler(InteractiveAPIError)
    async def interactive_error_handler(
        request: Request, error: InteractiveAPIError
    ) -> JSONResponse:
        LOGGER.warning(
            "interactive request failed request_id=%s path=%s code=%s",
            request.headers.get("X-Request-ID", "")[:128],
            request.url.path,
            error.code,
        )
        return JSONResponse(status_code=error.status_code, content=error_payload(error))

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        error = InteractiveAPIError(
            "invalid_request", "The HTTP request does not match the protocol schema.", 400
        )
        return JSONResponse(status_code=400, content=error_payload(error))

    @application.exception_handler(Exception)
    async def unexpected_error_handler(_request: Request, error: Exception) -> JSONResponse:
        LOGGER.exception("Unhandled interactive API error", exc_info=error)
        public_error = InteractiveAPIError(
            "inference_failed", "The interactive request could not be completed.", 500
        )
        return JSONResponse(status_code=500, content=error_payload(public_error))

    @application.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ready" if bool(getattr(model_adapter, "initialized", False)) else "starting",
            "service": "sampoly-interactive-refinement",
            "protocol_version": PROTOCOL_VERSION,
            "active_sessions": store.active_count,
        }

    @application.post("/v1/interactive/sessions", status_code=201)
    async def create_session(
        image: UploadFile | None = File(default=None),
        request_json: str | None = Form(default=None, alias="request"),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    ) -> dict[str, Any]:
        _authenticate(x_api_key, service_config.api_key)
        header_request_id = _require_request_id(x_request_id)
        if image is None:
            raise InteractiveAPIError("invalid_request", "Multipart field 'image' is required.", 400)
        started = time.monotonic()
        try:
            if request_json is None:
                raise InteractiveAPIError(
                    "invalid_request", "Multipart field 'request' is required.", 400
                )
            metadata = _parse_model(SessionCreateRequest, request_json)
            _validate_identity(metadata, header_request_id)
            if image.filename != metadata.image.file_name:
                raise InteractiveAPIError(
                    "invalid_image_metadata",
                    "The uploaded filename does not match request image metadata.",
                    400,
                )
            payload = await _read_upload(image, service_config.max_upload_bytes)
            image_rgb, decoded_format = await run_in_threadpool(
                _decode_image, payload, metadata.image.file_name
            )
            actual_height, actual_width = image_rgb.shape[:2]
            if (actual_width, actual_height) != (
                metadata.image.width,
                metadata.image.height,
            ):
                raise InteractiveAPIError(
                    "image_size_mismatch",
                    "Declared image dimensions do not match the decoded image.",
                    422,
                    {
                        "declared_width": metadata.image.width,
                        "declared_height": metadata.image.height,
                        "actual_width": actual_width,
                        "actual_height": actual_height,
                    },
                )
            if store.active_count >= service_config.max_sessions:
                raise InteractiveAPIError(
                    "session_limit_reached",
                    "The maximum number of active image sessions has been reached.",
                    429,
                )
            prepared = await run_in_threadpool(model_adapter.prepare_image, image_rgb)
            image_id = hashlib.sha256(payload).hexdigest()
            entry = store.create(
                image_id=image_id,
                file_name=metadata.image.file_name,
                width=actual_width,
                height=actual_height,
                prepared_image=prepared,
            )
            LOGGER.info(
                "interactive session created request_id=%s session=%s format=%s elapsed_ms=%.1f",
                header_request_id,
                entry.session_id[:10],
                decoded_format,
                (time.monotonic() - started) * 1000.0,
            )
            return {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": metadata.request_id,
                "session_id": entry.session_id,
                "image": {
                    "file_name": entry.file_name,
                    "image_id": entry.image_id,
                    "width": entry.width,
                    "height": entry.height,
                },
                "coordinate_space": COORDINATE_SPACE,
                "expires_in_seconds": service_config.session_ttl_seconds,
            }
        finally:
            await image.close()

    @application.post("/v1/interactive/sessions/{session_id:path}/refine")
    async def refine(
        session_id: str,
        request_body: dict[str, Any] = Body(...),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    ) -> dict[str, Any]:
        _authenticate(x_api_key, service_config.api_key)
        header_request_id = _require_request_id(x_request_id)
        request_model = _parse_model(RefineRequest, request_body)
        _validate_identity(request_model, header_request_id)
        decoded_session_id = unquote(session_id)
        if request_model.session_id != decoded_session_id:
            raise InteractiveAPIError(
                "invalid_request", "URL and JSON session IDs do not match.", 400
            )
        started = time.monotonic()
        response = await run_in_threadpool(
            _run_refine,
            store,
            model_adapter,
            decoded_session_id,
            request_model,
        )
        LOGGER.info(
            "interactive refine request_id=%s session=%s polygon=%s revision=%d positive=%d negative=%d elapsed_ms=%.1f",
            header_request_id,
            decoded_session_id[:10],
            request_model.polygon_id,
            request_model.revision,
            len(request_model.positive_points),
            len(request_model.negative_points),
            (time.monotonic() - started) * 1000.0,
        )
        return response

    @application.delete("/v1/interactive/sessions/{session_id:path}", status_code=204)
    async def delete_session(
        session_id: str,
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
        _x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    ) -> Response:
        _authenticate(x_api_key, service_config.api_key)
        decoded_session_id = unquote(session_id)
        await run_in_threadpool(store.delete, decoded_session_id)
        return Response(status_code=204)

    return application


def _authenticate(received: str | None, expected: str) -> None:
    if not expected or not hmac.compare_digest(str(received or ""), expected):
        raise InteractiveAPIError(
            "unauthorized", "A valid X-API-Key header is required.", 401
        )


def _require_request_id(value: str | None) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise InteractiveAPIError(
            "missing_request_id", "X-Request-ID must be non-empty.", 400
        )
    return normalized


def _parse_model(model_type, value):
    try:
        if isinstance(value, str):
            value = json.loads(value)
        return model_type.parse_obj(value)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise InteractiveAPIError(
            "invalid_request", "The request JSON does not match the protocol schema.", 400
        ) from exc


def _validate_identity(model: Any, header_request_id: str) -> None:
    if model.protocol_version != PROTOCOL_VERSION:
        raise InteractiveAPIError(
            "invalid_protocol_version", "Only interactive protocol 2.0 is supported.", 400
        )
    if not model.request_id.strip():
        raise InteractiveAPIError("missing_request_id", "request_id must be non-empty.", 400)
    if model.request_id != header_request_id:
        raise InteractiveAPIError(
            "request_id_mismatch", "Header and JSON request IDs do not match.", 400
        )
    if model.coordinate_space != COORDINATE_SPACE:
        raise InteractiveAPIError(
            "invalid_request", "coordinate_space must be original_image_pixel.", 400
        )


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        size += len(chunk)
        if size > limit:
            raise InteractiveAPIError(
                "upload_too_large", "The uploaded image exceeds the configured limit.", 413
            )
        chunks.append(chunk)
    if size == 0:
        raise InteractiveAPIError("invalid_request", "The uploaded image is empty.", 400)
    return b"".join(chunks)


def _decode_image(payload: bytes, file_name: str) -> tuple[np.ndarray, str]:
    suffix = Path(file_name.replace("\\", "/").rsplit("/", 1)[-1]).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise InteractiveAPIError(
            "unsupported_image_format", "The uploaded image extension is unsupported.", 415
        )
    try:
        with Image.open(io.BytesIO(payload)) as source:
            decoded_format = str(source.format or "").upper()
            if decoded_format not in SUPPORTED_PIL_FORMATS:
                raise InteractiveAPIError(
                    "unsupported_image_format", "The uploaded image content is unsupported.", 415
                )
            image_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    except InteractiveAPIError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InteractiveAPIError(
            "unsupported_image_format", "The uploaded image cannot be decoded.", 415
        ) from exc
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise InteractiveAPIError(
            "unsupported_image_format", "The uploaded image cannot be converted to RGB.", 415
        )
    return image_rgb, decoded_format


def _run_refine(
    store: SessionStore,
    predictor: Any,
    session_id: str,
    request: RefineRequest,
) -> dict[str, Any]:
    with store.locked_session(session_id) as entry:
        _validate_session_metadata(entry, request)
        bbox = _validate_bbox(request.bbox_xyxy, entry.width, entry.height)
        positive = _validate_points(
            request.positive_points, entry.width, entry.height, bbox, "positive_points", True
        )
        negative = _validate_points(
            request.negative_points, entry.width, entry.height, bbox, "negative_points", False
        )
        original_polygon = _validate_original_polygon(
            request.original_polygon, entry.width, entry.height
        )
        if request.revision <= 0:
            raise InteractiveAPIError(
                "invalid_revision", "revision must be greater than zero.", 400
            )

        previous_logits = None
        existing = entry.interaction_states.get(request.polygon_id)
        if request.interaction_token:
            if existing is None or not hmac.compare_digest(
                request.interaction_token, existing.interaction_token
            ):
                raise InteractiveAPIError(
                    "interaction_token_invalid",
                    "The interaction token is invalid for this session or polygon.",
                    409,
                )
            if request.revision != existing.successful_revision + 1:
                raise InteractiveAPIError(
                    "interaction_state_conflict",
                    "The revision does not follow the current successful interaction.",
                    409,
                )
            if bbox != existing.bbox or original_polygon != existing.original_polygon:
                raise InteractiveAPIError(
                    "interaction_state_conflict",
                    "BBox or original polygon changed while continuing an interaction token.",
                    409,
                )
            if not _is_prefix(existing.positive_points, positive) or not _is_prefix(
                existing.negative_points, negative
            ):
                raise InteractiveAPIError(
                    "interaction_state_conflict",
                    "Cumulative prompt points do not preserve the previous interaction state.",
                    409,
                )
            previous_logits = existing.mask_logits
        elif existing is not None and request.revision <= existing.successful_revision:
            raise InteractiveAPIError(
                "interaction_state_conflict",
                "An older revision cannot replace newer successful state.",
                409,
            )

        prediction = predictor.predict(
            prepared_image=entry.prepared_image,
            bbox=bbox,
            positive_points=positive,
            negative_points=negative,
            original_polygon=original_polygon,
            previous_logits=previous_logits,
        )
        output_polygon, output_bbox, output_score, output_logits, strategy = (
            _validate_prediction_output(prediction, entry.width, entry.height)
        )
        token = secrets.token_urlsafe(32)
        entry.interaction_states[request.polygon_id] = InteractionState(
            interaction_token=token,
            polygon_id=request.polygon_id,
            successful_revision=request.revision,
            bbox=bbox,
            positive_points=positive,
            negative_points=negative,
            original_polygon=original_polygon,
            mask_logits=output_logits,
        )
        store.touch(entry)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request.request_id,
            "session_id": request.session_id,
            "polygon_id": request.polygon_id,
            "revision": request.revision,
            "interaction_token": token,
            "image": {
                "file_name": request.image.file_name,
                "width": request.image.width,
                "height": request.image.height,
            },
            "coordinate_space": COORDINATE_SPACE,
            "polygon": {
                "id": request.polygon_id,
                "score": output_score,
                "bbox_xywh": output_bbox,
                "points": output_polygon,
                "metadata": {
                    "source": "sampoly_interactive",
                    "score_type": "predicted_iou",
                    "selection_strategy": strategy,
                },
            },
        }


def _validate_session_metadata(entry: SessionEntry, request: RefineRequest) -> None:
    if not request.polygon_id.strip():
        raise InteractiveAPIError("invalid_request", "polygon_id must be non-empty.", 400)
    if (
        request.image.file_name != entry.file_name
        or request.image.width != entry.width
        or request.image.height != entry.height
    ):
        raise InteractiveAPIError(
            "invalid_image_metadata", "Image metadata does not match the session.", 400
        )


def _finite_number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    result = float(value)
    if not math.isfinite(result):
        raise ValueError
    return result


def _validate_bbox(values: list[Any], width: int, height: int) -> tuple[float, float, float, float]:
    try:
        if len(values) != 4:
            raise ValueError
        x1, y1, x2, y2 = (_finite_number(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise InteractiveAPIError(
            "invalid_bbox", "bbox_xyxy must contain four finite numbers.", 400
        ) from exc
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise InteractiveAPIError(
            "invalid_bbox", "bbox_xyxy is outside the original image bounds.", 400
        )
    return x1, y1, x2, y2


def _validate_points(
    values: list[Any],
    width: int,
    height: int,
    bbox: tuple[float, float, float, float],
    field_name: str,
    required: bool,
) -> tuple[tuple[float, float], ...]:
    error_code = "invalid_positive_points" if required else "invalid_negative_points"
    if required and not values:
        raise InteractiveAPIError(error_code, "At least one positive point is required.", 400)
    result = []
    try:
        for value in values:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError
            x, y = _finite_number(value[0]), _finite_number(value[1])
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError
            if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
                raise ValueError
            result.append((x, y))
    except (TypeError, ValueError) as exc:
        raise InteractiveAPIError(
            error_code,
            f"{field_name} must contain finite points inside the image and bbox.",
            400,
        ) from exc
    return tuple(result)


def _validate_original_polygon(
    values: list[Any], width: int, height: int
) -> tuple[tuple[float, float], ...]:
    result = []
    try:
        for value in values:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError
            x, y = _finite_number(value[0]), _finite_number(value[1])
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError
            result.append((x, y))
    except (TypeError, ValueError) as exc:
        raise InteractiveAPIError(
            "invalid_polygon", "original_polygon contains invalid coordinates.", 400
        ) from exc
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    if len(set(result)) < 3:
        raise InteractiveAPIError(
            "invalid_polygon", "original_polygon needs at least three distinct points.", 400
        )
    points = np.asarray(result, dtype=np.float64)
    area = 0.5 * abs(
        float(np.dot(points[:, 0], np.roll(points[:, 1], -1)))
        - float(np.dot(points[:, 1], np.roll(points[:, 0], -1)))
    )
    if area <= 0:
        raise InteractiveAPIError(
            "invalid_geometry", "original_polygon must have a positive area.", 422
        )
    return tuple(result)


def _is_prefix(previous: tuple[Any, ...], current: tuple[Any, ...]) -> bool:
    return len(current) >= len(previous) and current[: len(previous)] == previous


def _validate_prediction_output(
    prediction: Any, width: int, height: int
) -> tuple[list[list[float]], list[float], float, np.ndarray, str]:
    try:
        score = _finite_number(prediction.score)
        raw_points = prediction.polygon
        strategy = str(prediction.selection_strategy)
        logits = np.asarray(prediction.mask_logits, dtype=np.float32)
    except (AttributeError, TypeError, ValueError) as exc:
        raise InteractiveAPIError(
            "invalid_model_output", "The predictor returned an invalid result.", 500
        ) from exc
    if not 0.0 <= score <= 1.0 or logits.ndim != 3 or not np.isfinite(logits).all():
        raise InteractiveAPIError(
            "invalid_model_output", "The predictor returned an invalid score or logits.", 500
        )
    points = []
    try:
        for raw_point in raw_points:
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
                raise ValueError
            x, y = _finite_number(raw_point[0]), _finite_number(raw_point[1])
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError
            points.append([x, y])
    except (TypeError, ValueError) as exc:
        raise InteractiveAPIError(
            "invalid_model_output", "The predictor returned invalid polygon coordinates.", 500
        ) from exc
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    if len({(x, y) for x, y in points}) < 3:
        raise InteractiveAPIError(
            "invalid_model_output", "The predicted polygon has fewer than three points.", 500
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
    if bbox[2] <= 0 or bbox[3] <= 0:
        raise InteractiveAPIError(
            "invalid_model_output", "The predicted polygon has an invalid extent.", 500
        )
    return points, bbox, score, logits, strategy


app = create_app()
