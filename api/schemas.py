"""Stable response schemas exposed to native GUI clients."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Readiness information returned without constructing the model runner."""

    status: str
    service: str
    config_exists: bool
    checkpoint_exists: bool
    cuda_available: bool
    cuda_error: str | None = None
    device: str
    busy: bool
    inference_lifecycle: str


class AutoImageResponse(BaseModel):
    """Identity and original pixel dimensions of the uploaded image."""

    file_name: str
    image_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class AutoPolygonResponse(BaseModel):
    """One validated building polygon returned in original image coordinates."""

    id: str
    score: float
    score_cls: float | None = None
    bbox_xywh: list[float]
    points: list[list[float]]
    metadata: dict[str, Any]


class AutoSummaryResponse(BaseModel):
    """Compact statistics for one successful inference request."""

    polygon_count: int = Field(ge=0)
    score_threshold: float = Field(ge=0.0, le=1.0)


class AutoInferenceResponse(BaseModel):
    """Successful single-image Auto inference response."""

    success: bool
    request_id: str
    image: AutoImageResponse
    coordinate_space: str
    polygons: list[AutoPolygonResponse]
    summary: AutoSummaryResponse


class ErrorDetail(BaseModel):
    """Machine-readable error payload shared by all API failures."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Top-level failure envelope returned to SHORTER."""

    success: bool = False
    error: ErrorDetail
