"""Strict Pydantic schemas for SHORTER interactive protocol 2.0."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, StrictInt, StrictStr, validator


class StrictModel(BaseModel):
    class Config:
        extra = "forbid"


class ImageMetadata(StrictModel):
    file_name: StrictStr
    width: StrictInt
    height: StrictInt

    @validator("file_name")
    def nonempty_file_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("file_name must be non-empty")
        return value

    @validator("width", "height")
    def positive_dimension(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("image dimensions must be positive")
        return value


class SessionCreateRequest(StrictModel):
    protocol_version: StrictStr
    request_id: StrictStr
    coordinate_space: StrictStr
    image: ImageMetadata


class RefineRequest(StrictModel):
    protocol_version: StrictStr
    request_id: StrictStr
    session_id: StrictStr
    polygon_id: StrictStr
    revision: StrictInt
    coordinate_space: StrictStr
    image: ImageMetadata
    bbox_xyxy: list[Any]
    positive_points: list[Any]
    negative_points: list[Any]
    original_polygon: list[Any]
    interaction_token: StrictStr


class ErrorDetail(StrictModel):
    code: str
    message: str
    details: dict[str, Any]


class ErrorResponse(StrictModel):
    error: ErrorDetail
