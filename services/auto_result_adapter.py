"""Adapt existing SAMPoly Auto COCO records to the HTTP polygon contract."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


def _flattened_ring(segmentation: Any, raw_index: int) -> list[float]:
    """Select the longest COCO ring and validate its numeric coordinates."""
    candidate = segmentation
    if isinstance(candidate, list) and candidate and isinstance(candidate[0], list):
        candidate = max(candidate, key=len)
    if not isinstance(candidate, list) or len(candidate) < 6 or len(candidate) % 2:
        raise ValueError(
            f"Auto result {raw_index} segmentation must contain at least three x/y pairs."
        )
    try:
        values = [float(value) for value in candidate]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Auto result {raw_index} segmentation coordinates must be numeric."
        ) from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError(
            f"Auto result {raw_index} segmentation coordinates must be finite."
        )
    return values


def _bounded_points(
    segmentation: Any,
    raw_index: int,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    """Convert one COCO ring to bounded original-image pixel coordinates."""
    values = _flattened_ring(segmentation, raw_index)
    points = []
    for index in range(0, len(values), 2):
        x, y = values[index], values[index + 1]
        if x < -1.0 or y < -1.0 or x > image_width + 1.0 or y > image_height + 1.0:
            raise ValueError(
                f"Auto result {raw_index} polygon coordinate ({x}, {y}) exceeds "
                f"image bounds ({image_width}, {image_height})."
            )
        points.append(
            [
                min(max(x, 0.0), float(image_width)),
                min(max(y, 0.0), float(image_height)),
            ]
        )
    return points


def _bbox_xywh(
    raw_bbox: Any,
    points: list[list[float]],
    raw_index: int,
    image_width: int,
    image_height: int,
) -> list[float]:
    """Validate a COCO xywh bbox or derive one from polygon coordinates."""
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            x, y, width, height = [float(value) for value in raw_bbox]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Auto result {raw_index} bbox values must be numeric.") from exc
        if not all(math.isfinite(value) for value in (x, y, width, height)):
            raise ValueError(f"Auto result {raw_index} bbox values must be finite.")
        if width < 0 or height < 0:
            raise ValueError(
                f"Auto result {raw_index} bbox width and height must be non-negative."
            )
        if (
            x < -1.0
            or y < -1.0
            or x + width > image_width + 1.0
            or y + height > image_height + 1.0
        ):
            raise ValueError(
                f"Auto result {raw_index} bbox {[x, y, width, height]} exceeds "
                f"image bounds ({image_width}, {image_height})."
            )
        bounded_x = min(max(x, 0.0), float(image_width))
        bounded_y = min(max(y, 0.0), float(image_height))
        bounded_x2 = min(max(x + width, 0.0), float(image_width))
        bounded_y2 = min(max(y + height, 0.0), float(image_height))
        return [bounded_x, bounded_y, bounded_x2 - bounded_x, bounded_y2 - bounded_y]

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _matches_image_id(raw_image_id: Any, expected_image_id: str) -> bool:
    """Match direct inference ids, including ids containing subdirectories."""
    return Path(str(raw_image_id)).name == expected_image_id


def adapt_auto_results(
    raw_results: list[dict[str, Any]],
    *,
    image_id: str,
    image_width: int,
    image_height: int,
    score_threshold: float,
) -> list[dict[str, Any]]:
    """Return validated HTTP polygon records in original image pixel coordinates."""
    if not isinstance(raw_results, list):
        raise ValueError("Auto results JSON root must be a list.")
    if not image_id:
        raise ValueError("image_id must not be empty.")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1.")

    polygons = []
    for raw_index, raw in enumerate(raw_results):
        if not isinstance(raw, dict):
            raise ValueError(f"Auto result {raw_index} must be an object.")
        if not _matches_image_id(raw.get("image_id", ""), image_id):
            continue
        try:
            threshold_score = float(raw.get("score_cls", raw.get("score", 0.0)))
            polygon_score = float(raw.get("score", threshold_score))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Auto result {raw_index} score must be numeric.") from exc
        if not math.isfinite(threshold_score) or not math.isfinite(polygon_score):
            raise ValueError(f"Auto result {raw_index} score must be finite.")
        if threshold_score < score_threshold:
            continue

        points = _bounded_points(
            raw.get("segmentation"), raw_index, image_width, image_height
        )
        polygons.append(
            {
                "id": f"{image_id}_poly_{len(polygons) + 1:04d}",
                "score": polygon_score,
                "score_cls": (
                    float(raw["score_cls"]) if raw.get("score_cls") is not None else None
                ),
                "bbox_xywh": _bbox_xywh(
                    raw.get("bbox"), points, raw_index, image_width, image_height
                ),
                "points": points,
                "metadata": {
                    "raw_index": raw_index,
                    "category_id": raw.get("category_id"),
                    "area": raw.get("area"),
                },
            }
        )
    return polygons
