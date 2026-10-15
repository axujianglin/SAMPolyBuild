"""Adapter from protocol prompts to the existing SAMPoly predictor."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from tools.point_refine_utils import extract_main_polygon_from_mask
from tools.prompt_inference_utils import build_prompt_predictor, load_prompt_settings

from .config import InteractiveServiceConfig
from .errors import InteractiveAPIError


@dataclass
class PreparedImage:
    """CPU-resident image features and transform metadata for one session."""

    embedding: Any
    original_size: tuple[int, int]
    input_size: tuple[int, int]


@dataclass
class PredictionResult:
    polygon: list[list[float]]
    bbox_xywh: list[float]
    score: float
    mask_logits: np.ndarray
    selection_strategy: str


class InteractivePredictorAdapter:
    """Own one model and safely swap cached session embeddings into its predictor."""

    def __init__(self, config: InteractiveServiceConfig) -> None:
        self.config = config
        self.settings = None
        self.predictor = None
        self.device = None
        self._initialized = False
        self._capacity = threading.BoundedSemaphore(config.max_concurrency)
        self._model_lock = threading.Lock()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        self.config.validate_model_files()
        self.settings = load_prompt_settings(
            self.config.model_config,
            checkpoint=self.config.checkpoint,
        )
        self.predictor, self.device = build_prompt_predictor(
            self.settings,
            device=self.config.device,
        )
        self._initialized = True

    def _enter_capacity(self) -> None:
        if not self._capacity.acquire(blocking=False):
            raise InteractiveAPIError(
                "service_busy",
                "The interactive inference service is busy. Please retry shortly.",
                429,
            )

    def prepare_image(self, image_rgb: np.ndarray) -> PreparedImage:
        if not self._initialized or self.predictor is None:
            raise InteractiveAPIError(
                "model_unavailable", "The interactive model is unavailable.", 503
            )
        self._enter_capacity()
        try:
            with self._model_lock:
                self.predictor.set_image(image_rgb, image_format="RGB")
                embedding = self.predictor.get_image_embedding().detach().to("cpu").clone()
                prepared = PreparedImage(
                    embedding=embedding,
                    original_size=tuple(self.predictor.original_size),
                    input_size=tuple(self.predictor.input_size),
                )
                self.predictor.reset_image()
                return prepared
        finally:
            self._capacity.release()

    def predict(
        self,
        *,
        prepared_image: PreparedImage,
        bbox: tuple[float, float, float, float],
        positive_points: tuple[tuple[float, float], ...],
        negative_points: tuple[tuple[float, float], ...],
        original_polygon: tuple[tuple[float, float], ...],
        previous_logits: Any = None,
    ) -> PredictionResult:
        if not self._initialized or self.predictor is None:
            raise InteractiveAPIError(
                "model_unavailable", "The interactive model is unavailable.", 503
            )
        self._enter_capacity()
        try:
            with self._model_lock:
                self._restore_prepared_image(prepared_image)
                point_coords = np.asarray(
                    positive_points + negative_points, dtype=np.float32
                )
                point_labels = np.asarray(
                    [1] * len(positive_points) + [0] * len(negative_points),
                    dtype=np.int32,
                )
                mask_input = None
                if previous_logits is not None:
                    mask_input = np.asarray(previous_logits, dtype=np.float32)
                outputs = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=np.asarray(bbox, dtype=np.float32),
                    mask_input=mask_input,
                    multimask_output=bool(getattr(self.settings, "multi_mask", True)),
                )
                if not isinstance(outputs, tuple) or len(outputs) not in (3, 4):
                    raise InteractiveAPIError(
                        "invalid_model_output",
                        "SAMPoly returned an unexpected prediction structure.",
                        500,
                    )
                masks, scores, low_res_logits = outputs[:3]
                result = self._select_and_vectorize(
                    masks=np.asarray(masks),
                    scores=np.asarray(scores),
                    low_res_logits=np.asarray(low_res_logits),
                    bbox=bbox,
                    positive_points=positive_points,
                    negative_points=negative_points,
                    original_polygon=original_polygon,
                    image_size=prepared_image.original_size,
                )
                self.predictor.reset_image()
                return result
        except InteractiveAPIError:
            raise
        except Exception as exc:
            raise InteractiveAPIError(
                "inference_failed", "SAMPoly interactive inference failed.", 500
            ) from exc
        finally:
            self._capacity.release()

    def _restore_prepared_image(self, prepared: PreparedImage) -> None:
        embedding = prepared.embedding
        if hasattr(embedding, "to"):
            embedding = embedding.to(self.predictor.device)
        self.predictor.reset_image()
        self.predictor.features = embedding
        self.predictor.original_size = tuple(prepared.original_size)
        self.predictor.input_size = tuple(prepared.input_size)
        self.predictor.is_image_set = True

    @staticmethod
    def _select_and_vectorize(
        *,
        masks: np.ndarray,
        scores: np.ndarray,
        low_res_logits: np.ndarray,
        bbox: tuple[float, float, float, float],
        positive_points: tuple[tuple[float, float], ...],
        negative_points: tuple[tuple[float, float], ...],
        original_polygon: tuple[tuple[float, float], ...],
        image_size: tuple[int, int],
    ) -> PredictionResult:
        height, width = image_size
        if masks.ndim != 3 or masks.shape[0] == 0:
            raise InteractiveAPIError(
                "invalid_model_output", "SAMPoly returned no valid masks.", 500
            )
        if scores.ndim != 1 or scores.shape[0] != masks.shape[0]:
            raise InteractiveAPIError(
                "invalid_model_output", "SAMPoly returned invalid mask scores.", 500
            )
        if low_res_logits.ndim != 3 or low_res_logits.shape[0] != masks.shape[0]:
            raise InteractiveAPIError(
                "invalid_model_output", "SAMPoly returned invalid mask logits.", 500
            )

        original_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(
            original_mask,
            [np.rint(np.asarray(original_polygon)).astype(np.int32)],
            1,
        )
        x1, y1, x2, y2 = bbox
        bx1, by1 = max(0, int(np.floor(x1))), max(0, int(np.floor(y1)))
        bx2, by2 = min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2)))

        candidates: list[tuple[tuple[float, ...], int, np.ndarray, str]] = []
        for mask_index, raw_mask in enumerate(masks):
            binary = (raw_mask > 0).astype(np.uint8)
            component_count, labels = cv2.connectedComponents(binary, connectivity=8)
            for component_id in range(1, component_count):
                component = (labels == component_id).astype(np.uint8)
                area = int(component.sum())
                if area <= 0:
                    continue
                positive_hits = sum(
                    _point_in_mask(component, point) for point in positive_points
                )
                negative_hits = sum(
                    _point_in_mask(component, point) for point in negative_points
                )
                original_intersection = int(np.logical_and(component, original_mask).sum())
                original_union = int(np.logical_or(component, original_mask).sum())
                original_iou = (
                    original_intersection / original_union if original_union else 0.0
                )
                bbox_overlap = int(component[by1:by2, bx1:bx2].sum()) / area
                model_score = float(scores[mask_index])
                rank = (
                    float(positive_hits),
                    float(-negative_hits),
                    float(original_iou),
                    float(bbox_overlap),
                    model_score,
                    float(area),
                )
                strategy = (
                    "positive_negative_prompts"
                    if negative_points
                    else "positive_prompts_overlap"
                )
                candidates.append((rank, mask_index, component, strategy))

        if not candidates:
            raise InteractiveAPIError(
                "invalid_model_output", "SAMPoly returned an empty building mask.", 500
            )
        rank, mask_index, selected_mask, strategy = max(candidates, key=lambda item: item[0])
        if rank[0] <= 0:
            raise InteractiveAPIError(
                "invalid_model_output",
                "No predicted component contains a positive prompt point.",
                500,
            )

        polygon, metadata = extract_main_polygon_from_mask(
            selected_mask,
            positive_points=[(int(round(x)), int(round(y))) for x, y in positive_points],
        )
        if polygon is None or len(polygon) < 3:
            raise InteractiveAPIError(
                "invalid_model_output", "The selected mask could not form a polygon.", 500
            )
        points = _normalize_output_polygon(polygon, width, height)
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        bbox_xywh = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
        if bbox_xywh[2] <= 0 or bbox_xywh[3] <= 0:
            raise InteractiveAPIError(
                "invalid_model_output", "The predicted polygon has an invalid extent.", 500
            )
        score = float(scores[mask_index])
        if not np.isfinite(score):
            raise InteractiveAPIError(
                "invalid_model_output", "SAMPoly returned a non-finite score.", 500
            )
        score = float(np.clip(score, 0.0, 1.0))

        logits = np.asarray(low_res_logits[mask_index], dtype=np.float32)
        component_low_res = cv2.resize(
            selected_mask,
            (logits.shape[1], logits.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        next_logits = np.where(component_low_res > 0, logits, -32.0).astype(np.float32)[None]
        if not np.isfinite(next_logits).all():
            raise InteractiveAPIError(
                "invalid_model_output", "SAMPoly returned non-finite mask logits.", 500
            )
        if metadata.get("reason"):
            strategy = f"{strategy}:{metadata['reason']}"
        return PredictionResult(points, bbox_xywh, score, next_logits, strategy)

    def close(self) -> None:
        with self._model_lock:
            if self.predictor is not None:
                self.predictor.reset_image()
            self.predictor = None
            self.settings = None
            self.device = None
            self._initialized = False
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def _point_in_mask(mask: np.ndarray, point: tuple[float, float]) -> int:
    x = min(max(int(round(point[0])), 0), mask.shape[1] - 1)
    y = min(max(int(round(point[1])), 0), mask.shape[0] - 1)
    return int(mask[y, x] > 0)


def _normalize_output_polygon(
    polygon: list[list[float]], width: int, height: int
) -> list[list[float]]:
    points = np.asarray(polygon, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise InteractiveAPIError(
            "invalid_model_output", "SAMPoly returned invalid polygon coordinates.", 500
        )
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    points[:, 0] = np.clip(points[:, 0], 0.0, float(width - 1))
    points[:, 1] = np.clip(points[:, 1], 0.0, float(height - 1))
    if len({(float(x), float(y)) for x, y in points}) < 3:
        raise InteractiveAPIError(
            "invalid_model_output", "SAMPoly returned fewer than three polygon points.", 500
        )
    return [[float(x), float(y)] for x, y in points]
