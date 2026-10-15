"""Model-preserving service wrapper for one-image SAMPoly Auto inference."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .auto_result_adapter import adapt_auto_results


class AutoServiceError(RuntimeError):
    """Structured error that the HTTP adapter can map to a stable response."""

    def __init__(self, code: str, message: str, status_code: int = 500):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)


@dataclass(frozen=True)
class AutoServiceConfig:
    """Filesystem, device, and request limits for the Auto API service."""

    model_config: Path
    checkpoint: Path
    runtime_root: Path
    device: str = "cuda"
    gpu: int = 0
    batch_size: int = 1
    num_workers: int = 1
    default_score_threshold: float = 0.1
    max_upload_bytes: int = 100 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_config", Path(self.model_config).expanduser().resolve())
        object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser().resolve())
        object.__setattr__(self, "runtime_root", Path(self.runtime_root).expanduser().resolve())
        normalized_device = str(self.device).strip().lower()
        if normalized_device not in ("cuda", "cpu"):
            raise ValueError("device must be 'cuda' or 'cpu'.")
        object.__setattr__(self, "device", normalized_device)
        if self.gpu < 0:
            raise ValueError("gpu must be non-negative.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        if not 0.0 <= float(self.default_score_threshold) <= 1.0:
            raise ValueError("default_score_threshold must be between 0 and 1.")
        if self.max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive.")


class AutoInferenceService:
    """Serialize GPU calls while reusing the existing Auto inference function."""

    def __init__(
        self,
        config: AutoServiceConfig,
        infer_runner: Callable[..., dict[str, Any]] | None = None,
        cuda_probe: Callable[[], bool] | None = None,
    ) -> None:
        self.config = config
        self._infer_runner = infer_runner
        self._cuda_probe = cuda_probe
        self._inference_lock = threading.Lock()

    @property
    def busy(self) -> bool:
        """Return whether another request currently owns the GPU inference lock."""
        return self._inference_lock.locked()

    def _probe_cuda(self) -> tuple[bool, str | None]:
        """Probe CUDA lazily so health checks remain independent from model imports."""
        try:
            if self._cuda_probe is not None:
                return bool(self._cuda_probe()), None
            import torch

            return bool(torch.cuda.is_available()), None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def health(self) -> dict[str, Any]:
        """Return readiness without constructing the SAMPoly runner."""
        config_exists = self.config.model_config.is_file()
        checkpoint_exists = self.config.checkpoint.is_file()
        cuda_available, cuda_error = self._probe_cuda()
        device_ready = self.config.device == "cpu" or cuda_available
        ready = config_exists and checkpoint_exists and device_ready
        return {
            "status": "ready" if ready else "not_ready",
            "service": "sampoly-auto-inference",
            "config_exists": config_exists,
            "checkpoint_exists": checkpoint_exists,
            "cuda_available": cuda_available,
            "cuda_error": cuda_error,
            "device": (
                "cpu" if self.config.device == "cpu" else f"cuda:{self.config.gpu}"
            ),
            "busy": self.busy,
            "inference_lifecycle": "runner_per_request",
        }

    def _validate_ready(self) -> None:
        """Raise explicit service errors for missing runtime prerequisites."""
        health = self.health()
        if not health["config_exists"]:
            raise AutoServiceError(
                "config_not_found",
                f"Auto config not found: {self.config.model_config}",
                503,
            )
        if not health["checkpoint_exists"]:
            raise AutoServiceError(
                "checkpoint_not_found",
                f"Auto checkpoint not found: {self.config.checkpoint}",
                503,
            )
        if self.config.device == "cuda" and not health["cuda_available"]:
            detail = f" ({health['cuda_error']})" if health["cuda_error"] else ""
            raise AutoServiceError(
                "cuda_unavailable",
                f"CUDA is unavailable inside the service container{detail}.",
                503,
            )

    def _resolve_runner(self) -> Callable[..., dict[str, Any]]:
        """Import the existing model runner only when a request needs inference."""
        if self._infer_runner is None:
            from infer_auto import run_auto_inference

            self._infer_runner = run_auto_inference
        return self._infer_runner

    @staticmethod
    def _read_image_size(image_path: Path) -> tuple[int, int]:
        """Read dimensions without copying the uploaded image into memory twice."""
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            raise AutoServiceError(
                "invalid_image",
                f"Unable to read uploaded image: {type(exc).__name__}: {exc}",
                400,
            ) from exc
        if width <= 0 or height <= 0:
            raise AutoServiceError("invalid_image", "Uploaded image has invalid dimensions.", 400)
        return int(width), int(height)

    @staticmethod
    def _load_raw_results(results_path: Path) -> list[dict[str, Any]]:
        """Load the existing results artifact, treating absence as no detections."""
        if not results_path.is_file():
            return []
        try:
            with results_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as exc:
            raise AutoServiceError(
                "invalid_inference_results",
                f"Auto inference produced invalid JSON: {exc}",
            ) from exc
        if not isinstance(payload, list):
            raise AutoServiceError(
                "invalid_inference_results", "Auto inference results root must be a list."
            )
        return payload

    def infer(
        self,
        image_path: Path,
        *,
        request_id: str,
        original_file_name: str,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Run one uploaded image through the unchanged Auto prediction entry point."""
        self._validate_ready()
        image_path = Path(image_path).expanduser().resolve()
        if not image_path.is_file():
            raise AutoServiceError("image_not_found", f"Uploaded image not found: {image_path}", 400)
        threshold = (
            float(self.config.default_score_threshold)
            if score_threshold is None
            else float(score_threshold)
        )
        if not 0.0 <= threshold <= 1.0:
            raise AutoServiceError(
                "invalid_score_threshold", "score_threshold must be between 0 and 1.", 400
            )
        width, height = self._read_image_size(image_path)
        result_dir = image_path.parent.parent / "results"
        work_dir = image_path.parent.parent / "work"

        try:
            with self._inference_lock:
                artifacts = self._resolve_runner()(
                    config=str(self.config.model_config),
                    ckpt_path=str(self.config.checkpoint),
                    img_dir=str(image_path.parent),
                    work_dir=str(work_dir),
                    img_suffix=image_path.suffix.lower(),
                    score_thr=threshold,
                    gpu=self.config.gpu,
                    batch_size=self.config.batch_size,
                    num_workers=self.config.num_workers,
                    status="predict",
                    result_dir=result_dir,
                    replace_predict_loader=True,
                    device=self.config.device,
                )
        except AutoServiceError:
            raise
        except Exception as exc:
            raise AutoServiceError(
                "inference_failed", f"{type(exc).__name__}: {exc}"
            ) from exc

        results_path = Path(artifacts.get("results_path", result_dir / "results.json"))
        raw_results = self._load_raw_results(results_path)
        image_id = image_path.stem
        try:
            polygons = adapt_auto_results(
                raw_results,
                image_id=image_id,
                image_width=width,
                image_height=height,
                score_threshold=threshold,
            )
        except ValueError as exc:
            raise AutoServiceError("invalid_inference_results", str(exc)) from exc
        return {
            "success": True,
            "request_id": request_id,
            "image": {
                "file_name": original_file_name,
                "image_id": image_id,
                "width": width,
                "height": height,
            },
            "coordinate_space": "original_image_pixel",
            "polygons": polygons,
            "summary": {
                "polygon_count": len(polygons),
                "score_threshold": threshold,
            },
        }
