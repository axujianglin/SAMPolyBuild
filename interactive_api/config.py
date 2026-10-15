"""Environment-backed configuration for the independent interactive API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


@dataclass(frozen=True)
class InteractiveServiceConfig:
    api_key: str
    model_config: Path
    checkpoint: Path
    device: str = "cuda"
    host: str = "0.0.0.0"
    port: int = 8010
    max_upload_bytes: int = 100 * 1024 * 1024
    max_concurrency: int = 1
    session_ttl_seconds: int = 1800
    max_sessions: int = 4

    @classmethod
    def from_env(cls) -> "InteractiveServiceConfig":
        api_key = os.getenv("SAMPOLY_API_KEY", "").strip()
        return cls(
            api_key=api_key,
            model_config=Path(
                os.getenv("SAMPOLY_CONFIG", "configs/prompt_instance_spacenet.json")
            ),
            checkpoint=Path(os.getenv("SAMPOLY_CHECKPOINT", "prompt_interactive.pth")),
            device=os.getenv("SAMPOLY_DEVICE", "cuda").strip() or "cuda",
            host=os.getenv("SAMPOLY_HOST", "0.0.0.0").strip() or "0.0.0.0",
            port=_positive_int("SAMPOLY_PORT", 8010),
            max_upload_bytes=_positive_int("SAMPOLY_MAX_UPLOAD_MB", 100) * 1024 * 1024,
            max_concurrency=_positive_int("SAMPOLY_MAX_CONCURRENCY", 1),
            session_ttl_seconds=_positive_int("SAMPOLY_SESSION_TTL_SECONDS", 1800),
            max_sessions=_positive_int("SAMPOLY_MAX_SESSIONS", 4),
        )

    def validate_model_files(self) -> None:
        if not self.model_config.is_file():
            raise FileNotFoundError(
                f"SAMPoly interactive model configuration was not found: {self.model_config}"
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"SAMPoly interactive checkpoint was not found: {self.checkpoint}"
            )
