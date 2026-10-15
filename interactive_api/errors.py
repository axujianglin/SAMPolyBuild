"""Stable protocol errors for the interactive refinement API."""

from __future__ import annotations

from typing import Any


class InteractiveAPIError(Exception):
    """An expected client-visible error with a stable code and status."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = int(status_code)
        self.details = dict(details or {})


def error_payload(error: InteractiveAPIError) -> dict[str, Any]:
    return {
        "error": {
            "code": error.code,
            "message": error.message,
            "details": error.details,
        }
    }
