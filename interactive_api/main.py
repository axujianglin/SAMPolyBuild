"""Command-line entry point for the interactive refinement API."""

from __future__ import annotations

import uvicorn

from .config import InteractiveServiceConfig


def main() -> None:
    config = InteractiveServiceConfig.from_env()
    uvicorn.run(
        "interactive_api.app:app",
        host=config.host,
        port=config.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
