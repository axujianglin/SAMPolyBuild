"""Thread-safe bounded session and interaction-state storage."""

from __future__ import annotations

import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from .errors import InteractiveAPIError


@dataclass
class InteractionState:
    interaction_token: str
    polygon_id: str
    successful_revision: int
    bbox: tuple[float, float, float, float]
    positive_points: tuple[tuple[float, float], ...]
    negative_points: tuple[tuple[float, float], ...]
    original_polygon: tuple[tuple[float, float], ...]
    mask_logits: Any


@dataclass
class SessionEntry:
    session_id: str
    image_id: str
    file_name: str
    width: int
    height: int
    prepared_image: Any
    created_at: float
    last_accessed_at: float
    expires_at: float
    interaction_states: dict[str, InteractionState] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def release(self) -> None:
        if hasattr(self.prepared_image, "crop_cache"):
            self.prepared_image.crop_cache.clear()
            self.prepared_image.image_rgb = None
        self.prepared_image = None
        for state in self.interaction_states.values():
            state.mask_logits = None
        self.interaction_states.clear()


class SessionStore:
    """Bounded in-memory store using monotonic time and sliding idle TTL."""

    def __init__(
        self,
        ttl_seconds: int,
        max_sessions: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_seconds = int(ttl_seconds)
        self.max_sessions = int(max_sessions)
        self._clock = clock
        self._sessions: dict[str, SessionEntry] = {}
        self._expired_ids: dict[str, float] = {}
        self._lock = threading.RLock()

    def _prune_tombstones_locked(self, now: float) -> None:
        cutoff = now - self.ttl_seconds
        stale = [key for key, expired_at in self._expired_ids.items() if expired_at < cutoff]
        for key in stale:
            self._expired_ids.pop(key, None)
        while len(self._expired_ids) > max(32, self.max_sessions * 4):
            self._expired_ids.pop(next(iter(self._expired_ids)))

    def _expire_locked(self, session_id: str, entry: SessionEntry, now: float) -> None:
        self._sessions.pop(session_id, None)
        self._expired_ids[session_id] = now
        entry.release()
        self._prune_tombstones_locked(now)

    def cleanup_expired(self) -> int:
        now = self._clock()
        expired: list[SessionEntry] = []
        with self._lock:
            for session_id, entry in list(self._sessions.items()):
                if now >= entry.expires_at and entry.lock.acquire(blocking=False):
                    try:
                        if self._sessions.get(session_id) is entry:
                            self._sessions.pop(session_id, None)
                            self._expired_ids[session_id] = now
                            expired.append(entry)
                    finally:
                        entry.lock.release()
            self._prune_tombstones_locked(now)
        for entry in expired:
            entry.release()
        return len(expired)

    def create(
        self,
        *,
        image_id: str,
        file_name: str,
        width: int,
        height: int,
        prepared_image: Any,
    ) -> SessionEntry:
        self.cleanup_expired()
        now = self._clock()
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise InteractiveAPIError(
                    "session_limit_reached",
                    "The maximum number of active image sessions has been reached.",
                    429,
                )
            session_id = secrets.token_urlsafe(32)
            while session_id in self._sessions:
                session_id = secrets.token_urlsafe(32)
            entry = SessionEntry(
                session_id=session_id,
                image_id=image_id,
                file_name=file_name,
                width=int(width),
                height=int(height),
                prepared_image=prepared_image,
                created_at=now,
                last_accessed_at=now,
                expires_at=now + self.ttl_seconds,
            )
            self._sessions[session_id] = entry
            return entry

    def _get_locked(self, session_id: str, now: float) -> SessionEntry:
        entry = self._sessions.get(session_id)
        if entry is None:
            if session_id in self._expired_ids:
                raise InteractiveAPIError(
                    "session_expired", "The image session has expired.", 410
                )
            raise InteractiveAPIError(
                "session_not_found", "The image session was not found.", 404
            )
        if now >= entry.expires_at:
            self._expire_locked(session_id, entry, now)
            raise InteractiveAPIError(
                "session_expired", "The image session has expired.", 410
            )
        return entry

    @contextmanager
    def locked_session(self, session_id: str) -> Iterator[SessionEntry]:
        now = self._clock()
        with self._lock:
            entry = self._get_locked(session_id, now)
        with entry.lock:
            with self._lock:
                entry = self._get_locked(session_id, self._clock())
            yield entry

    def touch(self, entry: SessionEntry) -> None:
        now = self._clock()
        with self._lock:
            if self._sessions.get(entry.session_id) is not entry:
                raise InteractiveAPIError(
                    "session_not_found", "The image session was not found.", 404
                )
            entry.last_accessed_at = now
            entry.expires_at = now + self.ttl_seconds

    def delete(self, session_id: str) -> bool:
        with self._lock:
            entry = self._sessions.get(session_id)
        if entry is None:
            return False
        with entry.lock:
            with self._lock:
                removed = self._sessions.pop(session_id, None)
                self._expired_ids.pop(session_id, None)
            if removed is not None:
                removed.release()
                return True
        return False

    def clear(self) -> None:
        with self._lock:
            entries = list(self._sessions.values())
            self._sessions.clear()
            self._expired_ids.clear()
        for entry in entries:
            with entry.lock:
                entry.release()

    @property
    def active_count(self) -> int:
        self.cleanup_expired()
        with self._lock:
            return len(self._sessions)
