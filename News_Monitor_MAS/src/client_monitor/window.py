"""15-minute (configurable) ISO timestamp windows for retrieval and MAS."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass(frozen=True)
class TimeWindow:
    """Inclusive UTC window ``[start, end]`` for Bigdata timestamp filters."""

    start: datetime
    end: datetime
    minutes: int

    @property
    def start_iso(self) -> str:
        return _to_iso(self.start)

    @property
    def end_iso(self) -> str:
        return _to_iso(self.end)


def _to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_window_end(value: str | None) -> datetime:
    """Parse ``--window-end`` as ISO datetime; default is now (UTC)."""
    if not value:
        return datetime.now(UTC)
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def build_time_window(*, window_end: datetime, window_minutes: int) -> TimeWindow:
    """Build ``[end - minutes, end]`` in UTC."""
    if window_minutes <= 0:
        msg = f"window_minutes must be positive, got {window_minutes}"
        raise ValueError(msg)
    end = window_end.astimezone(UTC)
    start = end - timedelta(minutes=window_minutes)
    return TimeWindow(start=start, end=end, minutes=window_minutes)


def build_time_window_from_range(*, window_start: datetime, window_end: datetime) -> TimeWindow:
    """Build a window from explicit UTC start and end timestamps."""
    start = window_start.astimezone(UTC)
    end = window_end.astimezone(UTC)
    if end < start:
        msg = f"window_end must be >= window_start, got {end.isoformat()} < {start.isoformat()}"
        raise ValueError(msg)
    minutes = max(int((end - start).total_seconds() // 60), 1)
    return TimeWindow(start=start, end=end, minutes=minutes)


def parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO datetime string as UTC."""
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
