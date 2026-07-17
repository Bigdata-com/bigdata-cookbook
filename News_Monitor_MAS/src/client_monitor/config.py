"""Default search configuration for the client news monitor."""

from __future__ import annotations

from typing import Any

NEWS_PREMIUM_CATEGORY: dict[str, Any] = {
    "mode": "INCLUDE",
    "values": ["news_premium"],
}

NEWS_CATEGORY: dict[str, Any] = {
    "mode": "INCLUDE",
    "values": ["news"],
}

SEARCH_CATEGORIES: dict[str, dict[str, Any]] = {
    "news_premium": NEWS_PREMIUM_CATEGORY,
    "news": NEWS_CATEGORY,
}

DEFAULT_SEARCH_CATEGORY: dict[str, Any] = NEWS_CATEGORY
DEFAULT_CATEGORY_PROFILE = "news"


def resolve_search_category(profile: str) -> tuple[str, dict[str, Any]]:
    """Resolve a category profile name to Bigdata ``filters.category`` payload."""
    normalized = profile.strip().lower()
    if normalized not in SEARCH_CATEGORIES:
        msg = f"unknown category profile: {profile!r} (expected news_premium or news)"
        raise ValueError(msg)
    return normalized, SEARCH_CATEGORIES[normalized]
