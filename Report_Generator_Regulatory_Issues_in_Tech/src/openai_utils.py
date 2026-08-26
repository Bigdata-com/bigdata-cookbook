"""OpenAI helpers for migrated cookbooks."""

from __future__ import annotations

from typing import Any


def resolve_model(llm_model: str) -> str:
    """Return the bare OpenAI model id (strip optional ``provider::`` prefix)."""
    return llm_model.split("::")[-1] if "::" in llm_model else llm_model


def sampling_params_for_model(model: str, **params: Any) -> dict[str, Any]:
    """Return sampling kwargs that ``model`` accepts.

    ``gpt-5.6-luna`` only supports default sampling, so temperature, top_p,
    seed, and penalty fields are omitted for luna models.
    """
    if "luna" in model.lower():
        return {}
    return {key: value for key, value in params.items() if value is not None}


def completion_token_params_for_model(model: str, max_tokens: int) -> dict[str, int]:
    """Return the correct max-token parameter name for ``model``."""
    if "luna" in model.lower():
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}
