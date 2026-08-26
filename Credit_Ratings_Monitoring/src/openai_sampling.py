"""OpenAI sampling helpers for model-specific API constraints."""

from __future__ import annotations

from typing import Any


def sampling_params_for_model(model: str, **params: Any) -> dict[str, Any]:
    """Return sampling kwargs that ``model`` accepts.

    ``gpt-5.6-luna`` only supports default sampling, so temperature, top_p,
    seed, and penalty fields are omitted for luna models.
    """
    if "luna" in model.lower():
        return {}
    return {key: value for key, value in params.items() if value is not None}
