"""Small OpenAI wrapper shared by the FX Forecast Report modules.

Provides plain-text and Pydantic-structured completions. The model is read from
`OPENAI_MODEL` (default `gpt-4o-mini`); the key from `OPENAI_API_KEY`.
"""

from __future__ import annotations

import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

DEFAULT_MODEL = "gpt-4o-mini"

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Thin synchronous OpenAI client for text and structured completions."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.model = model or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        self._client = OpenAI(api_key=key)

    def complete_text(
        self, system: str, user: str, temperature: float = 0.2
    ) -> str:
        """Return a plain-text completion."""
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def complete_structured(
        self, system: str, user: str, schema: type[T], temperature: float = 0.1
    ) -> T:
        """Return a completion parsed into a Pydantic model (structured output)."""
        response = self._client.beta.chat.completions.parse(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=schema,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned no structured output.")
        return parsed
