"""Ensure the demo stack does not pull in OpenAI client libraries."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def pyproject_deps() -> str:
    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps: list[str] = list(data["project"]["dependencies"])
    optional = data["project"].get("optional-dependencies") or {}
    for group in optional.values():
        deps.extend(group)
    return "\n".join(deps).lower()


def test_pyproject_has_no_openai_packages(pyproject_deps: str) -> None:
    assert "openai" not in pyproject_deps
    assert "langchain-openai" not in pyproject_deps


def test_uv_lock_has_no_openai_wheel() -> None:
    lock_text = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")
    # uv uses TOML [[package]] name = "..."
    assert not re.search(r'name = "openai"', lock_text)
    assert not re.search(r'name = "langchain-openai"', lock_text)
