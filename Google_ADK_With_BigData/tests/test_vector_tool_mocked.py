"""Vector search without calling Gemini embedding API."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings


class _FakeEmbeddings(Embeddings):
    """LangChain-compatible fake so FAISS builds and queries without calling Gemini."""

    _dim: int = 16

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float((i + j) % 7) / 10.0 for j in range(self._dim)] for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        return [0.25] * self._dim


def _clear_financial_agent_modules() -> None:
    for name in list(sys.modules):
        if name == "financial_agent" or name.startswith("financial_agent."):
            del sys.modules[name]


@pytest.fixture
def agent_with_fake_embeddings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FINANCIAL_AGENT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FINANCIAL_AGENT_SKIP_EMBED_WARMUP", "1")
    monkeypatch.delenv("BIGDATA_API_KEY", raising=False)

    _clear_financial_agent_modules()

    import financial_agent.agent as ag

    importlib.reload(ag)
    monkeypatch.setattr(ag, "GoogleGenerativeAIEmbeddings", lambda **_kw: _FakeEmbeddings())  # type: ignore[misc]
    ag._vector_store = None
    return ag


def test_internal_search_returns_documents(agent_with_fake_embeddings) -> None:
    ag = agent_with_fake_embeddings
    out = ag.internal_search_research("NVIDIA investment thesis", top_k=2)
    assert "error" not in out, out
    assert out["results_count"] >= 1
    assert "documents" in out
    first = out["documents"][0]
    assert "content" in first
    assert len(first["content"]) > 0
