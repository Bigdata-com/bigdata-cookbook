"""ADK entrypoint and agent naming."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_root_agent_name_and_model(agent_module) -> None:
    assert hasattr(agent_module, "root_agent")
    assert agent_module.root_agent.name == "financial_agent"
    # ADK Agent exposes configured model id on the public ``model`` field.
    assert getattr(agent_module.root_agent, "model", None) == agent_module.MODEL


def test_research_documents_include_samples(agent_module) -> None:
    assert len(agent_module.RESEARCH_DOCUMENTS) >= 3
    sources = {
        d.metadata.get("source_file")
        for d in agent_module.RESEARCH_DOCUMENTS
        if d.metadata.get("source_file")
    }
    assert "nvda_investment_thesis.md" in sources


def test_sample_markdown_dir_exists() -> None:
    sample_dir = PROJECT_ROOT / "financial_agent" / "sample_documents"
    assert sample_dir.is_dir()
    assert any(sample_dir.glob("*.md"))
