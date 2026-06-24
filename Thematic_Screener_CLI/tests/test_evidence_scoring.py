from __future__ import annotations

from src.mcp_workflow import (
    MAX_TOOL_RESPONSE_BYTES,
    _cap_tool_response,
    _chunk_evidence_score,
)


def test_chunk_evidence_score_uses_relevance_times_abs_sentiment() -> None:
    assert _chunk_evidence_score(0.2, -0.5) == 0.1
    assert _chunk_evidence_score(0.2, 0.5) == 0.1


def test_chunk_evidence_score_returns_zero_for_missing_values() -> None:
    assert _chunk_evidence_score(None, -0.5) == 0.0
    assert _chunk_evidence_score(0.2, "invalid") == 0.0


def test_cap_tool_response_trims_large_lists() -> None:
    payload = {
        "run_id": "demo",
        "stage": "artifact_access",
        "status": "completed",
        "summary": "preview",
        "rows": [{"text": "x" * 50_000} for _ in range(40)],
    }
    capped = _cap_tool_response(payload)
    assert len(capped["rows"]) < len(payload["rows"])
    assert capped.get("response_truncated") is True
    assert _payload_size(capped) <= MAX_TOOL_RESPONSE_BYTES


def _payload_size(payload: dict[str, object]) -> int:
    import json

    return len(json.dumps(payload, default=str).encode("utf-8"))
