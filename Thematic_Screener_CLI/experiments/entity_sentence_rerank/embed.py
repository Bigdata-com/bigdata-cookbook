"""Embedding-based reranking with OpenAI text-embedding-3-small."""

from __future__ import annotations

import math
from typing import Any

from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    embeddings: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings.extend(item.embedding for item in response.data)
    return embeddings


def add_embedding_scores(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add embedding_relevance = cosine(extracted_sentence, search_query)."""
    client = OpenAI()
    unique_queries = sorted({str(row.get("search_query") or "") for row in records})
    query_embeddings = _embed_texts(client, unique_queries)
    query_map = dict(zip(unique_queries, query_embeddings, strict=True))

    sentence_texts = [str(row.get("extracted_sentence") or "") for row in records]
    sentence_embeddings = _embed_texts(client, sentence_texts)

    scored: list[dict[str, Any]] = []
    for row, sentence_embedding in zip(records, sentence_embeddings, strict=True):
        query = str(row.get("search_query") or "")
        query_embedding = query_map[query]
        search_relevance = float(row.get("search_relevance") or 0.0)
        sentiment = float(row.get("sentiment") or 0.0)
        scored.append(
            {
                **row,
                "embedding_relevance": round(
                    _cosine_similarity(sentence_embedding, query_embedding),
                    6,
                ),
                "evidence_score": round(search_relevance * abs(sentiment), 6),
            }
        )
    return scored
