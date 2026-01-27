from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from bigdata_client.document import Document
from bigdata_client.models.document import (
    DocumentChunk,
    DocumentSentenceEntity,
    DocumentSentence,
    DocumentSource,
    DocumentScope,
)
from bigdata_client.query_type import QueryType


def convert_smart_batching_to_documents(raw_results: List[Dict]) -> List[Document]:
    """
    Convert Smart_batching output (list of dicts) to Document objects.
    
    Args:
        raw_results: List of raw documents from Smart_batching execute_search()
        
    Returns:
        List of Document objects compatible with ThematicScreener
    """
    documents = []
    
    for raw_doc in raw_results:
        # Convert chunks
        chunks = []
        for raw_chunk in raw_doc.get("chunks", []):
            # Convert detections to DocumentSentenceEntity
            entities = []
            for detection in raw_chunk.get("detections", []):
                query_type = QueryType.ENTITY if detection.get("type") == "entity" else QueryType.TOPIC
                entity = DocumentSentenceEntity(
                    key=detection.get("id", ""),
                    start=detection.get("start", 0),
                    end=detection.get("end", 0),
                    query_type=query_type
                )
                entities.append(entity)
            
            chunk = DocumentChunk(
                text=raw_chunk.get("text", ""),
                chunk=raw_chunk.get("cnum", 0),
                entities=entities,
                sentences=[],  # Not available in Smart_batching
                relevance=raw_chunk.get("relevance", 0.0),
                sentiment=raw_chunk.get("sentiment", 0.0),
                section_metadata=None,
                speaker=None
            )
            chunks.append(chunk)
        
        # Convert source
        raw_source = raw_doc.get("source", {})
        source = DocumentSource(
            key=raw_source.get("id", ""),
            name=raw_source.get("name", "Unknown"),
            rank=_parse_rank(raw_source.get("rank", "RANK_1"))
        )
        
        # Convert timestamp
        timestamp = _parse_timestamp(raw_doc.get("timestamp", ""))
        
        # Calculate average document sentiment
        doc_sentiment = _calculate_doc_sentiment(chunks)
        
        # Create Document
        doc = Document(
            id=raw_doc.get("id", ""),
            headline=raw_doc.get("headline", ""),
            sentiment=doc_sentiment,
            document_scope=DocumentScope.NEWS,
            source=source,
            timestamp=timestamp,
            chunks=chunks,
            language="English",
            cluster=None,
            reporting_period=None,
            document_type=None,
            reporting_entities=raw_doc.get("reporting_entities"),
            url=raw_doc.get("url")
        )
        documents.append(doc)
    
    return documents


def _parse_rank(rank_str) -> int:
    """Convert 'RANK_1' -> 0, 'RANK_2' -> 1, etc."""
    if isinstance(rank_str, int):
        return rank_str
    if rank_str and isinstance(rank_str, str) and rank_str.startswith("RANK_"):
        try:
            return int(rank_str.split("_")[1]) - 1
        except (IndexError, ValueError):
            return 0
    return 0


def _parse_timestamp(ts_str: str) -> datetime:
    """Convert timestamp string to datetime with UTC timezone."""
    if not ts_str:
        return datetime.now(timezone.utc)
    try:
        # Format: "2021-01-04T14:32:06" or "2021-01-04T14:32:06Z"
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return datetime.now(timezone.utc)


def _calculate_doc_sentiment(chunks: List[DocumentChunk]) -> float:
    """Calculate average document sentiment from chunks."""
    sentiments = [c.sentiment for c in chunks if c.sentiment is not None]
    if sentiments:
        return sum(sentiments) / len(sentiments)
    return 0.0


def convert_to_dataframe(raw_results: List[Dict]) -> pd.DataFrame:
    """
    Convert Smart_batching output to a DataFrame exploded by chunk.
    
    Each document is expanded into multiple rows, one per chunk.
    
    Args:
        raw_results: List of raw documents from Smart_batching execute_search()
        
    Returns:
        DataFrame with one row per chunk, containing:
        - Document columns: date, doc_id, headline, source_id, source_name, source_rank
        - Chunk columns: chunk_index, chunk_text, chunk_relevance, chunk_sentiment
        - Entity columns: entity_ids
        - Final columns: url, reporting_entities
    """
    rows = []
    
    for raw_doc in raw_results:
        # Extract document info
        doc_id = raw_doc.get("id", "")
        headline = raw_doc.get("headline", "")
        timestamp = raw_doc.get("timestamp", "")
        url = raw_doc.get("url", "")
        reporting_entities = raw_doc.get("reporting_entities", [])
        
        # Extract source info
        raw_source = raw_doc.get("source", {})
        source_id = raw_source.get("id", "")
        source_name = raw_source.get("name", "")
        source_rank = raw_source.get("rank", "")
        
        # Explode chunks
        chunks = raw_doc.get("chunks", [])
        
        if not chunks:
            # If no chunks, still create a row with empty values
            rows.append({
                # Document
                "date": timestamp,
                "doc_id": doc_id,
                "headline": headline,
                "source_id": source_id,
                "source_name": source_name,
                "source_rank": source_rank,
                # Chunk (empty)
                "chunk_index": None,
                "chunk_text": "",
                "chunk_relevance": None,
                "chunk_sentiment": None,
                # Entity
                "entity_ids": [],
                # Final
                "url": url,
                "reporting_entities": reporting_entities,
            })
        else:
            for chunk in chunks:
                # Entity IDs (added by execute_search)
                entity_ids = chunk.get("entity_ids", [])
                
                rows.append({
                    # Document
                    "date": timestamp,
                    "doc_id": doc_id,
                    "headline": headline,
                    "source_id": source_id,
                    "source_name": source_name,
                    "source_rank": source_rank,
                    # Chunk
                    "chunk_index": chunk.get("cnum"),
                    "chunk_text": chunk.get("text", ""),
                    "chunk_relevance": chunk.get("relevance"),
                    "chunk_sentiment": chunk.get("sentiment"),
                    # Entity
                    "entity_ids": entity_ids,
                    # Final
                    "url": url,
                    "reporting_entities": reporting_entities,
                })
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Convert date to Date (date only, no time)
    if "date" in df.columns and not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    
    return df
