from typing import Dict, List

import pandas as pd



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
        - Entity columns: entity_ids, detections (list of dicts with id, start, end, type)
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
            # No chunks => no entities detected => skip this document (no row)
            continue

        for chunk in chunks:
            # Only detections with type "entity" (exclude topic etc.)
            detections = chunk.get("detections", [])
            entity_detections = [d for d in detections if d.get("type") == "entity"]
            entity_ids = [d["id"] for d in entity_detections]

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
                # Entity: ids and full detections (id, start, end, type)
                "entity_ids": entity_ids,
                "detections": entity_detections,
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
