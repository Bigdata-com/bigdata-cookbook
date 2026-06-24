# Artifact Model

The MCP should treat artifacts as first-class objects. Claude should receive concise summaries and
artifact handles, while full data remains available through preview, query, pagination, and export
tools.

## Artifact Manifest

Each run should maintain an artifact manifest:

```json
{
  "run_id": "ai_product_dev_20260618",
  "artifacts": [
    {
      "artifact_id": "evidence_digest",
      "stage": "evidence_digest",
      "path": "evidence_digest.json",
      "content_type": "application/json",
      "created_at": "2026-06-18T15:30:00Z",
      "description": "Lightweight synthesis of retrieved chunks.",
      "preview_available": true,
      "queryable": true
    }
  ]
}
```

## Standard Metadata

Every artifact should expose:

- `artifact_id`: stable logical ID.
- `run_id`: owning run.
- `stage`: workflow stage that produced it.
- `path`: run-relative path.
- `content_type`: MIME-style type.
- `schema`: field definitions when tabular or structured.
- `size_bytes`: file size.
- `row_count`: tabular row count when applicable.
- `document_count`: document count when applicable.
- `chunk_count`: chunk count when applicable.
- `created_at`: ISO timestamp.
- `description`: short natural-language description.
- `preview_available`: whether bounded previews are supported.
- `queryable`: whether filtered retrieval is supported.
- `large_artifact`: true when `size_bytes` exceeds 1MB.
- `access_note`: guidance when inline previews should not be used.

## Core Artifacts

| Artifact ID | Path | Stage | Purpose |
| --- | --- | --- | --- |
| `run_config` | `config.json` | intent capture | Persisted run settings and selected defaults. |
| `universe_manifest` | `universe_manifest.json` | intent capture | Universe source mode, row count, normalized columns, and validation warnings. |
| `normalized_universe` | `universe.csv` | intent capture | Normalized company/entity universe used for planning. |
| `taxonomy_tree` | `taxonomy_tree.json` | mindmap creation | Full structured taxonomy tree. |
| `themes` | `themes.txt` | mindmap creation | Leaf exposure labels used for sentence labeling. |
| `search_queries` | `search_queries.txt` | mindmap creation | Document-voice retrieval text for Bigdata search planning. |
| `mindmap_validation` | `mindmap_validation.json` | validation | Findings and suggested taxonomy edits. |
| `plans` | `plans/*.json` | planning | Bigdata search plans. |
| `budget_summary` | `budget_summary.json` | budget preview | Expected chunks, presets, and estimated costs. |
| `budget_approval` | `budget_approval.json` | budget approval | Frozen selected retrieval budget. |
| `enrichment_summary` | `enrichment_summary.json` | enrichment preview | Estimated OpenAI cost and latency for labeling + summaries. |
| `enrichment_approval` | `enrichment_approval.json` | enrichment approval | Frozen approved enrichment budget. |
| `results` | `results/results.json` | retrieval | Deduplicated retrieved documents and chunks. |
| `evidence_digest` | `evidence_digest.json` | evidence digest | Retrieval-only synthesis of evidence. |
| `evidence_preview` | `evidence_preview.json` | evidence digest | Representative chunks and source mix for chat preview. |
| `labeled_sentences` | `labeled_sentences.csv` | enrichment | Optional taxonomy labels, revenue/cost fields, and materiality for chunks/sentences. |
| `company_summaries` | `company_summaries.csv` | enrichment | Optional company-level synthesis. |
| `screener_results` | `screener_results.csv` | enrichment | Optional final company screener output. |

## Preview Rules

Default preview limits:

- JSON: 20 top-level records or 12,000 characters.
- CSV: 20 rows and 30 columns.
- Text: 12,000 characters.
- Evidence chunks: 30 representative chunks.
- Images: export path and metadata; full image rendering is host-side.
- MCP inline responses: keep under 1MB. Large artifacts must use `query_artifact`,
  paginated previews, or `export_artifact`.

Preview responses should include:

```json
{
  "artifact_id": "labeled_sentences",
  "preview": [],
  "truncated": true,
  "next_cursor": "offset:20",
  "summary": "Previewing 20 of 3,420 labeled rows."
}
```

## Query Rules

Queryable artifacts should support field filters:

```json
{
  "artifact_id": "labeled_sentences",
  "filters": {
    "company_name": ["Autodesk Inc."],
    "label": ["AI-assisted product design"],
    "revenue_generation": ["medium", "high"],
    "materiality": ["medium", "high"]
  },
  "text_query": "design cycle acceleration",
  "limit": 25,
  "cursor": null
}
```

Responses should include matching rows, `next_cursor`, and a compact explanation of the filter.

## Evidence Digest Shape

The evidence digest is required after retrieval, even when labeling is skipped.

Recommended structure:

```json
{
  "run_id": "ai_product_dev_20260618",
  "retrieval_stats": {
    "document_count": 810,
    "chunk_count": 3420,
    "company_count": 145
  },
  "source_mix": [
    {"source_category": "news_premium", "document_count": 420},
    {"source_category": "transcripts", "document_count": 270},
    {"source_category": "filings", "document_count": 120}
  ],
  "top_signals": [
    {
      "signal": "AI-assisted engineering and design workflows",
      "summary": "Companies discuss AI as a way to accelerate product design and development.",
      "supporting_chunk_count": 118,
      "representative_chunk_ids": ["chunk_001", "chunk_017"]
    }
  ],
  "top_companies": [
    {
      "company_name": "Autodesk Inc.",
      "evidence_count": 32,
      "document_count": 18,
      "relevance_score": 4.82,
      "summary": "Evidence focuses on AI features embedded in design and engineering tools.",
      "representative_chunk_ids": ["chunk_042", "chunk_057"]
    }
  ],
  "representative_chunks": [
    {
      "chunk_id": "chunk_042",
      "company_name": "Autodesk Inc.",
      "source_category": "transcripts",
      "document_id": "doc_123",
      "relevance": 0.28,
      "sentiment": -0.84,
      "score": 0.2352,
      "text": "Representative excerpt..."
    }
  ]
}
```

## Export Rules

Export tools should return paths or handles, not inline file contents:

```json
{
  "artifact_id": "screener_results",
  "export_path": "runs/ai_product_dev_20260618/screener_results.csv",
  "content_type": "text/csv",
  "size_bytes": 1234567
}
```

## Provenance

Preserve provenance wherever possible:

- `document_id`
- `chunk_id`
- `source_category`
- `document_date`
- `company_id`
- `company_name`
- `theme_label`
- `sentence_id`
- `retrieval_plan_id`

Claude should be able to answer follow-up questions like: "Show me the evidence behind Autodesk"
or "Why did this company rank highly?" without rerunning retrieval.
