# MCP Tool Contracts

These contracts describe the intended MCP surface for a Claude-driven Thematic Screener.
They are designed around resumable stages and summary-first responses.

## Shared Conventions

Every tool response should include:

- `run_id`: stable ID for the workflow run.
- `stage`: current workflow stage.
- `status`: one of `created`, `pending_approval`, `running`, `completed`, `failed`.
- `summary`: short chat-safe explanation.
- `artifacts`: optional list of artifact handles.
- `next_actions`: suggested user-facing actions.

Large data should never be returned inline. Return artifact handles and previews instead.

## Universe Inputs

The MCP should accept multiple universe input modes:

```json
{
  "mode": "default_global_all_caps"
}
```

```json
{
  "mode": "default_europe_ml_caps"
}
```

```json
{
  "mode": "sample_xnas"
}
```

```json
{
  "mode": "csv_path",
  "path": "/path/to/companies.csv"
}
```

```json
{
  "mode": "inline_entity_ids",
  "entity_ids": ["7F3A5F", "ABC123"]
}
```

CSV universes should accept `RP_ENTITY_ID` or `RP_COMPANY_ID` as the entity ID column and
`COMPANY_NAME` as an optional name column. The MCP should normalize the result before planning and
persist `universe_manifest.json`.

## `create_run`

Create a run brief and initialize run metadata.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "main_theme": "AI disruption in product development",
  "analyst_focus": "How companies are including AI in their development cycle",
  "universe": {
    "mode": "default_global_all_caps"
  },
  "start_date": "2025-06-01",
  "end_date": "2026-06-09",
  "output_goal": "Rank exposed companies with supporting evidence"
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "intent_capture",
  "status": "created",
  "summary": "Run created for AI disruption in product development over the default global all-caps universe.",
  "universe_summary": {
    "mode": "default_global_all_caps",
    "row_count": 10001,
    "id_column": "RP_COMPANY_ID",
    "name_column": "COMPANY_NAME"
  },
  "artifacts": [
    {
      "artifact_id": "universe_manifest",
      "path": "universe_manifest.json",
      "preview_available": true
    }
  ],
  "next_actions": ["generate_mindmap"]
}
```

## `validate_universe`

Validate and normalize the selected universe before planning.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "universe": {
    "mode": "inline_entity_ids",
    "entity_ids": ["7F3A5F", "ABC123"]
  }
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "intent_capture",
  "status": "completed",
  "summary": "Validated 2 unique RP entity IDs for the run universe.",
  "universe_summary": {
    "mode": "inline_entity_ids",
    "row_count": 2,
    "id_column": "RP_ENTITY_ID",
    "name_column": null,
    "warnings": []
  },
  "artifacts": [
    {
      "artifact_id": "universe_manifest",
      "path": "universe_manifest.json",
      "preview_available": true
    },
    {
      "artifact_id": "normalized_universe",
      "path": "universe.csv",
      "preview_available": true
    }
  ],
  "next_actions": ["generate_mindmap", "build_search_plans"]
}
```

## `generate_mindmap`

Generate and store a taxonomy tree for the main theme.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "max_leaf_labels": 12,
  "style": "analyst_screening"
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "mindmap_creation",
  "status": "completed",
  "summary": "Generated a taxonomy with 5 branches and 12 leaf labels.",
  "mindmap_preview": {
    "root_label": "AI disruption in product development",
    "branch_count": 5,
    "leaf_count": 12
  },
  "artifacts": [
    {
      "artifact_id": "taxonomy_tree",
      "path": "taxonomy_tree.json",
      "preview_available": true
    },
    {
      "artifact_id": "themes",
      "path": "themes.txt",
      "preview_available": true
    }
  ],
  "next_actions": ["validate_mindmap", "update_mindmap"]
}
```

## `validate_mindmap`

Check taxonomy quality before any retrieval planning.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "checks": ["overlap", "searchability", "focus_alignment", "budget_risk"]
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "mindmap_validation",
  "status": "completed",
  "summary": "Validation found 2 overlapping leaves and 1 vague search label.",
  "findings": [
    {
      "severity": "medium",
      "check": "overlap",
      "message": "Developer tooling and code generation overlap heavily.",
      "suggested_action": "Merge into AI-assisted development tooling."
    }
  ],
  "next_actions": ["update_mindmap", "build_search_plans"]
}
```

## `update_mindmap`

Apply user edits to the taxonomy.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "instructions": "Merge developer tooling and code generation. Keep under 10 leaf labels."
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "mindmap_revision",
  "status": "completed",
  "summary": "Updated taxonomy now has 4 branches and 9 leaf labels.",
  "next_actions": ["validate_mindmap", "build_search_plans"]
}
```

## `build_search_plans`

Build search plans without retrieval.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "category": {
    "mode": "INCLUDE",
    "values": ["news_premium", "transcripts", "filings"]
  }
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "planning",
  "status": "completed",
  "summary": "Built 9 search plans. Retrieval has not started.",
  "artifacts": [
    {
      "artifact_id": "plans",
      "path": "plans/",
      "preview_available": true
    }
  ],
  "next_actions": ["estimate_retrieval_budget"]
}
```

## `estimate_retrieval_budget`

Estimate chunks and cost before retrieval.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "retrieval_cost_usd_per_10_chunks": 0.015,
  "presets": [
    {"name": "quick_scan", "chunk_percentage": 0.005},
    {"name": "balanced", "chunk_percentage": 0.02},
    {"name": "deep_dive", "chunk_percentage": 0.05}
  ]
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "budget_preview",
  "status": "pending_approval",
  "summary": "Planning found 184,000 expected chunks before sampling.",
  "total_expected_chunks": 184000,
  "retrieval_cost_usd_per_10_chunks": 0.015,
  "retrieval_chunks_per_cost_unit": 10,
  "presets": [
    {
      "name": "quick_scan",
      "chunk_percentage": 0.005,
      "selected_chunks": 920,
      "estimated_cost_usd": 1.38
    },
    {
      "name": "balanced",
      "chunk_percentage": 0.02,
      "selected_chunks": 3680,
      "estimated_cost_usd": 5.52
    },
    {
      "name": "deep_dive",
      "chunk_percentage": 0.05,
      "selected_chunks": 9200,
      "estimated_cost_usd": 13.8
    }
  ],
  "next_actions": ["approve_budget"]
}
```

## `approve_budget`

Freeze retrieval budget before spend.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "selection": {
    "preset": "balanced"
  }
}
```

Alternative custom request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "selection": {
    "max_cost_usd": 75.0
  }
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "budget_approval",
  "status": "completed",
  "summary": "Approved balanced retrieval: about 3,680 chunks and $5.52.",
  "approved_budget": {
    "chunk_percentage": 0.02,
    "selected_chunks": 3680,
    "estimated_cost_usd": 5.52
  },
  "next_actions": ["run_retrieval"]
}
```

## `run_retrieval`

Execute approved plans.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "requests_per_minute": 350
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "retrieval",
  "status": "completed",
  "summary": "Retrieved 3,420 chunks across 810 deduplicated documents.",
  "retrieval_stats": {
    "document_count": 810,
    "chunk_count": 3420
  },
  "artifacts": [
    {
      "artifact_id": "results",
      "path": "results/results.json",
      "preview_available": true
    }
  ],
  "next_actions": ["summarize_retrieval"]
}
```

## `summarize_retrieval`

Required lightweight synthesis over retrieved chunks. This does not replace the optional labeling
pipeline; it makes retrieval-only mode useful.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "max_representative_chunks": 30,
  "group_by": ["theme", "company", "source_category"]
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "evidence_digest",
  "status": "completed",
  "summary": "Evidence digest created with top signals, representative chunks, and source mix.",
  "digest_preview": {
    "top_signals": [
      "Companies describe AI-assisted design and engineering workflows.",
      "Earnings transcripts emphasize productivity and product cycle acceleration."
    ],
    "representative_chunk_count": 30
  },
  "artifacts": [
    {
      "artifact_id": "evidence_digest",
      "path": "evidence_digest.json",
      "preview_available": true
    }
  ],
  "next_actions": ["get_run_summary", "estimate_enrichment_budget"]
}
```

## `estimate_enrichment_budget`

Estimate OpenAI cost and latency for the merged enrichment step (labeling + company summaries).

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "labeling_model": "gpt-5.4-nano",
  "summary_model": "gpt-5.4-nano",
  "requests_per_minute": 10000,
  "max_concurrent_requests": 40,
  "labeling_input_usd_per_mtok": 0.10,
  "labeling_output_usd_per_mtok": 0.40,
  "summary_input_usd_per_mtok": 0.10,
  "summary_output_usd_per_mtok": 0.40,
  "safety_margin": 1.5
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "enrichment_preview",
  "status": "pending_approval",
  "summary": "Enrichment covers 104 labeling requests and about 31 company summaries (57 companies in retrieved evidence). Estimated cost $0.013; latency about 28 seconds.",
  "enrichment_budget": {
    "sentence_count": 104,
    "raw_company_count": 57,
    "estimated_summary_company_count": 31,
    "labeling_input_tokens": 31200,
    "labeling_output_tokens": 18720,
    "summary_input_tokens": 18000,
    "summary_output_tokens": 3720,
    "estimated_labeling_cost_usd": 0.009,
    "estimated_summary_cost_usd": 0.004,
    "estimated_total_cost_usd": 0.013,
    "estimated_total_seconds": 28,
    "estimated_latency_human": "about 28 seconds",
    "estimation_method": "token_based_with_safety_margin",
    "safety_margin": 1.5
  },
  "artifacts": [
    {
      "artifact_id": "enrichment_summary",
      "path": "enrichment_summary.json",
      "preview_available": true
    }
  ],
  "next_actions": ["approve_enrichment"]
}
```

## `approve_enrichment`

Persist user approval for enrichment cost and latency.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "approved": true
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "enrichment_approval",
  "status": "completed",
  "summary": "Approved enrichment: $0.13 estimated cost, about 23 seconds estimated latency.",
  "next_actions": ["run_enrichment"]
}
```

## `run_enrichment`

Run labeling and company summaries after enrichment approval.

Request:

```json
{
  "run_id": "ai_product_dev_20260618"
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "enrichment",
  "status": "completed",
  "summary": "Labeled 98 sentences and generated 11 company summaries.",
  "artifacts": [
    {
      "artifact_id": "labeled_sentences",
      "path": "labeled_sentences.csv",
      "preview_available": true
    },
    {
      "artifact_id": "company_summaries",
      "path": "company_summaries.csv",
      "preview_available": true
    },
    {
      "artifact_id": "screener_results",
      "path": "screener_results.csv",
      "preview_available": true
    }
  ],
  "next_actions": ["get_run_summary", "query_artifact", "export_artifact"]
}
```

## `run_labeling`

Legacy split step: taxonomy-based sentence/chunk labeling. Requires enrichment approval.

Each retained row in `labeled_sentences.csv` includes `materiality`, a generic score for the
strength of business exposure:

- `high`: direct revenue, cost, valuation, capex, supply-chain, regulatory, or operational impact.
- `medium`: clear exposure through a relevant role, but without strong magnitude evidence.
- `low`: indirect customer/adopter/proxy/market-sentiment exposure.
- `unclear`: not enough evidence; these rows are filtered out of retained labeled results.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "model": "gpt-5.4-nano",
  "requests_per_minute": 10000,
  "max_concurrent_requests": 40
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "labeling",
  "status": "completed",
  "summary": "Labeled retrieved chunks and wrote labeled_sentences.csv.",
  "artifacts": [
    {
      "artifact_id": "labeled_sentences",
      "path": "labeled_sentences.csv",
      "preview_available": true
    }
  ],
  "next_actions": ["run_company_summaries", "get_run_summary"]
}
```

## `run_company_summaries`

Legacy split step: company-level synthesis after labeling. Requires enrichment approval.

Request:

```json
{
  "run_id": "ai_product_dev_20260618",
  "model": "gpt-5.4-nano",
  "source": "labeled_sentences"
}
```

Response:

```json
{
  "run_id": "ai_product_dev_20260618",
  "stage": "company_summaries",
  "status": "completed",
  "summary": "Generated company summaries and final screener results.",
  "artifacts": [
    {
      "artifact_id": "company_summaries",
      "path": "company_summaries.csv",
      "preview_available": true
    },
    {
      "artifact_id": "screener_results",
      "path": "screener_results.csv",
      "preview_available": true
    }
  ],
  "next_actions": ["get_run_summary", "query_artifact", "export_artifact"]
}
```

## Artifact Tools

### `get_run_summary`

Return a compact summary of the run state and most important findings.

### `list_artifacts`

Return artifact metadata, row counts, schemas, sizes, and preview availability.

### `get_artifact_preview`

Return a bounded preview for a given artifact.

### `query_artifact`

Filter large artifacts by fields such as `company_name`, `label`, `source_category`,
`document_date`, `sentence_id`, or `text_query`.

### `export_artifact`

Return an exported file path or downloadable handle for full artifacts.
