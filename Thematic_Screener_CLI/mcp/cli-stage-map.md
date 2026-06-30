# Existing CLI to MCP Stage Map

This map shows how the current Thematic Screener CLI can be exposed through the proposed MCP
workflow without changing the core pipeline semantics.

## Current CLI Stages

The CLI currently exposes:

- `generate-labels`: generate leaf theme labels from a main theme and analyst focus.
- `plans`: build one Bigdata search plan per theme.
- `summarize-plans`: print expected chunk counts without retrieval.
- `search`: execute plans and write deduplicated search results.
- `label-sentences`: label retrieved chunks, summarize companies, and write final CSV outputs.
- `run-all`: run all stages in sequence.

## MCP Stage Mapping

| MCP stage | MCP tool | Existing implementation anchor | Notes |
| --- | --- | --- | --- |
| Intent capture | `create_run` | `RunContext.create`, `config.json` | Add run brief fields without changing CLI behavior. MCP default should be `global_all_caps.csv`; CLI smoke-test default can remain `XNAS_companies.csv`. |
| Universe validation | `validate_universe` | `screener.load_universe` plus new normalization layer | Accept `global_all_caps.csv`, `europe_ml_caps.csv`, `XNAS_companies.csv`, CSV paths/uploads, or inline `RP_ENTITY_ID` values. Normalize aliases before planning. |
| Mindmap creation | `generate_mindmap` | `screener.generate_labels` | Persists taxonomy tree, leaf labels (`themes.txt`), and search queries (`search_queries.txt`). |
| Mindmap validation | `validate_mindmap` | New validation layer | Validation is product logic around the generated taxonomy. |
| Mindmap revision | `update_mindmap` | New revision layer | Can update `taxonomy_tree.json` and regenerate `themes.txt` and `search_queries.txt`. |
| Planning | `build_search_plans` | `screener.build_plans` | Uses `search_queries.txt` for retrieval text; plan filenames derive from leaf labels. |
| Budget preview | `estimate_retrieval_budget` | `screener.summarize_plans`, `format_plans_summary` | Extend summary with presets and cost at `$0.015` per 10 chunks. |
| Budget approval | `approve_budget` | `config.json` | Persist selected chunk percentage, chunk cap, or dollar cap. |
| Retrieval | `run_retrieval` | `screener.run_search` | Same retrieval, using approved budget only. |
| Evidence digest | `summarize_retrieval` | New digest layer over `results/results.json` | Required even when labeling is skipped. |
| Enrichment preview | `estimate_enrichment_budget` | `extract_sentences`, `summarize_companies` request counts | Estimate OpenAI cost and latency for labeling + summaries. |
| Enrichment approval | `approve_enrichment` | New approval artifact | Persist approved enrichment budget before OpenAI spend. |
| Enrichment | `run_labeling` + `run_company_summaries` | split labeling/summary steps | Preferred MCP path; avoids ~4 minute Claude Desktop tool timeouts. |
| Enrichment (combined) | `run_enrichment` | `label_sentences`, `summarize_companies`, `build_screener_dataframe` | Small/fast runs only; may time out in Claude Desktop. |
| Labeling (legacy) | `run_labeling` | `extract_sentences`, `label_sentences`, `build_labeled_dataframe` | First split step after enrichment approval. |
| Company summaries (legacy) | `run_company_summaries` | `summarize_companies`, `build_screener_dataframe` | Second split step after labeling. |
| Artifact access | `list_artifacts`, `get_artifact_preview`, `query_artifact`, `export_artifact` | `RunContext` paths plus new manifest | Add bounded retrieval for large outputs. |

## Current Artifact Paths

Existing `RunContext` paths:

```text
runs/<run_name>/
  config.json
  themes.txt
  plans/
  results/results.json
  labeled_sentences.csv
  company_summaries.csv
  screener_results.csv
```

Recommended MCP additions:

```text
runs/<run_name>/
  universe.csv
  universe_manifest.json
  taxonomy_tree.json
  mindmap_validation.json
  budget_summary.json
  budget_approval.json
  enrichment_summary.json
  enrichment_approval.json
  evidence_digest.json
  evidence_preview.json
  artifact_manifest.json
```

## Implementation Notes

- Keep `run-all` for CLI users, but the MCP should prefer gated stage-by-stage execution.
- Do not allow `run_retrieval` unless `budget_approval.json` exists.
- Do not allow `run_enrichment` unless `enrichment_approval.json` exists.
- `summarize_retrieval` should run immediately after retrieval and should be safe to rerun.
- `run_enrichment` should not require retrieval to run again if `results/results.json` exists.
- Legacy split tools remain for partial retries after enrichment approval.
- Artifact tools should use bounded previews by default to protect Claude context.

## Cost Calculation

Use constants:

```text
RETRIEVAL_CHUNKS_PER_COST_UNIT = 10
RETRIEVAL_COST_USD_PER_UNIT = 0.015
```

For each budget preset:

```text
selected_chunks = round(total_expected_chunks * chunk_percentage)
estimated_cost_usd = (selected_chunks / 10) * 0.015
```

For a dollar cap:

```text
selected_chunks = floor(max_cost_usd * 10 / 0.015)
chunk_percentage = selected_chunks / total_expected_chunks
```

## Enrichment Cost Calculation

Use token-based pricing with a safety margin:

```text
DEFAULT_LABELING_INPUT_USD_PER_MTOK = 0.10
DEFAULT_LABELING_OUTPUT_USD_PER_MTOK = 0.40
DEFAULT_SUMMARY_INPUT_USD_PER_MTOK = 0.10
DEFAULT_SUMMARY_OUTPUT_USD_PER_MTOK = 0.40
ENRICHMENT_COST_SAFETY_MARGIN = 1.5
MIN_ESTIMATED_COST_USD = 0.01
```

```text
estimated_cost = (
  input_tokens * input_usd_per_mtok + output_tokens * output_usd_per_mtok
) / 1_000_000 * safety_margin
```

Summary company count uses a 55% retention factor over companies found in retrieved evidence,
ranked by sentence volume. If `labeled_sentences.csv` already exists, summary estimates use
actual labeled payloads.

Latency uses separate wave-based models for labeling and summaries:

- Total labeling: `ceil(sentence_count / 40) * 4s` average per concurrency wave
- MCP batch budget: `ceil(batch_size / 40) * 30s` straggler per wave (p99/max; target ≤90s)
- Summaries: `3s + 0.15s per 1k input chars`, same 40-request concurrency cap

Each completed `run_labeling` batch writes timing to `labeling_progress.json`
(`last_batch_elapsed_seconds`, `last_batch_latency_p99_seconds`,
`last_batch_latency_max_seconds`, `labeling_straggler_seconds`).

## Recommended Stage Statuses

- `created`: run exists, no work has started.
- `completed`: stage completed successfully.
- `pending_approval`: stage is waiting for user approval.
- `running`: long-running stage is active.
- `failed`: stage failed but run artifacts should remain resumable.
