# Changelog

All notable changes to Thematic Screener CLI are documented in this file.

## [Unreleased] — 2026-08-24

### Added

- **Derivatives taxonomy style** (`--taxonomy-style derivatives`): 1st / 2nd / 3rd hop mindmaps with theme-level Bigdata.com grounding (`--ground-with-bigdata`), CLI/MCP flags, and `grounding.json` / `derivative_preview.json`
- **Client-facing notebook** `notebooks/01_derivative_thematic_screener.ipynb`: narrated end-to-end derivatives run on `tsx_top150_rp_entities.csv` (grounding → mindmap → planning → retrieval → labeling → narratives → exports) with six charts, writing to `runs/tsx_oil_derivatives/`
- **Client-facing notebook** `notebooks/02_derivative_eu_parcel_tariffs.ipynb`: same workflow on `europe_ml_caps.csv` for EU tariffs on low-value Chinese e-commerce parcels (AliExpress / Temu / Shein), writing to `runs/eu_parcel_tariff_derivatives/` (`RETRIEVAL_DEPTH = 0.1`)
- CLI `--labeling-concurrency`, `--labeling-rpm`, `--summary-concurrency`
- Jupyter optional dependency group (`uv sync --group jupyter`), now including `matplotlib`
- **`src/viz.py`**: derivative-hop charts (mindmap, hop coverage, pathway and company rankings, company × hop exposure matrix, evidence timeline) with empty-data fallbacks. `plot_top_companies` and `plot_company_hop_matrix` accept `hops=` (with `viz.INDIRECT_HOPS` for 2nd/3rd only) so later hops can be ranked on their own terms instead of being crowded out of the top-N by direct-exposure volume
- **`src/notebook_support.py`**: `quiet_output` context manager suppressing planner/retrieval progress chatter in notebooks
- `derivative_taxonomy.leaf_branch_map` mapping each taxonomy leaf to its derivative hop
- Unit tests (7) for chart builders, hop attribution, and quiet-output log restoration

### Fixed

- `build_plans` now deletes plan files left over from an earlier taxonomy in the same run directory. Re-running a run after the mindmap changed left retired plans on disk, and `run_search` executes every plan file it finds, so retrieval covered pathways absent from the current mindmap (affected the CLI and MCP server as well as notebooks)
- `export_excel` truncates cells over Excel's 32,767-character limit with an explicit marker instead of letting openpyxl cut them silently; full text remains in the CSV and JSON exports

### Changed

- Default OpenAI model for taxonomy, labeling, and summaries: **`gpt-5.6-luna`**
- Labeling concurrency default **40 → 80**; summary concurrency **20 → 40**; MCP wave estimates 6s avg / 40s straggler (luna, not re-measured)

## [Unreleased] — 2026-06-24

### Added

- **MCP server** (`thematic-screener-mcp`) with a staged, resumable analyst workflow:
  - Run lifecycle: `create_run`, `validate_universe`, `generate_mindmap`, `validate_mindmap`, `update_mindmap`
  - Retrieval: `build_search_plans`, `estimate_retrieval_budget`, `approve_budget`, `run_retrieval`, `summarize_retrieval`
  - Enrichment: `estimate_enrichment_budget`, `approve_enrichment`, `run_labeling`, `run_company_summaries`, `run_enrichment`
  - Recovery: `get_run_summary` with `enrichment_status` and labeling progress
  - Artifacts: `list_artifacts`, `get_artifact_preview`, `query_artifact`, `export_artifact`
- **MCP documentation** under `mcp/`: README, `claude-skill.md`, `tool-contracts.md`, `schemas.json`, `cli-stage-map.md`, `artifact-model.md`
- **Cursor skill** at `.cursor/skills/thematic-screener/SKILL.md` for human-in-the-loop MCP orchestration
- **uv project layout** (`pyproject.toml`, `uv.lock`) with entry points `screener` and `thematic-screener-mcp`
- **Taxonomy `search_query` split**: leaf labels (`themes.txt`) are separate from document-voice retrieval text (`search_queries.txt`, `taxonomy_tree.json`)
- **`src/search_query.py`**: exposure-meta detection and summary→query normalization heuristics
- **`scripts/benchmark_search_queries.py`**: A/B benchmark for taxonomy vs document-voice query phrasing (incl. `--suite xnas`)
- **`global_all_caps.csv`**: default production universe for MCP runs
- **Labeling `materiality`** field (`high` / `medium` / `low` / `unclear`) on enriched evidence rows
- **Enrichment budget model**: token-based cost/latency estimates with safety margin and split-vs-combined execution guidance
- **MCP-safe batched labeling**: resumable `run_labeling` with `labeling_progress.json` and append-only `labeling_responses.jsonl`
- **Batch timing telemetry**: per-batch `elapsed_seconds`, throughput, p50/p95/p99/max latency in progress files
- **Straggler-aware batch sizing**: 30s/wave straggler model, 90s MCP-safe target, max batch 120, auto-calibration from observed tails
- **Unit tests** (29): enrichment budget, labeling batch metrics, labeling prompt scope, taxonomy/search-query split, evidence scoring / response capping

### Changed

- **Mindmap prompts** (`src/prompts.py`): generic exposure-taxonomy rules; separate `label`, `summary`, and `search_query` fields; document-voice search guidance
- **Labeling prompt**: `analyst_focus` is a mandatory scope gate (mechanism match without scope → `unclear`)
- **Plan building** uses `search_queries.txt` for `plan_search`, not analyst labels
- **`helpers.py`**: `get_leaf_labels`, `get_leaf_search_queries`, `get_leaf_pairs`; tree printing shows query preview
- **`run_context.py`**: paths for `search_queries.txt` and `taxonomy_tree.json`
- **`screener.py`**: `Node.search_query`, `write_taxonomy_artifacts`, parallel labeling default concurrency **40**, optional `metrics_out` for batch telemetry
- **`openai_parallel.py`**: `ChatResponse.elapsed_seconds` and improved async/sync executor ergonomics
- **`cli.py`**: threads `analyst_focus` through `label-sentences`
- **Default enrichment concurrency**: 20 → **40** in MCP workflow and schemas
- **README**: MCP setup, Cursor skill pointer, uv quick start

### Removed

- Heatmap generation tools, artifacts, and related MCP surface area (legacy heatmap outputs may remain in old run folders)

### Fixed

- Retrieval budget used `$0.015` per chunk; pricing is **`$0.015` per 10 chunks** (`estimated_cost_usd = selected_chunks / 10 * 0.015`).
- MCP labeling `nest_asyncio` failure when invoked from Claude Desktop’s async tool context
- MCP tool timeouts on long enrichment: split labeling into resumable batches; defer `labeled_sentences.csv` until labeling completes
- Optimistic labeling latency estimates replaced with wave-based models calibrated from live benchmarks

### Verified

- CLI end-to-end smoke (`cli_smoke_20260624`): `generate-labels` → `plans` → `search` → `label-sentences` on 1-company universe
- All **29** pytest tests passing

### Not for commit (local artifacts)

- `runs/` — run outputs from smoke tests and live MCP sessions (exclude from git unless intentionally versioning fixtures)
