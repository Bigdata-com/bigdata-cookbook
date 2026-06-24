# Claude Skill: Guided Thematic Screener MCP

Use this skill when a user wants to run a thematic screen through the MCP.

## Operating Principles

- Work stage by stage.
- **Human-in-the-loop is mandatory** at mindmap confirmation, retrieval budget approval, and
  enrichment budget approval.
- Do not retrieve documents until the user approves a budget.
- Do not build search plans until the user confirms the taxonomy.
- Treat the taxonomy/mindmap as the main quality lever.
- Treat retrieved evidence as useful even if labeling is skipped.
- Keep chat responses concise and return artifact handles for full data.
- Never paste large CSV or JSON outputs into chat.
- Artifacts over 1MB must not be loaded inline. Use `query_artifact` with filters,
  paginated `get_artifact_preview`, or `export_artifact` for the local file path.

## Stage 1: Intent Capture

Collect or infer:

- Main theme.
- Analyst focus.
- Company universe.
- Date range.
- Desired output depth.

If important inputs are missing, propose reasonable defaults and ask only for the critical decision.

Call `create_run`, then confirm universe and dates if non-default.

Example:

```text
I will screen for "AI disruption in product development" over the default global all-caps
universe from 2025-06-01 to 2026-06-09. I will create a mindmap first and pause for your
confirmation before planning or spending retrieval budget.
```

## Stage 2: Mindmap Creation — STOP FOR USER

Call `generate_mindmap`.

Show:

- Branch count.
- Leaf label count.
- For each leaf: exposure **label** plus document-voice **search_query** preview.
- Any obvious concerns before validation.

Do not build search plans until the taxonomy is reviewed or validated.

## Stage 3: Mindmap Validation — STOP FOR USER

Call `validate_mindmap`.

Evaluate:

- Overlapping leaves.
- Vague exposure labels.
- Search queries that still use analyst meta-language (for example "exposed to", "benefiting from").
- Missing angles relevant to the analyst focus.
- Label count relative to the user budget.

If findings are material, recommend edits and ask whether to apply them via `update_mindmap`.

**Required gate:** ask the user to confirm the mindmap or request revisions. Do not proceed
until they respond explicitly, for example: "looks good", "approve", or specific edit instructions.

```text
Here is the proposed exposure taxonomy: 8 leaves across 4 branches.

Validation notes:
- Two leaves may overlap on satellite broadband vs aerospace competitors.
- One label may be too broad for a quick scan budget.

Please confirm this mindmap, or tell me what to change before I build search plans.
```

## Stage 4: Planning and Budget — STOP FOR USER

Only after mindmap confirmation, call `build_search_plans`, then `estimate_retrieval_budget`.

Present budget options using chunks and dollars:

```text
Planning found 184,000 expected chunks.

Quick scan: 0.5%, about 920 chunks, about $13.80
Balanced: 2.0%, about 3,680 chunks, about $55.20
Deep dive: 5.0%, about 9,200 chunks, about $138.00
Custom: choose a chunk count, percentage, or dollar cap
```

The cost formula is:

```text
estimated_cost_usd = selected_chunks * 0.015
```

**Required gate:** do not call `approve_budget` or `run_retrieval` until the user picks a budget.

## Stage 5: Retrieval

After budget approval, call `approve_budget`, then `run_retrieval`.

Summarize status by stage:

- Search plans executed.
- Documents retrieved.
- Chunks retrieved.
- Deduplication complete.

Avoid exposing raw logs unless the user asks.

## Stage 6: Evidence Digest

Always call `summarize_retrieval` after retrieval.

The evidence digest is the default stopping point when the user does not need full labeling.

Show:

- Top signals.
- Representative chunks.
- Source mix.
- Top companies by evidence count.
- Artifact handles for drill-down.

Ask what to do next:

```text
Would you like to review the evidence digest, run optional enrichment (labeling + company
summaries), or stop here?
```

## Stage 7: Optional Enrichment — STOP FOR USER

When the user wants taxonomy-based labels and company summaries, call
`estimate_enrichment_budget` before spending on OpenAI.

Present cost and latency like retrieval budget:

```text
Enrichment covers 104 labeling requests and about 31 company summaries
(57 companies in retrieved evidence).

Labeling: about $0.009 (31k input / 19k output tokens, 1.5× safety margin)
Summaries: about $0.004 (18k input / 4k output tokens, 1.5× safety margin)
Total: about $0.013; latency about 28 seconds
```

The default cost model is token-based:

```text
estimated_cost = (
  input_tokens * input_usd_per_mtok + output_tokens * output_usd_per_mtok
) / 1_000_000 * safety_margin
```

Default pricing assumes `gpt-5.4-nano`-like rates ($0.10 input / $0.40 output per 1M tokens)
with a 1.5× safety margin. Summary company count applies a 55% retention factor to the
companies found in retrieved evidence, prioritizing companies with more supporting sentences.
If `labeled_sentences.csv` already exists, summary estimates use the actual labeled company
payloads instead.

Latency uses a wave-based concurrency model measured on `gpt-5.4-nano` labeling:

```text
summary_seconds_per_request = 3 + (input_chars / 1000) * 0.15
labeling_total_seconds = max(rpm_bound, ceil(sentence_count / 40) * 4)
mcp_batch_seconds = max(rpm_bound, ceil(batch_size / 40) * 30)   # straggler per wave
```

Default concurrency is **40** in-flight requests. Batch sizing targets **≤90s estimated**
wall time using a **30s straggler-per-wave** model (p99/max tails, not p95). Measured batches
at size **120** typically finish in **~10–15s**, but stragglers can reach **~60s** on larger
batches — hence the conservative cap.

**Required gate:** do not call `approve_enrichment` or `run_enrichment` until the user approves
the enrichment budget.

After approval:

- **Claude Desktop default:** call `run_labeling` repeatedly until labeling completes (~120
  sentences per call at default settings), then `run_company_summaries` once.
- Each `run_labeling` call processes one MCP-safe batch. After timeouts, check
  `labeling_progress.last_batch_elapsed_seconds`, `last_batch_latency_max_seconds`, and
  `processed_sentences` before retrying.
- **Small/fast runs only:** `run_enrichment` may be used when estimated latency is well under
  the MCP tool timeout.

`run_enrichment` runs labeling and company summaries in one blocking MCP call. Claude Desktop
often times out on long labeling even when artifacts are written successfully. If any enrichment
tool times out, call `get_run_summary` first to check `enrichment_status` before retrying.

Explain that enrichment:

- Costs additional LLM time (shown in the estimate).
- Can be rerun with a revised taxonomy without repeating retrieval.
- Uses the persisted `analyst_focus` as mandatory scope during labeling; mechanism matches without
  scope support should become `unclear`, not a labeled hit.
- Adds a generic `materiality` field (`high`, `medium`, `low`, `unclear`).

When presenting company results, prioritize high-materiality evidence, then medium-materiality
evidence. Mention low-materiality rows as weak/indirect exposure rather than treating them as core
findings.

Each company summary must be grounded in retrieved/labeled evidence. If the user asks why a company
appears in the result, retrieve the supporting artifact rows instead of guessing.

Legacy split tools `run_labeling` and `run_company_summaries` are the **normal MCP path** after
enrichment approval. Prefer them over `run_enrichment` unless the enrichment estimate is clearly
under the MCP timeout.

If enrichment fails partway through, report the failed stage and retry the smallest resumable step
(`run_labeling` if summaries did not start, otherwise `run_company_summaries`).

## Follow-Up Retrieval

Use artifact tools for follow-up questions:

- `list_artifacts` to show available outputs.
- `get_artifact_preview` for small samples.
- `query_artifact` for company, label, source, date, or text filters.
- `export_artifact` for full data.

Examples:

```text
User: Show me the Autodesk evidence.
Action: query_artifact with company_name = "Autodesk Inc."
```

```text
User: Give me the full screener CSV.
Action: export_artifact with artifact_id = "screener_results"
```

## Failure Handling

If a stage fails:

- Report the failed stage.
- Keep the run ID visible.
- Mention which artifacts already exist.
- Suggest the smallest resumable next action.

Example:

```text
Retrieval completed, but evidence digest generation failed. The raw results artifact is available,
so I can retry `summarize_retrieval` without rerunning retrieval.
```

```text
`run_labeling` timed out after 4 minutes. Call `get_run_summary` and inspect
`enrichment_status`. If `labeled_sentences.csv` exists or `labeling_complete` is true, continue
with `run_company_summaries` or call `run_labeling` again only if labeling is incomplete.
```
