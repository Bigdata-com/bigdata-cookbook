# Claude MCP Workflow UX

This folder describes a Claude-native MCP experience for the Thematic Screener workflow.
It is a product and implementation scaffold for turning the current staged CLI into a
resumable analyst workflow exposed through MCP tools.

## Goals

- Make the workflow feel like a guided analyst conversation instead of a black-box batch run.
- Validate the mindmap/taxonomy before any retrieval spend.
- Estimate retrieval cost before search execution.
- Treat retrieved chunks as a usable deliverable through an evidence digest.
- Keep taxonomy-based labeling and company summaries optional.
- Return compact summaries in chat and expose full results through artifact handles.

## Workflow

1. **Intent capture**
   - Capture the main theme, analyst focus, company universe, date range, and output goal.
   - Normalize those inputs into a short run brief.
   - Default the company universe to `global_all_caps.csv` for production-style Claude runs,
     while keeping `XNAS_companies.csv` useful for small smoke tests.

2. **Mindmap creation**
   - Claude uses a skill to create a structured taxonomy/mindmap.
   - The MCP stores the full tree and returns a compact preview.
   - The user can edit the taxonomy before planning.

3. **Mindmap validation**
   - Validate duplicate or overlapping leaves.
   - Flag leaves that are not independently searchable.
   - Check alignment with the analyst focus.
   - Warn when taxonomy size or vagueness increases budget risk.

4. **Planning and budget preview**
   - Build Bigdata search plans without retrieval.
   - Estimate total expected chunks and per-label chunks.
   - Estimate cost with:

     ```text
     estimated_cost_usd = (selected_chunks / 10) * 0.015
     ```

   - Present budget presets:
     - `quick_scan`: small sample for directional review.
     - `balanced`: default evidence coverage.
     - `deep_dive`: broader evidence coverage.
     - `full`: retrieve every planned chunk (100%).
     - `custom`: explicit chunk count, percentage, or dollar cap.

5. **Approved retrieval**
   - Retrieval only starts after explicit budget approval.
   - The selected budget is persisted so the run is auditable.

6. **Evidence digest**
   - After retrieval, always create an evidence digest.
   - The digest summarizes and organizes retrieved chunks without requiring full labeling.
   - It should include top signals, representative chunks, source mix, company mentions,
     and artifact handles for drill-down.

7. **Optional enrichment**
   - `estimate_enrichment_budget`: estimate OpenAI cost and latency for labeling + summaries.
   - `approve_enrichment`: persist user approval before spending on enrichment.
   - `run_labeling` then `run_company_summaries`: preferred MCP path (avoids ~4 minute tool timeouts).
   - `run_enrichment`: combined step for small/fast runs only.
   - These steps can be skipped, resumed, or rerun without repeating retrieval.

8. **Summary-first delivery**
   - Claude shows concise results in chat.
   - Full CSV/JSON outputs are retrieved by artifact ID, filters, pagination, or export.
   - Claude must pause for human confirmation at mindmap review and budget approval.

## Universe Selection

The MCP should treat the universe as a validated input artifact, not just a file path.

Supported universe modes:

- `default_global_all_caps`: use `global_all_caps.csv` from this project (~10k companies).
- `default_europe_ml_caps`: use `europe_ml_caps.csv` for European mid/large caps (~646 companies).
- `sample_xnas`: use `XNAS_companies.csv` for quick smoke tests.
- `csv_path`: use a user-provided CSV path.
- `uploaded_csv`: use a CSV uploaded through the host client, then persist a normalized copy.
- `inline_entity_ids`: use a list of `RP_ENTITY_ID` values supplied in chat or tool input.

Validation rules:

- Require an entity ID column. Accept ``RP_ENTITY_ID`` (preferred) and legacy ``RP_COMPANY_ID`` as aliases, then
  normalize internally to the field expected by the current pipeline.
- Use `COMPANY_NAME` when available; otherwise keep company names blank until search results or
  downstream enrichment can resolve them.
- Deduplicate IDs before planning.
- Report universe size and a small sample before building search plans.
- Persist `universe_manifest.json` with source mode, row count, ID column, name column, and
  validation warnings.

Example Claude prompt:

```text
I will use the default global all-caps universe: 10,001 companies from global_all_caps.csv.
For European mid/large caps, use default_europe_ml_caps: 646 companies from europe_ml_caps.csv.
You can also provide a CSV or paste a list of RP_ENTITY_ID values if you want a narrower screen.
```

## Example Conversation

```text
User: Screen global software companies for AI disruption in product development.
Claude: I will use the default global all-caps universe, create a taxonomy first, then validate it before spending retrieval budget.

Claude: Here is the proposed mindmap: 5 branches, 14 leaf search labels. Three leaves may overlap.
User: Merge developer tooling and code generation. Keep it under 10 labels.

Claude: Updated taxonomy has 9 leaves. I can now build plans and estimate retrieval volume.
User: Continue.

Claude: Planning found 184,000 expected chunks. Budget options:
Quick scan: 0.5%, about 920 chunks, about $1.38
Balanced: 2.0%, about 3,680 chunks, about $5.52
Deep dive: 5.0%, about 9,200 chunks, about $13.80
Which budget should I use?

User: Balanced.
Claude: Running retrieval. I will create an evidence digest before asking about optional enrichment.

Claude: Retrieval finished with 3,420 chunks across 810 documents. I created an evidence digest
with the top signals, representative chunks, and source mix. Would you like to review the digest,
run enrichment (labeling + company summaries), or stop here?
```

## Claude Desktop Install

Add the server to `~/Library/Application Support/Claude/claude_desktop_config.json` using the same
`uv --directory` pattern as other local MCP servers:

```json
{
  "mcpServers": {
    "thematic-screener": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/franciscogomez/git/bigdata/bigdata-cookbook/Thematic_Screener_CLI",
        "run",
        "--quiet",
        "thematic-screener-mcp"
      ]
    }
  }
}
```

Restart Claude Desktop after saving the config. The `thematic-screener-mcp` entrypoint is defined
in `pyproject.toml` and runs the FastMCP stdio server.

## Files

- `.cursor/skills/thematic-screener/SKILL.md`: Cursor skill with human-in-the-loop gates.
- `tool-contracts.md`: proposed MCP tools and request/response shapes.
- `artifact-model.md`: artifact IDs, metadata, previews, pagination, and retrieval semantics.
- `claude-skill.md`: Claude Desktop operating instructions for the guided workflow.
- `cli-stage-map.md`: mapping from existing CLI stages to the MCP workflow.
- `schemas.json`: compact JSON Schema-style definitions for the main MCP payloads.
