# AI in Go-To-Market (GTM) — Official-Source Evidence Report

Scans **official company sources only** for verifiable, quote-level evidence that AI is
used in a company's **Go-To-Market (GTM) strategy** over the **last 12 months**, for an
input list of companies, and produces two reconciled Markdown tables:

- **A) Company Summary** — `| Company | Docs with AI-GTM Mentions | Total AI-GTM Mentions | Illustrative Quote | Citation |`
- **B) Citations** — `| Company | Article/Document Title (verbatim) | Document Type | Date (ISO) | URL | Quote |`

Built on the [Bigdata.com](https://bigdata.com) API, following the architecture of
`Index_MA_Activity_Report` (shared `services/` layer, `config/topics.py` topic set,
`config/prompts.yaml` extraction prompt).

## What counts as an AI-GTM mention

A verbatim passage (≤ 2–3 sentences) where an **explicit AI term** — `AI`,
`artificial intelligence`, `GenAI`, `generative AI`, `LLM`, `machine learning` — is
directly linked to a **GTM activity or outcome** in the same or adjacent sentence
(sales execution, lead gen, marketing/campaigns, customer acquisition/expansion,
pricing/packaging, channel/partner motions, product launch/commercialization,
segmentation/targeting/personalization, sales enablement, CRM optimization,
pipeline/forecasting, quote-to-cash).

**Excluded:** investor/capital-markets day decks; third-party media/aggregators; sell-side
notes; AI for portfolio valuation/trading/market analysis; internal-only ops (finance/HR/IT)
without an explicit GTM outcome; generic claims; AI in customer products with no GTM link.

## How it works

| Stage | Mechanism |
|---|---|
| **Universe** | `COMPANIES` list of names (Step 3) → resolved to Bigdata entity IDs via the Knowledge Graph API (Step 4) |
| **Search — Pass A** | 8 AI×GTM keyword topics, category filter `["filings", "transcripts"]` (regulatory filings, annual/interim reports, earnings-call transcripts) |
| **Search — Pass B** | Same topics, source filter `["DFF004"]` (PubT Corporate Communications: company-issued press releases) |
| **Pre-filters** | Merge + de-dup by document ID; drop investor/capital-markets-day titles; drop excerpts with no explicit AI term |
| **Extraction** | `ai_gtm_extract` prompt (OpenAI `gpt-4o-mini` by default) emits one JSON row per verbatim quote, with document title/type/date/URL |
| **Validation** | Deterministic: AI-term regex on the quote; **verbatim check** against retrieved source text (normalized containment, fuzzy ≥ 85% fallback); 12-month window; near-duplicate de-dup within a document |
| **Counts** | Computed in pandas from citation rows — docs = distinct titles, mentions = quote rows — then reconciled with assertions |

General news media and sell-side research categories are **never searched**, so
third-party sources cannot leak in.

## Quick start

```bash
cd AI_GTM_Evidence_Report
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# credentials: export in your shell, or copy .env.example -> .env and fill in
export BIGDATA_API_KEY=...   # official-source search
export OPENAI_API_KEY=...    # quote extraction

uv run jupyter lab           # open ai_gtm_evidence.ipynb and Run All
```

The input universe is the `COMPANIES` list in Step 3 (default:
`["Atlas Copco", "ABB", "Sandvik", "Schneider Electric"]`); it can also be loaded from a
CSV with a `COMPANY_NAME` column.

## Outputs (`output/`)

| File | Contents |
|---|---|
| `ai_gtm_report_*.md` | Both Markdown tables (Company Summary + Citations) |
| `ai_gtm_summary_*.csv` | Per-company counts + illustrative quote |
| `ai_gtm_citations_*.csv` | One row per validated quote (with AI-term/GTM-activity audit columns) |
| `ai_gtm_quotes_*.json` | Validated quote rows (full fidelity) |
| `ai_gtm_search_results_*.json` | Raw pre-filtered search excerpts (re-run extraction without re-querying) |

## Notes on coverage

The default universe is European industrials; their evidence comes mostly from
earnings-call transcripts and official press releases (`DFF004`). SEC-style filings
(10-K/10-Q/8-K/20-F) appear only for companies with US listings/ADR programs — the
`filings` category pass covers them when present.
