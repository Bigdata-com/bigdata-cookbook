# Google ADK with Bigdata.com and local data — reference demo

This project is a **working reference** for **[Google Agent Development Kit (ADK)](https://google.github.io/adk-docs/)** integrated with local structured and unstructured data and with **[Bigdata.com](https://bigdata.com)** over MCP. It combines:

1. **Structured local data** — a sample **SQLite** portfolio (holdings, transactions, P&L) queried with precision.  
2. **Unstructured local knowledge** — research-style content as **Markdown** under [`financial_agent/sample_documents/`](financial_agent/sample_documents/), indexed with **FAISS** and **Gemini embeddings** (Google stack end-to-end; no third-party LLM or embedding vendors).  
3. **Live market intelligence** — **MCP** at `https://mcp.bigdata.com/` (search, tearsheets, events, entity resolution, and other tools as exposed by the server; see [Bigdata.com](https://bigdata.com)).  

Together, these pieces show how one ADK agent can use SQL, semantic retrieval over local documents, and Bigdata MCP tools in a single session, with a clear split between on-box data and remote Bigdata calls.

---

## What you need to run it

| Requirement | Notes |
|---------------|-------|
| **Bigdata.com API key** | **Required** for MCP. Requests go to **`https://mcp.bigdata.com/`** with the **`x-api-key`** header. Obtain credentials from **[Bigdata.com](https://bigdata.com)**. |
| **Google Gemini API key** | **Required** for the chat model and for embeddings used with local semantic search. See **[Google AI Studio](https://aistudio.google.com/apikey)** (or your organization’s Gemini setup). |
| **Python 3.11+** | Matches current ADK and dependency stacks. |
| **[uv](https://docs.astral.sh/uv/)** | Recommended for creating the virtual environment and running commands reproducibly. |

---

## Quick start

```bash
cd Google_ADK_With_BigData
uv sync
cp financial_agent/.env.example financial_agent/.env
```

Edit **`financial_agent/.env`**: set **`BIGDATA_API_KEY`**, **`GOOGLE_API_KEY`** (or **`GEMINI_API_KEY`**), and optional model overrides if needed.

```bash
uv run adk web .
```

Open the URL printed in the terminal (commonly `http://127.0.0.1:8000`). Select the **`financial_agent`** application in the ADK UI.

**Run location:** start `adk web` from the **`Google_ADK_With_BigData`** project root—the directory that **contains** the `financial_agent/` package—not from inside `financial_agent/`.

---

## Sample queries

Paste these into the ADK web chat after **`financial_agent`** is running. They exercise local SQL, local semantic search, and Bigdata MCP together or in isolation.

### Query A — PF002: holdings, internal risk, tearsheets, news

```text
Analyze the AI & Semiconductor Focus portfolio (PF002):
1. What are our current holdings and their performance?
2. What risks does our internal research identify?
3. For each holding, get us the pricing information from tearsheet
4. For each holding, get us negative news
```

| Step | What the agent should do | Data source |
|------|---------------------------|-------------|
| 1 | Holdings, weights, and performance context | **Local SQLite** — `internal_portfolio_summary("PF002")` and/or `internal_query_database` (SELECT on `holdings`, `transactions`) |
| 2 | Theses, risk memos, sector risk language | **Local FAISS** — `internal_search_research` over `sample_documents/` + embedded seed memos |
| 3 | Per-ticker quotes, multiples, profile fields | **Bigdata MCP** — typically `find_companies` (or equivalent) then **company tearsheet** tools as exposed by the server |
| 4 | Adverse or cautionary headlines | **Bigdata MCP** — **search** (and related tools) with a negative-sentiment or “risks / litigation / downgrade” style query per ticker |

---

### Query B — NVIDIA earnings and revenue (Bigdata MCP)

```text
Find the latest news about NVIDIA's earnings and revenue growth using Bigdata tools.
```

| Aspect | Notes |
|--------|-------|
| **Primary load** | **Bigdata MCP** — search and possibly tearsheet-style tools for earnings, revenue, and recent headlines |
| **Local data** | Optional; internal NVDA memos may still appear if the model chooses `internal_search_research` |

---

### Query C — NVIDIA: database, internal research, and Bigdata news

```text
For our NVIDIA holdings:
1. Check our internal database to see which portfolios hold NVDA and how much
2. Search our internal research for our investment thesis
3. Use Bigdata tools to find recent news about NVIDIA
4. Provide a comprehensive summary combining all sources
```

| Step | What the agent should do | Data source |
|------|---------------------------|-------------|
| 1 | Which `portfolio_id` rows include `NVDA`, shares, market value | **Local SQLite** — `internal_query_database` (e.g. `holdings` joined to `portfolios`) |
| 2 | NVDA thesis, risks, ratings language | **Local FAISS** — `internal_search_research` (sample NVDA memo + related seed docs) |
| 3 | Fresh external context | **Bigdata MCP** — search (and entity resolution if needed) |
| 4 | Single narrative with **inline attribution** for external facts | **Gemini** synthesis over tool outputs; system prompt asks for **Bigdata.com** inline citations where applicable |

---

## Environment variables

| Variable | Required | Role |
|----------|----------|------|
| `BIGDATA_API_KEY` | **Yes** | Authenticates MCP requests to **`https://mcp.bigdata.com/`** via the **`x-api-key`** header. |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | **Yes** | Gemini for chat and for **Gemini embeddings** used with FAISS over local documents. |
| `GEMINI_MODEL` | No | Chat model id (default: `gemini-2.5-flash`). |
| `GEMINI_EMBEDDING_MODEL` | No | Embedding model id (default: `gemini-embedding-001`). |

**Engineering / CI only** (omit for normal `adk web` runs):

| Variable | Purpose |
|----------|---------|
| `FINANCIAL_AGENT_DATA_DIR` | Absolute path for isolated SQLite and runtime data. |
| `FINANCIAL_AGENT_SKIP_EMBED_WARMUP` | Set to `1` to skip embedding calls at import (used by automated tests). |

---

## Customization

- **Local memos:** add or replace **`.md`** files under [`financial_agent/sample_documents/`](financial_agent/sample_documents/). Filename-based metadata is mapped in [`financial_agent/agent.py`](financial_agent/agent.py) (`_SAMPLE_DOC_METADATA`); additional `*.md` files still ingest with sensible defaults.  
- **Structured data:** the bundled SQLite dataset is recreated on each process start from code; replace the seed logic in code when you adopt your own schema.  

---

## Development and quality checks

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
```

---

## Troubleshooting

- **MCP or Bigdata errors** — Confirm `BIGDATA_API_KEY` is present, not expired, and allowed for MCP. Verify outbound HTTPS to `https://mcp.bigdata.com/`.  
- **Internal semantic search** — Confirm `GOOGLE_API_KEY` / `GEMINI_API_KEY`. Default embedding model is `gemini-embedding-001`. On startup, watch the terminal for FAISS build logs unless `FINANCIAL_AGENT_SKIP_EMBED_WARMUP=1` is set (tests only).  

---

## Security

- Do **not** commit **`.env`** or share API keys in issues, PRs, or public artifacts.  
- Sample portfolios and research text are **synthetic**—for illustration only, not investment advice or production data.  

---

## License

If you redistribute this demo inside a broader repository, follow that repository’s license terms. This folder does not ship a standalone license file by default.
