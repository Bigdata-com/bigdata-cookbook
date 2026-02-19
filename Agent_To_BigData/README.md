# Agentic AI with Bigdata.com API

A modular framework for building AI agents that integrate Bigdata.com APIs with internal data sources, featuring **LangSmith** observability for production monitoring. Suitable for equity research, credit research, and risk workflows.

---

## Overview

This repository showcases a multi-source AI agent that can:

- Query **Bigdata.com Search API** for real-time financial news, filings, and transcripts
- Use **Bigdata.com Knowledge Graph API** for entity resolution (ticker/company → entity ID)
- Use **Bigdata.com Research Agent API** for deep research with citations
- Connect via **Bigdata.com MCP** for automatic tool discovery (search, tearsheets, events calendar)
- Query **SQLite** for portfolio and transaction data
- Search a **FAISS vector store** for internal research documents

The demos use LangChain and LangSmith; the integration patterns apply to other agentic frameworks (CrewAI, AutoGen, etc.).

---

## Files

| File | Description |
|------|-------------|
| `langgraph_core.py` | Reusable core module: logging, retry, KG cache, and tool definitions |
| `research_client.py` | Bigdata.com Research Agent client with retry, logging, and chat_id handling |
| `agent_to_search.ipynb` | Demo: Agent using Search & Knowledge Graph APIs with internal DB and vector store |
| `agent_to_research_agent.ipynb` | Demo: Hierarchical agent—internal sources first, then Research Agent escalation |
| `agent_to_bigdata_mcp.ipynb` | Demo: Agent using Bigdata.com via MCP (automatic tool discovery) |

---

## Available Tools

### External (Bigdata.com)

| Tool | API | Description |
|------|-----|-------------|
| `bigdata_lookup_company` | Knowledge Graph | Resolve ticker/company name to entity ID (cached) |
| `bigdata_search_news` | Search | Search financial news and headlines (with retry) |
| `bigdata_research_agent` | Research Agent | Deep research with citations (20–60s); retry and full chat_id logging |

### Internal (Company Systems)

| Tool | Source | Description |
|------|--------|-------------|
| `internal_query_database` | SQLite | Execute SQL on portfolios |
| `internal_portfolio_summary` | SQLite | Portfolio holdings summary |
| `internal_search_research` | FAISS | Semantic search on research documents |

---

## MCP Integration

You can connect to **Bigdata.com via MCP (Model Context Protocol)** for automatic tool discovery. With MCP, your agent gets current and future Bigdata.com tools (search, tearsheets, events calendar, company lookup) without code changes when new capabilities are added.

**Benefits:**

- **Automatic tool discovery** — MCP exposes tools dynamically; new Bigdata.com capabilities become available to your agent automatically
- **Single integration** — One MCP connection replaces multiple API wrappers
- **Framework-agnostic** — Works with LangChain, CrewAI, AutoGen, and others via MCP adapters

**Configuration:**

- **URL:** `https://mcp.bigdata.com/`
- **Transport:** HTTP (streamable)
- **Authentication:** `x-api-key` header with your Bigdata API key

For full setup and usage, see **`agent_to_bigdata_mcp.ipynb`** and install `langchain-mcp-adapters` as noted in Quick Start.

---

## Hierarchical Agent

For complex research workflows, a **hierarchical agent**:

1. **Checks internal sources first** (DB, vector store) for speed and proprietary data
2. **Escalates to the Research Agent** when internal data is insufficient

Use cases: equity research (thesis validation, competitive analysis), credit research (debt analysis, refinancing risk), and credit risk (counterparty exposure, stress testing).

Full system prompt and examples are in **`agent_to_research_agent.ipynb`**.

---

## Sample Data

### SQLite database

- 3 accounts (Institutional, Hedge Fund, Pension)
- 3 portfolios (PF001, PF002, PF003)
- 15 holdings (NVDA, AAPL, MSFT, etc.)
- 100+ transactions

### Vector store documents

- Investment theses (NVDA, AAPL, MSFT, AMD)
- Q1 2025 Portfolio Strategy memo
- Technology Sector Risk Assessment

---

## LangSmith Integration

The framework supports **LangSmith** for agent observability. When enabled, traces at [smith.langchain.com](https://smith.langchain.com) include:

- Tool calls and their inputs/outputs
- LLM reasoning steps
- Token usage and latency
- Error handling

Set `LANGSMITH_API_KEY` and optionally configure the project name in your environment or in `setup_environment()` (see the notebooks).

---

## Quick Start

Follow these steps to run the notebooks locally. No sample code is required in this section—each notebook is self-contained.

### 1. Environment setup with uv

From the project root, create a virtual environment and install dependencies using [uv](https://docs.astral.sh/uv/):

```bash
# Create venv and install dependencies (uses pyproject.toml and uv.lock)
uv sync
```

For the **MCP notebook** (`agent_to_bigdata_mcp.ipynb`), `langchain-mcp-adapters` is in `pyproject.toml` and will be installed by the above. Ensure your Python version matches the project (e.g. `>=3.13` per `pyproject.toml`; uv will use `.python-version` if present).

### 2. Environment variables

Set these before launching Jupyter or running any notebook:

| Variable | Required | Description |
|----------|----------|-------------|
| `BIGDATA_API_KEY` | Yes | Your Bigdata.com API key |
| `OPENAI_API_KEY` | Yes | OpenAI API key for the LLM |
| `LANGSMITH_API_KEY` | No | For LangSmith tracing; omit to disable |

Example (Unix-like shells):

```bash
export BIGDATA_API_KEY="your-bigdata-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export LANGSMITH_API_KEY="your-langsmith-api-key"   # optional
```

Use a `.env` file or your platform’s standard way to set variables if you prefer.

### 3. Launch the notebooks

Start Jupyter from the project root:

```bash
uv run jupyter notebook
```

Or, if the environment is already activated:

```bash
jupyter notebook
```

Then open one of the following:

- **`agent_to_search.ipynb`** — Introduces the agent wired to Bigdata.com Search and Knowledge Graph APIs plus internal SQLite and FAISS. Use this to see how to combine external market data with internal portfolios and research in a single, cited flow.
- **`agent_to_research_agent.ipynb`** — Shows the hierarchical pattern: the agent uses internal DB and vector store first, then escalates to the Bigdata.com Research Agent for deep, cited research when needed. Best for understanding when and how to call the Research Agent.
- **`agent_to_bigdata_mcp.ipynb`** — Connects to Bigdata.com via MCP so tools (search, tearsheets, events, company lookup) are discovered automatically. Use this to integrate without hard-coding each API and to get new Bigdata.com tools as they are added.

Run the cells in order; each notebook includes its own setup and usage instructions.

---

## License

See the repository’s LICENSE file.
