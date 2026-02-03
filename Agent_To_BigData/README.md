# 🤖 Agentic AI with Bigdata.com API

A modular framework for building AI agents that integrate Bigdata.com APIs with internal data sources, featuring **LangSmith** observability for production monitoring.

## Overview

This demo showcases a multi-source AI agent that can:
- Query **Bigdata.com Search API** for real-time financial news
- Use **Bigdata.com Knowledge Graph API** for entity resolution
- Use **Bigdata.com Research Agent API** for deep research with citations
- Connect via **Bigdata.com MCP** for automatic tool discovery (search, tearsheets, events calendar)
- Query **SQLite Database** for portfolio/transaction data
- Search **FAISS Vector Store** for internal research documents

## Files

| File | Description |
|------|-------------|
| `langgraph_core.py` | Reusable core module with logging, retry, KG cache, and tools |
| `research_client.py` | Bigdata.com Research Agent client with retry, logging, and full chat_id |
| `agent_to_search.ipynb` | Demo: Agent using Search & Knowledge Graph APIs |
| `agent_to_research_agent.ipynb` | Demo: Hierarchical agent with Research Agent escalation |
| `agent_to_bigdata_mcp.ipynb` | Demo: Agent using Bigdata.com via MCP (automatic tool discovery) |

## Quick Start

### 1. Set Environment Variables

```bash
export BIGDATA_API_KEY="your-bigdata-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export LANGSMITH_API_KEY="your-langsmith-api-key"  # Optional, for tracing
```

### 2. Run the Demo

```python
from langchain.agents import create_agent as langchain_create_agent
from langchain_openai import ChatOpenAI
from langgraph_core import (
    setup_environment,
    create_financial_database,
    create_vector_store,
    get_bigdata_tools,
    get_database_tools,
    get_vectorstore_tools,
    run_agent_query,
    display_query,
    display_response,
)

# Initialize environment and data sources
config = setup_environment()
create_financial_database()
create_vector_store()

# Build agent with all tools
llm = ChatOpenAI(model="gpt-5", temperature=0)
tools = get_bigdata_tools() + get_database_tools() + get_vectorstore_tools()
agent = langchain_create_agent(llm, tools)

# Run a query (in Jupyter: use display_query, then run_agent_query, then display_response)
display_query("What are our NVIDIA holdings across all portfolios?")
result = run_agent_query(agent, "What are our NVIDIA holdings across all portfolios?")
display_response(result)
```

## Core Module Usage

### Setup Environment

```python
from langgraph_core import setup_environment

config = setup_environment(
    langsmith_project='my-project',  # LangSmith project name
    enable_tracing=True              # Enable observability
)
```

### Create Data Sources

```python
from langgraph_core import create_financial_database, create_vector_store

# Create SQLite DB with sample financial data
create_financial_database()

# Create FAISS vector store with research docs
create_vector_store()
```

### Create Agent with Custom Tools

```python
from langchain.agents import create_agent as langchain_create_agent
from langchain_openai import ChatOpenAI
from langgraph_core import (
    get_bigdata_tools,
    get_database_tools,
    get_vectorstore_tools,
)

llm = ChatOpenAI(model="gpt-5", temperature=0)

# Use all default tools (Search, KG, DB, vector store)
tools = get_bigdata_tools() + get_database_tools() + get_vectorstore_tools()
agent = langchain_create_agent(llm, tools)

# Or customize which tools to include
tools = get_bigdata_tools() + get_database_tools()
agent = langchain_create_agent(llm, tools)
```

### Run Queries

```python
from langgraph_core import run_agent_query, display_query, display_response, display_tools_used, display_citations

# Programmatic access (returns dict)
result = run_agent_query(agent, "Analyze NVIDIA position")
print(result["response"])

# Jupyter display (formatted HTML: query, response, tools used, citations)
display_query("Analyze NVIDIA position")
result = run_agent_query(agent, "Analyze NVIDIA position", return_tools=True)
display_response(result)
display_tools_used(result)
display_citations(result)  # When Research Agent or search was used
```

## Production Features

- **Logging**: Module-level logging for Bigdata API calls; full `chat_id` logged for research tool completion and in Research Client.
- **Retry**: Exponential backoff for Bigdata Search and Knowledge Graph calls; Research Agent uses built-in retry and stream timeout detection.
- **Knowledge Graph cache**: Entity lookups by ticker/company name are cached in memory to avoid repeated API calls for the same entity.

## Available Tools

### External Tools (Bigdata.com)

| Tool | API | Description |
|------|-----|-------------|
| `bigdata_lookup_company` | Knowledge Graph | Resolve ticker/company name to entity ID (cached by ticker/name) |
| `bigdata_search_news` | Search | Search financial news and headlines (retry on transient failures) |
| `bigdata_research_agent` | Research Agent | Deep research with citations (20-60s); retry and full chat_id logging |

### Internal Tools (Company Systems)

| Tool | Source | Description |
|------|--------|-------------|
| `internal_query_database` | SQLite | Execute SQL queries on portfolios |
| `internal_portfolio_summary` | SQLite | Get portfolio holdings summary |
| `internal_search_research` | FAISS | Semantic search on research documents |

## Hierarchical Agent (Agent-to-Agent)

For complex research workflows, use the hierarchical agent that:
1. **Checks internal sources first** (faster, proprietary data)
2. **Escalates to Research Agent** when internal data is insufficient

Build the agent in code by combining internal tools, Bigdata tools, and the Research Agent tool with a system prompt that enforces the hierarchy:

```python
from langchain.agents import create_agent as langchain_create_agent
from langchain_openai import ChatOpenAI
from langgraph_core import (
    get_bigdata_tools,
    get_database_tools,
    get_vectorstore_tools,
    get_research_agent_tool,
    run_agent_query,
    display_query,
    display_response,
)

llm = ChatOpenAI(model="gpt-5", temperature=0)
tools = (
    get_database_tools()
    + get_vectorstore_tools()
    + get_bigdata_tools()
    + get_research_agent_tool()
)
# Use a system prompt that instructs: check internal first, then escalate to Research Agent
agent = langchain_create_agent(llm, tools, system_prompt=HIERARCHICAL_SYSTEM_PROMPT)

display_query("Analyze our NVIDIA position vs market sentiment")
result = run_agent_query(agent, "Analyze our NVIDIA position vs market sentiment")
display_response(result)
```

**Use Cases:**
- 📈 **Equity Research**: Thesis validation, competitive analysis
- 💳 **Credit Research**: Debt analysis, refinancing risk
- ⚠️ **Credit Risk**: Counterparty exposure, stress testing

See `agent_to_research_agent.ipynb` for the full system prompt and detailed examples.

## MCP Integration

You can connect to **Bigdata.com via MCP (Model Context Protocol)** for automatic tool discovery. With MCP, your agent gets all current and future Bigdata.com tools (search, tearsheets, events calendar, company lookup) without code changes when new capabilities are added.

**Benefits:**
- **Automatic tool discovery** — MCP exposes tools dynamically; new Bigdata.com capabilities (e.g. tearsheets, calendars, screeners) become available to your agent automatically
- **Single integration** — One MCP connection replaces multiple API wrappers
- **Framework-agnostic** — MCP works with LangChain, CrewAI, AutoGen, and other frameworks via their MCP adapters

**Configuration:**
- **URL**: `https://mcp.bigdata.com/`
- **Transport**: HTTP (streamable)
- **Authentication**: `x-api-key` header with your Bigdata API key

**Example (LangChain + langchain-mcp-adapters):**

```python
# Install: pip install langchain-mcp-adapters
from langchain_mcp_adapters.client import MultiServerMCPClient

os.environ["ANYIO_BACKEND"] = "asyncio"  # Required in Jupyter

mcp_client = MultiServerMCPClient({
    "bigdata": {
        "url": "https://mcp.bigdata.com/",
        "transport": "http",
        "headers": {"x-api-key": os.getenv("BIGDATA_API_KEY")},
    }
})

# Discover all Bigdata tools (company lookup, search, tearsheets, events calendar, etc.)
bigdata_mcp_tools = await mcp_client.get_tools()
# Combine with langgraph_core local tools (DB, vector store) and create agent
agent = langchain_create_agent(llm, bigdata_mcp_tools + get_database_tools() + get_vectorstore_tools())
```

**Tools exposed via Bigdata MCP** (discovered at runtime): company/country tearsheets, events calendar, search, and `find_companies` for entity resolution. See `agent_to_bigdata_mcp.ipynb` for the full demo and dependency setup (`langchain-mcp-adapters`).

## LangSmith Integration

The framework includes built-in LangSmith support for agent observability:

```python
# Enable tracing
config = setup_environment(
    langsmith_api_key="your-key",
    langsmith_project="bigdata-agent-demo",
    enable_tracing=True
)
```

View traces at: **https://smith.langchain.com**

Traces include:
- Tool calls and their inputs/outputs
- LLM reasoning steps
- Token usage and latency
- Error handling


## Sample Data

### SQLite Database
- 3 Accounts (Institutional, Hedge Fund, Pension)
- 3 Portfolios (PF001, PF002, PF003)
- 15 Holdings (NVDA, AAPL, MSFT, etc.)
- 100+ Transactions

### Vector Store Documents
- Investment theses (NVDA, AAPL, MSFT, AMD)
- Q1 2025 Portfolio Strategy memo
- Technology Sector Risk Assessment

## Dependencies

Install from requirements file:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install langchain langchain-openai langchain-community faiss-cpu requests python-dotenv
```

**For MCP demo** (`agent_to_bigdata_mcp.ipynb`), also install:

```bash
pip install langchain-mcp-adapters
```

## License

See the main repository LICENSE file.
