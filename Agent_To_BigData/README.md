# 🤖 Agentic AI with Bigdata.com API

A modular framework for building AI agents that integrate Bigdata.com APIs with internal data sources, featuring **LangSmith** observability for production monitoring.

## Overview

This demo showcases a multi-source AI agent that can:
- Query **Bigdata.com Search API** for real-time financial news
- Use **Bigdata.com Knowledge Graph API** for entity resolution
- Use **Bigdata.com Research Agent API** for deep research with citations
- Query **SQLite Database** for portfolio/transaction data
- Search **FAISS Vector Store** for internal research documents

## Files

| File | Description |
|------|-------------|
| `langgraph_core.py` | Reusable core module with logging, retry, KG cache, and tools |
| `research_client.py` | Bigdata.com Research Agent client with retry, logging, and full chat_id |
| `agent_to_search.ipynb` | Demo: Agent using Search & Knowledge Graph APIs |
| `agent_to_research_agent.ipynb` | Demo: Hierarchical agent with Research Agent escalation |

## Quick Start

### 1. Set Environment Variables

```bash
export BIGDATA_API_KEY="your-bigdata-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export LANGSMITH_API_KEY="your-langsmith-api-key"  # Optional, for tracing
```

### 2. Run the Demo

```python
from langgraph_core import quick_setup, display_agent_response

# Initialize everything
config, agent = quick_setup()

# Run a query
display_agent_response(agent, "What are our NVIDIA holdings across all portfolios?")
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
from langgraph_core import (
    create_agent,
    get_bigdata_tools,
    get_database_tools,
    get_vectorstore_tools
)

# Use all default tools
agent = create_agent()

# Or customize which tools to include
tools = get_bigdata_tools() + get_database_tools()
agent = create_agent(tools=tools)
```

### Run Queries

```python
from langgraph_core import run_agent_query, display_agent_response

# Programmatic access (returns dict)
result = run_agent_query(agent, "Analyze NVIDIA position")
print(result['response'])

# Jupyter display (formatted HTML output)
display_agent_response(agent, "Analyze NVIDIA position")
```

## Production Features

- **Logging**: Module-level logging for Bigdata API calls; full `chat_id` logged for research tool completion and Research Client (when `setup_logging()` is used).
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

```python
from langgraph_core import create_hierarchical_agent, display_agent_response

# Create hierarchical agent
agent = create_hierarchical_agent(include_research_agent=True)

# Queries automatically route to appropriate tools
display_agent_response(agent, "Analyze our NVIDIA position vs market sentiment")
```

**Use Cases:**
- 📈 **Equity Research**: Thesis validation, competitive analysis
- 💳 **Credit Research**: Debt analysis, refinancing risk
- ⚠️ **Credit Risk**: Counterparty exposure, stress testing

See `agent_to_research_agent.ipynb` for detailed examples.

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

## Reusing Core Module

Import `langgraph_core.py` in other notebooks or scripts:

```python
# In another notebook
import sys
sys.path.append('../Agent_To_BigData')

from langgraph_core import (
    setup_environment,
    create_agent,
    run_agent_query,
    get_bigdata_tools,
    get_research_agent_tool
)

# Create a specialized agent
config = setup_environment()
research_agent = create_agent(
    tools=get_bigdata_tools() + get_research_agent_tool(),
    system_prompt="You are a market research specialist..."
)

# Use in multi-agent workflow
result = run_agent_query(research_agent, "Research NVDA competitive position")
```

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

## License

See the main repository LICENSE file.
