"""Financial research ADK agent (local SQLite, local docs + FAISS, Bigdata.com MCP).

Import ``financial_agent.agent`` (or let ADK load it) after environment variables are set.
This package intentionally does not import ``agent`` at import time so tests can set
``FINANCIAL_AGENT_DATA_DIR`` before the module body runs.
"""
