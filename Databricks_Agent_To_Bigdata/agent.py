"""
Databricks + Bigdata.com financial-intelligence agent (MLflow models-from-code).

A single Mosaic AI agent that reasons over three data sources in one conversation:

  Internal — structured    : Unity Catalog SQL functions (and optionally an AI/BI Genie space)
  Internal — unstructured  : Mosaic AI Vector Search over proprietary research notes
  External — real-time      : Bigdata.com MCP (news, filings, transcripts, tearsheets, events)

The agent is a LangGraph tool-calling graph driven by an LLM served through Databricks
Model Serving (by default OpenAI gpt-5.4 via an AI Gateway external-model endpoint, or a
Databricks-hosted foundation model — see LLM_PROVIDER/LLM_ENDPOINT below), wrapped in
MLflow's `ChatAgent` interface so it can be logged, evaluated, and deployed to
Model Serving. This file is loaded via `mlflow.models.set_model(...)`. It follows the
canonical Databricks LangGraph `ChatAgent` template (using `ChatAgentState` /
`ChatAgentToolNode`, which handle message conversion for deployment).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Generator, Optional, Sequence

import mlflow
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentChunk,
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)

# =============================================================================
# Configuration — keep in sync with the notebooks (00–05)
# =============================================================================
CATALOG = "bigdata_demo"
SCHEMA = "financial_intelligence"

# LLM that drives the agent's reasoning.
#   LLM_PROVIDER = "databricks" -> use a Databricks Model Serving endpoint (LLM_ENDPOINT).
#       This covers both Databricks-hosted foundation models AND an OpenAI-backed
#       *external-model* endpoint governed by AI Gateway (recommended — the OpenAI key
#       stays in a Databricks secret and never enters the agent process). Just set
#       LLM_ENDPOINT to your external endpoint's name (e.g. "openai-chat").
#   LLM_PROVIDER = "openai"     -> call OpenAI directly via langchain-openai (OPENAI_MODEL).
#       Simplest fallback; the key is read from env var OPENAI_API_KEY, or the Databricks
#       secret bigdata/openai_api_key.
LLM_PROVIDER = "databricks"
# Default: an OpenAI-backed external-model endpoint governed by AI Gateway (created by
# 00_setup.ipynb; see README.md's "Set up the agent's LLM" section for the full write-up).
# Serves OpenAI gpt-5.4 while keeping the key in a Databricks secret.
# To use a Databricks-hosted model instead, set this to e.g. "databricks-claude-sonnet-4-5".
LLM_ENDPOINT = "openai-chat"
OPENAI_MODEL = "gpt-5.4"  # used only when LLM_PROVIDER == "openai" (direct path)

# Internal unstructured — Vector Search index built in notebook 02
VS_INDEX = f"{CATALOG}.{SCHEMA}.research_docs_index"

# Internal structured — Unity Catalog SQL functions built in notebook 01
UC_FUNCTIONS = [
    f"{CATALOG}.{SCHEMA}.get_top_holdings",
    f"{CATALOG}.{SCHEMA}.get_portfolio_positions",
    f"{CATALOG}.{SCHEMA}.get_ticker_exposure",
]

# How the internal structured tools EXECUTE those UC functions:
#   "sql_warehouse" -> Statement Execution REST API via a serverless SQL warehouse.
#                      No databricks-connect required; works on Free Edition serverless
#                      notebooks and in Model Serving. (Default — most portable.)
#   "uc_toolkit"    -> databricks-langchain UCFunctionToolkit. On serverless this needs the
#                      databricks-connect package (>=15.1.0), otherwise you get
#                      "No package metadata was found" when the client starts.
STRUCTURED_TOOLS = "sql_warehouse"
SQL_WAREHOUSE_ID = ""  # optional; auto-discovered from the workspace when left empty

# External — Bigdata.com MCP.
#   USE_MCP_SERVICE = True  -> governed MCP Service via Unity AI Gateway (notebook 03, Path A)
#   USE_MCP_SERVICE = False -> direct connection to https://mcp.bigdata.com (notebook 03, Path B)
USE_MCP_SERVICE = False
BIGDATA_MCP_URL = "https://mcp.bigdata.com/"
MCP_SERVICE_NAME = f"{CATALOG}.{SCHEMA}.bigdata_mcp"  # used when USE_MCP_SERVICE = True

# Optional AI/BI Genie space for open-ended structured Q&A (notebook 04)
USE_GENIE = False
GENIE_SPACE_ID = ""  # e.g. "01ef..."

SYSTEM_PROMPT = """You are a financial research analyst for an institutional asset manager, \
powered by Databricks and Bigdata.com.

You can combine three kinds of data in a single answer:
1. Internal structured data — portfolio accounts, holdings, transactions, P&L \
(via the portfolio SQL functions / Genie).
2. Internal unstructured research — the firm's own investment theses, strategy memos, and \
risk assessments (via the research search tool).
3. External real-time intelligence from Bigdata.com — news/filings/transcripts search, \
security & company resolution, and rich snapshots: company, ETF, country, and market \
tearsheets, media sentiment, an earnings/events calendar, and credit-factor and \
fund-manager screeners.

Guidance:
- Decompose multi-part questions and call the right tool for each part.
- To resolve a company, ETF, or fund from a name/ticker/ISIN, use find_securities \
(this replaces the deprecated find_companies) to get its entity id, then fetch the \
relevant tearsheet (bigdata_company_tearsheet, bigdata_etf_tearsheet, etc.). Use \
get_securities when you already have exact identifiers (ISIN/CUSIP/SEDOL) to batch-resolve.
- Match the tool to the question: bigdata_sentiment_tearsheet for media sentiment and \
narratives, bigdata_events_calendar for upcoming earnings/conference calls, \
bigdata_country_tearsheet / bigdata_market_tearsheet for macro and cross-asset context, \
bigdata_screen_credit_factor for credit risk factors, bigdata_screen_fund_managers for \
institutional ownership. Prefer these purpose-built tools over a plain search when they fit.
- When you use Bigdata.com results, cite them inline (source, headline, date, URL when \
available) and end with a numbered "Sources" section listing every Bigdata.com citation.
- Be explicit about which figures come from internal data vs. external Bigdata.com data.
"""


# =============================================================================
# Tool assembly
# =============================================================================
def _uc_toolkit_tools() -> list[BaseTool]:
    """UC SQL functions via UCFunctionToolkit (serverless needs databricks-connect>=15.1)."""
    from databricks_langchain import DatabricksFunctionClient, UCFunctionToolkit

    try:
        uc_client = DatabricksFunctionClient()
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Could not initialize DatabricksFunctionClient for the portfolio SQL-function "
            "tools. On serverless this needs the 'databricks-connect' package (>=15.1.0): "
            "run `%uv pip install databricks-connect` and restart, or set "
            "STRUCTURED_TOOLS = 'sql_warehouse'."
        ) from exc
    return list(UCFunctionToolkit(function_names=UC_FUNCTIONS, client=uc_client).tools)


def _sql_warehouse_tools() -> list[BaseTool]:
    """The same UC SQL functions, executed through the Statement Execution REST API.

    Needs no databricks-connect — works on Free Edition serverless notebooks and in Model
    Serving. Runs against a serverless SQL warehouse (auto-discovered when SQL_WAREHOUSE_ID
    is empty).
    """
    from databricks.sdk import WorkspaceClient
    from langchain_core.tools import tool

    ws = WorkspaceClient()

    def _warehouse_id() -> str:
        if SQL_WAREHOUSE_ID:
            return SQL_WAREHOUSE_ID
        warehouses = list(ws.warehouses.list())
        if not warehouses:
            raise RuntimeError(
                "No SQL warehouse available — start one, or set SQL_WAREHOUSE_ID in agent.py."
            )
        return warehouses[0].id

    def _run(sql: str) -> str:
        resp = ws.statement_execution.execute_statement(
            warehouse_id=_warehouse_id(), statement=sql, wait_timeout="30s"
        )
        manifest, result = resp.manifest, resp.result
        if not (manifest and manifest.schema and result and result.data_array):
            return "No rows."
        cols = [c.name for c in manifest.schema.columns]
        header = [" | ".join(cols), " | ".join("---" for _ in cols)]
        body = [
            " | ".join("" if v is None else str(v) for v in row)
            for row in result.data_array
        ]
        return "\n".join(header + body)

    def _lit(s: str) -> str:  # minimal SQL-literal escaping
        return s.replace("'", "''")

    @tool
    def get_top_holdings(max_rows: int = 10) -> str:
        """Top holdings across all portfolios ranked by total market value. Use for
        questions about largest positions, top holdings, or biggest gainers."""
        return _run(f"SELECT * FROM {CATALOG}.{SCHEMA}.get_top_holdings({int(max_rows)})")

    @tool
    def get_portfolio_positions(portfolio: str) -> str:
        """All positions in a given portfolio (e.g. PF001, PF002, PF003), including market
        value, unrealized P&L, and weight."""
        return _run(
            f"SELECT * FROM {CATALOG}.{SCHEMA}.get_portfolio_positions('{_lit(portfolio)}')"
        )

    @tool
    def get_ticker_exposure(symbol: str) -> str:
        """Total exposure to a single ticker (e.g. NVDA, AAPL) across all portfolios."""
        return _run(
            f"SELECT * FROM {CATALOG}.{SCHEMA}.get_ticker_exposure('{_lit(symbol)}')"
        )

    return [get_top_holdings, get_portfolio_positions, get_ticker_exposure]


def _internal_tools() -> list[BaseTool]:
    """Governed internal tools: structured SQL-function tools + Vector Search retriever."""
    tools: list[BaseTool] = (
        _uc_toolkit_tools() if STRUCTURED_TOOLS == "uc_toolkit" else _sql_warehouse_tools()
    )

    tools.append(
        VectorSearchRetrieverTool(
            index_name=VS_INDEX,
            num_results=5,
            columns=["doc_id", "title", "company", "ticker", "doc_type", "content"],
            tool_name="search_internal_research",
            tool_description=(
                "Semantic search over the firm's proprietary research notes — investment "
                "theses, strategy memos, and risk assessments. Use for questions about our "
                "internal view, thesis, or risk assessment on a company or the portfolio."
            ),
        )
    )

    if USE_GENIE and GENIE_SPACE_ID:
        from databricks_langchain.genie import GenieAgent

        genie = GenieAgent(
            genie_space_id=GENIE_SPACE_ID,
            genie_agent_name="portfolio_genie",
            description=(
                "Answers open-ended questions about portfolio accounts, holdings, "
                "transactions, AUM, weights, and P&L using natural-language SQL."
            ),
        )
        tools.append(
            genie.as_tool(
                name="query_portfolio_genie",
                description=(
                    "Natural-language analytics over internal portfolio data (accounts, "
                    "portfolios, holdings, transactions). Use for structured questions not "
                    "covered by the specific portfolio functions."
                ),
            )
        )

    return tools


async def _bigdata_mcp_tools() -> list[BaseTool]:
    """External tools discovered from the Bigdata.com MCP server."""
    if USE_MCP_SERVICE:
        # Governed path — Unity AI Gateway MCP Service (Databricks OAuth handled for us).
        from databricks.sdk import WorkspaceClient
        from databricks_langchain import (
            DatabricksMCPServer,
            DatabricksMultiServerMCPClient,
        )

        ws = WorkspaceClient()
        service_url = f"{ws.config.host}/ai-gateway/mcp-services/{MCP_SERVICE_NAME}"
        client = DatabricksMultiServerMCPClient(
            [DatabricksMCPServer(name="bigdata", url=service_url, workspace_client=ws)]
        )
        return await client.get_tools()

    # Direct path — remote Bigdata.com MCP, authenticated with x-api-key from a secret.
    from langchain_mcp_adapters.client import MultiServerMCPClient

    api_key = os.environ.get("BIGDATA_API_KEY")
    if not api_key:
        try:
            from databricks.sdk.runtime import dbutils  # noqa: PLC0415

            api_key = dbutils.secrets.get(scope="bigdata", key="api_key")
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "BIGDATA_API_KEY not set. Provide it as a secret-backed env var on the "
                "serving endpoint, or a Databricks secret 'bigdata/api_key' in a notebook."
            ) from exc

    client = MultiServerMCPClient(
        {
            "bigdata": {
                "url": BIGDATA_MCP_URL,
                "transport": "streamable_http",
                "headers": {"x-api-key": api_key},
            }
        }
    )
    return await client.get_tools()


def _run_async(coro):
    """Run an async coroutine from sync code, with or without a running event loop.

    Databricks notebooks execute cells *inside* a live event loop, so a bare
    ``asyncio.run()`` raises "cannot be called from a running event loop". Model Serving
    has no running loop, where ``asyncio.run()`` is correct. When a loop is already
    running we apply ``nest_asyncio`` (which also lets the LangGraph graph invoke the
    async MCP tools later during ``predict``); if it is unavailable we fall back to a
    one-shot background thread with its own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # no running loop (e.g. Model Serving)

    try:
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.run(coro)
    except Exception:
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()


def _all_tools() -> list[BaseTool]:
    return _internal_tools() + _run_async(_bigdata_mcp_tools())


# =============================================================================
# LangGraph tool-calling graph (canonical Databricks ChatAgent template)
# =============================================================================
def _build_graph(
    model: LanguageModelLike,
    tools: Sequence[BaseTool],
    system_prompt: Optional[str] = None,
) -> CompiledStateGraph:
    model = model.bind_tools(tools)

    def should_continue(state: ChatAgentState):
        last = state["messages"][-1]
        return "continue" if last.get("tool_calls") else "end"

    if system_prompt:
        preprocessor = RunnableLambda(
            lambda state: [{"role": "system", "content": system_prompt}] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(state: ChatAgentState, config: RunnableConfig):
        return {"messages": [model_runnable.invoke(state, config)]}

    workflow = StateGraph(ChatAgentState)
    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")
    return workflow.compile()


def _build_llm() -> LanguageModelLike:
    """Return the chat model based on LLM_PROVIDER."""
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            try:
                from databricks.sdk.runtime import dbutils  # noqa: PLC0415

                api_key = dbutils.secrets.get(scope="bigdata", key="openai_api_key")
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Provide it as a secret-backed env var on the "
                    "serving endpoint, or a Databricks secret 'bigdata/openai_api_key' in a "
                    "notebook."
                ) from exc
        return ChatOpenAI(model=OPENAI_MODEL, api_key=api_key, temperature=0)

    # Databricks Model Serving endpoint (foundation model or OpenAI external model).
    return ChatDatabricks(endpoint=LLM_ENDPOINT)


class FinancialIntelligenceAgent(ChatAgent):
    """Wraps the LangGraph tool-calling graph in the MLflow ChatAgent interface."""

    def __init__(self) -> None:
        self.agent = _build_graph(_build_llm(), _all_tools(), system_prompt=SYSTEM_PROMPT)

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        request = {"messages": self._convert_messages_to_dict(messages)}
        out: list[ChatAgentMessage] = []
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                out.extend(ChatAgentMessage(**msg) for msg in node_data["messages"])
        return ChatAgentResponse(messages=out)

    def predict_stream(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> Generator[ChatAgentChunk, None, None]:
        request = {"messages": self._convert_messages_to_dict(messages)}
        for event in self.agent.stream(request, stream_mode="updates"):
            for node_data in event.values():
                yield from (
                    ChatAgentChunk(delta=msg) for msg in node_data["messages"]
                )


mlflow.langchain.autolog()
AGENT = FinancialIntelligenceAgent()
mlflow.models.set_model(AGENT)
