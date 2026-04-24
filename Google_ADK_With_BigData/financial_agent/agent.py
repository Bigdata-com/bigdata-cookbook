"""Google ADK financial agent: local SQLite, local documents + FAISS, Bigdata.com MCP.

Served with ``adk web`` from the repository root (the directory that contains
``financial_agent/``):

    cd Google_ADK_With_BigData
    uv sync
    uv run adk web .

Data sources:

1. **SQLite** — Sample portfolio holdings, transactions, and P&L
2. **FAISS + Gemini embeddings** — Markdown files under ``sample_documents/`` plus
   embedded seed memos for semantic search
3. **Bigdata.com MCP** — Market tools (search, tearsheets, events) when
   ``BIGDATA_API_KEY`` is set

MCP follows Google ADK guidance for tool integration (``McpToolset``). For full
agent-to-agent delegation, see the A2A pattern in ADK docs.
"""

from __future__ import annotations

import logging
import os
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPServerParams
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)

# Package layout: Google_ADK_With_BigData/financial_agent/agent.py
_AGENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _AGENT_DIR.parent

# Load env: agent-local first, then repo root (standalone project; no parent cookbook paths).
load_dotenv(_AGENT_DIR / ".env")
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()


def _resolve_data_dir() -> Path:
    """Writable directory for SQLite (and optional future artifacts).

    Override with ``FINANCIAL_AGENT_DATA_DIR`` for tests or custom deployments.
    """
    override = (os.getenv("FINANCIAL_AGENT_DATA_DIR") or "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path
    default = _AGENT_DIR / "data"
    default.mkdir(parents=True, exist_ok=True)
    return default


_DATA_DIR = _resolve_data_dir()
DB_PATH = str(_DATA_DIR / "financial_transactions.db")

BIGDATA_API_KEY = (os.getenv("BIGDATA_API_KEY") or "").strip()
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GOOGLE_GENAI_API_KEY = GOOGLE_API_KEY or GEMINI_API_KEY

_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
MODEL = (os.getenv("GEMINI_MODEL") or "").strip() or _DEFAULT_GEMINI_MODEL

_DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
GEMINI_EMBEDDING_MODEL = (
    (os.getenv("GEMINI_EMBEDDING_MODEL") or "").strip() or _DEFAULT_GEMINI_EMBEDDING_MODEL
)


# ---------------------------------------------------------------------------
# SQLite sample database
# ---------------------------------------------------------------------------


def _create_financial_database() -> None:
    """Create and populate the SQLite database with sample portfolio data.

    Tables: accounts, portfolios, holdings, transactions.
    Idempotent: drops and recreates tables on each call.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.executescript("""
        DROP TABLE IF EXISTS transactions;
        DROP TABLE IF EXISTS holdings;
        DROP TABLE IF EXISTS portfolios;
        DROP TABLE IF EXISTS accounts;

        CREATE TABLE accounts (
            account_id TEXT PRIMARY KEY,
            account_name TEXT NOT NULL,
            account_type TEXT NOT NULL,
            currency TEXT DEFAULT 'USD',
            balance REAL DEFAULT 0
        );

        CREATE TABLE portfolios (
            portfolio_id TEXT PRIMARY KEY,
            portfolio_name TEXT NOT NULL,
            account_id TEXT,
            strategy TEXT,
            risk_profile TEXT,
            aum REAL DEFAULT 0
        );

        CREATE TABLE holdings (
            holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id TEXT,
            ticker TEXT NOT NULL,
            company_name TEXT,
            shares REAL,
            avg_cost REAL,
            current_price REAL,
            market_value REAL,
            unrealized_pnl REAL,
            weight_pct REAL
        );

        CREATE TABLE transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id TEXT,
            ticker TEXT NOT NULL,
            transaction_type TEXT NOT NULL,
            shares REAL,
            price REAL,
            amount REAL,
            fees REAL DEFAULT 0,
            transaction_date TIMESTAMP,
            notes TEXT
        );
    """)

    accounts = [
        ("ACC001", "Institutional Growth Fund", "Institutional", "USD", 50_000_000),
        ("ACC002", "Tech Innovation Portfolio", "Hedge Fund", "USD", 25_000_000),
        ("ACC003", "Global Macro Strategy", "Pension Fund", "USD", 100_000_000),
    ]

    portfolios = [
        ("PF001", "US Large Cap Growth", "ACC001", "Growth", "Moderate", 30_000_000),
        ("PF002", "AI & Semiconductor Focus", "ACC002", "Sector Focus", "Aggressive", 15_000_000),
        ("PF003", "Diversified Tech Leaders", "ACC003", "Value Growth", "Moderate", 50_000_000),
    ]

    holdings = [
        ("PF001", "AAPL", "Apple Inc.", 15000, 142.50, 185.25, 2_778_750, 641_250, 9.26),
        ("PF001", "MSFT", "Microsoft Corporation", 8000, 285.00, 415.50, 3_324_000, 1_044_000, 11.08),
        ("PF001", "GOOGL", "Alphabet Inc.", 5000, 125.00, 175.25, 876_250, 251_250, 2.92),
        ("PF001", "AMZN", "Amazon.com Inc.", 6000, 145.00, 225.75, 1_354_500, 484_500, 4.52),
        ("PF001", "META", "Meta Platforms Inc.", 4500, 280.00, 585.00, 2_632_500, 1_372_500, 8.78),
        ("PF002", "NVDA", "NVIDIA Corporation", 12000, 450.00, 875.50, 10_506_000, 5_106_000, 70.04),
        ("PF002", "AMD", "Advanced Micro Devices", 8000, 95.00, 145.25, 1_162_000, 402_000, 7.75),
        ("PF002", "AVGO", "Broadcom Inc.", 1500, 850.00, 1425.00, 2_137_500, 862_500, 14.25),
        ("PF002", "TSM", "Taiwan Semiconductor", 3000, 110.00, 185.75, 557_250, 227_250, 3.72),
        ("PF002", "PLTR", "Palantir Technologies", 25000, 18.50, 65.25, 1_631_250, 1_168_750, 10.88),
        ("PF003", "AAPL", "Apple Inc.", 25000, 155.00, 185.25, 4_631_250, 756_250, 9.26),
        ("PF003", "MSFT", "Microsoft Corporation", 15000, 310.00, 415.50, 6_232_500, 1_582_500, 12.47),
        ("PF003", "NVDA", "NVIDIA Corporation", 8000, 520.00, 875.50, 7_004_000, 2_844_000, 14.01),
        ("PF003", "CRM", "Salesforce Inc.", 10000, 215.00, 325.50, 3_255_000, 1_105_000, 6.51),
        ("PF003", "ORCL", "Oracle Corporation", 12000, 95.00, 175.25, 2_103_000, 963_000, 4.21),
    ]

    txn_types = ["BUY", "SELL", "DIVIDEND"]
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "AVGO", "PLTR", "CRM"]
    transactions: list[tuple] = []
    base_date = datetime.now() - timedelta(days=90)

    for _ in range(100):
        portfolio_id = random.choice(["PF001", "PF002", "PF003"])
        ticker = random.choice(tickers)
        txn_type = random.choices(txn_types, weights=[50, 35, 15])[0]
        shares = random.randint(50, 1500)
        price = random.uniform(80, 800)
        amount = shares * price
        txn_date = base_date + timedelta(days=random.randint(0, 90))
        notes = f"{txn_type} order for {ticker}"
        transactions.append((
            portfolio_id, ticker, txn_type, shares, round(price, 2),
            round(amount, 2), round(amount * 0.0005, 2), txn_date.isoformat(), notes,
        ))

    cursor.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?, ?)", accounts)
    cursor.executemany("INSERT INTO portfolios VALUES (?, ?, ?, ?, ?, ?)", portfolios)
    cursor.executemany(
        "INSERT INTO holdings "
        "(portfolio_id, ticker, company_name, shares, avg_cost, current_price, "
        "market_value, unrealized_pnl, weight_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        holdings,
    )
    cursor.executemany(
        "INSERT INTO transactions "
        "(portfolio_id, ticker, transaction_type, shares, price, amount, fees, "
        "transaction_date, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        transactions,
    )

    conn.commit()
    conn.close()
    logger.info("SQLite database created at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Local markdown + embedded seed documents → FAISS
# ---------------------------------------------------------------------------

_SAMPLE_DOC_METADATA: dict[str, dict[str, str]] = {
    "nvda_investment_thesis.md": {
        "ticker": "NVDA",
        "company": "NVIDIA",
        "doc_type": "investment_thesis",
        "date": "2024-12-15",
    },
    "aapl_strategic_analysis.md": {
        "ticker": "AAPL",
        "company": "Apple",
        "doc_type": "strategic_analysis",
        "date": "2024-12-12",
    },
    "msft_azure_ai.md": {
        "ticker": "MSFT",
        "company": "Microsoft",
        "doc_type": "segment_analysis",
        "date": "2024-12-08",
    },
}


def _load_sample_markdown_documents() -> list[Document]:
    """Load ``*.md`` from ``financial_agent/sample_documents/`` into LangChain documents."""
    sample_dir = _AGENT_DIR / "sample_documents"
    if not sample_dir.is_dir():
        return []
    out: list[Document] = []
    for path in sorted(sample_dir.glob("*.md")):
        body = path.read_text(encoding="utf-8").strip()
        if not body:
            continue
        meta = dict(_SAMPLE_DOC_METADATA.get(path.name, {}))
        meta.setdefault("source_file", path.name)
        meta.setdefault("doc_type", "sample_markdown")
        out.append(Document(page_content=body, metadata=meta))
    return out


_EMBEDDED_SEED_DOCUMENTS: list[Document] = [
    Document(
        page_content=(
            "AMD - Data Center & AI Opportunity Assessment\n\n"
            "Competitive Positioning:\n\n"
            "1. MI300X GPU Performance:\n"
            "   - 192GB HBM3 memory (1.5x NVIDIA H100)\n"
            "   - Strong inference performance for LLM workloads\n"
            "   - Microsoft Azure, Oracle Cloud deployments confirmed\n"
            "   - $5B+ AI GPU revenue target for 2025\n\n"
            "2. EPYC Server CPU Dominance:\n"
            "   - 33%+ server CPU market share (up from 5% in 2018)\n"
            "   - Turin (Zen 5) launching H1 2025 with 192 cores\n\n"
            "Challenges:\n"
            "   - ROCm software ecosystem still lagging CUDA\n"
            "   - NVIDIA mindshare advantage with AI developers\n\n"
            "Valuation: Trading at 35x FY25E, premium for AI optionality\n"
            "Rating: HOLD | PT: $165"
        ),
        metadata={"ticker": "AMD", "company": "AMD", "doc_type": "investment_thesis", "date": "2024-12-11"},
    ),
    Document(
        page_content=(
            "Q1 2025 Portfolio Strategy - Technology Sector Allocation\n\n"
            "Recommended Allocation Changes:\n\n"
            "INCREASE:\n"
            "- NVDA: +3% weight (AI training demand exceeds supply)\n"
            "- META: +2% weight (undervalued relative to AI investments)\n"
            "- PLTR: +1% weight (government AI contracts accelerating)\n\n"
            "MAINTAIN:\n"
            "- MSFT: Current weight (balanced growth/value)\n"
            "- AAPL: Current weight (services growth offsetting hardware)\n\n"
            "REDUCE:\n"
            "- AMD: -1% weight (valuation stretched vs execution risk)\n"
            "- CRM: -1% weight (Agentforce adoption uncertain)\n\n"
            "Key Themes to Monitor:\n"
            "1. AI inference scaling in enterprise\n"
            "2. Cloud spending reacceleration\n"
            "3. China tech policy changes"
        ),
        metadata={
            "ticker": "PORTFOLIO",
            "company": "Internal Strategy",
            "doc_type": "strategy_memo",
            "date": "2025-01-05",
        },
    ),
    Document(
        page_content=(
            "Technology Sector Risk Assessment - January 2025\n\n"
            "KEY RISKS:\n\n"
            "1. Valuation Risk (HIGH):\n"
            "   - Magnificent 7 trading at 30x+ forward P/E\n"
            "   - AI premium may compress if monetization disappoints\n\n"
            "2. Regulatory Risk (MEDIUM-HIGH):\n"
            "   - Google antitrust remedy could impact ad revenue\n"
            "   - Apple App Store ruling may reduce services margin\n"
            "   - EU Digital Markets Act enforcement increasing\n\n"
            "3. China Exposure (MEDIUM):\n"
            "   - NVDA: 20-25% revenue at risk from export controls\n"
            "   - AAPL: 18% revenue, supply chain concentration\n\n"
            "4. AI Bubble Risk (MEDIUM):\n"
            "   - Infrastructure spend may front-run actual demand\n"
            "   - ROI on enterprise AI investments still unproven\n\n"
            "HEDGING RECOMMENDATIONS:\n"
            "- Consider put spreads on QQQ for portfolio protection\n"
            "- Maintain 5-10% cash allocation for opportunities"
        ),
        metadata={
            "ticker": "PORTFOLIO",
            "company": "Risk Management",
            "doc_type": "risk_assessment",
            "date": "2025-01-10",
        },
    ),
]

RESEARCH_DOCUMENTS: list[Document] = _load_sample_markdown_documents() + _EMBEDDED_SEED_DOCUMENTS

_vector_store: FAISS | None = None


def _create_vector_store() -> FAISS:
    """Build an in-memory FAISS index over ``RESEARCH_DOCUMENTS`` using Gemini embeddings."""
    global _vector_store
    if not GOOGLE_GENAI_API_KEY:
        raise ValueError(
            "Set GOOGLE_API_KEY or GEMINI_API_KEY to embed internal research docs (Gemini embeddings)."
        )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL,
        google_api_key=GOOGLE_GENAI_API_KEY,
    )
    _vector_store = FAISS.from_documents(RESEARCH_DOCUMENTS, embeddings)
    logger.info(
        "FAISS vector store created with %d documents (embedding model=%s)",
        len(RESEARCH_DOCUMENTS),
        GEMINI_EMBEDDING_MODEL,
    )
    return _vector_store


def _warm_vector_store() -> None:
    """Build FAISS at import so embedding issues surface in server logs early."""
    try:
        _create_vector_store()
    except Exception:
        logger.exception(
            "Failed to build internal FAISS index at startup. "
            "Fix GOOGLE_API_KEY / GEMINI_API_KEY and GEMINI_EMBEDDING_MODEL, then restart `adk web`. "
            "internal_search_research will retry on first use."
        )


def _get_vector_store() -> FAISS:
    """Return the shared FAISS store, building on first access."""
    global _vector_store
    if _vector_store is None:
        _create_vector_store()
    assert _vector_store is not None
    return _vector_store


# ---------------------------------------------------------------------------
# Internal tools
# ---------------------------------------------------------------------------


def internal_query_database(sql_query: str) -> dict:
    """Run a read-only SQL query against the sample portfolio database."""
    if not sql_query.strip().upper().startswith("SELECT"):
        return {"error": "Only SELECT queries are allowed"}
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return {"query": sql_query, "row_count": len(results), "results": results[:50]}
    except Exception as e:
        return {"error": str(e)}


def internal_portfolio_summary(portfolio_id: str) -> dict:
    """Summarize one portfolio: PF001, PF002, or PF003."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT p.*, a.account_name FROM portfolios p "
            "JOIN accounts a ON p.account_id = a.account_id "
            "WHERE p.portfolio_id = ?",
            (portfolio_id,),
        )
        row = cursor.fetchone()
        portfolio = dict(row) if row else None

        cursor.execute(
            "SELECT ticker, company_name, shares, avg_cost, current_price, "
            "market_value, unrealized_pnl, weight_pct "
            "FROM holdings WHERE portfolio_id = ? ORDER BY market_value DESC",
            (portfolio_id,),
        )
        holdings = [dict(row) for row in cursor.fetchall()]

        cursor.execute(
            "SELECT ticker, transaction_type, shares, price, amount, transaction_date "
            "FROM transactions WHERE portfolio_id = ? ORDER BY transaction_date DESC LIMIT 10",
            (portfolio_id,),
        )
        transactions = [dict(row) for row in cursor.fetchall()]
        conn.close()

        total_value = sum(h["market_value"] for h in holdings)
        total_pnl = sum(h["unrealized_pnl"] for h in holdings)

        return {
            "portfolio": portfolio,
            "holdings": holdings,
            "recent_transactions": transactions,
            "summary": {
                "total_market_value": round(total_value, 2),
                "total_unrealized_pnl": round(total_pnl, 2),
                "holdings_count": len(holdings),
            },
        }
    except Exception as e:
        return {"error": str(e)}


def internal_search_research(query: str, top_k: int = 3) -> dict:
    """Semantic search over local research content (FAISS + Gemini embeddings)."""
    try:
        vs = _get_vector_store()
        docs = vs.similarity_search(query, k=top_k)
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content[:1000],
                "ticker": doc.metadata.get("ticker"),
                "company": doc.metadata.get("company"),
                "doc_type": doc.metadata.get("doc_type"),
                "date": doc.metadata.get("date"),
                "source_file": doc.metadata.get("source_file"),
            })
        return {"query": query, "results_count": len(results), "documents": results}
    except Exception as e:
        logger.exception("internal_search_research failed (query=%r)", query)
        return {
            "error": str(e),
            "error_type": type(e).__name__,
            "query": query,
            "hint": (
                "Embedding or FAISS failed. See terminal logs. "
                "Confirm GOOGLE_API_KEY or GEMINI_API_KEY and GEMINI_EMBEDDING_MODEL=gemini-embedding-001. "
                "Restart adk web after changing .env."
            ),
        }


# ---------------------------------------------------------------------------
# Bigdata.com MCP (optional)
# ---------------------------------------------------------------------------

if BIGDATA_API_KEY:
    bigdata_mcp_toolset = McpToolset(
        connection_params=StreamableHTTPServerParams(
            url="https://mcp.bigdata.com/",
            headers={"x-api-key": BIGDATA_API_KEY},
        ),
    )
else:
    logger.warning(
        "BIGDATA_API_KEY is not set. Bigdata MCP tools are disabled. "
        "Set BIGDATA_API_KEY in financial_agent/.env (or the shell) to enable."
    )
    bigdata_mcp_toolset = None

SYSTEM_PROMPT = """\
You are an intelligent financial research assistant with access to multiple data sources.

## Internal Data (Company Systems)

| Tool | Purpose |
|------|---------|
| `internal_query_database` | Execute SQL queries on portfolio/transaction database |
| `internal_portfolio_summary` | Get portfolio holdings and performance summary |
| `internal_search_research` | Search internal investment research documents (theses, risk notes, strategy memos) |

## External Data (Bigdata.com MCP)

Tools dynamically loaded from the Bigdata MCP server, including:
- **bigdata_search** — Search financial news and market intelligence
- **bigdata_company_tearsheet** — Get structured company profile, financials, and key metrics
- **bigdata_country_tearsheet** — Get macroeconomic and country-level data
- **bigdata_events_calendar** — Upcoming earnings dates and corporate events
- **find_companies** — Resolve company names / tickers to Bigdata entity IDs

## Tool Selection Guidelines

1. **Portfolio / holdings / transactions** → use `internal_query_database` or `internal_portfolio_summary`
2. **Investment research / theses / risk notes** → use `internal_search_research`
3. **Live market news / recent developments** → use `bigdata_search`
4. **Company profile / financials / key metrics** → use `bigdata_company_tearsheet` (call `find_companies` first to get the entity ID)
5. **Earnings dates / events** → use `bigdata_events_calendar`
6. **Multi-source analysis** → combine internal + external tools for comprehensive answers

If a tool returns an `error` field, quote that message (and `hint` if present) so the user can fix configuration.

## Inline Attribution (IMPORTANT)

When using data from Bigdata.com MCP tools, you MUST:
- Add **inline citations** with the source name as clickable link text: [Source Name](url)
- Weave citations naturally into the text next to the fact they support
- Do NOT add a separate "Sources" or "References" section at the end
- Every material claim from an external source must have an inline citation

Example: NVIDIA reported record data center revenue of $18.4B in Q4 ([Reuters](https://reuters.com/...)).

## Formatting

- Use tables for structured comparisons (holdings, metrics, P&L)
- Use bullet points for lists of findings
- Bold key numbers and ratings
- Keep responses concise and actionable

## Available Portfolios

- **PF001** — US Large Cap Growth (ACC001)
- **PF002** — AI & Semiconductor Focus (ACC002)
- **PF003** — Diversified Tech Leaders (ACC003)

Do NOT offer follow-up questions at the end of your response.
"""

# ---------------------------------------------------------------------------
# Startup: SQLite always; FAISS warmup optional for tests
# ---------------------------------------------------------------------------

_create_financial_database()

_skip_warm = (os.getenv("FINANCIAL_AGENT_SKIP_EMBED_WARMUP") or "").strip().lower() in (
    "1",
    "true",
    "yes",
)
if _skip_warm:
    logger.info("Skipping FAISS warmup (FINANCIAL_AGENT_SKIP_EMBED_WARMUP is set).")
else:
    _warm_vector_store()

_internal_tools = [
    FunctionTool(func=internal_query_database),
    FunctionTool(func=internal_portfolio_summary),
    FunctionTool(func=internal_search_research),
]

root_agent = Agent(
    model=MODEL,
    name="financial_agent",
    description=(
        "Financial research assistant combining internal portfolio data "
        "(SQLite + FAISS over local documents) with Bigdata.com market intelligence via MCP."
    ),
    instruction=SYSTEM_PROMPT,
    tools=_internal_tools + ([bigdata_mcp_toolset] if bigdata_mcp_toolset else []),
)
