"""
Agentic AI Demo Core Module

Reusable components for building AI agents that integrate:
- Bigdata.com Search & Knowledge Graph APIs
- SQLite financial database
- FAISS vector store for research documents
- LangSmith observability

Usage:
    from langgraph_core import (
        setup_environment,
        create_financial_database,
        create_vector_store,
        get_bigdata_tools,
        get_database_tools,
        get_vectorstore_tools,
        create_agent,
        run_agent_query
    )
"""

import os
import json
import sqlite3
import random
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

# LangChain imports
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.agents import create_agent as langchain_create_agent

# Local imports for Research Agent
from research_client import ResearchClient, ResearchResult

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Configuration for the demo environment."""
    bigdata_api_key: str = ""
    openai_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = "bigdata-agent-demo"
    bigdata_base_url: str = "https://api.bigdata.com/v1"
    db_path: str = "output/financial_transactions.db"
    enable_tracing: bool = True


# Global config instance
_config: Optional[Config] = None
_vector_store: Optional[FAISS] = None


def setup_environment(
    bigdata_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    langsmith_api_key: Optional[str] = None,
    langsmith_project: str = "bigdata-agent-demo",
    db_path: str = "output/financial_transactions.db",
    enable_tracing: bool = True
) -> Config:
    """
    Setup environment with API keys and configure LangSmith tracing.
    
    Args:
        bigdata_api_key: Bigdata.com API key (or use BIGDATA_API_KEY env var)
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        langsmith_api_key: LangSmith API key (or use LANGSMITH_API_KEY env var)
        langsmith_project: LangSmith project name for tracing
        db_path: Path to SQLite database
        enable_tracing: Enable LangSmith tracing
    
    Returns:
        Config object with all settings
    """
    global _config
    
    # Load from env if not provided
    _config = Config(
        bigdata_api_key=bigdata_api_key or os.getenv("BIGDATA_API_KEY", ""),
        openai_api_key=openai_api_key or os.getenv("OPENAI_API_KEY", ""),
        langsmith_api_key=langsmith_api_key or os.getenv("LANGSMITH_API_KEY", ""),
        langsmith_project=langsmith_project,
        db_path=db_path,
        enable_tracing=enable_tracing
    )
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Setup LangSmith tracing
    if enable_tracing and _config.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = _config.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = _config.langsmith_project
        print(f"✅ LangSmith tracing enabled → Project: {_config.langsmith_project}")
    elif enable_tracing:
        print("⚠️ LangSmith API key not set. Tracing disabled.")
        print("   Set via: export LANGSMITH_API_KEY='your-key'")
    
    # Validate keys
    if _config.bigdata_api_key:
        print(f"✅ Bigdata API Key: {_config.bigdata_api_key[:10]}...")
    else:
        print("⚠️ BIGDATA_API_KEY not set")
    
    if _config.openai_api_key:
        print(f"✅ OpenAI API Key: {_config.openai_api_key[:10]}...")
    else:
        print("⚠️ OPENAI_API_KEY not set")
    
    return _config


def get_config() -> Config:
    """Get current configuration."""
    global _config
    if _config is None:
        _config = setup_environment()
    return _config


# ============================================================================
# DATABASE SETUP
# ============================================================================

def create_financial_database(db_path: Optional[str] = None) -> str:
    """
    Create and populate SQLite database with realistic financial transactions.
    
    Args:
        db_path: Path to database file (uses config default if not provided)
    
    Returns:
        Path to created database
    """
    config = get_config()
    db_path = db_path or config.db_path
    
    conn = sqlite3.connect(db_path)
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
    
    # Sample data
    accounts = [
        ('ACC001', 'Institutional Growth Fund', 'Institutional', 'USD', 50000000),
        ('ACC002', 'Tech Innovation Portfolio', 'Hedge Fund', 'USD', 25000000),
        ('ACC003', 'Global Macro Strategy', 'Pension Fund', 'USD', 100000000),
    ]
    
    portfolios = [
        ('PF001', 'US Large Cap Growth', 'ACC001', 'Growth', 'Moderate', 30000000),
        ('PF002', 'AI & Semiconductor Focus', 'ACC002', 'Sector Focus', 'Aggressive', 15000000),
        ('PF003', 'Diversified Tech Leaders', 'ACC003', 'Value Growth', 'Moderate', 50000000),
    ]
    
    holdings = [
        ('PF001', 'AAPL', 'Apple Inc.', 15000, 142.50, 185.25, 2778750, 641250, 9.26),
        ('PF001', 'MSFT', 'Microsoft Corporation', 8000, 285.00, 415.50, 3324000, 1044000, 11.08),
        ('PF001', 'GOOGL', 'Alphabet Inc.', 5000, 125.00, 175.25, 876250, 251250, 2.92),
        ('PF001', 'AMZN', 'Amazon.com Inc.', 6000, 145.00, 225.75, 1354500, 484500, 4.52),
        ('PF001', 'META', 'Meta Platforms Inc.', 4500, 280.00, 585.00, 2632500, 1372500, 8.78),
        ('PF002', 'NVDA', 'NVIDIA Corporation', 12000, 450.00, 875.50, 10506000, 5106000, 70.04),
        ('PF002', 'AMD', 'Advanced Micro Devices', 8000, 95.00, 145.25, 1162000, 402000, 7.75),
        ('PF002', 'AVGO', 'Broadcom Inc.', 1500, 850.00, 1425.00, 2137500, 862500, 14.25),
        ('PF002', 'TSM', 'Taiwan Semiconductor', 3000, 110.00, 185.75, 557250, 227250, 3.72),
        ('PF002', 'PLTR', 'Palantir Technologies', 25000, 18.50, 65.25, 1631250, 1168750, 10.88),
        ('PF003', 'AAPL', 'Apple Inc.', 25000, 155.00, 185.25, 4631250, 756250, 9.26),
        ('PF003', 'MSFT', 'Microsoft Corporation', 15000, 310.00, 415.50, 6232500, 1582500, 12.47),
        ('PF003', 'NVDA', 'NVIDIA Corporation', 8000, 520.00, 875.50, 7004000, 2844000, 14.01),
        ('PF003', 'CRM', 'Salesforce Inc.', 10000, 215.00, 325.50, 3255000, 1105000, 6.51),
        ('PF003', 'ORCL', 'Oracle Corporation', 12000, 95.00, 175.25, 2103000, 963000, 4.21),
    ]
    
    # Generate transactions
    txn_types = ['BUY', 'SELL', 'DIVIDEND']
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'AMD', 'AVGO', 'PLTR', 'CRM']
    transactions = []
    base_date = datetime.now() - timedelta(days=90)
    
    for i in range(100):
        portfolio_id = random.choice(['PF001', 'PF002', 'PF003'])
        ticker = random.choice(tickers)
        txn_type = random.choices(txn_types, weights=[50, 35, 15])[0]
        shares = random.randint(50, 1500)
        price = random.uniform(80, 800)
        amount = shares * price
        txn_date = base_date + timedelta(days=random.randint(0, 90))
        notes = f'{txn_type} order for {ticker}'
        transactions.append((portfolio_id, ticker, txn_type, shares, round(price, 2), 
                           round(amount, 2), round(amount * 0.0005, 2), txn_date.isoformat(), notes))
    
    cursor.executemany("INSERT INTO accounts VALUES (?, ?, ?, ?, ?)", accounts)
    cursor.executemany("INSERT INTO portfolios VALUES (?, ?, ?, ?, ?, ?)", portfolios)
    cursor.executemany("INSERT INTO holdings (portfolio_id, ticker, company_name, shares, avg_cost, current_price, market_value, unrealized_pnl, weight_pct) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", holdings)
    cursor.executemany("INSERT INTO transactions (portfolio_id, ticker, transaction_type, shares, price, amount, fees, transaction_date, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", transactions)
    
    conn.commit()
    
    # Summary
    counts = {}
    for table in ['accounts', 'portfolios', 'holdings', 'transactions']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        counts[table] = cursor.fetchone()[0]
        print(f"✅ Created {counts[table]} {table}")
    
    conn.close()
    return db_path


# ============================================================================
# VECTOR STORE SETUP
# ============================================================================

# Sample research documents
RESEARCH_DOCUMENTS = [
    Document(
        page_content="""NVIDIA Q4 2024 Investment Thesis Update

NVIDIA remains our top pick in the semiconductor space. Key highlights:

1. Data Center Revenue: $18.4B (+409% YoY) driven by H100/H200 GPU demand for AI training
2. Blackwell Architecture: Next-gen B100/B200 GPUs launching Q2 2025 with 2.5x performance
3. Software Moat: CUDA ecosystem has 4M+ developers, creating significant switching costs
4. AI Inference Opportunity: $150B TAM by 2027 as enterprises deploy AI at scale

Risk Factors: China export restrictions, AMD competition, supply constraints
Price Target: $950 (25x FY26E EPS)
Rating: STRONG BUY""",
        metadata={"ticker": "NVDA", "company": "NVIDIA", "doc_type": "investment_thesis", "date": "2024-12-15"}
    ),
    Document(
        page_content="""Apple Inc. Strategic Analysis - Services & AI Focus

Key Investment Points:

1. Services Segment ($96B ARR): Highest-margin business (70%+ gross margin)
   - App Store, Apple Music, iCloud, Apple TV+, Apple Pay
   - 1B+ paid subscriptions across ecosystem

2. Apple Intelligence (AI Strategy):
   - On-device AI processing preserving privacy
   - Partnership with OpenAI for ChatGPT integration
   - Siri 2.0 with LLM capabilities launching iOS 18.4

3. iPhone 16 Cycle:
   - AI features driving upgrade demand
   - Pro models with A18 Pro chip outperforming

Valuation: Trading at 28x FY25E P/E, premium justified by ecosystem strength
Price Target: $210""",
        metadata={"ticker": "AAPL", "company": "Apple", "doc_type": "strategic_analysis", "date": "2024-12-12"}
    ),
    Document(
        page_content="""Microsoft Azure & AI Monetization Analysis

Cloud & AI Revenue Breakdown:

1. Azure Growth: +29% YoY (Q1 FY25)
   - AI services contributing 12 percentage points to growth
   - 60K+ Azure AI customers (2x YoY)
   - OpenAI partnership generating $3B+ annual revenue

2. Copilot Monetization:
   - Microsoft 365 Copilot: $30/user/month (400K+ enterprise customers)
   - GitHub Copilot: 1.8M paid subscribers (+40% QoQ)
   - Security Copilot: Fastest-growing enterprise product

3. Enterprise Moat:
   - Office 365 installed base: 400M+ users
   - Teams MAU: 320M (dominant collaboration platform)

Price Target: $475 | Rating: OVERWEIGHT""",
        metadata={"ticker": "MSFT", "company": "Microsoft", "doc_type": "segment_analysis", "date": "2024-12-08"}
    ),
    Document(
        page_content="""AMD - Data Center & AI Opportunity Assessment

Competitive Positioning:

1. MI300X GPU Performance:
   - 192GB HBM3 memory (1.5x NVIDIA H100)
   - Strong inference performance for LLM workloads
   - Microsoft Azure, Oracle Cloud deployments confirmed
   - $5B+ AI GPU revenue target for 2025

2. EPYC Server CPU Dominance:
   - 33%+ server CPU market share (up from 5% in 2018)
   - Turin (Zen 5) launching H1 2025 with 192 cores

Challenges:
   - ROCm software ecosystem still lagging CUDA
   - NVIDIA mindshare advantage with AI developers

Valuation: Trading at 35x FY25E, premium for AI optionality
Rating: HOLD | PT: $165""",
        metadata={"ticker": "AMD", "company": "AMD", "doc_type": "investment_thesis", "date": "2024-12-11"}
    ),
    Document(
        page_content="""Q1 2025 Portfolio Strategy - Technology Sector Allocation

Recommended Allocation Changes:

INCREASE:
- NVDA: +3% weight (AI training demand exceeds supply)
- META: +2% weight (undervalued relative to AI investments)
- PLTR: +1% weight (government AI contracts accelerating)

MAINTAIN:
- MSFT: Current weight (balanced growth/value)
- AAPL: Current weight (services growth offsetting hardware)

REDUCE:
- AMD: -1% weight (valuation stretched vs execution risk)
- CRM: -1% weight (Agentforce adoption uncertain)

Key Themes to Monitor:
1. AI inference scaling in enterprise
2. Cloud spending reacceleration
3. China tech policy changes""",
        metadata={"ticker": "PORTFOLIO", "company": "Internal Strategy", "doc_type": "strategy_memo", "date": "2025-01-05"}
    ),
    Document(
        page_content="""Technology Sector Risk Assessment - January 2025

KEY RISKS:

1. Valuation Risk (HIGH):
   - Magnificent 7 trading at 30x+ forward P/E
   - AI premium may compress if monetization disappoints

2. Regulatory Risk (MEDIUM-HIGH):
   - Google antitrust remedy could impact ad revenue
   - Apple App Store ruling may reduce services margin
   - EU Digital Markets Act enforcement increasing

3. China Exposure (MEDIUM):
   - NVDA: 20-25% revenue at risk from export controls
   - AAPL: 18% revenue, supply chain concentration

4. AI Bubble Risk (MEDIUM):
   - Infrastructure spend may front-run actual demand
   - ROI on enterprise AI investments still unproven

HEDGING RECOMMENDATIONS:
- Consider put spreads on QQQ for portfolio protection
- Maintain 5-10% cash allocation for opportunities""",
        metadata={"ticker": "PORTFOLIO", "company": "Risk Management", "doc_type": "risk_assessment", "date": "2025-01-10"}
    ),
]


def create_vector_store(documents: Optional[List[Document]] = None) -> FAISS:
    """
    Create FAISS vector store from research documents.
    
    Args:
        documents: List of Document objects (uses default research docs if not provided)
    
    Returns:
        FAISS vector store instance
    """
    global _vector_store
    
    config = get_config()
    docs = documents or RESEARCH_DOCUMENTS
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=config.openai_api_key
    )
    
    _vector_store = FAISS.from_documents(docs, embeddings)
    print(f"✅ Created vector store with {len(docs)} documents")
    
    return _vector_store


def get_vector_store() -> FAISS:
    """Get current vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = create_vector_store()
    return _vector_store


# ============================================================================
# BIGDATA.COM API TOOLS
# ============================================================================

def get_bigdata_tools() -> List[Callable]:
    """
    Get Bigdata.com API tools for the agent.
    
    Returns:
        List of tool functions for Knowledge Graph and Search APIs
    """
    config = get_config()
    
    @tool
    def bigdata_lookup_company(ticker: str) -> str:
        """Look up a company's Bigdata entity ID using the Knowledge Graph API.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'NVDA')
        
        Returns:
            JSON string with entity ID and company details
        """
        try:
            url = f"{config.bigdata_base_url}/knowledge-graph/companies"
            headers = {"X-API-KEY": config.bigdata_api_key, "Content-Type": "application/json"}
            
            response = requests.post(url, headers=headers, json={"query": ticker}, timeout=30)
            response.raise_for_status()
            
            results = response.json().get("results", [])
            if results:
                company = results[0]
                return json.dumps({
                    "ticker": ticker,
                    "entity_id": company.get("id"),
                    "name": company.get("name"),
                    "country": company.get("country"),
                    "description": company.get("description", "")[:200]
                }, indent=2)
            return json.dumps({"error": f"No entity found for ticker {ticker}"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def bigdata_search_news(
        query: str, 
        entity_id: Optional[str] = None, 
        days_back: int = 90, 
        max_results: int = 10
    ) -> str:
        """Search Bigdata.com for financial news and market intelligence.
        
        Args:
            query: Natural language search query (e.g., 'AI chip demand', 'earnings guidance')
            entity_id: Optional Bigdata entity ID to filter by company
            days_back: Number of days to search back (default: 90)
            max_results: Maximum number of results to return (default: 10)
        
        Returns:
            JSON string with search results including headlines, sources, and relevant text
        """
        try:
            url = f"{config.bigdata_base_url}/search"
            headers = {"X-API-KEY": config.bigdata_api_key, "Content-Type": "application/json"}
            
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days_back)
            
            request_body = {
                "query": {
                    "text": query,
                    "filters": {
                        "timestamp": {
                            "start": start_date.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                            "end": end_date.strftime("%Y-%m-%dT%H:%M:%S.999Z")
                        },
                        "category": {
                            "mode": "INCLUDE",
                            "values": ["news_public"]
                        }
                    },
                    "max_chunks": max_results * 2
                }
            }
            
            if entity_id:
                request_body["query"]["filters"]["entity"] = {"any_of": [entity_id]}
            
            response = requests.post(url, headers=headers, json=request_body, timeout=60)
            response.raise_for_status()
            
            results = response.json().get("results", [])
            
            formatted = []
            for doc in results[:max_results]:
                chunks_text = " | ".join([c.get("text", "")[:300] for c in doc.get("chunks", [])[:2]])
                formatted.append({
                    "headline": doc.get("headline"),
                    "source": doc.get("source", {}).get("name"),
                    "timestamp": doc.get("timestamp"),
                    "sentiment": doc.get("chunks", [{}])[0].get("sentiment") if doc.get("chunks") else None,
                    "url": doc.get("url"),
                    "relevant_text": chunks_text[:500]
                })
            
            return json.dumps({"query": query, "results_count": len(formatted), "results": formatted}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return [bigdata_lookup_company, bigdata_search_news]


def get_research_agent_tool() -> List[Callable]:
    """
    Get the Bigdata Research Agent tool for deep research queries.
    
    The Research Agent performs multi-step reasoning with RAG across
    web, premium sources, and uploaded content.
    
    Uses the ResearchClient from research_client.py for robust citation handling.
    
    Returns:
        List containing the research agent tool
    """
    config = get_config()
    
    @tool
    def bigdata_research_agent(
        query: str,
        research_effort: str = "standard"
    ) -> str:
        """Perform deep research using Bigdata.com Research Agent with full citations.
        
        Use this tool for complex research questions that require:
        - Multi-step reasoning and analysis
        - Synthesis across multiple sources
        - Comprehensive answers with inline citations [1], [2], etc.
        
        This is more powerful than bigdata_search_news but takes longer (20-60 seconds).
        
        Args:
            query: Research question or analysis request. Can include formatting instructions.
            research_effort: "lite" (10-20s, quick facts) or "standard" (20-60s, deep analysis)
        
        Returns:
            JSON string with answer containing inline citations and numbered source details
        """
        try:
            # Use ResearchClient for robust citation handling
            client = ResearchClient(api_key=config.bigdata_api_key)
            result: ResearchResult = client.research(
                message=query,
                research_effort=research_effort,
                days_back=90,  # Last 90 days
                source_categories=["news_public"]  # Filter to public news only
            )
            
            # Get answer with inline citation numbers [1], [2], etc.
            answer_with_citations = result.get_answer_with_citations()
            
            # Get numbered citations that match the inline numbers
            numbered_citations = result.get_numbered_citations()
            
            # Format citations for display
            formatted_citations = []
            for citation in numbered_citations[:25]:  # Limit to 25
                formatted_citation = {
                    "number": citation.get("number"),
                    "headline": citation.get("headline"),
                    "source": citation.get("source", {}).get("name") if isinstance(citation.get("source"), dict) else citation.get("source"),
                    "timestamp": citation.get("timestamp"),
                    "url": citation.get("url"),
                }
                # Add text excerpt if available from chunks
                if citation.get("chunks"):
                    first_chunk = citation["chunks"][0] if citation["chunks"] else {}
                    formatted_citation["text"] = first_chunk.get("text", "")[:300]
                formatted_citations.append(formatted_citation)
            
            return json.dumps({
                "query": query,
                "answer": answer_with_citations,
                "citations_count": len(formatted_citations),
                "citations": formatted_citations,
                "processing_time_ms": result.processing_time_ms,
                "chat_id": result.chat_id
            }, indent=2)
            
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return [bigdata_research_agent]


# ============================================================================
# DATABASE TOOLS
# ============================================================================

def get_database_tools(db_path: Optional[str] = None) -> List[Callable]:
    """
    Get SQLite database tools for the agent.
    
    Args:
        db_path: Path to SQLite database (uses config default if not provided)
    
    Returns:
        List of tool functions for database queries
    """
    config = get_config()
    _db_path = db_path or config.db_path
    
    @tool
    def internal_query_database(sql_query: str) -> str:
        """Execute SQL query against the internal financial transactions database.
        
        Available tables:
        - accounts: account_id, account_name, account_type, currency, balance
        - portfolios: portfolio_id, portfolio_name, account_id, strategy, risk_profile, aum
        - holdings: portfolio_id, ticker, company_name, shares, avg_cost, current_price, market_value, unrealized_pnl
        - transactions: portfolio_id, ticker, transaction_type, shares, price, amount, fees, transaction_date, notes
        
        Args:
            sql_query: SQL SELECT query to execute
        
        Returns:
            JSON string with query results
        """
        try:
            if not sql_query.strip().upper().startswith("SELECT"):
                return json.dumps({"error": "Only SELECT queries are allowed"})
            
            conn = sqlite3.connect(_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(sql_query)
            results = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return json.dumps({"query": sql_query, "row_count": len(results), "results": results[:50]}, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @tool
    def internal_portfolio_summary(portfolio_id: str) -> str:
        """Get a summary of a specific portfolio from internal database including holdings and performance.
        
        Args:
            portfolio_id: Portfolio identifier (e.g., 'PF001', 'PF002', 'PF003')
        
        Returns:
            JSON string with portfolio details, holdings, and recent transactions
        """
        try:
            conn = sqlite3.connect(_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT p.*, a.account_name FROM portfolios p JOIN accounts a ON p.account_id = a.account_id WHERE p.portfolio_id = ?", (portfolio_id,))
            row = cursor.fetchone()
            portfolio = dict(row) if row else None
            
            cursor.execute("SELECT ticker, company_name, shares, avg_cost, current_price, market_value, unrealized_pnl FROM holdings WHERE portfolio_id = ? ORDER BY market_value DESC", (portfolio_id,))
            holdings = [dict(row) for row in cursor.fetchall()]
            
            cursor.execute("SELECT ticker, transaction_type, shares, price, amount, transaction_date FROM transactions WHERE portfolio_id = ? ORDER BY transaction_date DESC LIMIT 10", (portfolio_id,))
            transactions = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            
            total_value = sum(h["market_value"] for h in holdings)
            total_pnl = sum(h["unrealized_pnl"] for h in holdings)
            
            return json.dumps({
                "portfolio": portfolio,
                "holdings": holdings,
                "recent_transactions": transactions,
                "summary": {"total_market_value": round(total_value, 2), "total_unrealized_pnl": round(total_pnl, 2), "holdings_count": len(holdings)}
            }, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return [internal_query_database, internal_portfolio_summary]


# ============================================================================
# VECTOR STORE TOOLS
# ============================================================================

def get_vectorstore_tools() -> List[Callable]:
    """
    Get vector store search tools for the agent.
    
    Returns:
        List of tool functions for semantic search
    """
    @tool
    def internal_search_research(query: str, top_k: int = 3) -> str:
        """Search internal research documents using semantic similarity.
        
        This searches through investment theses, competitive analyses, strategy memos, and risk assessments.
        
        Args:
            query: Natural language query about companies or investment topics
            top_k: Number of documents to return (default: 3)
        
        Returns:
            JSON string with relevant research document excerpts
        """
        try:
            vs = get_vector_store()
            docs = vs.similarity_search(query, k=top_k)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content[:1000],
                    "ticker": doc.metadata.get("ticker"),
                    "company": doc.metadata.get("company"),
                    "doc_type": doc.metadata.get("doc_type"),
                    "date": doc.metadata.get("date")
                })
            
            return json.dumps({"query": query, "results_count": len(results), "documents": results}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    return [internal_search_research]


# ============================================================================
# AGENT CREATION
# ============================================================================

DEFAULT_SYSTEM_PROMPT = """You are an intelligent financial research assistant with access to multiple data sources:

**External Data (Bigdata.com APIs):**
1. `bigdata_lookup_company` - Look up company entity IDs from the Knowledge Graph
2. `bigdata_search_news` - Search financial news and market intelligence with source citations

**Internal Data (Company Systems):**
3. `internal_query_database` - Execute SQL queries on portfolio/transaction database
4. `internal_portfolio_summary` - Get portfolio holdings and performance summary
5. `internal_search_research` - Search internal investment research documents

Guidelines:
- For company news, first use `bigdata_lookup_company` to get the entity_id, then use it in `bigdata_search_news`
- For portfolio questions, use `internal_portfolio_summary` or `internal_query_database` with SQL
- Combine external market data with internal holdings/research for comprehensive analysis

**IMPORTANT - Source Attribution:**
- ALWAYS cite your sources clearly in your response
- For Bigdata.com data: Include inline citation numbers [1], [2] if provided in the tool response
- For internal data: Mention "From internal database" or "According to our internal research"
- Preserve any citation markers from Bigdata.com Research Agent responses
- When presenting external data, attribute it to Bigdata.com

Available portfolios: PF001 (US Large Cap Growth), PF002 (AI & Semiconductor Focus), PF003 (Diversified Tech Leaders)
"""


def create_agent(
    tools: Optional[List[Callable]] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    model: str = "gpt-4o",
    temperature: float = 0
):
    """
    Create a LangGraph ReAct agent with the specified tools.
    
    Args:
        tools: List of tool functions (uses all default tools if not provided)
        system_prompt: System prompt for the agent
        model: OpenAI model to use
        temperature: Model temperature
    
    Returns:
        LangGraph agent instance
    """
    config = get_config()
    
    # Default tools if not provided
    if tools is None:
        tools = (
            get_bigdata_tools() + 
            get_database_tools() + 
            get_vectorstore_tools()
        )
    
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=config.openai_api_key
    )
    
    agent = langchain_create_agent(llm, tools, system_prompt=system_prompt)
    
    print(f"✅ Agent created with {len(tools)} tools")
    return agent


# ============================================================================
# HIERARCHICAL AGENT (Agent-to-Agent)
# ============================================================================

HIERARCHICAL_SYSTEM_PROMPT = """You are a senior financial research analyst with access to both internal company systems and the Bigdata.com Research Agent for external research.

**IMPORTANT: Follow this research hierarchy:**

1. **ALWAYS check internal sources FIRST:**
   - `internal_query_database` - Check our portfolio positions, transactions, and account data
   - `internal_portfolio_summary` - Get quick overview of specific portfolios  
   - `internal_search_research` - Search our internal research documents, investment theses, and analyst notes

2. **For entity resolution (optional):**
   - `bigdata_lookup_company` - Get Bigdata entity IDs for companies (useful for identification)

3. **ESCALATE to Research Agent when internal sources are insufficient:**
   - `bigdata_research_agent` - Use when you need:
     * Deep analysis requiring synthesis across many external sources
     * Market-wide trends, macro analysis, or competitive intelligence
     * Information about companies we don't hold or haven't researched internally
     * Current events, regulatory changes, breaking news, or credit analysis
     * Any question that internal sources cannot answer
   - Note: Research Agent takes 20-60 seconds but provides comprehensive analysis with citations

**Decision Framework:**
- Portfolio/holdings questions → Internal DB first
- Company we hold → Internal research first, escalate to Research Agent if more info needed
- Company we don't hold → Research Agent
- Market trends/macro/credit analysis → Research Agent
- Current news/events → Research Agent

**CRITICAL - Source Attribution:**
- ALWAYS preserve inline citation numbers [1], [2], [3] from Bigdata.com Research Agent responses
- These citations link to verified sources - DO NOT remove or modify them
- For internal data: Clearly state "From internal database" or "According to internal research"
- When synthesizing from multiple sources, attribute each fact to its source
- The citations demonstrate data provenance and build client confidence

**Available Portfolios:** PF001 (US Large Cap Growth), PF002 (AI & Semiconductor Focus), PF003 (Diversified Tech Leaders)

Always explain which sources you consulted and why. Check internal sources first, then escalate to Research Agent for external data.
"""


def create_hierarchical_agent(
    model: str = "gpt-4o",
    temperature: float = 0,
    include_research_agent: bool = True
):
    """
    Create a hierarchical agent that checks internal sources first, then escalates to Research Agent.
    
    This implements an agent-to-agent pattern where:
    1. Internal tools (DB, vector store) are checked first
    2. Entity lookup for company identification
    3. Bigdata Research Agent for deep research when internal sources are insufficient
    
    Note: This agent does NOT include bigdata_search_news - it's designed for
    internal-first research with escalation to the Research Agent for external data.
    
    Args:
        model: OpenAI model to use
        temperature: Model temperature
        include_research_agent: Whether to include the deep research agent tool
    
    Returns:
        LangGraph agent instance configured for hierarchical research
    """
    config = get_config()
    
    # Build tool list - internal first
    tools = (
        get_database_tools() +           # Internal DB
        get_vectorstore_tools()          # Internal research docs
    )
    
    # Add only bigdata_lookup_company (not search) for entity resolution
    bigdata_tools = get_bigdata_tools()
    lookup_tool = [t for t in bigdata_tools if t.name == 'bigdata_lookup_company']
    tools += lookup_tool
    
    # Add research agent for escalation (the main external tool)
    if include_research_agent:
        tools += get_research_agent_tool()
    
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=config.openai_api_key
    )
    
    agent = langchain_create_agent(llm, tools, system_prompt=HIERARCHICAL_SYSTEM_PROMPT)
    
    tool_names = [t.name for t in tools]
    print(f"✅ Hierarchical agent created with {len(tools)} tools:")
    print(f"   Internal: {[n for n in tool_names if n.startswith('internal_')]}")
    print(f"   External: {[n for n in tool_names if n.startswith('bigdata_')]}")
    
    return agent


# ============================================================================
# AGENT EXECUTION
# ============================================================================

def run_agent_query(
    agent,
    query: str,
    verbose: bool = True,
    return_tools: bool = False
) -> Dict[str, Any]:
    """
    Run a query through the agent and return results.
    
    Args:
        agent: LangGraph agent instance
        query: User query string
        verbose: Print tool calls during execution
        return_tools: Include tool call details in return value
    
    Returns:
        Dict with 'response' and optionally 'tools' keys
    """
    messages = [{"role": "user", "content": query}]
    final_response = None
    tool_calls = []
    tool_results = []
    
    for event in agent.stream({"messages": messages}, stream_mode="values"):
        last_message = event["messages"][-1]
        msg_type = type(last_message).__name__
        
        # Capture tool calls from AI messages
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tc in last_message.tool_calls:
                tool_calls.append({
                    "name": tc['name'],
                    "args": tc['args']
                })
                if verbose:
                    args_str = json.dumps(tc['args'], indent=2)[:150]
                    print(f"🔧 {tc['name']}: {args_str}...")
        
        # Capture tool results
        if msg_type == 'ToolMessage':
            tool_results.append(last_message.content)
        
        # Capture final AI response (AIMessage without tool_calls)
        if msg_type == 'AIMessage':
            if hasattr(last_message, 'content') and last_message.content:
                if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                    final_response = last_message.content
    
    # If no final response but we have tool results, use the last tool result
    if not final_response and tool_results:
        final_response = tool_results[-1]
    
    result = {"response": final_response, "tools_count": len(tool_calls)}
    if return_tools:
        result["tools"] = tool_calls
        result["tool_results"] = tool_results
    
    return result


def display_agent_response(
    agent,
    query: str,
    verbose: bool = True,
    show_json: bool = False
):
    """
    Run agent query and display results with nice formatting (for Jupyter notebooks).
    Shows citations prominently for Bigdata.com Research Agent responses.
    
    Args:
        agent: LangGraph agent instance
        query: User query string
        verbose: Print tool calls during execution
        show_json: Show raw JSON response at the end (default: False)
    """
    from IPython.display import display, Markdown, HTML
    import html as html_lib
    
    # Header
    display(HTML(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h3 style="color: white; margin: 0;">🔍 Query</h3>
        <p style="color: #e0e0e0; margin: 10px 0 0 0; font-size: 14px;">{html_lib.escape(query.strip())}</p>
    </div>
    """))
    
    result = run_agent_query(agent, query, verbose=verbose, return_tools=True)
    
    # Tools summary
    if result.get("tools"):
        tools_html = "".join([
            f"<span style='background: #1565C0; color: #ffffff; padding: 6px 12px; border-radius: 6px; margin: 4px; display: inline-block; font-family: monospace; font-size: 13px; font-weight: 500;'>{html_lib.escape(t['name'])}</span>" 
            for t in result["tools"]
        ])
        display(HTML(f"""
        <div style="background: #263238; padding: 16px 20px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #42A5F5;">
            <b style="color: #90CAF9; font-size: 14px;">📊 Tools Used ({result['tools_count']}):</b>
            <div style="margin-top: 12px;">{tools_html}</div>
        </div>
        """))
    
    # Response - format nicely
    response_text = result.get("response") or "No response generated"
    
    # Check if response is JSON and format it nicely
    try:
        if response_text.strip().startswith('{') or response_text.strip().startswith('['):
            parsed = json.loads(response_text)
            # Format as a nice summary if it's a DB result
            if isinstance(parsed, dict) and 'results' in parsed:
                formatted_parts = []
                if 'query' in parsed:
                    formatted_parts.append(f"**Query:** `{parsed['query']}`")
                if 'row_count' in parsed:
                    formatted_parts.append(f"**Results:** {parsed['row_count']} rows")
                if parsed.get('results'):
                    formatted_parts.append("\n**Data:**\n```json\n" + json.dumps(parsed['results'], indent=2) + "\n```")
                response_text = "\n\n".join(formatted_parts)
            else:
                response_text = "```json\n" + json.dumps(parsed, indent=2) + "\n```"
    except (json.JSONDecodeError, TypeError):
        pass  # Not JSON, use as-is
    
    display(HTML(f"""
    <div style="background: #e8f5e9; padding: 15px 20px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #4CAF50;">
        <b style="color: #2E7D32;">📝 Response:</b>
    </div>
    """))
    
    display(Markdown(response_text))
    
    # Extract sources from both Research Agent (citations) and Search API (results)
    all_sources = []
    citation_counter = 1
    
    for tool_result in result.get("tool_results", []):
        try:
            parsed_result = json.loads(tool_result)
            
            # Extract from Research Agent citations
            if "citations" in parsed_result and parsed_result.get("citations"):
                for citation in parsed_result["citations"]:
                    all_sources.append({
                        "number": citation.get("number", citation_counter),
                        "headline": citation.get("headline", "Untitled"),
                        "source": citation.get("source", "Bigdata.com"),
                        "url": citation.get("url"),
                        "timestamp": citation.get("timestamp"),
                        "text": citation.get("text", ""),
                        "type": "research"
                    })
                    citation_counter += 1
            
            # Extract from Search API results
            elif "results" in parsed_result and parsed_result.get("results"):
                for item in parsed_result["results"]:
                    if item.get("headline"):  # Only include items with headlines
                        all_sources.append({
                            "number": citation_counter,
                            "headline": item.get("headline", "Untitled"),
                            "source": item.get("source", "Bigdata.com"),
                            "url": item.get("url"),
                            "timestamp": item.get("timestamp"),
                            "text": item.get("relevant_text", ""),
                            "sentiment": item.get("sentiment"),
                            "type": "search"
                        })
                        citation_counter += 1
                        
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Display sources if found
    if all_sources:
        # Deduplicate by headline
        seen_headlines = set()
        unique_sources = []
        for src in all_sources:
            headline_key = src.get("headline", "")[:50]
            if headline_key not in seen_headlines:
                seen_headlines.add(headline_key)
                unique_sources.append(src)
        
        # Re-number after deduplication
        for i, src in enumerate(unique_sources, 1):
            src["number"] = i
        
        citations_html = []
        for source in unique_sources[:15]:  # Limit to 15 for display
            num = source.get("number", "")
            headline = html_lib.escape(str(source.get("headline", "Untitled"))[:100])
            src_name = html_lib.escape(str(source.get("source", "Bigdata.com")))
            url = source.get("url", "")
            timestamp = str(source.get("timestamp", ""))[:10] if source.get("timestamp") else ""
            text = source.get("text", "")
            excerpt = html_lib.escape(text[:150]) + "..." if text else ""
            
            # Sentiment indicator for search results
            sentiment = source.get("sentiment")
            sentiment_badge = ""
            if sentiment is not None:
                if sentiment > 0.3:
                    sentiment_badge = '<span style="background: #4CAF50; color: white; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-left: 8px;">POSITIVE</span>'
                elif sentiment < -0.3:
                    sentiment_badge = '<span style="background: #F44336; color: white; padding: 1px 6px; border-radius: 3px; font-size: 10px; margin-left: 8px;">NEGATIVE</span>'
            
            # Format date if present
            date_str = f" • {timestamp}" if timestamp else ""
            
            # Build source HTML
            if url:
                source_html = f"""
                <div style="padding: 12px; margin: 8px 0; background: #fafafa; border-radius: 6px; border-left: 3px solid #1976D2;">
                    <div style="display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;">
                        <span style="background: #1976D2; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">{num}</span>
                        <a href="{url}" target="_blank" style="color: #1565C0; text-decoration: none; font-weight: 500;">{headline}</a>
                        {sentiment_badge}
                    </div>
                    <div style="margin-top: 6px; font-size: 12px; color: #666;">
                        <span style="font-weight: 500;">{src_name}</span>{date_str}
                    </div>
                    {f'<div style="margin-top: 6px; font-size: 12px; color: #444; font-style: italic;">{excerpt}</div>' if excerpt else ''}
                </div>
                """
            else:
                source_html = f"""
                <div style="padding: 12px; margin: 8px 0; background: #fafafa; border-radius: 6px; border-left: 3px solid #1976D2;">
                    <div style="display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;">
                        <span style="background: #1976D2; color: white; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 12px;">{num}</span>
                        <span style="color: #333; font-weight: 500;">{headline}</span>
                        {sentiment_badge}
                    </div>
                    <div style="margin-top: 6px; font-size: 12px; color: #666;">
                        <span style="font-weight: 500;">{src_name}</span>{date_str}
                    </div>
                    {f'<div style="margin-top: 6px; font-size: 12px; color: #444; font-style: italic;">{excerpt}</div>' if excerpt else ''}
                </div>
                """
            citations_html.append(source_html)
        
        if citations_html:
            source_type = "citations" if any(s.get("type") == "research" for s in unique_sources) else "sources"
            display(HTML(f"""
            <div style="background: #E3F2FD; padding: 16px 20px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #1976D2;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <b style="color: #1565C0; font-size: 14px;">📚 Sources from Bigdata.com ({len(unique_sources)} {source_type})</b>
                    <span style="background: #1976D2; color: white; padding: 2px 8px; border-radius: 4px; font-size: 11px;">VERIFIED</span>
                </div>
                {''.join(citations_html)}
                {f'<div style="margin-top: 8px; font-size: 11px; color: #666;">... and {len(unique_sources) - 15} more sources</div>' if len(unique_sources) > 15 else ''}
            </div>
            """))
    
    # Only return JSON result if explicitly requested (reduces notebook output clutter)
    if show_json:
        return result
    return None


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def quick_setup() -> tuple:
    """
    Quick setup for demos - initializes everything and returns agent.
    
    Returns:
        Tuple of (config, agent)
    """
    config = setup_environment()
    create_financial_database()
    create_vector_store()
    agent = create_agent()
    return config, agent

