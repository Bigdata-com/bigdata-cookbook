# Bigdata Cookbook

A comprehensive collection of financial analysis tools and report generators built on the [Bigdata.com](https://bigdata.com) REST API, smart-batching search, and OpenAI. This repository contains ready-to-use notebooks and CLIs for thematic screening, narrative mining, sovereign and crypto analysis, and sector-specific workflows including pricing power, AI disruption risks, and regulatory issues in the technology sector.

Most migrated cookbooks use **`BIGDATA_API_KEY`** (REST) and **`OPENAI_API_KEY`** instead of the legacy `bigdata-client` / `bigdata-research-tools` SDK stack. See [`Thematic_Screener_CLI`](./Thematic_Screener_CLI/) and per-project READMEs for setup details.

## Features

- **Client-Ready**: Each project is self-contained with its own dependencies and documentation
- **Easy Setup**: Uses Docker for containerized deployment or uv for fast, reliable dependency management
- **Comprehensive Analysis**: Combines multiple data sources for robust insights
- **Professional Output**: Generates Excel reports, HTML visualizations, and structured data
- **GitHub-Friendly Notebooks**: Many cookbooks commit a static `.html` export alongside the `.ipynb` so charts render on GitHub without running cells
- **Modular Design**: Each project can be run independently

## GitHub notebook exports

These cookbooks ship a committed HTML export next to the notebook (refreshed after re-execution). Open the `.html` on GitHub for a read-only view with Plotly charts as PNGs:

| Cookbook | HTML export |
|----------|-------------|
| [AI Cost Cutting](./AI_Cost_Cutting_Market_Analysis/AI_Cost_Cutting_Market_Analysis.html) | `AI_Cost_Cutting_Market_Analysis.html` |
| [AI Revenue Generation](./AI_Revenue_Generation_Market_Analysis/AI_Revenue_Generation_Market_Analysis.html) | `AI_Revenue_Generation_Market_Analysis.html` |
| [Board Management Monitoring](./Board_Management_Monitoring/Board_Management_Monitoring.html) | `Board_Management_Monitoring.html` |
| [Credit Ratings Monitoring](./Credit_Ratings_Monitoring/Credit_Ratings_Monitoring.html) | `Credit_Ratings_Monitoring.html` |
| [Daily Digest Central Banks](./Daily_Digest_Central_Banks/Daily_Digest_Central_Banks.html) | `Daily_Digest_Central_Banks.html` |
| [Daily Digest Crude Oil](./Daily_Digest_Crude_Oil/Daily_Digest_Crude_Oil.html) | `Daily_Digest_Crude_Oil.html` |
| [Election Monitor](./Election_Monitor/Trump_Reelection_Impact_Analysis.html) | `Trump_Reelection_Impact_Analysis.html` |
| [Liquid Cooling Market Watch](./Liquid_Cooling_Market_Watch/Liquid_Cooling_Market_Watch.html) | `Liquid_Cooling_Market_Watch.html` |
| [Narrative Miners](./Narrative_Miners/NarrativeMiner.html) | `NarrativeMiner.html` |
| [Pricing Power Analysis](./Pricing_Power_Analysis/Pricing%20Power.html) | `Pricing Power.html` |
| [Report Generator AI Threats](./Report_Generator_AI_Threats/Report%20Generator_%20AI%20Disruption%20Risk.html) | `Report Generator_ AI Disruption Risk.html` |
| [Report Generator Regulatory Issues](./Report_Generator_Regulatory_Issues_in_Tech/Report%20Generator_%20Regulatory%20Issues.html) | `Report Generator_ Regulatory Issues.html` |
| [Report Generator Tariffs](./Report_Generator_Specialized_Report_Tariffs/Report_Generator_Specialized_Report_Tariffs.html) | `Report_Generator_Specialized_Report_Tariffs.html` |
| [Rising Bond Spread Risks](./Rising_Bond_Spread_Risks/Rising_Bond_Spread_Risks.html) | `Rising_Bond_Spread_Risks.html` |
| [Risk Analyzer](./Risk_Analyzer/Risk_Analyzer.html) | `Risk_Analyzer.html` |
| [Screener for Crypto](./Screener_for_Crypto/Screener_for_Crypto.html) | `Screener_for_Crypto.html` |
| [Thematic Screener](./Thematic_Screener/ThematicScreener.html) *(deprecated SDK)* | `ThematicScreener.html` |
| [Tracking Inflation Drivers](./Tracking_Inflation_Drivers/Tracking_Inflation_Drivers.html) | `Tracking_Inflation_Drivers.html` |

## Projects

### 🔍 [Thematic Screener CLI](./Thematic_Screener_CLI/)
**REST-based thematic screening (CLI, MCP, derivative-hop notebooks)**

- Screen a CSV company universe via Bigdata.com REST API and `bigdata-smart-batching`
- Exposure or derivatives taxonomy (`--taxonomy-style derivatives`); risk-analyzer export mode
- Client notebooks for TSX oil and EU parcel-tariff derivative screens

### 🔍 [Thematic Screener](./Thematic_Screener/) *(deprecated)*
**Legacy SDK notebook — use Thematic Screener CLI instead**

- Thematic identification and categorization across multiple sectors
- Automated screening based on thematic criteria
- Theme tracking and evolution analysis
- Investment opportunity identification through thematic lenses

> **Deprecated:** Relied on the removed `bigdata-research-tools` SDK and platform watchlists. Kept for reference only.

### 📊 [Pricing Power Analysis](./Pricing_Power_Analysis/)
**Automated Analysis of Pricing Power Narratives and Competitive Positioning**

- Assesses competitive positioning across a CSV company universe (`RP_ENTITY_ID`, `COMPANY_NAME`)
- Provides sector-wide comparative analysis
- Tracks temporal evolution of pricing narratives
- Implements confidence scoring system for pricing power signals

### 🤖 [AI Threats Report Generator](./Report_Generator_AI_Threats/)
**Automated Analysis of AI Threats and Opportunities in Technology Companies**

- Evaluates AI disruption risks and proactive AI adoption
- Provides standardized scoring for cross-company comparison
- Generates investment intelligence from AI transformation narratives
- Creates structured reports ranking companies by AI resilience

### ⚖️ [Regulatory Issues in Tech Report Generator](./Report_Generator_Regulatory_Issues_in_Tech/)
**Automated Analysis of Regulatory Risks and Company Mitigation Strategies**

- Maps sector-wide regulatory issues across technology domains
- Quantifies company-specific regulatory risks
- Extracts mitigation strategies from corporate communications
- Provides structured reporting on regulatory intensity and business impact

### 🎯 [Risk Analyzer](./Risk_Analyzer/)
**Automated Risk Analysis and Assessment Tool**

- Comprehensive risk assessment across multiple risk dimensions
- Quantitative risk modeling with statistical analysis
- Risk visualization and reporting capabilities
- Automated risk scoring and ranking systems

### 📖 [Narrative Miners](./Narrative_Miners/)
**Automated Narrative Analysis and Mining Tool**

- Narrative extraction and pattern recognition from unstructured data
- Sentiment analysis and narrative sentiment tracking
- Narrative evolution and temporal analysis
- Automated narrative scoring and ranking systems

### 📈 [Sentiment Pulse](./Sentiment_Pulse/)
**Single-Ticker Company Sentiment Dashboard**

- Turns one company stock ticker into a complete sentiment dashboard
- Qualitative sentiment tearsheet (executive summary, bullish/risk drivers, outlook, ranked evidence) via Bigdata.com MCP
- Quantitative daily time series of sentiment, sentiment pressure, and abnormal media attention via the REST Entity Sentiment API
- Current sentiment snapshot with direction gauge (Bullish / Neutral / Bearish)
- Self-contained notebook — change only the `TICKER` and re-run

### 🎙️ [Earnings Call Tone Analyzer](./Earnings_Call_Tone_Analyzer/)
**Automated management tone scoring from earnings call transcripts**

- Russell 1000 (or custom) universe via Bigdata.com entity IDs
- Transcript retrieval, de-duplication by quarter, and LLM-based tone scoring (`gpt-5.6-luna`)
- Batch CLI with CSV/JSON output for portfolio-wide tone trends

### 📰 [News Monitor (Edge MRVR)](./News_Monitor_MAS/)
**Entity-scoped web/public news pull from RavenPack Edge (provider MRVR)**

- Deterministic analytics per company–document row: relevance, sentiment, novelty, document ID, URL
- Optional post-processing via Bigdata document fetch or URL scrape
- `uv`-based runner with universe CSV input (`us_sml.csv`, etc.)

### 👥 [Board Management Monitoring](./Board_Management_Monitoring/)
**Automated Analysis of Board Member and Management Activity Exposure**

- Comprehensive person tracking across multiple name variations and contexts
- Company-specific filtering ensuring relevance to monitored organizations
- Multi-mode search precision from strict entity matching to broader coverage
- Temporal analysis showing how coverage patterns evolve over time
- Entity-specific monitoring using bigdata's entity tracking capabilities

### 🌊 [Liquid Cooling Market Watch](./Liquid_Cooling_Market_Watch/)
**Automated Analysis of Liquid Cooling Technology Providers and Adopters**

- Dual-role classification distinguishing technology providers from adopters
- Network analysis mapping provider-customer relationships in the cooling ecosystem
- Temporal tracking of adoption patterns and market evolution
- Market positioning analysis with confidence scoring for investment decisions
- Comprehensive ecosystem mapping for infrastructure investment intelligence

### 🗳️ [Election Monitor](./Election_Monitor/)
**Automated Analysis of Corporate Perspectives on Electoral Outcomes**

- Positive vs. negative impact assessment distinguishing companies that expect benefits from those anticipating challenges under new elected officials' policies
- Sector-wide political exposure mapping revealing industry patterns in positioning toward electoral results
- Temporal positioning tracking showing how political expectations evolve over time
- Corporate-political topic networks identifying key policy themes and company concerns through relationship analysis

### 📊 [Credit Ratings Monitoring](./Credit_Ratings_Monitoring/)
**Automated Detection and Analysis of Credit Rating Events**

- Event detection and classification for credit rating updates, outlook changes, and watch list events
- Entity relationship mapping distinguishing between rating agencies and rated entities with validation workflows
- Multi-feature extraction capturing credit ratings, outlooks, watchlist status, debt instruments, and key drivers
- Timeline analysis generating chronological reports showing rating evolution over time
- Interactive visualizations creating HTML reports with charts for rating timeline analysis

### 📉 [Credit Factor Analysis](./Credit_Factor_Analysis/)
**Screen a universe on credit-news sentiment, drill into catalysts, and write a grounded narrative**

- Rank a portfolio or sector list with `bigdata_screen_credit_factor` (worst credit-news names first)
- Drill into a deteriorating name with `bigdata_get_credit_factor` to see event-type catalysts
- Retrieve supporting news via `bigdata_search` and synthesize a credit narrative with `gpt-5.6-terra`
- Talks to the Bigdata.com Remote MCP server over Streamable HTTP (`BIGDATA_API_KEY`)
- Swap the coverage list and horizon (`daily` / `weekly` / `monthly`) to reuse the notebook as-is

### 💰 [AI Cost Cutting Market Analysis](./AI_Cost_Cutting_Market_Analysis/)
**Automated Analysis of AI Cost Cutting Providers and Users**

- Dual-role classification distinguishing companies developing AI cost cutting solutions from those implementing them
- Technology ecosystem mapping revealing relationships between solution providers and corporate users
- Adoption timeline tracking showing how AI cost cutting implementation evolves across different sectors
- Market positioning analysis quantifying each company's role and exposure in the AI cost cutting ecosystem

### 📈 [AI Revenue Generation Market Analysis](./AI_Revenue_Generation_Market_Analysis/)
**Automated Analysis of AI Revenue Generation Providers and Users**

- Dual-role classification distinguishing companies developing AI revenue generation solutions from those implementing them
- Technology ecosystem mapping revealing relationships between solution providers and corporate users
- Adoption timeline tracking showing how AI revenue generation implementation evolves across different companies
- Market positioning analysis quantifying each company's role and exposure in the AI revenue generation ecosystem

### 📊 [Tracking Inflation Drivers](./Tracking_Inflation_Drivers/)
**Automated Macroeconomic Inflation Analysis Tool**

- Automated theme breakdown into specific inflation components and drivers
- Systematic document analysis using embeddings-based search and classification
- Economic categorization that turns narrative signals into structured insights
- Comprehensive reporting with analytical summaries for each inflation driver covering demand-pull, cost-push, wage increases, global factors, and monetary policy impacts

### 🏦 [Daily Digest Central Banks](./Daily_Digest_Central_Banks/)
**Automated Central Bank Announcements Monitoring and Analysis Tool**

- Lexicon generation of monetary policy and central bank-specific terminology
- Real-time content retrieval via Bigdata API with parallelized keyword searches
- Topic clustering and selection with AI-powered verification and ranking
- Custom report generation with configurable ranking systems for trending topics
- Market impact assessment scoring topics for trendiness, novelty, and magnitude

### 🛢️ [Daily Digest Crude Oil](./Daily_Digest_Crude_Oil/)
**Automated Crude Oil Market Monitoring and Analysis Tool**

- Lexicon generation of crude oil industry-specific terminology and jargon
- Real-time content retrieval via Bigdata API with parallelized keyword searches
- Topic clustering and selection with AI-powered verification and ranking
- Custom report generation with configurable ranking systems for trending topics
- Market impact assessment scoring topics for trendiness, novelty, and magnitude

### 📋 [Large-Scale Portfolio Briefs Generation](./Briefs_Generation_Large_Scale/)
**Automated Brief Generation for Large Company Portfolios**

- Batch processing for hundreds or thousands of companies in configurable batches
- CSV-based input for easy portfolio management
- Customizable topics and research questions tailored to analysis needs
- Progress tracking with status polling and error handling
- Multiple export formats including JSON and Excel for further analysis
- Source attribution with full metadata including URLs, headlines, and publication dates

### ☀️ [Morning Brief CLI](./morning_brief_cli/)
**Daily institutional morning brief for equity portfolios (CLI)**

- Up to 50 companies across five pre-configured research topics (earnings, macro, analyst sentiment, M&A, supply chain)
- Bigdata.com smart-batching search plus OpenAI summarisation
- Markdown and HTML output under `runs/<run_name>/briefs/`
- Portfolio CSV format: `TICKER`, `RP_COMPANY_ID`, `COMPANY_NAME`

### 🧾 [Specialized Report Tariffs](./Report_Generator_Specialized_Report_Tariffs/)
**Automated Analysis of Trade Tariff Risks and Corporate Mitigation Strategies**

- Generates sector-wide and company-specific risk reports
- Extracts mitigation plans from SEC filings and earnings transcripts
- Produces executive and detailed HTML reports
- Exports structured CSVs for further analysis

### 📉 [Rising Bond Spread Risks](./Rising_Bond_Spread_Risks/)
**Analyzing Spillover Risks from Rising Bond Spreads in Western Europe**

- Sovereign universe: Western European **countries + central banks** from `data/western_europe_countries_banks.csv` (dual search, bank→country remap)
- Predefined bond-spillover risk taxonomy with REST smart-batching search (`chunk_percentage`, `requests_per_minute`)
- Country-level risk scoring across bond spread sub-scenarios
- Rolling sentiment indicators, volume tracking, and interactive dashboards with AI-powered narrative summaries

### 🪙 [Screener for Crypto](./Screener_for_Crypto/)
**Automated Cryptocurrency Thematic Screening and Analysis Tool**

- Screens the **Top 15 cryptocurrencies** from `data/top_15_cryptos.csv` (not crypto-exposed public companies)
- OpenAI-generated theme taxonomy and chunk labeling via REST search
- Cross-crypto comparison enabling portfolio-level thematic assessment
- Interactive visualizations with heatmaps, bar charts, and scatter plots
- Committed HTML export for GitHub-friendly chart viewing

### 🔧 [Build Your Own MCP](./Build_Your_Own_MCP/)
**MCP Server Integration for Bigdata Research Tools**

- Integration of Bigdata.com REST APIs with MCP (Model Context Protocol) server patterns
- Thematic screening and search workflows exposed as MCP tools
- Compatible with Cursor, Claude Desktop, and other MCP clients
- Example grounded dashboards and HTML assets under `assets/`

### 📌 [MCP Dashboard Demo](./MCP_Dashboard_Demo/)
**Illustration: MCP-grounded dashboard (frozen snapshot)**

- React + Vite demo: typed **`GROUNDED_DATA`** in `src/dashboard.jsx` populated via **Bigdata.com MCP** (market tearsheet, search, country tearsheets)—**no in-browser API**
- Shows source-attributed panels (Iran–Gulf example); **cookbook copy frozen 2026-03-18**; deploy and live refresh live in a **separate production repo**
- Example GitHub Actions and Fly.io workflows are **reference-only** under [`MCP_Dashboard_Demo/docs/reference-workflows/`](./MCP_Dashboard_Demo/docs/reference-workflows/) (not active CI here)

### 🔬 [Research Agent Sync Response](./Research_Agent_Sync_Response/)
**Python Client for Research Agent API with Citation Support**

- Simple synchronous interface wrapping the Research Agent streaming API
- Bigdata.com standard citation format with full source metadata
- Inline citation markers `[1]`, `[2]` with numbered reference lists
- Multiple output formats: plain answer, citations JSON, or combined results
- Follow-up conversation support with chat ID continuation
- Configurable research effort levels (lite/standard) for speed vs. depth tradeoff

### 🤖 [Agent to Bigdata](./Agent_To_Bigdata/)
**Modular Framework for Building AI Agents with Bigdata.com Integration**

- Multi-source AI agent integrating Bigdata.com Search, Knowledge Graph, and Research Agent APIs
- Internal data integration with SQLite databases and FAISS vector stores
- Hierarchical agent architecture with smart tool routing (internal-first, external escalation)
- LangSmith observability for production monitoring and tracing
- Reusable core module for building custom agent workflows
- Citation support with inline markers and numbered references

### 🤖 [Google ADK with Bigdata and local data](./Google_ADK_With_BigData/)
**Standalone Google ADK agent with SQLite, local Markdown research files (FAISS + Gemini embeddings), and Bigdata.com MCP**

- Multi-source AI agent integrating Bigdata.com Search, Knowledge Graph, and Research Agent APIs
- Internal data integration with SQLite databases and FAISS vector stores
- Citation support with inline markers and numbered references

### 🧱 [Databricks Agent to Bigdata](./Databricks_Agent_To_Bigdata/)
**Financial-intelligence agent on Databricks combining internal lakehouse data with Bigdata.com over MCP**

- One Mosaic AI agent over three sources: internal structured (Unity Catalog SQL functions + AI/BI Genie), internal unstructured (Vector Search), and external real-time (Bigdata.com MCP)
- Model Context Protocol integration with automatic tool discovery — governed via Unity AI Gateway MCP Services, or a direct connection
- Built on the latest Databricks agent stack: Mosaic AI Agent Framework, MLflow `ChatAgent`, LangGraph, Databricks-hosted Claude
- End-to-end: Unity Catalog setup → Vector Search → MCP → deploy to Model Serving with the AI Playground / Review App
- Cited, cross-source answers separating proprietary signal from public market intelligence

### ❄️ [Snowflake Agent to Bigdata](./Snowflake_Agent_To_Bigdata/)
**Snowflake Intelligence demo combining live Snowflake data with Bigdata.com MCP**

- Cortex Agent (`SNOWFLAKE_BIGDATA_AGENT`) over portfolio SQL, internal research search, and Bigdata.com MCP tools
- Search financial news and filings, resolve securities, and pull company tearsheets from one chat UI
- End-to-end Snowflake setup guide: External Access Integration, MCP connector, demo data, and agent deployment

### 🔍 [Large Scale Search](./Search_Large_Scale/)
**High-Performance Portfolio Search Tool**

- Entity resolution with CSV caching for ticker-to-entity ID mapping
- Parallel processing with ThreadPoolExecutor for searching hundreds of tickers
- Multi-layered rate limiting (sliding window + concurrency semaphore + auto-retry)
- SQLite storage with indexed queries for fast result retrieval
- Customizable research topics with company name placeholders
- Query interface to filter results by ticker, topic, or custom criteria

### 📊 [Index M&A Activity Report](./Index_MA_Activity_Report/)
**Automated M&A Analysis and Report Generation Tool**

- M&A news search for specified tickers using Bigdata.com API
- AI-powered executive briefs summarizing key M&A developments
- Structured deal analysis tables identifying acquisition targets
- Desk notes per ticker with source attribution
- Automated report generation with deal tables, summaries, and source links

### 📦 [Smart Batching](./Smart_Batching/)
**Optimized Semantic Search with Intelligent Query Planning and Large-Scale Execution**

- **Two-Step System**: Planning phase creates optimized baskets, execution phase performs search with proportional sampling
- **Query Optimization**: Reduces API queries by 96-99% (varies by topic specificity) through intelligent company grouping
  - Niche topics: Up to 99.85% reduction (e.g., "Customer Trust Erosion": 17 queries vs 11,357 naive)
  - Specialized topics: 96-97% reduction (e.g., "Higher ESG Compliance Costs": 435 queries)
- **Large-Scale Search Execution**: Follows Search_Large_Scale pattern with:
  - Parallel processing using ThreadPoolExecutor for high-throughput searches
  - Multi-layered rate limiting (sliding window algorithm + concurrency semaphore)
  - Automatic retry with exponential backoff for robust error handling
  - Proportional sampling to retrieve percentage of results while preserving distribution
- **Volume-Based Batching**: Automatic granularity determination and basket creation maximizing efficiency
- **Production Ready**: Comprehensive error handling, logging, and plan persistence for reuse
- **Scalable**: Efficiently handles universes with 10,000+ companies

### 🔍 [Batch Search API](./Batch_Search_API/)
**One Batch Job for Large-Scale Search Across Full Universes**

- Scale to full universes (e.g. Global All-Cap, 10,000+ companies) without client-side rate limits or thousands of round-trips
- Single batch job: submit one JSONL file with all queries; the service runs them asynchronously and returns one result file
- No client-side rate limiting: no QPS management, connection pools, or thousands of round-trips
- Entity-level post-processing: deduplicate chunks, assign to query entities only, aggregate score and volume per entity
- Sector–country heatmap: optional bottom-up macro view by sector and country (e.g. G12)

### 📚 [Bigdata.com API Examples](./API_Tutorials/)
**Notebook and script examples for key Bigdata.com APIs**

- Five notebook examples: Search, Volume, Knowledge Graph, Co-mentions, and an end-to-end workflow example
- Client-ready script library: [Sample_Scripts](./API_Tutorials/Sample_Scripts/) — full folder catalog, quickstart, and step-by-step workflow patterns are in [`API_Tutorials/Sample_Scripts/README.md`](./API_Tutorials/Sample_Scripts/README.md)
- Standardized auth via `BIGDATA_API_KEY` loaded from `.env`
- Progressive path from API fundamentals to workflow-level signal construction
- Designed as a practical onboarding and execution path for teams integrating Bigdata.com APIs

## Quick Start

### Prerequisites

#### For Docker Installation
- Docker installed on your system
- Bigdata API access
- OpenAI API key (for advanced features)

#### For Local Installation
- Python 3.11 or higher (3.8+ may work for older projects; check each `pyproject.toml` / `requirements.txt`)
- [uv](https://github.com/astral-sh/uv) package manager
- Bigdata.com API key (`BIGDATA_API_KEY`)
- OpenAI API key (`OPENAI_API_KEY`) for LLM labeling and summarization in most cookbooks

#### Clone repository

Clone the repository to your local computer. Please follow the below steps:

- Navigate your local computer to the folder where you want to clone the repo and run the following command:
```bash
git clone https://github.com/Bigdata-com/bigdata-cookbook.git
```


### Installation

Each project supports both Docker and local installation methods:

- **Docker Installation**: Each project includes a Dockerfile for containerized deployment
- **Local Installation**: Traditional installation using Python and uv package manager

Each project has its own detailed README with specific installation and usage instructions for both methods.

## Project Structure

```
bigdata-cookbook/
├── Pricing_Power_Analysis/                          # Pricing power analysis
│   ├── Pricing Power.ipynb
│   ├── Pricing Power.html
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Report_Generator_AI_Threats/                      # AI risk analysis
│   ├── Report Generator_ AI Disruption Risk.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Report_Generator_Regulatory_Issues_in_Tech/        # Regulatory analysis
│   ├── Report Generator_ Regulatory Issues.ipynb
│   ├── Report Generator_ Regulatory Issues.html
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Risk_Analyzer/                                    # Risk analysis tool
│   ├── Risk_Analyzer.ipynb
│   ├── Risk_Analyzer.html
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Thematic_Screener_CLI/                            # Thematic screener CLI + MCP (REST)
│   ├── notebooks/
│   ├── src/
│   ├── pyproject.toml
│   └── README.md
├── Thematic_Screener/                                # Deprecated SDK notebook
│   ├── ThematicScreener.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Narrative_Miners/                                 # Narrative analysis tool
│   ├── NarrativeMiner.ipynb
│   ├── NarrativeMiner.html
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Sentiment_Pulse/                                  # Single-ticker sentiment dashboard
│   ├── company_sentiment_dashboard.ipynb
│   ├── requirements.txt
│   └── README.md
├── Earnings_Call_Tone_Analyzer/                      # Earnings call tone scoring CLI
│   ├── pyproject.toml
│   └── README.md
├── News_Monitor_MAS/                                 # Edge MRVR news monitor
│   ├── pyproject.toml
│   └── README.md
├── Board_Management_Monitoring/                      # Board monitoring tool
│   ├── Board_Management_Monitoring.ipynb
│   ├── Board_Management_Monitoring.html
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Liquid_Cooling_Market_Watch/                      # Liquid cooling analysis
│   ├── Liquid_Cooling_Market_Watch.ipynb
│   ├── Liquid_Cooling_Market_Watch.html
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Election_Monitor/                                 # Election monitoring tool
│   ├── Trump_Reelection_Impact_Analysis.ipynb
│   ├── Trump_Reelection_Impact_Analysis.html
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Credit_Ratings_Monitoring/                       # Credit rating event monitoring
│   ├── Credit_Ratings_Monitoring.ipynb
│   ├── Credit_Ratings_Monitoring.html
│   ├── report/
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Credit_Factor_Analysis/                          # Credit-news factor screen + narrative
│   ├── Credit_Factor_Analysis.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── AI_Cost_Cutting_Market_Analysis/                # AI cost cutting analysis
│   ├── AI_Cost_Cutting_Market_Analysis.ipynb
│   ├── AI_Cost_Cutting_Market_Analysis.html
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── AI_Revenue_Generation_Market_Analysis/          # AI revenue generation analysis
│   ├── AI_Revenue_Generation_Market_Analysis.ipynb
│   ├── AI_Revenue_Generation_Market_Analysis.html
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Tracking_Inflation_Drivers/                     # Inflation analysis tool
│   ├── Tracking_Inflation_Drivers.ipynb
│   ├── Tracking_Inflation_Drivers.html
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Daily_Digest_Central_Banks/                      # Central bank monitoring
│   ├── Daily_Digest_Central_Banks.ipynb
│   ├── Daily_Digest_Central_Banks.html
│   ├── src/
│   ├── assets/
│   ├── report/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── Daily_Digest_Crude_Oil/                          # Crude oil market analysis
│   ├── Daily_Digest_Crude_Oil.ipynb
│   ├── Daily_Digest_Crude_Oil.html
│   ├── src/
│   ├── assets/
│   ├── report/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── Briefs_Generation_Large_Scale/                    # Large-scale portfolio briefs generation
│   ├── portfolio_briefs_generation.ipynb
│   ├── static/
│   │   └── data/
│   ├── requirements.txt
│   └── README.md
├── morning_brief_cli/                               # Daily morning brief CLI (≤50 names)
│   ├── pyproject.toml
│   └── README.md
├── Report_Generator_Specialized_Report_Tariffs/      # Tariffs risk report generator
│   ├── Report_Generator_Specialized_Report_Tariffs.ipynb
│   ├── Report_Generator_Specialized_Report_Tariffs.html
│   ├── src/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── Rising_Bond_Spread_Risks/                        # Bond spread spillover analysis
│   ├── Rising_Bond_Spread_Risks.ipynb
│   ├── Rising_Bond_Spread_Risks.html
│   ├── data/western_europe_countries_banks.csv
│   ├── src/
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   └── README.md
├── Screener_for_Crypto/                             # Cryptocurrency thematic screening
│   ├── Screener_for_Crypto.ipynb
│   ├── Screener_for_Crypto.html
│   ├── data/top_15_cryptos.csv
│   ├── src/
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   └── README.md
├── Build_Your_Own_MCP/                              # MCP server integration
│   ├── build_your_mcp.py
│   ├── assets/
│   ├── Dockerfile
│   └── README.md
├── MCP_Dashboard_Demo/                            # MCP-grounded dashboard illustration (frozen snapshot)
│   ├── src/
│   ├── docs/reference-workflows/
│   ├── Dockerfile
│   └── README.md
├── Research_Agent_Sync_Response/                    # Research Agent API client
│   ├── research_client_usage.ipynb
│   ├── research_client.py
│   ├── output/
│   └── README.md
├── Agent_To_Bigdata/                                # AI agent framework with Bigdata.com integration
│   ├── agent_to_research_agent.ipynb
│   ├── agent_to_search.ipynb
│   ├── langgraph_core.py
│   ├── research_client.py
│   ├── requirements.txt
│   ├── static/
│   └── README.md
├── Google_ADK_With_BigData/                         # Google ADK + Bigdata + local FAISS demo
│   └── README.md
├── Databricks_Agent_To_Bigdata/                     # Databricks + Bigdata MCP agent
│   └── README.md
├── Snowflake_Agent_To_Bigdata/                      # Snowflake Intelligence + Bigdata MCP demo
│   └── README.md
├── Search_Large_Scale/                              # Large-scale portfolio search
│   ├── large_search.ipynb
│   ├── output/
│   └── README.md
├── Index_MA_Activity_Report/                        # M&A activity report generation
│   ├── index_ma_report.ipynb
│   ├── config/
│   ├── services/
│   ├── requirements.txt
│   └── README.md
├── Smart_Batching/                                  # Optimized query planning
│   ├── ...
│   └── README.md
├── Batch_Search_API/                                # Batch Search API — one job for thousands of queries
│   ├── Batch_Search_API.ipynb
│   ├── src/
│   ├── data/
│   ├── requirements.txt
│   └── README.md
├── API_Tutorials/                                   # Bigdata.com API examples bundle
│   ├── Search_API/
│   ├── Volume_API/
│   ├── Knowledge_Graph_API/
│   ├── CoMentions_API/
│   ├── Workflow_example/
│   ├── Sample_Scripts/
│   └── README.md
└── README.md                                        # This file
```

## Requirements

### Typical dependencies (REST cookbooks)

Each project declares its own dependencies in `requirements.txt` or `pyproject.toml`. Migrated cookbooks commonly use:

- **`BIGDATA_API_KEY`** — [Bigdata.com REST API](https://docs.bigdata.com/api-reference/introduction#api-key) (`X-API-KEY` header)
- **`OPENAI_API_KEY`** — LLM labeling, summarisation, and theme generation
- **`pandas`**, **`requests`**, **`python-dotenv`** — data handling and auth
- **`bigdata-smart-batching`** — cost-controlled universe search (where applicable)
- **`plotly`** + **`kaleido`** — interactive charts; PNG export for GitHub HTML notebooks
- **`jupyter`** / **`nbconvert`** — notebook execution and HTML export

Legacy SDK cookbooks (e.g. deprecated [Thematic Screener](./Thematic_Screener/)) may still list `bigdata-client` and `bigdata-research-tools` in their own requirements files.

## Usage

Each project follows a similar workflow:

1. **Setup**: Copy `.env.example` to `.env`, set `BIGDATA_API_KEY` and (where needed) `OPENAI_API_KEY`
2. **Data Collection**: Fetch relevant content via Bigdata.com REST search or smart-batching
3. **Analysis**: Run the analysis pipeline (LLM labeling, scoring, aggregation)
4. **Reporting**: Generate Excel, CSV, and HTML reports
5. **Visualization**: Review charts in the notebook or open the committed `.html` export on GitHub

## Support

- Each project has its own detailed README with specific instructions
- Check the individual project documentation for troubleshooting
- Ensure you have valid Bigdata API credentials before running analyses

## License

This project is licensed under the terms specified in the LICENSE file.

---

**Note**: This repository contains financial analysis tools. Please ensure compliance with relevant regulations and use appropriate risk management practices when making investment decisions based on these analyses.
