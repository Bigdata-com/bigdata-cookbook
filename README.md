# Bigdata Cookbook

Turn unstructured financial news, filings, and transcripts into investment-ready intelligence. This repository is a library of **ready-to-run notebooks and workflows** built on [Bigdata.com](https://bigdata.com)—thematic screeners, sovereign and crypto analysis, credit and risk monitors, narrative miners, daily digests, and agent integrations for research teams, portfolio managers, and strategists.

Each cookbook is self-contained: open the notebook, point it at your universe or theme, and produce scores, dashboards, briefs, and reports you can share with stakeholders.

## Features

- **Client-ready**: Self-contained projects with their own setup guides
- **Fast to try**: Docker or local install; many notebooks include a browsable HTML preview on GitHub
- **Institutional depth**: Thematic scoring, risk taxonomies, narrative summaries, and exportable reports
- **Broad coverage**: Equities, sovereigns, crypto, credit, macro, M&A, and sector themes
- **Composable**: Use one cookbook on its own or combine outputs across your research stack

## Preview on GitHub

Many cookbooks include a static HTML export next to the notebook so you can **browse charts and results on GitHub** without running cells:

| Cookbook | Preview |
|----------|---------|
| [AI Cost Cutting](./AI_Cost_Cutting_Market_Analysis/AI_Cost_Cutting_Market_Analysis.html) | Provider vs. adopter map for AI cost-cutting narratives |
| [AI Revenue Generation](./AI_Revenue_Generation_Market_Analysis/AI_Revenue_Generation_Market_Analysis.html) | Who is selling vs. adopting AI revenue tools |
| [Board Management Monitoring](./Board_Management_Monitoring/Board_Management_Monitoring.html) | Leadership and board activity exposure |
| [Credit Ratings Monitoring](./Credit_Ratings_Monitoring/Credit_Ratings_Monitoring.html) | Rating actions, outlooks, and timeline views |
| [Daily Digest Central Banks](./Daily_Digest_Central_Banks/Daily_Digest_Central_Banks.html) | What central banks are saying, ranked by impact |
| [Daily Digest Crude Oil](./Daily_Digest_Crude_Oil/Daily_Digest_Crude_Oil.html) | Oil-market narrative digest |
| [Election Monitor](./Election_Monitor/Trump_Reelection_Impact_Analysis.html) | Corporate positioning on electoral outcomes |
| [Liquid Cooling Market Watch](./Liquid_Cooling_Market_Watch/Liquid_Cooling_Market_Watch.html) | Liquid cooling providers, adopters, and ecosystem |
| [Narrative Miners](./Narrative_Miners/NarrativeMiner.html) | Theme discovery and narrative ranking |
| [Pricing Power Analysis](./Pricing_Power_Analysis/Pricing%20Power.html) | Pricing power signals across a company universe |
| [Report Generator AI Threats](./Report_Generator_AI_Threats/Report%20Generator_%20AI%20Disruption%20Risk.html) | AI disruption risk by company |
| [Report Generator Regulatory Issues](./Report_Generator_Regulatory_Issues_in_Tech/Report%20Generator_%20Regulatory%20Issues.html) | Regulatory exposure in tech |
| [Report Generator Tariffs](./Report_Generator_Specialized_Report_Tariffs/Report_Generator_Specialized_Report_Tariffs.html) | Tariff risk and mitigation narratives |
| [Rising Bond Spread Risks](./Rising_Bond_Spread_Risks/Rising_Bond_Spread_Risks.html) | Western Europe sovereign spillover from bond spreads |
| [Risk Analyzer](./Risk_Analyzer/Risk_Analyzer.html) | Corporate exposure to a defined risk scenario |
| [Screener for Crypto](./Screener_for_Crypto/Screener_for_Crypto.html) | Institutional adoption themes across major cryptos |
| [Thematic Screener](./Thematic_Screener/ThematicScreener.html) | Legacy thematic screener (reference) |
| [Tracking Inflation Drivers](./Tracking_Inflation_Drivers/Tracking_Inflation_Drivers.html) | Inflation driver taxonomy and narrative scores |

## Projects

### 🔍 [Thematic Screener CLI](./Thematic_Screener_CLI/)
**Screen any theme across a company universe—in the terminal, via MCP, or in client notebooks**

- Rank companies by exposure to an investment theme or derivative narrative
- Export heatmaps and scored universes for portfolio and sector work
- Example client workflows for commodity and tariff derivative screens

### 🔍 [Thematic Screener](./Thematic_Screener/)
**Classic notebook workflow for thematic identification and scoring** *(legacy reference; prefer [Thematic Screener CLI](./Thematic_Screener_CLI/))*

- Thematic identification and categorization across multiple sectors
- Automated screening based on thematic criteria
- Theme tracking and evolution analysis
- Investment opportunity identification through thematic lenses

### 📊 [Pricing Power Analysis](./Pricing_Power_Analysis/)
**Automated Analysis of Pricing Power Narratives and Competitive Positioning**

- Assesses competitive positioning across your company universe
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
- Qualitative tearsheet: executive summary, bullish and risk drivers, outlook, ranked evidence
- Quantitative sentiment time series, pressure, and abnormal media attention
- At-a-glance direction gauge (Bullish / Neutral / Bearish)
- Change the ticker and re-run for any single name

### 🎙️ [Earnings Call Tone Analyzer](./Earnings_Call_Tone_Analyzer/)
**Score management tone from earnings calls at scale**

- Pull the latest earnings transcripts for your coverage list
- LLM-based tone scoring with quarter-over-quarter comparability
- Portfolio-wide tone trends for idea generation and risk monitoring

### 📰 [News Monitor (Edge MRVR)](./News_Monitor_MAS/)
**Fresh, entity-scoped news with relevance, sentiment, and novelty scores**

- Monitor breaking stories across a defined company universe
- Structured outputs ready for alerts, dashboards, or downstream research
- Enrich with full document text when you need the underlying article

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
**Find deteriorating credit names, explain the catalysts, and draft a grounded narrative**

- Rank a portfolio or sector on credit-news sentiment
- Drill into the names that moved and see event-type drivers
- Pull supporting news and synthesize an analyst-ready credit story

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

### ☀️ [Morning Brief](./morning_brief_cli/)
**Start the day with a portfolio morning brief across five research lenses**

- Earnings and guidance, macro and policy, analyst sentiment, M&A and corporate actions, supply chain and operations
- One command produces shareable Markdown and HTML briefs with sourced evidence
- Built for PM morning meetings, sector pods, and coverage-team standups

### 🧾 [Specialized Report Tariffs](./Report_Generator_Specialized_Report_Tariffs/)
**Automated Analysis of Trade Tariff Risks and Corporate Mitigation Strategies**

- Generates sector-wide and company-specific risk reports
- Extracts mitigation plans from SEC filings and earnings transcripts
- Produces executive and detailed HTML reports
- Exports structured CSVs for further analysis

### 📉 [Rising Bond Spread Risks](./Rising_Bond_Spread_Risks/)
**Quantify sovereign spillover risk as Western European bond spreads widen**

- Score Western European countries on bond-spillover and contagion narratives
- Compare relative exposure across the region with standardized risk metrics
- Rolling sentiment, volume spikes, and AI-written peak-risk narratives per country
- Interactive country dashboards for committee packs and sovereign research

### 🪙 [Screener for Crypto](./Screener_for_Crypto/)
**Identify cryptocurrencies aligned with institutional adoption before the crowd**

- Screen major digital assets against institutional adoption themes (KYC/AML, custody, regulation, enterprise use)
- Rank cryptos by thematic exposure with heatmaps and composite scores
- Purpose-built for crypto wire intelligence and early trend detection
- Interactive visualizations for portfolio and research presentations

### 🔧 [Build Your Own MCP](./Build_Your_Own_MCP/)
**Connect Bigdata.com research workflows to Cursor, Claude, and other MCP clients**

- Expose search and screening as tools your AI assistant can call
- Example grounded dashboards and HTML assets for client-ready deliverables
- A starting point for custom agent and automation workflows

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
**Run thematic search across thousands of companies efficiently**

- Intelligent query planning that groups similar companies and themes
- Large-universe execution with parallel search and robust retries
- Proportional sampling so results stay representative at scale
- Reusable search plans for recurring screens and monitors

### 🔍 [Batch Search API](./Batch_Search_API/)
**One batch job to search an entire universe asynchronously**

- Submit all queries in a single job and receive one consolidated result file
- Entity-level scores and volumes for ranking and heatmaps
- Optional sector–country views for macro-style screens

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
- Python 3.11+ recommended
- [uv](https://github.com/astral-sh/uv) package manager
- Bigdata.com and OpenAI API keys (see each project's `.env.example`)

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
├── morning_brief_cli/                               # Portfolio morning brief generator
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

Each project lists its own dependencies in `requirements.txt` or `pyproject.toml`. Setup details and credentials are in every project README.

## Usage

Typical flow across cookbooks:

1. **Choose a workflow** that matches your question (theme, risk, credit, macro, etc.)
2. **Configure** your universe, dates, and theme in the notebook
3. **Run** the pipeline to retrieve, label, and score unstructured content
4. **Share** Excel, CSV, HTML, or dashboard outputs with your team

## Support

- Each project has its own detailed README with specific instructions
- Check the individual project documentation for troubleshooting
- Ensure you have valid Bigdata API credentials before running analyses

## License

This project is licensed under the terms specified in the LICENSE file.

---

**Note**: This repository contains financial analysis tools. Please ensure compliance with relevant regulations and use appropriate risk management practices when making investment decisions based on these analyses.
