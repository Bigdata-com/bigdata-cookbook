# Bigdata Cookbook

A comprehensive collection of financial analysis tools and report generators built on the Bigdata API and research tools. This repository contains ready-to-use notebooks for thematic screening, narrative mining, and various sector-specific analyses including pricing power, AI disruption risks, and regulatory issues in the technology sector.

## Features

- **Client-Ready**: Each project is self-contained with its own dependencies and documentation
- **Easy Setup**: Uses Docker for containerized deployment or uv for fast, reliable dependency management
- **Comprehensive Analysis**: Combines multiple data sources for robust insights
- **Professional Output**: Generates Excel reports, HTML visualizations, and structured data
- **Modular Design**: Each project can be run independently

## Projects

### 🔍 [Thematic Screener](./Thematic_Screener/)
**Automated Thematic Analysis and Screening Tool**

- Thematic identification and categorization across multiple sectors
- Automated screening based on thematic criteria
- Theme tracking and evolution analysis
- Investment opportunity identification through thematic lenses

### 📊 [Pricing Power Analysis](./Pricing_Power_Analysis/)
**Automated Analysis of Pricing Power Narratives and Competitive Positioning**

- Assesses competitive positioning across company watchlists
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

### 🧾 [Specialized Report Tariffs](./Report_Generator_Specialized_Report_Tariffs/)
**Automated Analysis of Trade Tariff Risks and Corporate Mitigation Strategies**

- Generates sector-wide and company-specific risk reports
- Extracts mitigation plans from SEC filings and earnings transcripts
- Produces executive and detailed HTML reports
- Exports structured CSVs for further analysis

### 📉 [Rising Bond Spread Risks](./Rising_Bond_Spread_Risks/)
**Analyzing Spillover Risks from Rising Bond Spreads in Western Europe**

- Risk taxonomy generation with LLM-powered mind mapping
- Country-level risk scoring across bond spread sub-scenarios
- Rolling sentiment indicators and volume tracking
- Interactive dashboards with AI-powered narrative summaries

### 🪙 [Screener for Crypto](./Screener_for_Crypto/)
**Automated Cryptocurrency Thematic Screening and Analysis Tool**

- Automated theme taxonomy generation using LLM to break down complex investment themes
- Systematic cryptocurrency screening against specific thematic criteria
- Cross-crypto comparison enabling portfolio-level thematic assessment
- Interactive visualizations with heatmaps, bar charts, and scatter plots
- Structured output generating Excel reports and HTML visualizations for investment intelligence

### 🔧 [Build Your Own MCP](./Build_Your_Own_MCP/)
**MCP Server Integration for Bigdata Research Tools**

- Integration of Bigdata research tools with MCP (Model Context Protocol) server
- Watchlist creation and management through MCP interface
- Thematic screening of companies via MCP tools
- Compatible with Cursor, Claude Desktop, and other MCP clients
- Enables AI agents to interact with Bigdata platform for research and analysis

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

- Self-contained `uv` project under `Google_ADK_With_BigData/` with an ADK `financial_agent` (run `uv run adk web .` from that directory)
- Requires a Bigdata.com API key for MCP (`https://mcp.bigdata.com/`) and a Google Gemini key for the model and embeddings
- Includes sample prompts and pytest tests; see the folder README for setup and environment variables

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
- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Bigdata API access
- OpenAI API key (for advanced features)

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
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Report_Generator_AI_Threats/                      # AI risk analysis
│   ├── Report Generator_ AI Disruption Risk.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Report_Generator_Regulatory_Isses_in_Tech/        # Regulatory analysis
│   ├── Report Generator_ Regulatory Issues.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Risk_Analyzer/                                    # Risk analysis tool
│   ├── Risk_Analyzer.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Thematic_Screener/                                # Thematic analysis tool
│   ├── ThematicScreener.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Narrative_Miners/                                 # Narrative analysis tool
│   ├── NarrativeMiner.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Board_Management_Monitoring/                      # Board monitoring tool
│   ├── Board_Management_Monitoring.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Liquid_Cooling_Market_Watch/                      # Liquid cooling analysis
│   ├── Liquid_Cooling_Market_Watch.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Election_Monitor/                               # Elecion Monitoring tool
│   ├── Trump_Reelection_Impact_Analisys.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Credit_Ratings_Monitoring/                       # Credit rating event monitoring
│   ├── Credit_Ratings_Monitoring.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── AI_Cost_Cutting_Market_Analysis/                # AI cost cutting analysis
│   ├── AI_Cost_Cutting_Market_Analysis.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── AI_Revenue_Generation_Market_Analysis/          # AI revenue generation analysis
│   ├── AI_Revenue_Generation_Market_Analysis.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Tracking_Inflation_Drivers/                     # Inflation analysis tool
│   ├── Tracking_Inflation_Drivers.ipynb
│   ├── src/
│   ├── requirements.txt
│   └── README.md
├── Daily_Digest_Central_Banks/                      # Central bank monitoring
│   ├── Daily_Digest_Central_Banks.ipynb
│   ├── src/
│   ├── assets/
│   ├── report/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── Daily_Digest_Crude_Oil/                          # Crude oil market analysis
│   ├── Daily_Digest_Crude_Oil.ipynb
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
├── Report_Generator_Specialized_Report_Tariffs/      # Tariffs risk report generator
│   ├── Report_Generator_Specialized_Report_Tariffs.ipynb
│   ├── src/
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
├── Rising_Bond_Spread_Risks/                        # Bond spread spillover analysis
│   ├── Rising_Bond_Spread_Risks.ipynb
│   ├── src/
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .dockerignore
│   └── README.md
├── Screener_for_Crypto/                             # Cryptocurrency thematic screening
│   ├── Screener_for_Crypto.ipynb
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

### Core Dependencies
- `bigdata-client>=2.17.0` - Bigdata API client
- `bigdata-research-tools>=0.17.3` - Research analysis tools
- `nest-asyncio>=1.6.0` - Async compatibility
- `matplotlib>=3.0.0` - Data visualization
- `numpy>=1.20.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `jupyter>=1.0.0` - Notebook environment

### Optional Dependencies
- `seaborn>=0.11.0` - Statistical visualizations
- `plotly>=5.0.0` - Interactive plots
- `ipython>=7.0.0` - Enhanced Python shell

## Usage

Each project follows a similar workflow:

1. **Setup**: Install dependencies and configure credentials
2. **Data Collection**: Fetch relevant data from Bigdata platform
3. **Analysis**: Run the analysis pipeline
4. **Reporting**: Generate Excel and HTML reports
5. **Visualization**: Create charts and insights

## Support

- Each project has its own detailed README with specific instructions
- Check the individual project documentation for troubleshooting
- Ensure you have valid Bigdata API credentials before running analyses

## License

This project is licensed under the terms specified in the LICENSE file.

---

**Note**: This repository contains financial analysis tools. Please ensure compliance with relevant regulations and use appropriate risk management practices when making investment decisions based on these analyses.
