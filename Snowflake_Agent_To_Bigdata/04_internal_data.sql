-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 04: Internal Data — Portfolio Tables + Research Documents
-- =============================================================================
-- Creates internal Snowflake data that the combined agent will query:
--
--   1. Structured data  — Financial tables (accounts, portfolios, holdings,
--      transactions) with a Semantic View for Cortex Analyst
--      (used by INTERNAL_PORTFOLIO_ANALYST)
--   2. Unstructured data — Research documents (investment theses, risk
--      assessments, strategy memos) with a Cortex Search Service
--      (used by INTERNAL_RESEARCH_SERVICE)
--
-- Run this AFTER 03_test_procedures.sql.
-- =============================================================================

-- *** CONFIGURATION — must match values from script 01 ***
SET db_name     = 'BIGDATA_DB';
SET schema_name = 'MCP_TOOLS';
SET wh_name     = 'BIGDATA_WH';
-- *** END CONFIGURATION ***

USE ROLE ACCOUNTADMIN;
USE DATABASE IDENTIFIER($db_name);
USE SCHEMA   IDENTIFIER($schema_name);
USE WAREHOUSE IDENTIFIER($wh_name);

-- =============================================================================
-- Step 1: Create and populate financial tables
-- =============================================================================

-- 1a. Accounts
CREATE OR REPLACE TABLE accounts (
    account_id   VARCHAR(10) PRIMARY KEY,
    account_name VARCHAR(100) NOT NULL,
    account_type VARCHAR(50)  NOT NULL,
    currency     VARCHAR(3)   DEFAULT 'USD',
    balance      NUMBER(18,2) DEFAULT 0
);

INSERT INTO accounts VALUES
    ('ACC001', 'Institutional Growth Fund', 'Institutional', 'USD', 50000000),
    ('ACC002', 'Tech Innovation Portfolio', 'Hedge Fund',    'USD', 25000000),
    ('ACC003', 'Global Macro Strategy',     'Pension Fund',  'USD', 100000000);

-- 1b. Portfolios
CREATE OR REPLACE TABLE portfolios (
    portfolio_id   VARCHAR(10)  PRIMARY KEY,
    portfolio_name VARCHAR(100) NOT NULL,
    account_id     VARCHAR(10),
    strategy       VARCHAR(50),
    risk_profile   VARCHAR(50),
    aum            NUMBER(18,2) DEFAULT 0,
    FOREIGN KEY (account_id) REFERENCES accounts(account_id)
);

INSERT INTO portfolios VALUES
    ('PF001', 'US Large Cap Growth',      'ACC001', 'Growth',       'Moderate',   30000000),
    ('PF002', 'AI & Semiconductor Focus', 'ACC002', 'Sector Focus', 'Aggressive', 15000000),
    ('PF003', 'Diversified Tech Leaders', 'ACC003', 'Value Growth', 'Moderate',   50000000);

-- 1c. Holdings
CREATE OR REPLACE TABLE holdings (
    holding_id     NUMBER AUTOINCREMENT PRIMARY KEY,
    portfolio_id   VARCHAR(10),
    ticker         VARCHAR(10)  NOT NULL,
    company_name   VARCHAR(100),
    shares         NUMBER(18,4),
    avg_cost       NUMBER(18,2),
    current_price  NUMBER(18,2),
    market_value   NUMBER(18,2),
    unrealized_pnl NUMBER(18,2),
    weight_pct     NUMBER(8,2),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
);

INSERT INTO holdings (portfolio_id, ticker, company_name, shares, avg_cost, current_price, market_value, unrealized_pnl, weight_pct) VALUES
    ('PF001', 'AAPL',  'Apple Inc.',              15000,  142.50, 185.25, 2778750,  641250,  9.26),
    ('PF001', 'MSFT',  'Microsoft Corporation',    8000,  285.00, 415.50, 3324000, 1044000, 11.08),
    ('PF001', 'GOOGL', 'Alphabet Inc.',            5000,  125.00, 175.25,  876250,  251250,  2.92),
    ('PF001', 'AMZN',  'Amazon.com Inc.',          6000,  145.00, 225.75, 1354500,  484500,  4.52),
    ('PF001', 'META',  'Meta Platforms Inc.',       4500,  280.00, 585.00, 2632500, 1372500,  8.78),
    ('PF002', 'NVDA',  'NVIDIA Corporation',      12000,  450.00, 875.50,10506000, 5106000, 70.04),
    ('PF002', 'AMD',   'Advanced Micro Devices',   8000,   95.00, 145.25, 1162000,  402000,  7.75),
    ('PF002', 'AVGO',  'Broadcom Inc.',            1500,  850.00,1425.00, 2137500,  862500, 14.25),
    ('PF002', 'TSM',   'Taiwan Semiconductor',     3000,  110.00, 185.75,  557250,  227250,  3.72),
    ('PF002', 'PLTR',  'Palantir Technologies',   25000,   18.50,  65.25, 1631250, 1168750, 10.88),
    ('PF003', 'AAPL',  'Apple Inc.',              25000,  155.00, 185.25, 4631250,  756250,  9.26),
    ('PF003', 'MSFT',  'Microsoft Corporation',   15000,  310.00, 415.50, 6232500, 1582500, 12.47),
    ('PF003', 'NVDA',  'NVIDIA Corporation',       8000,  520.00, 875.50, 7004000, 2844000, 14.01),
    ('PF003', 'CRM',   'Salesforce Inc.',         10000,  215.00, 325.50, 3255000, 1105000,  6.51),
    ('PF003', 'ORCL',  'Oracle Corporation',      12000,   95.00, 175.25, 2103000,  963000,  4.21);

-- 1d. Transactions (representative subset across portfolios and tickers)
CREATE OR REPLACE TABLE transactions (
    transaction_id   NUMBER AUTOINCREMENT PRIMARY KEY,
    portfolio_id     VARCHAR(10),
    ticker           VARCHAR(10)  NOT NULL,
    transaction_type VARCHAR(10)  NOT NULL,
    shares           NUMBER(18,4),
    price            NUMBER(18,2),
    amount           NUMBER(18,2),
    fees             NUMBER(18,2) DEFAULT 0,
    transaction_date TIMESTAMP,
    notes            VARCHAR(200),
    FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id)
);

INSERT INTO transactions (portfolio_id, ticker, transaction_type, shares, price, amount, fees, transaction_date, notes) VALUES
    ('PF001', 'AAPL',  'BUY',      500,  178.50,  89250.00,  44.63, '2025-01-08T10:30:00', 'BUY order for AAPL'),
    ('PF001', 'MSFT',  'BUY',      200,  405.25,  81050.00,  40.53, '2025-01-12T14:15:00', 'BUY order for MSFT'),
    ('PF001', 'GOOGL', 'SELL',     300,  172.00,  51600.00,  25.80, '2025-01-15T09:45:00', 'SELL order for GOOGL'),
    ('PF001', 'META',  'BUY',      150,  575.00,  86250.00,  43.13, '2025-01-20T11:00:00', 'BUY order for META'),
    ('PF001', 'AAPL',  'DIVIDEND', 15000,   0.25,   3750.00,   0.00, '2025-02-01T00:00:00', 'DIVIDEND order for AAPL'),
    ('PF001', 'AMZN',  'BUY',      400,  220.00,  88000.00,  44.00, '2025-02-05T13:20:00', 'BUY order for AMZN'),
    ('PF001', 'MSFT',  'SELL',     100,  420.00,  42000.00,  21.00, '2025-02-10T15:30:00', 'SELL order for MSFT'),
    ('PF002', 'NVDA',  'BUY',     1000,  850.00, 850000.00, 425.00, '2025-01-06T09:30:00', 'BUY order for NVDA'),
    ('PF002', 'AMD',   'SELL',     500,  148.00,  74000.00,  37.00, '2025-01-10T10:00:00', 'SELL order for AMD'),
    ('PF002', 'PLTR',  'BUY',     2000,   62.00, 124000.00,  62.00, '2025-01-18T14:00:00', 'BUY order for PLTR'),
    ('PF002', 'AVGO',  'BUY',      100, 1400.00, 140000.00,  70.00, '2025-01-25T11:30:00', 'BUY order for AVGO'),
    ('PF002', 'NVDA',  'SELL',     200,  890.00, 178000.00,  89.00, '2025-02-03T09:15:00', 'SELL order for NVDA'),
    ('PF002', 'TSM',   'BUY',      500,  180.00,  90000.00,  45.00, '2025-02-08T10:45:00', 'BUY order for TSM'),
    ('PF002', 'AMD',   'BUY',      800,  140.00, 112000.00,  56.00, '2025-02-15T13:00:00', 'BUY order for AMD'),
    ('PF003', 'AAPL',  'BUY',     1500,  180.00, 270000.00, 135.00, '2025-01-07T10:00:00', 'BUY order for AAPL'),
    ('PF003', 'MSFT',  'BUY',      500,  400.00, 200000.00, 100.00, '2025-01-14T11:30:00', 'BUY order for MSFT'),
    ('PF003', 'NVDA',  'BUY',      300,  860.00, 258000.00, 129.00, '2025-01-22T09:00:00', 'BUY order for NVDA'),
    ('PF003', 'CRM',   'SELL',     200,  330.00,  66000.00,  33.00, '2025-01-28T14:45:00', 'SELL order for CRM'),
    ('PF003', 'ORCL',  'BUY',     1000,  170.00, 170000.00,  85.00, '2025-02-06T10:30:00', 'BUY order for ORCL'),
    ('PF003', 'MSFT',  'DIVIDEND',15000,   0.75,  11250.00,   0.00, '2025-02-14T00:00:00', 'DIVIDEND order for MSFT');

-- Verify table counts
SELECT 'accounts' AS table_name, COUNT(*) AS row_count FROM accounts
UNION ALL SELECT 'portfolios', COUNT(*) FROM portfolios
UNION ALL SELECT 'holdings', COUNT(*) FROM holdings
UNION ALL SELECT 'transactions', COUNT(*) FROM transactions;

-- =============================================================================
-- Step 2: Create a Semantic View over the financial tables
-- =============================================================================
-- Used by the INTERNAL_PORTFOLIO_ANALYST tool (Cortex Analyst text-to-SQL)

CREATE OR REPLACE SEMANTIC VIEW portfolio_semantic_view
    TABLES (
        accounts     AS BIGDATA_DB.MCP_TOOLS.ACCOUNTS     PRIMARY KEY (account_id),
        portfolios   AS BIGDATA_DB.MCP_TOOLS.PORTFOLIOS   PRIMARY KEY (portfolio_id),
        holdings     AS BIGDATA_DB.MCP_TOOLS.HOLDINGS     PRIMARY KEY (holding_id),
        transactions AS BIGDATA_DB.MCP_TOOLS.TRANSACTIONS PRIMARY KEY (transaction_id)
    )
    RELATIONSHIPS (
        portfolios   (account_id)   REFERENCES accounts,
        holdings     (portfolio_id) REFERENCES portfolios,
        transactions (portfolio_id) REFERENCES portfolios
    )
    FACTS (
        accounts.balance             AS balance,
        portfolios.aum               AS aum,
        holdings.shares              AS shares,
        holdings.avg_cost            AS avg_cost,
        holdings.current_price       AS current_price,
        holdings.market_value        AS market_value,
        holdings.unrealized_pnl      AS unrealized_pnl,
        holdings.weight_pct          AS weight_pct,
        transactions.shares          AS shares,
        transactions.price           AS price,
        transactions.amount          AS amount,
        transactions.fees            AS fees
    )
    DIMENSIONS (
        accounts.account_id          AS account_id,
        accounts.account_name        AS account_name,
        accounts.account_type        AS account_type,
        accounts.currency            AS currency,
        portfolios.portfolio_id      AS portfolio_id,
        portfolios.portfolio_name    AS portfolio_name,
        portfolios.strategy          AS strategy,
        portfolios.risk_profile      AS risk_profile,
        holdings.ticker              AS ticker,
        holdings.company_name        AS company_name,
        transactions.transaction_type AS transaction_type,
        transactions.transaction_date AS transaction_date,
        transactions.notes           AS notes
    )
    METRICS (
        holdings.total_market_value  AS SUM(market_value),
        holdings.total_unrealized_pnl AS SUM(unrealized_pnl),
        transactions.total_amount    AS SUM(amount),
        transactions.total_fees      AS SUM(fees),
        holdings.avg_weight          AS AVG(weight_pct)
    );

GRANT SELECT ON SEMANTIC VIEW BIGDATA_DB.MCP_TOOLS.portfolio_semantic_view TO ROLE PUBLIC;

DESCRIBE SEMANTIC VIEW portfolio_semantic_view;

-- =============================================================================
-- Step 3: Create research documents table + Cortex Search Service
-- =============================================================================
-- Used by the INTERNAL_RESEARCH_SERVICE tool (Cortex Search)

CREATE OR REPLACE TABLE research_documents (
    doc_id   VARCHAR(10) PRIMARY KEY,
    title    VARCHAR(200) NOT NULL,
    content  VARCHAR(16777216),
    ticker   VARCHAR(20),
    company  VARCHAR(100),
    doc_type VARCHAR(50),
    doc_date DATE
);

INSERT INTO research_documents VALUES
('DOC001',
 'NVIDIA Q4 2024 Investment Thesis Update',
 'NVIDIA Q4 2024 Investment Thesis Update

NVIDIA remains our top pick in the semiconductor space. Key highlights:

1. Data Center Revenue: $18.4B (+409% YoY) driven by H100/H200 GPU demand for AI training
2. Blackwell Architecture: Next-gen B100/B200 GPUs launching Q2 2025 with 2.5x performance
3. Software Moat: CUDA ecosystem has 4M+ developers, creating significant switching costs
4. AI Inference Opportunity: $150B TAM by 2027 as enterprises deploy AI at scale

Risk Factors: China export restrictions, AMD competition, supply constraints
Price Target: $950 (25x FY26E EPS)
Rating: STRONG BUY',
 'NVDA', 'NVIDIA', 'investment_thesis', '2024-12-15'),

('DOC002',
 'Apple Inc. Strategic Analysis - Services & AI Focus',
 'Apple Inc. Strategic Analysis - Services & AI Focus

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
Price Target: $210',
 'AAPL', 'Apple', 'strategic_analysis', '2024-12-12'),

('DOC003',
 'Microsoft Azure & AI Monetization Analysis',
 'Microsoft Azure & AI Monetization Analysis

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

Price Target: $475 | Rating: OVERWEIGHT',
 'MSFT', 'Microsoft', 'segment_analysis', '2024-12-08'),

('DOC004',
 'AMD - Data Center & AI Opportunity Assessment',
 'AMD - Data Center & AI Opportunity Assessment

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
Rating: HOLD | PT: $165',
 'AMD', 'AMD', 'investment_thesis', '2024-12-11'),

('DOC005',
 'Q1 2025 Portfolio Strategy - Technology Sector Allocation',
 'Q1 2025 Portfolio Strategy - Technology Sector Allocation

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
3. China tech policy changes',
 'PORTFOLIO', 'Internal Strategy', 'strategy_memo', '2025-01-05'),

('DOC006',
 'Technology Sector Risk Assessment - January 2025',
 'Technology Sector Risk Assessment - January 2025

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
- Maintain 5-10% cash allocation for opportunities',
 'PORTFOLIO', 'Risk Management', 'risk_assessment', '2025-01-10');

SELECT 'research_documents' AS table_name, COUNT(*) AS row_count FROM research_documents;

-- Create Cortex Search Service over the research documents
CREATE OR REPLACE CORTEX SEARCH SERVICE research_search_service
    ON content
    ATTRIBUTES ticker, company, doc_type, doc_date
    WAREHOUSE = BIGDATA_WH
    TARGET_LAG = '1 hour'
AS (
    SELECT
        doc_id,
        title,
        content,
        ticker,
        company,
        doc_type,
        doc_date
    FROM research_documents
);

GRANT USAGE ON CORTEX SEARCH SERVICE BIGDATA_DB.MCP_TOOLS.research_search_service TO ROLE PUBLIC;

SELECT '04_internal_data complete — run 05_bigdata_mcp.sql next' AS status;
