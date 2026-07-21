-- Updated tool references from bigdata_find_companies to bigdata_find_securities
-- Co-authored with CoCo
-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 07: Connect SNOWFLAKE_BIGDATA_AGENT to Snowflake Intelligence
-- =============================================================================
-- After running scripts 01–06, use the instructions below to connect your
-- agent to Snowflake Intelligence.
-- =============================================================================

-- *** CONFIGURATION — must match values from previous scripts ***
SET db_name     = 'BIGDATA_DB';
SET schema_name = 'MCP_TOOLS';
SET wh_name     = 'BIGDATA_WH';
-- *** END CONFIGURATION ***

USE ROLE ACCOUNTADMIN;
USE DATABASE IDENTIFIER($db_name);
USE SCHEMA   IDENTIFIER($schema_name);
USE WAREHOUSE IDENTIFIER($wh_name);

-- =============================================================================
-- Connect to Snowflake Intelligence
-- =============================================================================
-- The SNOWFLAKE_BIGDATA_AGENT is already configured with six tools:
--   Internal:  INTERNAL_PORTFOLIO_ANALYST, INTERNAL_RESEARCH_SERVICE, DATA_TO_CHART
--   External:  BIGDATA_SEARCH, BIGDATA_FIND_SECURITIES, BIGDATA_COMPANY_TEARSHEET
--
-- To use it in Snowflake Intelligence:
--
-- 1. Open Snowsight (https://<account>.snowflakecomputing.com)
-- 2. Navigate to: AI & ML  ->  Snowflake Intelligence
-- 3. Click "+ New" to create a new Intelligence instance
-- 4. Under "Agents", add your Cortex Agent:
--      Database: BIGDATA_DB
--      Schema:   MCP_TOOLS
--      Agent:    SNOWFLAKE_BIGDATA_AGENT
-- 5. Save and start chatting
--
-- Example questions to try:
--   "What are the top holdings by market value across all portfolios?"
--   "Show unrealized P&L by ticker as a bar chart"
--   "What does our internal research say about NVIDIA?"
--   "Search for the latest Apple earnings news and compare with our internal thesis"
--   "Which portfolio has the highest AUM? Show the breakdown of its holdings."
--   "What are the key risks identified in our technology sector assessment?"

-- Verify the agent exists and is accessible
SHOW AGENTS IN SCHEMA BIGDATA_DB.MCP_TOOLS;

-- =============================================================================
-- Optional: Test the agent directly via SQL
-- =============================================================================
-- SELECT SNOWFLAKE.CORTEX.AGENT(
--     'BIGDATA_DB.MCP_TOOLS.SNOWFLAKE_BIGDATA_AGENT',
--     'What are the top holdings by market value across all portfolios?'
-- ) AS response;

SELECT '07_snowflake_intelligence complete — your agent is ready in Snowflake Intelligence' AS status;

-- see the agents at: AI & ML > Agents 

