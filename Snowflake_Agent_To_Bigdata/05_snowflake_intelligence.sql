-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 5: Connect SNOWFLAKE_BIGDATA_AGENT to Snowflake Intelligence
-- =============================================================================
-- After running scripts 01–04 (and optionally 04c), use the instructions
-- below to connect your agent to Snowflake Intelligence.
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
-- OPTION A: Using Cortex Agent (from script 04a or 04c)  [RECOMMENDED]
-- =============================================================================
-- The SNOWFLAKE_BIGDATA_AGENT is already configured. To use it in Snowflake Intelligence:
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
-- Example questions to try (after running 04a only — BigData MCP tools):
--   "What are the latest earnings results for Apple?"
--   "Give me a financial tearsheet for Microsoft"
--   "Search for NVIDIA AI chip demand news in 2024"
--
-- Example questions to try (after running 04c — combined Snowflake + BigData):
--   "What is the total revenue by nation? Show me a bar chart."
--   "Which customers have the highest order spend? Show top 10."
--   "What are the latest news about Apple? Also show me our top revenue nations as a chart."
--   "Get the financial tearsheet for the company with the most supplier activity in our data."

-- Verify the agent exists and is accessible
SHOW AGENTS IN SCHEMA BIGDATA_DB.MCP_TOOLS;

-- =============================================================================
-- OPTION B: Using MCP Server (from script 04b) [Private Preview]
-- =============================================================================
-- The MCP Server exposes BigData tools via the standard MCP protocol.
-- To use it in Snowflake Intelligence:
--
-- 1. Open Snowsight (https://<account>.snowflakecomputing.com)
-- 2. Navigate to: AI & ML  ->  Snowflake Intelligence
-- 3. Click "+ New" to create a new Intelligence instance
-- 4. Under "Tools", add your MCP Server tools:
--      Database: BIGDATA_DB
--      Schema:   MCP_TOOLS
--      MCP Server: BIGDATA_MCP_SERVER
-- 5. The three tools (bigdata_search, bigdata_find_companies, bigdata_company_tearsheet)
--    will be auto-discovered
-- 6. Save and start chatting
--
-- External MCP clients can also connect using:
--   https://<account_url>/api/v2/databases/BIGDATA_DB/schemas/MCP_TOOLS/mcp-servers/BIGDATA_MCP_SERVER
-- Authentication: Snowflake OAuth 2.0 (see Snowflake docs for setup)

-- Verify the MCP server exists
SHOW MCP SERVERS IN SCHEMA BIGDATA_DB.MCP_TOOLS;

-- =============================================================================
-- Optional: Test the agent directly via SQL (Option A only)
-- =============================================================================
-- SELECT SNOWFLAKE.CORTEX.AGENT(
--     'BIGDATA_DB.MCP_TOOLS.SNOWFLAKE_BIGDATA_AGENT',
--     'What are the latest earnings results for Apple?'
-- ) AS response;

SELECT '05_snowflake_intelligence complete — your agent is ready in Snowflake Intelligence' AS status;
