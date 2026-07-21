-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 08: Cleanup — Remove All Created Objects
-- =============================================================================
-- Run this to tear down everything created by scripts 01-07.
-- Objects are dropped in reverse dependency order.
--
-- WARNING: This will permanently delete all objects. Make sure you no longer
-- need them before running.
-- =============================================================================

-- *** CONFIGURATION — must match values from previous scripts ***
SET db_name     = 'BIGDATA_DB';
SET schema_name = 'MCP_TOOLS';
SET wh_name     = 'BIGDATA_WH';
-- *** END CONFIGURATION ***

USE ROLE ACCOUNTADMIN;

-- Step 1: Drop agent (created in 06)
DROP AGENT IF EXISTS BIGDATA_DB.MCP_TOOLS.SNOWFLAKE_BIGDATA_AGENT;

-- Step 2: Drop MCP server (if created via alternative_sf_mcp_server_path.sql)
DROP MCP SERVER IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_MCP_SERVER;

-- Step 3: Drop Cortex Search Service (created in 04)
DROP CORTEX SEARCH SERVICE IF EXISTS BIGDATA_DB.MCP_TOOLS.RESEARCH_SEARCH_SERVICE;

-- Step 4: Drop semantic view (created in 04)
DROP SEMANTIC VIEW IF EXISTS BIGDATA_DB.MCP_TOOLS.PORTFOLIO_SEMANTIC_VIEW;

-- Step 5: Drop financial tables (created in 04, reverse dependency order)
DROP TABLE IF EXISTS BIGDATA_DB.MCP_TOOLS.TRANSACTIONS;
DROP TABLE IF EXISTS BIGDATA_DB.MCP_TOOLS.HOLDINGS;
DROP TABLE IF EXISTS BIGDATA_DB.MCP_TOOLS.PORTFOLIOS;
DROP TABLE IF EXISTS BIGDATA_DB.MCP_TOOLS.ACCOUNTS;
DROP TABLE IF EXISTS BIGDATA_DB.MCP_TOOLS.RESEARCH_DOCUMENTS;

-- Step 6: Drop stored procedures (created in 02)
DROP PROCEDURE IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_COMPANY_TEARSHEET(STRING, STRING, STRING);
DROP PROCEDURE IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_FIND_SECURITIES(STRING, ARRAY, STRING, ARRAY, ARRAY);
DROP PROCEDURE IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_SEARCH(STRING, STRING, INT, VARIANT);
-- Legacy signatures (pre find_securities / search_mode update)
DROP PROCEDURE IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_FIND_COMPANIES(STRING);
DROP PROCEDURE IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_SEARCH(STRING, INT);
DROP PROCEDURE IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_MCP_CALL(STRING, VARIANT);

-- Step 7: Drop external access integration
DROP EXTERNAL ACCESS INTEGRATION IF EXISTS bigdata_mcp_eai;

-- Step 8: Drop secret
DROP SECRET IF EXISTS BIGDATA_DB.MCP_TOOLS.bigdata_api_key;

-- Step 9: Drop network rule
DROP NETWORK RULE IF EXISTS BIGDATA_DB.MCP_TOOLS.bigdata_mcp_rule;

-- Step 10: Drop schema, database, warehouse (optional — uncomment if desired)
-- DROP SCHEMA IF EXISTS BIGDATA_DB.MCP_TOOLS;
-- DROP DATABASE IF EXISTS BIGDATA_DB;
-- DROP WAREHOUSE IF EXISTS BIGDATA_WH;


SELECT '08_cleanup complete — all Snowflake + BigData MCP demo objects have been removed' AS status;
