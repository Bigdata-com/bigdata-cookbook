-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 6: Cleanup — Remove All Created Objects
-- =============================================================================
-- Run this to tear down everything created by scripts 01-05.
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

-- Step 1: Drop agent (created in 04a or 04c)
DROP AGENT IF EXISTS BIGDATA_DB.MCP_TOOLS.SNOWFLAKE_BIGDATA_AGENT;

-- Step 2: Drop MCP server (Approach B, created in 04b)
DROP MCP SERVER IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_MCP_SERVER;

-- Step 3: Drop semantic view (created in 04c)
DROP SEMANTIC VIEW IF EXISTS BIGDATA_DB.MCP_TOOLS.tpch_orders_semantic_view;

-- Step 4: Drop stored procedures (created in 02)
DROP PROCEDURE IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_COMPANY_TEARSHEET(STRING, STRING, STRING);
DROP PROCEDURE IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_FIND_COMPANIES(STRING);
DROP PROCEDURE IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_SEARCH(STRING, INT);
DROP PROCEDURE IF EXISTS BIGDATA_DB.MCP_TOOLS.BIGDATA_MCP_CALL(STRING, VARIANT);

-- Step 5: Drop external access integration
DROP EXTERNAL ACCESS INTEGRATION IF EXISTS bigdata_mcp_eai;

-- Step 6: Drop secret
DROP SECRET IF EXISTS BIGDATA_DB.MCP_TOOLS.bigdata_api_key;

-- Step 7: Drop network rule
DROP NETWORK RULE IF EXISTS BIGDATA_DB.MCP_TOOLS.bigdata_mcp_rule;

-- Step 8: Drop schema, database, warehouse (optional — uncomment if desired)
-- DROP SCHEMA IF EXISTS BIGDATA_DB.MCP_TOOLS;
-- DROP DATABASE IF EXISTS BIGDATA_DB;
-- DROP WAREHOUSE IF EXISTS BIGDATA_WH;

SELECT '06_cleanup complete — all Snowflake + BigData MCP demo objects have been removed' AS status;
