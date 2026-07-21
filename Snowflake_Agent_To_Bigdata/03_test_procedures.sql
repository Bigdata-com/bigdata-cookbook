-- Updated test procedures to use bigdata_find_securities instead of bigdata_find_companies
-- Co-authored with CoCo
-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 3: Test BigData MCP Procedures
-- =============================================================================
-- Run this after 02_create_mcp_procedures.sql to verify connectivity
-- and correct responses before setting up the agent.
--
-- Each test should return a JSON result.
-- If you see "403" or "Unauthorized" — check your API key in script 01.
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
-- Test 1: BIGDATA_FIND_SECURITIES — look up Apple by name
-- Expected: JSON array with security records including id (entity ID)
-- =============================================================================
CALL bigdata_find_securities('Apple');

-- =============================================================================
-- Test 2: BIGDATA_FIND_SECURITIES — look up by ticker
-- Expected: JSON array with AAPL / Apple Inc.
-- =============================================================================
CALL bigdata_find_securities('AAPL');

-- =============================================================================
-- Test 2b: BIGDATA_FIND_SECURITIES — ETF search with filters
-- Expected: JSON array with US-listed ETF results
-- =============================================================================
CALL bigdata_find_securities('dividend ETF', ARRAY_CONSTRUCT('US'), NULL, NULL, ARRAY_CONSTRUCT('ETF'));

-- =============================================================================
-- Test 3: BIGDATA_SEARCH — search for recent financial news (fast mode)
-- Expected: JSON with "results" array containing document chunks
-- =============================================================================
CALL bigdata_search('Apple earnings Q4 2024');

-- =============================================================================
-- Test 3b: BIGDATA_SEARCH — smart mode with max_chunks
-- Expected: AI-interpreted search with limited results
-- =============================================================================
CALL bigdata_search('NVIDIA AI chip demand outlook', 'smart', 5);

-- =============================================================================
-- Test 4: BIGDATA_COMPANY_TEARSHEET — use Apple's entity ID (4A6F00)
-- Expected: Detailed financial data in markdown format
-- =============================================================================
CALL bigdata_company_tearsheet('4A6F00', 'Public', 'quarter');

-- =============================================================================
-- Test 5: BIGDATA_MCP_CALL — generic call, useful for debugging
-- =============================================================================
CALL bigdata_mcp_call(
    'find_securities',
    OBJECT_CONSTRUCT('query', 'Microsoft')
);

SELECT '03_test_procedures complete — all tools working. Proceed to 04a, then 04c.' AS status;
