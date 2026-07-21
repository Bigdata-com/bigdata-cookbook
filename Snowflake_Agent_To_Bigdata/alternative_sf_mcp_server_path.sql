-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 4b: MCP Server — BigData MCP Tools (Private Preview)
-- =============================================================================
-- APPROACH B — Requires CREATE MCP SERVER feature access (Private Preview)
--
-- Creates a Snowflake-managed MCP Server object (BIGDATA_MCP_SERVER) that
-- wraps the BigData MCP stored procedures from script 02 as GENERIC tools.
-- MCP clients (including Snowflake Intelligence in Private Preview) can
-- discover and invoke these tools via the standard MCP protocol.
--
-- Run this after 03_test_procedures.sql confirms all tools are working.
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
-- Create the MCP Server wrapping all three BigData tools
-- =============================================================================
CREATE OR REPLACE MCP SERVER BIGDATA_MCP_SERVER
    FROM SPECIFICATION $$
    tools:
      - title: "BigData Search"
        name: "bigdata_search"
        type: "GENERIC"
        identifier: "BIGDATA_DB.MCP_TOOLS.BIGDATA_SEARCH"
        description: >
          Search engine for financial documents, earnings call transcripts,
          news articles, analyst reports, SEC filings, and business content.
          Returns document chunks with timestamps, source attribution, and URLs.
        config:
          type: "procedure"
          warehouse: "BIGDATA_WH"
          input_schema:
            type: "object"
            properties:
              search_text:
                type: "string"
                description: "Natural-language search query for financial and business content"
              search_mode:
                type: "string"
                description: "Search mode: 'fast' for direct semantic/lexical search, 'smart' for AI-interpreted search. Default is 'fast'."
              max_chunks:
                type: "number"
                description: "Maximum number of result chunks to retrieve"

      - title: "BigData Find Securities"
        name: "bigdata_find_securities"
        type: "GENERIC"
        identifier: "BIGDATA_DB.MCP_TOOLS.BIGDATA_FIND_SECURITIES"
        description: >
          Search for ETFs, funds, and securities using names, tickers, or
          identifiers. Returns the Knowledge Graph ID and security metadata.
          Call this tool first to resolve an entity identifier before using
          the company tearsheet tool.
        config:
          type: "procedure"
          warehouse: "BIGDATA_WH"
          input_schema:
            type: "object"
            properties:
              query:
                type: "string"
                description: "One focused search token: ETF/fund/company name or ticker, ISIN/CUSIP/SEDOL, or short theme (e.g., 'dividend ETF')"

      - title: "BigData Company Tearsheet"
        name: "bigdata_company_tearsheet"
        type: "GENERIC"
        identifier: "BIGDATA_DB.MCP_TOOLS.BIGDATA_COMPANY_TEARSHEET"
        description: >
          Get comprehensive financial data, market intelligence, and analyst
          coverage for both public and private companies. Requires the
          rp_entity_id obtained from the bigdata_find_securities tool.
        config:
          type: "procedure"
          warehouse: "BIGDATA_WH"
          input_schema:
            type: "object"
            properties:
              rp_entity_id:
                type: "string"
                description: "6-character RavenPack entity ID from find_securities (e.g., 4A6F00 for Alphabet)"
              company_type:
                type: "string"
                description: "Must be 'Public' or 'Private' — use the listing type from find_securities response"
              interval:
                type: "string"
                description: "For public companies only: 'quarter' (default) or 'annual' for financial statement periods"
    $$;

-- Grant usage so the MCP server is discoverable
GRANT USAGE ON MCP SERVER BIGDATA_DB.MCP_TOOLS.BIGDATA_MCP_SERVER TO ROLE PUBLIC;

-- Verify
DESCRIBE MCP SERVER BIGDATA_MCP_SERVER;
SHOW MCP SERVERS IN SCHEMA BIGDATA_DB.MCP_TOOLS;

SELECT '04b_mcp_server complete — proceed to 05_snowflake_intelligence.sql' AS status;
