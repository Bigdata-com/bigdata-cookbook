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
          Search for financial insights across news, SEC filings, earnings
          transcripts, and research documents using BigData.com MCP protocol.
          Returns relevant chunks with relevance scores. Use this tool when
          the user asks about financial news, market events, earnings, or
          any document-level research.
        config:
          type: "procedure"
          warehouse: "BIGDATA_WH"
          input_schema:
            type: "object"
            properties:
              search_text:
                type: "string"
                description: "Natural-language search query for financial and business content"
              max_chunks:
                type: "number"
                description: "Maximum number of result chunks to retrieve. Default is 10."

      - title: "BigData Find Companies"
        name: "bigdata_find_companies"
        type: "GENERIC"
        identifier: "BIGDATA_DB.MCP_TOOLS.BIGDATA_FIND_COMPANIES"
        description: >
          Identify a private or public company by name, ticker, ISIN, SEDOL,
          CUSIP, or webpage URL and retrieve its Knowledge Graph entity ID.
          Call this tool first to resolve a company identifier before using
          the company tearsheet tool.
        config:
          type: "procedure"
          warehouse: "BIGDATA_WH"
          input_schema:
            type: "object"
            properties:
              query:
                type: "string"
                description: "Partial or complete company name, webpage, ticker, ISIN, SEDOL, or CUSIP"

      - title: "BigData Company Tearsheet"
        name: "bigdata_company_tearsheet"
        type: "GENERIC"
        identifier: "BIGDATA_DB.MCP_TOOLS.BIGDATA_COMPANY_TEARSHEET"
        description: >
          Get comprehensive financial data, market intelligence, and analyst
          coverage for both public and private companies. Requires the
          rp_entity_id obtained from the bigdata_find_companies tool.
        config:
          type: "procedure"
          warehouse: "BIGDATA_WH"
          input_schema:
            type: "object"
            properties:
              rp_entity_id:
                type: "string"
                description: "6-character RavenPack entity ID from find_companies (e.g., 4A6F00 for Alphabet)"
              company_type:
                type: "string"
                description: "Must be 'Public' or 'Private' — use the exact type field from find_companies response"
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
