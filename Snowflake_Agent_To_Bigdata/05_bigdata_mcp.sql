-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 05: BigData MCP Agent (BigData Tools Only)
-- =============================================================================
-- Creates a Cortex Agent (SNOWFLAKE_BIGDATA_AGENT) that uses the three
-- BigData MCP stored procedures from script 02 as custom tools.
-- This agent demonstrates how BigData.com financial intelligence is surfaced
-- directly inside Snowflake Intelligence via natural-language queries.
--
-- Run this after 04_internal_data.sql.
-- After verifying BigData tools work, run 06_snowflake_bigdata_agent.sql to
-- replace this agent with the full combined version (internal + BigData).
--
-- NOTE: Agent spec uses JSON format inside the $$ block.
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
-- Create the Cortex Agent with BigData MCP tools (JSON specification format)
-- =============================================================================
CREATE OR REPLACE AGENT BIGDATA_DB.MCP_TOOLS.SNOWFLAKE_BIGDATA_AGENT
    FROM SPECIFICATION $$
{
    "models": {
        "orchestration": "auto"
    },
    "instructions": {
        "system": "You are a financial research analyst powered by Snowflake and BigData.com. You have access to BigData.com MCP tools that provide real-time financial intelligence including news, SEC filings, earnings transcripts, and company financial profiles.",
        "orchestration": "Use BIGDATA_FIND_COMPANIES to resolve company names or tickers to entity IDs before calling BIGDATA_COMPANY_TEARSHEET. Use BIGDATA_SEARCH for news, filings, transcripts, and research. Always cite data sources in your responses."
    },
    "tools": [
        {
            "tool_spec": {
                "type": "generic",
                "name": "BIGDATA_SEARCH",
                "description": "Search for financial insights across news, SEC filings, earnings transcripts, and research documents using BigData.com MCP protocol. Returns relevant chunks with relevance scores.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "search_text": {
                            "type": "string",
                            "description": "Natural-language search query for financial and business content"
                        },
                        "max_chunks": {
                            "type": "number",
                            "description": "Maximum number of chunks to retrieve. Default is 10."
                        }
                    },
                    "required": ["search_text"]
                }
            }
        },
        {
            "tool_spec": {
                "type": "generic",
                "name": "BIGDATA_FIND_COMPANIES",
                "description": "Identify a private or public company by name, ticker, ISIN, SEDOL, CUSIP, or webpage URL and retrieve its Knowledge Graph entity ID. Always call this before BIGDATA_COMPANY_TEARSHEET to get the rp_entity_id.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Partial or complete company name, webpage, ticker, ISIN, SEDOL, or CUSIP"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "tool_spec": {
                "type": "generic",
                "name": "BIGDATA_COMPANY_TEARSHEET",
                "description": "Get comprehensive financial data, market intelligence, and analyst coverage for both public and private companies. Requires the rp_entity_id from BIGDATA_FIND_COMPANIES.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "rp_entity_id": {
                            "type": "string",
                            "description": "6-character RavenPack entity ID from find_companies (e.g., 4A6F00 for Alphabet)"
                        },
                        "company_type": {
                            "type": "string",
                            "description": "Must be 'Public' or 'Private' — use the exact type field from find_companies response"
                        },
                        "interval": {
                            "type": "string",
                            "description": "For public companies only: 'quarter' (default) or 'annual' for financial statement periods"
                        }
                    },
                    "required": ["rp_entity_id", "company_type"]
                }
            }
        }
    ],
    "tool_resources": {
        "BIGDATA_SEARCH": {
            "type": "procedure",
            "identifier": "BIGDATA_DB.MCP_TOOLS.BIGDATA_SEARCH",
            "name": "BIGDATA_SEARCH(VARCHAR, DEFAULT NUMBER)",
            "execution_environment": {
                "type": "warehouse",
                "warehouse": "BIGDATA_WH",
                "query_timeout": 180
            }
        },
        "BIGDATA_FIND_COMPANIES": {
            "type": "procedure",
            "identifier": "BIGDATA_DB.MCP_TOOLS.BIGDATA_FIND_COMPANIES",
            "name": "BIGDATA_FIND_COMPANIES(VARCHAR)",
            "execution_environment": {
                "type": "warehouse",
                "warehouse": "BIGDATA_WH",
                "query_timeout": 180
            }
        },
        "BIGDATA_COMPANY_TEARSHEET": {
            "type": "procedure",
            "identifier": "BIGDATA_DB.MCP_TOOLS.BIGDATA_COMPANY_TEARSHEET",
            "name": "BIGDATA_COMPANY_TEARSHEET(VARCHAR, VARCHAR, DEFAULT VARCHAR)",
            "execution_environment": {
                "type": "warehouse",
                "warehouse": "BIGDATA_WH",
                "query_timeout": 180
            }
        }
    }
}
$$;

GRANT USAGE ON AGENT BIGDATA_DB.MCP_TOOLS.SNOWFLAKE_BIGDATA_AGENT TO ROLE PUBLIC;

-- Verify
SHOW AGENTS IN SCHEMA BIGDATA_DB.MCP_TOOLS;

SELECT '05_bigdata_mcp complete — run 06_snowflake_bigdata_agent.sql next' AS status;
