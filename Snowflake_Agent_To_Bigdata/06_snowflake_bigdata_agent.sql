-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 06: Combined Snowflake + BigData Agent
-- =============================================================================
-- Replaces the BigData-only agent from script 05 with the full combined agent.
-- The SNOWFLAKE_BIGDATA_AGENT now has six tools:
--
--   Internal (Snowflake):
--     INTERNAL_PORTFOLIO_ANALYST  — Cortex Analyst text-to-SQL over portfolio data
--     INTERNAL_RESEARCH_SERVICE   — Cortex Search over internal research documents
--     DATA_TO_CHART               — chart generation from tabular results
--
--   External (BigData.com MCP):
--     BIGDATA_SEARCH              — news, filings, transcripts search
--     BIGDATA_FIND_COMPANIES      — resolve company name/ticker to entity ID
--     BIGDATA_COMPANY_TEARSHEET   — full company financial profile
--
-- Run this AFTER 05_bigdata_mcp.sql (and 04_internal_data.sql for tables).
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
-- Recreate agent with all six tools
-- =============================================================================
CREATE OR REPLACE AGENT BIGDATA_DB.MCP_TOOLS.SNOWFLAKE_BIGDATA_AGENT
    FROM SPECIFICATION $$
{
    "models": {
        "orchestration": "auto"
    },
    "instructions": {
        "system": "You are a financial research analyst powered by Snowflake and BigData.com. You have access to internal portfolio data (accounts, portfolios, holdings, transactions), internal research documents (investment theses, risk assessments, strategy memos), and BigData.com real-time financial intelligence (news, SEC filings, earnings transcripts, company profiles). Combine these sources to deliver comprehensive financial insights. When using BigData.com tools (BIGDATA_SEARCH, BIGDATA_FIND_COMPANIES, BIGDATA_COMPANY_TEARSHEET), always provide citations including the source name, headline, URL, and date when available.",
        "orchestration": "For questions about portfolio holdings, accounts, transactions, AUM, P&L, or any structured financial data use INTERNAL_PORTFOLIO_ANALYST. For questions about internal research notes, investment theses, risk assessments, or strategy memos use INTERNAL_RESEARCH_SERVICE. For external financial news, SEC filings, or earnings transcripts use BIGDATA_SEARCH. For company lookup by name or ticker use BIGDATA_FIND_COMPANIES. For detailed company financials and analyst coverage use BIGDATA_COMPANY_TEARSHEET (always call BIGDATA_FIND_COMPANIES first to get the rp_entity_id). Generate charts with DATA_TO_CHART for data that benefits from visualization. Always cite sources."
    },
    "tools": [
        {
            "tool_spec": {
                "type": "cortex_analyst_text_to_sql",
                "name": "INTERNAL_PORTFOLIO_ANALYST",
                "description": "Answers structured data questions about portfolio holdings, accounts, transactions, unrealized P&L, market values, and asset allocation. Use for questions like: top holdings by market value, total AUM by account, recent buy/sell transactions, P&L by ticker."
            }
        },
        {
            "tool_spec": {
                "type": "cortex_search",
                "name": "INTERNAL_RESEARCH_SERVICE",
                "description": "Searches internal research documents including investment theses, strategic analyses, portfolio strategy memos, and risk assessments. Use for questions like: What is our thesis on NVIDIA? What are the key risks? What allocation changes are recommended?"
            }
        },
        {
            "tool_spec": {
                "type": "data_to_chart",
                "name": "DATA_TO_CHART",
                "description": "Generates visualizations and charts from tabular data. Use after INTERNAL_PORTFOLIO_ANALYST returns results when the user asks for a chart, graph, or visual representation."
            }
        },
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
        "INTERNAL_PORTFOLIO_ANALYST": {
            "semantic_view": "BIGDATA_DB.MCP_TOOLS.PORTFOLIO_SEMANTIC_VIEW",
            "execution_environment": {
                "type": "warehouse",
                "warehouse": "BIGDATA_WH",
                "query_timeout": 120
            }
        },
        "INTERNAL_RESEARCH_SERVICE": {
            "name": "BIGDATA_DB.MCP_TOOLS.RESEARCH_SEARCH_SERVICE",
            "max_results": 5,
            "title_column": "TITLE",
            "id_column": "DOC_ID",
            "columns_and_descriptions": {
                "CONTENT": {
                    "description": "The main text content of the research document including analysis, recommendations, and financial data",
                    "type": "string",
                    "searchable": true,
                    "filterable": false
                },
                "TICKER": {
                    "description": "Stock ticker symbol for the company covered. Values include: NVDA, AAPL, MSFT, AMD, PORTFOLIO.",
                    "type": "string",
                    "searchable": false,
                    "filterable": true
                },
                "DOC_TYPE": {
                    "description": "Type of research document. Values include: investment_thesis, strategic_analysis, segment_analysis, strategy_memo, risk_assessment.",
                    "type": "string",
                    "searchable": false,
                    "filterable": true
                },
                "COMPANY": {
                    "description": "Company name. Values include: NVIDIA, Apple, Microsoft, AMD, Internal Strategy, Risk Management.",
                    "type": "string",
                    "searchable": false,
                    "filterable": true
                }
            }
        },
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

SELECT '06_snowflake_bigdata_agent complete — run 07_snowflake_intelligence.sql next' AS status;
