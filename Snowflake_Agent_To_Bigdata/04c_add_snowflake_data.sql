-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 4c: Add Snowflake Sample Data to the Agent (Full Combined Demo)
-- =============================================================================
-- This script extends the agent created in 04a to also query Snowflake's
-- built-in sample data (SNOWFLAKE_SAMPLE_DATA.TPCH_SF1) using Cortex Analyst
-- (natural-language to SQL). This is the recommended demo configuration.
--
-- The combined SNOWFLAKE_BIGDATA_AGENT can:
--   • Query Snowflake orders/revenue data via Cortex Analyst (TPCH_ANALYST)
--   • Search external financial news & filings via BIGDATA_SEARCH
--   • Resolve company names/tickers via BIGDATA_FIND_COMPANIES
--   • Get company financial profiles via BIGDATA_COMPANY_TEARSHEET
--   • Generate charts from any query results (DATA_TO_CHART)
--
-- Run this AFTER 04a_cortex_agent.sql.
-- This replaces the agent from 04a with the full combined version.
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
-- Step 1: Create a Semantic View over SNOWFLAKE_SAMPLE_DATA TPC-H tables
-- =============================================================================
-- The semantic view defines business-friendly names and relationships so
-- Cortex Analyst can answer natural-language questions about orders, revenue,
-- customers, and suppliers without knowing raw column names.
-- =============================================================================
CREATE OR REPLACE SEMANTIC VIEW tpch_orders_semantic_view
    TABLES (
        orders AS SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.ORDERS PRIMARY KEY (o_orderkey),
        lineitem AS SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.LINEITEM PRIMARY KEY (l_orderkey, l_linenumber),
        customer AS SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.CUSTOMER PRIMARY KEY (c_custkey),
        supplier AS SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.SUPPLIER PRIMARY KEY (s_suppkey),
        nation AS SNOWFLAKE_SAMPLE_DATA.TPCH_SF1.NATION PRIMARY KEY (n_nationkey)
    )
    RELATIONSHIPS (
        lineitem (l_orderkey) REFERENCES orders,
        orders (o_custkey) REFERENCES customer,
        supplier (s_nationkey) REFERENCES nation,
        customer (c_nationkey) REFERENCES nation
    )
    FACTS (
        orders.o_totalprice AS o_totalprice,
        lineitem.l_quantity AS l_quantity,
        lineitem.l_extendedprice AS l_extendedprice,
        lineitem.l_discount AS l_discount,
        lineitem.l_tax AS l_tax
    )
    DIMENSIONS (
        orders.order_date AS o_orderdate,
        orders.order_status AS o_orderstatus,
        orders.order_priority AS o_orderpriority,
        lineitem.ship_date AS l_shipdate,
        lineitem.return_flag AS l_returnflag,
        lineitem.line_status AS l_linestatus,
        lineitem.ship_mode AS l_shipmode,
        customer.customer_name AS c_name,
        customer.customer_segment AS c_mktsegment,
        supplier.supplier_name AS s_name,
        nation.nation_name AS n_name
    )
    METRICS (
        orders.total_order_price AS SUM(o_totalprice),
        lineitem.total_quantity AS SUM(l_quantity),
        lineitem.total_revenue AS SUM(l_extendedprice)
    );

-- Grant access to the semantic view
GRANT SELECT ON SEMANTIC VIEW BIGDATA_DB.MCP_TOOLS.tpch_orders_semantic_view TO ROLE PUBLIC;

-- Verify
DESCRIBE SEMANTIC VIEW tpch_orders_semantic_view;

-- =============================================================================
-- Step 2: Recreate agent as SNOWFLAKE_BIGDATA_AGENT with all five tools
-- =============================================================================
-- Tools:
--   TPCH_ANALYST              — Cortex Analyst text-to-SQL over Snowflake sample data
--   DATA_TO_CHART             — chart generation from tabular results
--   BIGDATA_SEARCH            — BigData MCP: news, filings, transcripts search
--   BIGDATA_FIND_COMPANIES    — BigData MCP: resolve company name/ticker to entity ID
--   BIGDATA_COMPANY_TEARSHEET — BigData MCP: full company financial profile
-- =============================================================================
CREATE OR REPLACE AGENT BIGDATA_DB.MCP_TOOLS.SNOWFLAKE_BIGDATA_AGENT
    FROM SPECIFICATION $$
{
    "models": {
        "orchestration": "auto"
    },
    "instructions": {
        "system": "You are a financial research analyst powered by Snowflake and BigData.com. You can query internal Snowflake business data and access BigData.com's real-time financial intelligence including news, SEC filings, earnings transcripts, and company financial profiles. Combine both sources to deliver comprehensive financial insights.",
        "orchestration": "For questions about orders, revenue, customers, suppliers, shipping, or any structured business data use TPCH_ANALYST. For external financial news, SEC filings, earnings transcripts, or company research use BIGDATA_SEARCH. For company lookup by name or ticker use BIGDATA_FIND_COMPANIES. For detailed company financials and analyst coverage use BIGDATA_COMPANY_TEARSHEET (always call BIGDATA_FIND_COMPANIES first to get the rp_entity_id). Generate charts with DATA_TO_CHART for data that benefits from visualization. Always cite sources."
    },
    "tools": [
        {
            "tool_spec": {
                "type": "cortex_analyst_text_to_sql",
                "name": "TPCH_ANALYST",
                "description": "Answers structured data questions about orders, revenue, customers, suppliers, shipping modes, and geographic regions using the TPC-H sample dataset. Use for questions like: total revenue by region, top customers by spend, orders by status, shipping analysis."
            }
        },
        {
            "tool_spec": {
                "type": "data_to_chart",
                "name": "DATA_TO_CHART",
                "description": "Generates visualizations and charts from tabular data. Use after TPCH_ANALYST returns results when the user asks for a chart, graph, or visual representation."
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
        "TPCH_ANALYST": {
            "semantic_view": "BIGDATA_DB.MCP_TOOLS.tpch_orders_semantic_view",
            "execution_environment": {
                "type": "warehouse",
                "warehouse": "BIGDATA_WH",
                "query_timeout": 120
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

SELECT '04c complete — SNOWFLAKE_BIGDATA_AGENT now has Snowflake data + BigData MCP + charting' AS status;
