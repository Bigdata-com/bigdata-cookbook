-- =============================================================================
-- Snowflake + BigData MCP Demo
-- Script 2: Create BigData MCP Stored Procedures
-- =============================================================================
-- Run this after 01_setup_infrastructure.sql.
-- Creates four stored procedures that call https://mcp.bigdata.com via
-- JSON-RPC 2.0 and expose BigData MCP tools inside Snowflake.
--
-- Procedures created:
--   bigdata_mcp_call          — core generic caller (JSON-RPC 2.0 over SSE)
--   bigdata_search            — search financial news, filings, transcripts
--   bigdata_find_companies    — resolve company name/ticker to entity ID
--   bigdata_company_tearsheet — full financial profile for public/private companies
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
-- Core procedure: generic MCP tool caller (JSON-RPC 2.0 over SSE)
-- =============================================================================
CREATE OR REPLACE PROCEDURE bigdata_mcp_call(
    tool_name STRING,
    arguments VARIANT
)
RETURNS STRING
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
PACKAGES = ('requests', 'snowflake-snowpark-python')
EXTERNAL_ACCESS_INTEGRATIONS = (bigdata_mcp_eai)
SECRETS = ('api_key' = bigdata_api_key)
HANDLER = 'main'
AS $$
import _snowflake
import requests
import json

MCP_URL = "https://mcp.bigdata.com"

def main(session, tool_name: str, arguments: dict) -> str:
    api_key = _snowflake.get_generic_secret_string('api_key')

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-api-key": api_key
    }

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments or {}}
    }

    try:
        resp = requests.post(MCP_URL, headers=headers, json=payload, stream=True, timeout=180)
        resp.raise_for_status()

        result = None
        for line in resp.iter_lines(decode_unicode=True):
            if line and line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    try:
                        result = json.loads(data)
                    except json.JSONDecodeError:
                        pass

        if result:
            if "result" in result:
                return json.dumps(result["result"], indent=2)
            elif "error" in result:
                return json.dumps({"error": result["error"]}, indent=2)
        return json.dumps(result, indent=2)
    except requests.exceptions.HTTPError as e:
        return json.dumps({"error": str(e), "response": resp.text[:500]})
    except Exception as e:
        return json.dumps({"error": str(e)})
$$;

-- =============================================================================
-- BIGDATA_SEARCH — search financial news, filings, transcripts, research
-- =============================================================================
CREATE OR REPLACE PROCEDURE bigdata_search(
    search_text STRING,
    max_chunks INT DEFAULT 10
)
RETURNS STRING
LANGUAGE SQL
AS
DECLARE
    result STRING;
BEGIN
    CALL bigdata_mcp_call(
        'bigdata_search',
        OBJECT_CONSTRUCT('search_text', :search_text, 'max_chunks', :max_chunks)
    ) INTO result;
    RETURN result;
END;

-- =============================================================================
-- BIGDATA_FIND_COMPANIES — resolve company name/ticker/ISIN to entity ID
-- =============================================================================
CREATE OR REPLACE PROCEDURE bigdata_find_companies(
    query STRING
)
RETURNS STRING
LANGUAGE SQL
AS
DECLARE
    result STRING;
BEGIN
    CALL bigdata_mcp_call(
        'find_companies',
        OBJECT_CONSTRUCT('query', :query)
    ) INTO result;
    RETURN result;
END;

-- =============================================================================
-- BIGDATA_COMPANY_TEARSHEET — financial data and analyst coverage
-- =============================================================================
CREATE OR REPLACE PROCEDURE bigdata_company_tearsheet(
    rp_entity_id STRING,
    company_type STRING,
    interval STRING DEFAULT 'quarter'
)
RETURNS STRING
LANGUAGE SQL
AS
DECLARE
    result STRING;
BEGIN
    CALL bigdata_mcp_call(
        'bigdata_company_tearsheet',
        OBJECT_CONSTRUCT('rp_entity_id', :rp_entity_id, 'company_type', :company_type, 'interval', :interval)
    ) INTO result;
    RETURN result;
END;

-- Grant usage so non-admin roles can call these procedures
GRANT USAGE ON PROCEDURE bigdata_mcp_call(STRING, VARIANT)                    TO ROLE PUBLIC;
GRANT USAGE ON PROCEDURE bigdata_search(STRING, INT)                           TO ROLE PUBLIC;
GRANT USAGE ON PROCEDURE bigdata_find_companies(STRING)                        TO ROLE PUBLIC;
GRANT USAGE ON PROCEDURE bigdata_company_tearsheet(STRING, STRING, STRING)     TO ROLE PUBLIC;

SELECT '02_create_mcp_procedures complete — proceed to 03_test_procedures.sql' AS status;
