-- Replaced bigdata_find_companies with bigdata_find_securities supporting ETFs, funds, bonds, and additional filters
-- Co-authored with CoCo
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
--   bigdata_find_securities   — search ETFs, funds, and securities by name/ticker/ID
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
    search_mode STRING DEFAULT 'fast',
    max_chunks INT DEFAULT NULL,
    filters VARIANT DEFAULT NULL
)
RETURNS STRING
LANGUAGE SQL
AS
BEGIN
    LET query_obj VARIANT := OBJECT_CONSTRUCT('text', :search_text);
    IF (:max_chunks IS NOT NULL) THEN
        query_obj := OBJECT_INSERT(:query_obj, 'max_chunks', :max_chunks);
    END IF;
    IF (:filters IS NOT NULL) THEN
        query_obj := OBJECT_INSERT(:query_obj, 'filters', :filters);
    END IF;
    LET request_obj VARIANT := OBJECT_CONSTRUCT('search_mode', :search_mode, 'query', :query_obj);
    LET result STRING;
    CALL bigdata_mcp_call('bigdata_search', OBJECT_CONSTRUCT('request', :request_obj)) INTO :result;
    RETURN :result;
END;

-- =============================================================================
-- BIGDATA_FIND_SECURITIES — search ETFs, funds, and securities by name/ticker/ID
-- =============================================================================
CREATE OR REPLACE PROCEDURE bigdata_find_securities(
    query STRING,
    countries ARRAY DEFAULT NULL,
    listing_type STRING DEFAULT NULL,
    sectors ARRAY DEFAULT NULL,
    security_types ARRAY DEFAULT NULL
)
RETURNS STRING
LANGUAGE SQL
AS
BEGIN
    LET args VARIANT := OBJECT_CONSTRUCT('query', :query);
    IF (:countries IS NOT NULL) THEN
        args := OBJECT_INSERT(:args, 'countries', :countries);
    END IF;
    IF (:listing_type IS NOT NULL) THEN
        args := OBJECT_INSERT(:args, 'listing_type', :listing_type);
    END IF;
    IF (:sectors IS NOT NULL) THEN
        args := OBJECT_INSERT(:args, 'sectors', :sectors);
    END IF;
    IF (:security_types IS NOT NULL) THEN
        args := OBJECT_INSERT(:args, 'security_types', :security_types);
    END IF;
    LET result STRING;
    CALL bigdata_mcp_call('find_securities', :args) INTO :result;
    RETURN :result;
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
BEGIN
    LET result STRING;
    CALL bigdata_mcp_call(
        'bigdata_company_tearsheet',
        OBJECT_CONSTRUCT('rp_entity_id', :rp_entity_id, 'company_type', :company_type, 'interval', :interval)
    ) INTO :result;
    RETURN :result;
END;

-- Grant usage so non-admin roles can call these procedures
GRANT USAGE ON PROCEDURE bigdata_mcp_call(STRING, VARIANT)                    TO ROLE PUBLIC;
GRANT USAGE ON PROCEDURE bigdata_search(STRING, STRING, INT, VARIANT)           TO ROLE PUBLIC;
GRANT USAGE ON PROCEDURE bigdata_find_securities(STRING, ARRAY, STRING, ARRAY, ARRAY) TO ROLE PUBLIC;
GRANT USAGE ON PROCEDURE bigdata_company_tearsheet(STRING, STRING, STRING)     TO ROLE PUBLIC;

SELECT '02_create_mcp_procedures complete — proceed to 03_test_procedures.sql' AS status;
