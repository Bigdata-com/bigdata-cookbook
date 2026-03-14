-- =============================================================================
-- BigData MCP for Snowflake Intelligence
-- Script 1: Infrastructure Setup (Network Rule, Secret, External Access)
-- =============================================================================
-- Run this script in a Snowsight SQL Worksheet with ACCOUNTADMIN role.
--
-- Before running: replace <YOUR_BIGDATA_API_KEY> with your actual key
-- from https://platform.bigdata.com/api-keys
-- =============================================================================

-- *** CONFIGURATION — edit these values to match your environment ***
SET db_name     = 'BIGDATA_DB';
SET schema_name = 'MCP_TOOLS';
SET wh_name     = 'BIGDATA_WH';
SET api_key     = '<YOUR_BIGDATA_API_KEY>';   -- paste your key here

-- *** END CONFIGURATION ***

USE ROLE ACCOUNTADMIN;

-- 1. Create database and schema (idempotent)
CREATE DATABASE IF NOT EXISTS IDENTIFIER($db_name);
CREATE SCHEMA IF NOT EXISTS IDENTIFIER($db_name || '.' || $schema_name);

-- 2. Create warehouse (idempotent)
CREATE WAREHOUSE IF NOT EXISTS IDENTIFIER($wh_name)
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND   = 60
    AUTO_RESUME    = TRUE;

USE DATABASE IDENTIFIER($db_name);
USE SCHEMA   IDENTIFIER($schema_name);
USE WAREHOUSE IDENTIFIER($wh_name);

-- 3. Network rule — allow egress to BigData MCP endpoint
CREATE OR REPLACE NETWORK RULE bigdata_mcp_rule
    MODE       = EGRESS
    TYPE       = HOST_PORT
    VALUE_LIST = ('mcp.bigdata.com:443');

-- 4. Secret — store the BigData API key
CREATE OR REPLACE SECRET bigdata_api_key
    TYPE          = GENERIC_STRING
    SECRET_STRING = $api_key;

-- 5. External Access Integration — binds network rule + secret
CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION bigdata_mcp_eai
    ALLOWED_NETWORK_RULES          = (bigdata_mcp_rule)
    ALLOWED_AUTHENTICATION_SECRETS = (bigdata_api_key)
    ENABLED                        = TRUE;

-- Verify
SHOW NETWORK RULES;
SHOW SECRETS;
SHOW EXTERNAL ACCESS INTEGRATIONS;

SELECT '01_setup_infrastructure complete — proceed to 02_create_mcp_procedures.sql' AS status;
