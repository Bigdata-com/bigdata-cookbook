# Snowflake + BigData MCP Demo — End-to-End Guide

Demonstrate [BigData.com](https://bigdata.com) financial intelligence alongside **live Snowflake data** inside **Snowflake Intelligence** — all powered by the BigData MCP protocol and a single Cortex Agent (`SNOWFLAKE_BIGDATA_AGENT`).

The demo shows how Snowflake customers can combine their own internal data with real-time external financial intelligence (news, filings, earnings, company profiles) in one natural-language chat experience.

## What the Demo Covers

| Capability | Powered by | Example question |
|---|---|---|
| Query internal Snowflake data | `TPCH_ANALYST` (Cortex Analyst) | *"What is total revenue by nation?"* |
| Generate charts from query results | `DATA_TO_CHART` | *"Show that as a bar chart"* |
| Search financial news & filings | `BIGDATA_SEARCH` (BigData MCP) | *"Latest Apple earnings news"* |
| Resolve company name/ticker | `BIGDATA_FIND_COMPANIES` (BigData MCP) | *"Find the entity ID for Tesla"* |
| Full company financial profile | `BIGDATA_COMPANY_TEARSHEET` (BigData MCP) | *"Financial tearsheet for Microsoft"* |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Snowflake Account                           │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Foundation (scripts 01–03)                                 │   │
│  │                                                             │   │
│  │  Network Rule ──► External Access Integration ◄── Secret   │   │
│  │                                                             │   │
│  │  BigData MCP Procedures                                     │   │
│  │    • BIGDATA_SEARCH            (news, filings, transcripts) │   │
│  │    • BIGDATA_FIND_COMPANIES    (resolve name/ticker to ID)  │   │
│  │    • BIGDATA_COMPANY_TEARSHEET (financial data & coverage)  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│  ┌───────────────────────┼─────────────────────────────────────┐   │
│  │  SNOWFLAKE_BIGDATA_AGENT  (scripts 04a + 04c)               │   │
│  │                                                             │   │
│  │  TPCH_ANALYST         (Cortex Analyst over TPCH sample data)│   │
│  │  DATA_TO_CHART         (auto-generate charts)               │   │
│  │  BIGDATA_SEARCH        (→ mcp.bigdata.com)                  │   │
│  │  BIGDATA_FIND_COMPANIES (→ mcp.bigdata.com)                 │   │
│  │  BIGDATA_COMPANY_TEARSHEET (→ mcp.bigdata.com)              │   │
│  └───────────────────────┬─────────────────────────────────────┘   │
│                          │                                          │
│              Snowflake Intelligence (chat UI)                       │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                 https://mcp.bigdata.com
                 (BigData.com MCP API)
```

---

## Part 1: Snowflake Account and Basic Setup

Everything you need before running any scripts.

### 1.1 — Snowflake Account Requirements

> **IMPORTANT: A paid Snowflake Enterprise (or higher) account is required.**
> Trial accounts do NOT support External Access Integrations, which are needed
> to call the BigData MCP API. If you only have a trial account, you must add
> billing information (Admin → Billing in Snowsight) to convert it to a paid
> account before proceeding. Your remaining trial credits will still apply.

**If you need a new account:**

1. Go to [https://signup.snowflake.com/](https://signup.snowflake.com/)
2. Fill in your name and email
3. Choose these options:
   - **Edition**: **Enterprise** (required for Cortex Agents, External Access Integrations, and Snowflake Intelligence)
   - **Cloud Provider**: AWS or Azure (your preference)
   - **Region**: Closest to you (e.g., `US West (Oregon)` or `EU (Frankfurt)`)
4. Click **Get Started** and check your email for the activation link
5. Set your password and log in
6. **Add billing information**: Go to Admin → Billing → add a payment method to enable External Access Integrations

**If you already have a Snowflake account:** Confirm it is Enterprise edition or higher under Admin → Account → Edition.

### 1.2 — First Login and Snowsight Orientation

1. After activation, you land in **Snowsight** (the Snowflake web UI)
2. Note your **account identifier** from the URL: `https://<account_identifier>.snowflakecomputing.com`
3. Confirm you are using the **ACCOUNTADMIN** role — check the role selector in the bottom-left corner
4. Navigate to **SQL Worksheets** (left sidebar → Projects → Worksheets) — all scripts run here

### 1.3 — Create a Warehouse

Open a new SQL Worksheet and run:

```sql
CREATE WAREHOUSE IF NOT EXISTS BIGDATA_WH
    WAREHOUSE_SIZE = 'MEDIUM'
    AUTO_SUSPEND   = 60
    AUTO_RESUME    = TRUE;
```

### 1.4 — Create a Database and Schema

```sql
CREATE DATABASE IF NOT EXISTS BIGDATA_DB;
CREATE SCHEMA IF NOT EXISTS BIGDATA_DB.MCP_TOOLS;
```

These names are the defaults used in all scripts. Edit the `SET` block at the top of any script to change them.

### 1.5 — Get a BigData.com API Key

1. Go to [https://platform.bigdata.com/api-keys](https://platform.bigdata.com/api-keys)
2. Sign up or log in
3. Click **Generate API Key**
4. Copy the key — you will paste it into `01_setup_infrastructure.sql`

---

## Part 2: Standalone Demo Setup (No CLI Required)

> **No Snowflake CLI required.** Everything runs in Snowsight SQL Worksheets.
> No Native App, no Docker, no SPCS — just SQL scripts in order.

### Script Overview

| Script | What it does |
|--------|-------------|
| `01_setup_infrastructure.sql` | Network rule, secret (API key), external access integration |
| `02_create_mcp_procedures.sql` | BigData MCP stored procedures (`BIGDATA_SEARCH`, `BIGDATA_FIND_COMPANIES`, `BIGDATA_COMPANY_TEARSHEET`) |
| `03_test_procedures.sql` | Verify all BigData MCP procedures work |
| `04a_cortex_agent.sql` | **Approach A (GA)**: Cortex Agent with BigData MCP tools only |
| `04b_mcp_server.sql` | **Approach B (Preview)**: MCP Server object wrapping BigData tools |
| `04c_add_snowflake_data.sql` | **Extends 04a** — adds Snowflake data + charting for the full demo |
| `05_snowflake_intelligence.sql` | Instructions for connecting the agent to Snowflake Intelligence |
| `06_cleanup.sql` | Remove all objects when done |

**Recommended path**: Run `01` → `02` → `03` → `04a` → `04c` → `05`

`04c` is the key step that transforms the agent into the full combined demo: Snowflake data + BigData MCP + charting.

### Approach A vs. Approach B

| | Approach A: Cortex Agent | Approach B: MCP Server |
|---|---|---|
| **SQL command** | `CREATE AGENT` | `CREATE MCP SERVER` |
| **Availability** | Generally Available | Requires MCP Server feature access (Private Preview) |
| **How it works** | Agent uses procedures as custom tools | MCP Server wraps procedures as GENERIC tools |
| **Intelligence integration** | Add agent under "Agents" in Intelligence | Add MCP Server under "Tools" in Intelligence |
| **External MCP clients** | No (Snowflake-only) | Yes (via MCP protocol endpoint URL) |
| **Recommendation** | **Start here** | Use if you have MCP Server access |

---

### Step-by-Step: Script 01 — Infrastructure

Open `01_setup_infrastructure.sql` in a SQL Worksheet.

1. **Edit the configuration block** — paste your BigData API key:
   ```sql
   SET api_key = 'bd-abc123...your-actual-key...';
   ```
2. Run the entire script (`Ctrl+Shift+Enter` or `Cmd+Shift+Enter`)
3. Verify with the `SHOW` commands at the bottom — you should see:
   - `bigdata_mcp_rule` in network rules
   - `bigdata_api_key` in secrets
   - `bigdata_mcp_eai` in external access integrations

### Step-by-Step: Script 02 — BigData MCP Procedures

Open `02_create_mcp_procedures.sql` and run it.

This creates four procedures:

| Procedure | Purpose |
|-----------|---------|
| `bigdata_mcp_call(tool_name, arguments)` | Generic MCP caller — JSON-RPC 2.0 over SSE |
| `bigdata_search(search_text, max_chunks)` | Search financial news, SEC filings, transcripts |
| `bigdata_find_companies(query)` | Resolve company name/ticker to entity ID |
| `bigdata_company_tearsheet(rp_entity_id, company_type, interval)` | Get full company financial profile |

All procedures are bound to the EAI and secret from script 01 and can securely reach `https://mcp.bigdata.com`.

### Step-by-Step: Script 03 — Test

Open `03_test_procedures.sql` and run each `CALL` statement one at a time.

**Expected results:**

- `CALL bigdata_find_companies('Apple')` → JSON array with company records including `id`, `name`, `type`
- `CALL bigdata_search('Apple earnings Q4 2024', 5)` → JSON with search result chunks
- `CALL bigdata_company_tearsheet('4A6F00', 'Public', 'quarter')` → Detailed financial markdown

**If you get errors:**

| Error | Fix |
|-------|-----|
| `403 Forbidden` or `Unauthorized` | Check your API key in script 01 |
| `Network access denied` | Verify the EAI is enabled: `SHOW EXTERNAL ACCESS INTEGRATIONS` |
| `Procedure not found` | Confirm you ran script 02 in the same database/schema |
| `Timeout` | Retry — latency comes from the external BigData API, not Snowflake |

### Step-by-Step: Script 04a — Cortex Agent (BigData MCP Only)

Open `04a_cortex_agent.sql` and run it.

This creates `SNOWFLAKE_BIGDATA_AGENT` with three BigData MCP tools:

| Tool | Procedure |
|------|-----------|
| `BIGDATA_SEARCH` | Searches financial news, filings, transcripts |
| `BIGDATA_FIND_COMPANIES` | Resolves company name/ticker to entity ID |
| `BIGDATA_COMPANY_TEARSHEET` | Returns full company financial profile |

> **Note**: The `CREATE AGENT` spec uses **JSON format** inside the `$$` block (not YAML).
> `instructions` must be a top-level key — not nested inside `models` or `orchestration`.

### Step-by-Step: Script 04c — Add Snowflake Data + Charting (Full Demo)

Open `04c_add_snowflake_data.sql` and run it **after** 04a.

This is the step that completes the demo. It:

1. Creates a **semantic view** (`tpch_orders_semantic_view`) over `SNOWFLAKE_SAMPLE_DATA.TPCH_SF1` — orders, lineitems, customers, suppliers, nations — with business-friendly column names
2. Recreates `SNOWFLAKE_BIGDATA_AGENT` with two additional tools:
   - `TPCH_ANALYST` — answers structured questions about orders/revenue/customers by generating and running SQL
   - `DATA_TO_CHART` — generates charts from any tabular results

After running 04c, the agent orchestrates all five tools in one conversation:

| Question | Tool used |
|---|---|
| *"What is total revenue by region?"* | `TPCH_ANALYST` (internal Snowflake) |
| *"Show that as a bar chart"* | `DATA_TO_CHART` |
| *"What are analysts saying about Apple?"* | `BIGDATA_SEARCH` |
| *"Give me Apple's tearsheet"* | `BIGDATA_FIND_COMPANIES` → `BIGDATA_COMPANY_TEARSHEET` |

### Step-by-Step: Script 04b — MCP Server (Approach B, optional)

Open `04b_mcp_server.sql` and run it as an alternative to 04a.

Creates `BIGDATA_MCP_SERVER` — an MCP Server object that exposes the three BigData procedures as `GENERIC` tools discoverable by Snowflake Intelligence and any MCP-compatible client.

> If you get `Insufficient privileges to operate on MCP SERVER`, your account does not have this feature enabled. Use Approach A instead.

### Step-by-Step: Script 05 — Connect to Snowflake Intelligence

#### Using the Cortex Agent (Approach A — recommended)

1. Open Snowsight → **AI & ML → Snowflake Intelligence**
2. Click **"+ New"** to create a new Intelligence instance
3. Under **Agents**, add:
   - Database: `BIGDATA_DB`
   - Schema: `MCP_TOOLS`
   - Agent: `SNOWFLAKE_BIGDATA_AGENT`
4. Click **Save** and start chatting

#### Using the MCP Server (Approach B)

1. Open Snowsight → **AI & ML → Snowflake Intelligence**
2. Click **"+ New"** to create a new Intelligence instance
3. Under **Tools**, add the MCP Server:
   - Database: `BIGDATA_DB`
   - Schema: `MCP_TOOLS`
   - MCP Server: `BIGDATA_MCP_SERVER`
4. The three tools (`bigdata_search`, `bigdata_find_companies`, `bigdata_company_tearsheet`) will be auto-discovered
5. Click **Save** and start chatting

### Cleanup

When done testing, run `06_cleanup.sql` to remove all created objects. The database, schema, and warehouse are preserved by default — uncomment the last lines to remove those too.

---

## Demo Queries

These queries work in Snowflake Intelligence once you have run scripts 01–03, 04a, and 04c.

**Agent**: `SNOWFLAKE_BIGDATA_AGENT` &nbsp;|&nbsp; **5 tools available**:

| Tool | Data source |
|------|-------------|
| `TPCH_ANALYST` | `SNOWFLAKE_SAMPLE_DATA.TPCH_SF1` (internal Snowflake) |
| `DATA_TO_CHART` | Output of any query |
| `BIGDATA_SEARCH` | BigData.com MCP — news, filings, transcripts |
| `BIGDATA_FIND_COMPANIES` | BigData.com MCP — entity resolution |
| `BIGDATA_COMPANY_TEARSHEET` | BigData.com MCP — company profiles |

---

### Snowflake Data (TPCH_ANALYST)

- *"What is the total revenue by customer market segment?"*
- *"Show total order price by nation, top 10"*
- *"What is the breakdown of orders by order status?"*
- *"Which shipping mode has the highest total quantity shipped?"*
- *"Show monthly total revenue for 1995"*
- *"What is the total revenue for URGENT priority orders?"*

---

### Snowflake Data + Charts (TPCH_ANALYST + DATA_TO_CHART)

- *"Show total revenue by customer market segment as a bar chart"*
- *"Plot monthly total order price for 1996 as a line chart"*
- *"Show the share of total revenue by nation as a pie chart"*
- *"Compare total revenue across all shipping modes as a bar chart"*
- *"Show order count by order priority as a horizontal bar chart"*

---

### BigData MCP Only

- *"Search for the latest earnings news for Apple"*
- *"Find the company record for Tesla"*
- *"Give me a financial tearsheet for Microsoft"*
- *"Search for NVIDIA AI chip demand news from 2024"*
- *"Find Amazon and show me their analyst coverage"*

---

### Cross-Tool: Snowflake Data + BigData News

- *"What is our top customer market segment by total revenue? Then search for the latest news about that industry."*
  > Agent queries TPCH for segment revenue → identifies top segment → searches BigData for industry news

- *"Show me total revenue by nation, then search for recent economic news about the top performing nation."*
  > Agent queries TPCH → identifies top nation → searches BigData for that country's economic news

- *"Which shipping mode is most used? Search for news about that logistics sector."*
  > Agent queries TPCH for shipping mode counts → searches BigData for logistics news

---

### Cross-Tool: Snowflake Data + BigData Company Tearsheet

- *"What is our total revenue from the AUTOMOBILE customer segment? Also give me a financial tearsheet for Toyota."*
  > Agent queries TPCH for automobile segment revenue → finds Toyota entity ID → fetches tearsheet

- *"Show the top 3 nations by total order price, then give me a financial tearsheet for a major company from the top nation."*
  > Agent queries TPCH → user or agent picks a company → fetches tearsheet

- *"What is the total revenue for BUILDING segment customers? Compare that with the latest financials for Caterpillar."*
  > Agent queries TPCH → finds Caterpillar → fetches company tearsheet

---

### Cross-Tool: Snowflake Data + Charts + BigData MCP (Full Demo)

- *"Show total revenue by customer market segment as a bar chart, then search for the latest news about the top segment."*
  > Queries TPCH → renders bar chart → searches BigData news for top segment

- *"Plot monthly revenue for 1995 as a line chart. Are there any news events from that period that might explain spikes?"*
  > Queries TPCH monthly revenue → renders line chart → searches BigData for 1995 financial events

- *"Show revenue by nation as a pie chart, then give me a tearsheet for a major company from the leading nation."*
  > Queries TPCH → renders pie chart → finds company → fetches tearsheet

- *"What is the total revenue split by shipping mode? Show as a bar chart. Then search for recent disruptions in ocean shipping."*
  > Queries TPCH shipping modes → renders chart → searches BigData for logistics news

- *"Compare URGENT vs LOW priority order revenue as a chart, then search for supply chain news."*
  > Queries TPCH by priority → renders comparison chart → searches BigData

---

### Multi-Turn Conversations

These work as sequential follow-up messages in the same Intelligence thread:

**Chain 1**
1. *"What is the total revenue by customer market segment?"*
2. *"Show that as a pie chart"*
3. *"Which real-world companies are major players in the top segment?"*
4. *"Give me a financial tearsheet for one of those companies"*

**Chain 2**
1. *"Which nation has the highest total order revenue?"*
2. *"Now search for recent economic or trade news about that country"*
3. *"Find a major company headquartered there and show me their tearsheet"*

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| `External access is not supported for trial accounts` | Trial account limitation | Add billing info in Admin → Billing to convert to paid (remaining credits still apply) |
| `403 Forbidden` from BigData API | Invalid or expired API key | Regenerate at [platform.bigdata.com/api-keys](https://platform.bigdata.com/api-keys) and update the secret |
| `Insufficient privileges` | Wrong role | Switch to `ACCOUNTADMIN` |
| `External access not allowed` | EAI not enabled | Run `SHOW EXTERNAL ACCESS INTEGRATIONS` — ensure `bigdata_mcp_eai` exists and `ENABLED = TRUE` |
| `Operation failed since agent spec is invalid: unrecognized field instructions` | JSON format error | Use JSON (not YAML) with `instructions` as a top-level key. See `04a_cortex_agent.sql`. |
| `MCP SERVER not supported` | Feature not available | Use Approach A (Cortex Agent) instead |
| `Agent not visible in Intelligence` | Missing grants | Run `GRANT USAGE ON AGENT ... TO ROLE PUBLIC;` |
| Procedures return `null` | SSE stream not parsed | Verify API key is correct; check event logs |
| Slow responses (>30s) | Normal for external API calls | Latency comes from BigData API, not Snowflake |

### Verify Objects Exist

```sql
SHOW PROCEDURES IN SCHEMA BIGDATA_DB.MCP_TOOLS;
SHOW AGENTS IN SCHEMA BIGDATA_DB.MCP_TOOLS;
SHOW MCP SERVERS IN SCHEMA BIGDATA_DB.MCP_TOOLS;
SHOW EXTERNAL ACCESS INTEGRATIONS;
SHOW SECRETS IN SCHEMA BIGDATA_DB.MCP_TOOLS;
SHOW NETWORK RULES IN SCHEMA BIGDATA_DB.MCP_TOOLS;
```

### Check Query History

```sql
SELECT *
FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY())
WHERE QUERY_TEXT ILIKE '%bigdata%'
ORDER BY START_TIME DESC
LIMIT 20;
```

---

## Reference Links

### BigData.com
- [BigData API Documentation](https://docs.bigdata.com/)
- [MCP Reference — Introduction](https://docs.bigdata.com/mcp-reference/introduction)
- [MCP Tool: bigdata_search](https://docs.bigdata.com/mcp-reference/tools/bigdata-search)
- [MCP Tool: find_companies](https://docs.bigdata.com/mcp-reference/tools/find-companies)
- [MCP Tool: bigdata_company_tearsheet](https://docs.bigdata.com/mcp-reference/tools/bigdata-company-tearsheet)
- [Developer Platform (API Keys)](https://platform.bigdata.com/api-keys)

### Snowflake
- [Snowflake Intelligence Overview](https://docs.snowflake.com/en/user-guide/snowflake-cortex/snowflake-intelligence)
- [Cortex Agents](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents)
- [CREATE MCP SERVER](https://docs.snowflake.com/en/sql-reference/sql/create-mcp-server)
- [Snowflake-Managed MCP Server](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-mcp)
- [External Access Integrations](https://docs.snowflake.com/en/developer-guide/external-network-access/creating-using-external-network-access)
- [Integrating Third-Party APIs into Cortex Agents](https://medium.com/snowflake/integrating-third-party-apis-into-snowflake-cortex-agents-2802fe50ae9d)
- [Snowflake Trial Signup](https://signup.snowflake.com/)
