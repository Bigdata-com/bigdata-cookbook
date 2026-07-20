# 04 — (Optional) AI/BI Genie Space for Structured Data

The Unity Catalog SQL functions from notebook `01` (`get_top_holdings`,
`get_portfolio_positions`, `get_ticker_exposure`) already give the agent a governed,
reliable structured-data tool — that path is fully runnable with no UI steps.

This notebook is the **optional upgrade**: exposing the same portfolio tables through an
AI/BI **Genie** space so the agent can answer *open-ended* structured questions
("compare AUM across account types", "which portfolio has the most aggressive risk
profile?") with text-to-SQL, instead of only the three pre-built functions.

Genie spaces are created in the Databricks UI, so this step is documented rather than
scripted.

---

## Create the Genie space

1. In the sidebar, open **Genie Agents** (under **SQL**) → **New**.
2. **Connect data** — select these tables from `bigdata_demo.financial_intelligence`:
   - `accounts`
   - `portfolios`
   - `holdings`
   - `transactions`
3. Pick a **serverless SQL warehouse** for the space to run queries on.
4. Name the space **`Portfolio Intelligence`**.

## Add instructions (improves text-to-SQL accuracy)

Paste this into the space's **Instructions**:

> This space answers questions about an institutional investment book. `accounts` are
> top-level funds, each with one or more `portfolios`; `holdings` are current positions
> per portfolio; `transactions` is the trade blotter. Join on `account_id` and
> `portfolio_id`. `market_value`, `unrealized_pnl`, and `aum` are in USD. "Exposure" or
> "how much do we hold" means `SUM(market_value)` for a ticker across all portfolios.
> "P&L" means `unrealized_pnl`. Portfolio IDs are PF001, PF002, PF003.

## Add sample questions

- What are the top 5 holdings by market value across all portfolios?
- Show total AUM by account type.
- Which portfolio has the most aggressive risk profile?
- What is our total NVDA exposure across all portfolios?
- List all BUY transactions in February 2025 sorted by amount.

## Test, then copy the Space ID

Ask a couple of the sample questions to confirm the space returns correct results.
Then copy the **Space ID** from the space URL:

```
https://<workspace>/genie/rooms/<SPACE_ID>
                                 ^^^^^^^^^^
```

## Wire it into the agent

In **`agent.py`** and **`05_build_deploy_agent`**, set:

```python
USE_GENIE = True
GENIE_SPACE_ID = "<SPACE_ID>"
```

The agent adds Genie as a structured-data tool alongside (or in place of) the UC SQL
functions. When `USE_GENIE = False`, the agent uses the UC functions only — no Genie
space required.

## Grant access for deployment

When the agent is deployed to Model Serving, its service principal needs `CAN RUN` on
the Genie space (Genie space → **Share** → add the principal). Notebook `05` also lists
the space as a `DatabricksGenieSpace` resource at logging time so Databricks provisions
the credential automatically.

---

**Reference:** [AI/BI Genie](https://docs.databricks.com/aws/en/genie/) ·
[GenieAgent in databricks-langchain](https://api-docs.databricks.com/python/databricks-ai-bridge/latest/databricks_langchain.html)
