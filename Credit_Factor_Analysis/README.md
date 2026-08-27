# Credit News Factors — Screen, Drill, Discover Narrative

A worked example for credit analysts: rank a universe by credit-news sentiment, drill into
the worst name's catalysts, then have an LLM turn those catalysts into a grounded narrative —
using three [Bigdata.com](https://bigdata.com) MCP tools and `gpt-5.6-terra`.

| Step | Tool | What it does |
|---|---|---|
| 1. Rank the universe | `bigdata_screen_credit_factor` | Negative screen across a portfolio/sector list, worst names first. |
| 2. Drill into a name | `bigdata_get_credit_factor` | One name's most extreme catalyst rows, by event type. |
| 3. Build the narrative | Your LLM + `bigdata_search` | News evidence + catalyst rows → why the score moved, what to watch next. |

Open [`notebook/credit_narrative_workflow.ipynb`](notebook/credit_narrative_workflow.ipynb) to
see it run end-to-end against the "Magnificent Seven" (Apple, Microsoft, Alphabet, Amazon,
Meta, Nvidia, Tesla) — swap that list for your own coverage universe to reuse it as-is.

## Project layout

```
notebook/credit_narrative_workflow.ipynb   the walkthrough — run this
src/bigdata_mcp_client.py                  async client for the Bigdata.com Remote MCP server
src/narrative.py                           prompt construction + LLM call for Step 3
requirements.txt
```

## Setup

1. Get a Bigdata.com API key from the [Developer Platform](https://platform.bigdata.com/api-keys)
   and an OpenAI API key with access to `gpt-5.6-terra`.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Export both keys:

   ```bash
   export BIGDATA_API_KEY=...
   export OPENAI_API_KEY=...
   ```
4. Launch Jupyter and run `notebook/credit_narrative_workflow.ipynb` top to bottom.

## How it connects

The notebook talks to the Bigdata.com Remote MCP server (`https://mcp.bigdata.com/`) as a
standard MCP client over Streamable HTTP, authenticated with `x-api-key`. No proprietary
SDK required — any MCP client library works the same way.
