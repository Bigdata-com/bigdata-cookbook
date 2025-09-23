#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["mcp[cli]==1.11.0", "bigdata-research-tools==0.20.1", "nest-asyncio==1.6.0", "python-dotenv==1.1.1"]
# ///

import os

from datetime import datetime
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from bigdata_research_tools.watchlists import (
    create_watchlist as create_watchlist_internal,
    fuzzy_find_watchlist_by_name,
)
from bigdata_research_tools.workflows.thematic_screener import (
    ThematicScreener,
    DocumentType,
)
from bigdata_client import Bigdata
import nest_asyncio

# Select your LLM model here
LLM_MODEL = "openai::gpt-4o-mini"
# LLM_MODEL = "azure::gpt-4o-mini"
# LLM_MODEL = "bedrock::anthropic.claude-3-sonnet-20240229-v1:0"

nest_asyncio.apply()

# Create an MCP server
mcp = FastMCP("Demo", stateless_http=True, json_response=True, host="0.0.0.0")


load_dotenv('.env')
# Initialize Bigdata client
BIGDATA = Bigdata()


# Add an addition tool
@mcp.tool()
def create_watchlist(watchlist_name: str, companies: list[str]):
    """Create a watchlist for the given companies."""
    return create_watchlist_internal(watchlist_name, companies, BIGDATA)


@mcp.tool()
def screen_companies(
    watchlist_name: str, main_theme: str, fiscal_year: int, focus: str = ""
):
    """Screen companies in a watchlist for a given theme and fiscal year. This will return
    a JSON string with the results."""
    # Find the watchlist by name
    watchlist = fuzzy_find_watchlist_by_name(watchlist_name, BIGDATA)
    if not watchlist:
        return {"error": f"Watchlist '{watchlist_name}' not found."}

    # Extract companies from the watchlist
    companies = BIGDATA.knowledge_graph.get_entities(watchlist.items)

    # Configure and run the thematic screener
    them = ThematicScreener(
        llm_model=LLM_MODEL,
        main_theme=main_theme,
        focus=focus,
        companies=companies,
        start_date=datetime(fiscal_year - 1, 1, 1),
        end_date=datetime(fiscal_year + 1, 12, 31),
        document_type=DocumentType.TRANSCRIPTS,
        fiscal_year=fiscal_year,
    )
    result = them.screen_companies(
        document_limit=20,
        batch_size=10,
        frequency="3M",
    )

    # Extract and return the relevant data as JSON
    return str(result["df_company"].to_json(orient="records"))


def test_llm_model_configured():
    """Test that the LLM model is configured correctly."""
    from bigdata_research_tools.llm.base import LLMEngine

    try:
        test_answer = LLMEngine(LLM_MODEL).get_response(
            [{"role": "user", "content": "Hello, world!"}]
        )
    except Exception as e:
        print(
            "[ERROR] LLM model is not configured correctly. Read more here: https://github.com/Bigdata-com/bigdata-research-tools?tab=readme-ov-file#llm-integration"
        )
        exit(1)
    else:
        assert isinstance(test_answer, str), (
            "LLM model is not configured correctly. Read more here: https://github.com/Bigdata-com/bigdata-research-tools?tab=readme-ov-file#llm-integration"
        )


if __name__ == "__main__":
    test_llm_model_configured()
    assert "BIGDATA_API_KEY" in os.environ, (
        "Please set the BIGDATA_API_KEY environment variable."
    )
    mcp.run(transport="streamable-http")
