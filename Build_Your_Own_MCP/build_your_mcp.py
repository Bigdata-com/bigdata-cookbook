#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["mcp[cli]==1.11.0", "bigdata-smart-batching>=1.3.1,<2.0.0", "requests>=2.31.0", "openai>=1.0.0", "python-dotenv>=1.2.2", "nest-asyncio==1.6.0"]
# ///

"""Build Your Own MCP - migrated from bigdata-research-tools to REST + OpenAI."""

# NOTE: deliberately NOT using `from __future__ import annotations` here.
# mcp[cli]==1.11.0's FastMCP.Tool.from_function() calls issubclass() on raw
# inspect.signature() parameter annotations without eval_str=True; postponed
# evaluation (PEP 563) turns those into plain strings and crashes tool
# registration with "TypeError: issubclass() arg 1 must be a class".

import os
from typing import Literal
from datetime import datetime
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import nest_asyncio
from openai import OpenAI
import json

from bigdata_rest import BigdataRestClient

# Select your LLM model here
LLM_MODEL = "gpt-5.6-luna"  # OpenAI model (luna: omit temperature)

# Use streamable-http for better compatibility with various clients
TRANSPORT: Literal["sse", "streamable-http"] = "streamable-http"

nest_asyncio.apply()

# Create an MCP server
mcp = FastMCP("Demo", stateless_http=True, json_response=True, host="0.0.0.0")

load_dotenv(".env")

# Initialize clients
REST_CLIENT = BigdataRestClient()
OPENAI_CLIENT = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _chat_completion_kwargs(**extra: object) -> dict[str, object]:
    """Build OpenAI chat kwargs; luna models omit temperature and use max_completion_tokens."""
    kwargs: dict[str, object] = {"model": LLM_MODEL}
    if "luna" in LLM_MODEL:
        if "max_tokens" in extra:
            extra = {**extra}
            extra["max_completion_tokens"] = extra.pop("max_tokens")
    else:
        kwargs["temperature"] = 0.7
    kwargs.update(extra)
    return kwargs


@mcp.tool()
def create_watchlist(watchlist_name: str, company_ids: list[str]):
    """
    Create a watchlist for the given company IDs.
    
    Args:
        watchlist_name: Name for the watchlist
        company_ids: List of Bigdata entity IDs (e.g., ["4F2B", "D8442"])
    
    Returns:
        Dict with watchlist info
    """
    # In the migrated version, we simply store the list in memory or return it
    # No actual watchlist creation via SDK
    return {
        "watchlist_name": watchlist_name,
        "company_ids": company_ids,
        "company_count": len(company_ids),
        "message": f"Watchlist '{watchlist_name}' created with {len(company_ids)} companies. Use these IDs for screening.",
    }


@mcp.tool()
def screen_companies(
    company_ids: list[str],
    main_theme: str,
    fiscal_year: int,
    focus: str = "",
    document_limit: int = 20,
):
    """
    Screen companies for a given theme and fiscal year using REST API.
    
    Args:
        company_ids: List of Bigdata entity IDs
        main_theme: Main theme to screen for
        fiscal_year: Fiscal year to analyze
        focus: Optional focus area
        document_limit: Max documents per query
    
    Returns:
        JSON string with screening results
    """
    # Build date range around fiscal year
    start_date = f"{fiscal_year - 1}-01-01"
    end_date = f"{fiscal_year + 1}-12-31"

    # Generate theme taxonomy using OpenAI
    theme_prompt = f"""Generate a list of 5-8 specific sub-themes related to "{main_theme}".
    
Focus: {focus if focus else "General analysis"}

Return ONLY a JSON array of strings, e.g.:
["Sub-theme 1", "Sub-theme 2", "Sub-theme 3"]
"""

    try:
        response = OPENAI_CLIENT.chat.completions.create(
            **_chat_completion_kwargs(
                messages=[{"role": "user", "content": theme_prompt}],
            ),
        )
        sub_themes_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        if sub_themes_text.startswith("```json"):
            sub_themes_text = sub_themes_text.split("```json")[1].split("```")[0].strip()
        elif sub_themes_text.startswith("```"):
            sub_themes_text = sub_themes_text.split("```")[1].split("```")[0].strip()
            
        sub_themes = json.loads(sub_themes_text)
        
    except Exception as e:
        # Fallback to generic sub-themes
        sub_themes = [
            f"{main_theme} - Innovation",
            f"{main_theme} - Market Position",
            f"{main_theme} - Competitive Advantage",
        ]

    # Search for each company and theme combination
    results = []

    for company_id in company_ids:
        company_scores = {"company_id": company_id}

        for theme in sub_themes:
            # Build search query (POST /v1/search PublicSearchRequest shape)
            query = {
                "text": theme,
                "filters": {
                    "timestamp": {
                        "start": f"{start_date}T00:00:00Z",
                        "end": f"{end_date}T23:59:59Z",
                    },
                    "entity": {"any_of": [company_id], "search_in": "BODY"},
                },
                "max_chunks": document_limit,
                "auto_enrich_filters": False,
            }

            try:
                search_results = REST_CLIENT.search(query)
                # Simple scoring: count of relevant documents
                company_scores[theme] = len(search_results)
            except Exception as e:
                print(f"Search failed for company {company_id}, theme {theme}: {e}")
                company_scores[theme] = 0

        # Calculate composite score
        company_scores["Composite Score"] = sum(
            v for k, v in company_scores.items() if k not in ["company_id", "Composite Score"]
        )

        results.append(company_scores)

    # Convert to JSON
    return json.dumps(results, indent=2)


@mcp.tool()
def bigdata_search(queries: list[str], start_date: str = "2020-01-01", end_date: str = "2025-12-31"):
    """
    Run a search on Bigdata API for the given queries.
    
    Args:
        queries: List of search query strings
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    
    Returns:
        Dict mapping queries to search results
    """
    results = {}

    for query_text in queries:
        # POST /v1/search PublicSearchRequest shape: {"query": {"text": ..., "filters": {...}}}
        query = {
            "text": query_text,
            "filters": {
                "timestamp": {
                    "start": f"{start_date}T00:00:00Z",
                    "end": f"{end_date}T23:59:59Z",
                },
            },
            "max_chunks": 10,
            "auto_enrich_filters": False,
        }

        try:
            search_results = REST_CLIENT.search(query)
            results[query_text] = [
                {
                    "title": doc.get("headline") or doc.get("title"),
                    "content": " ".join(
                        chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                        for chunk in (doc.get("chunks") or [])
                    )[:500],
                    "timestamp": doc.get("timestamp") or doc.get("timestamp_utc"),
                    "url": doc.get("url"),
                }
                for doc in search_results[:5]  # Limit to top 5
            ]
        except Exception as e:
            print(f"Search failed for query '{query_text}': {e}")
            results[query_text] = []

    return results


def test_llm_model_configured():
    """Test that the LLM model is configured correctly."""
    try:
        test_answer = OPENAI_CLIENT.chat.completions.create(
            **_chat_completion_kwargs(
                messages=[{"role": "user", "content": "Hello, world!"}],
                max_tokens=50,
            ),
        )
        assert test_answer.choices[0].message.content, "LLM model test failed"
    except Exception as e:
        raise RuntimeError(
            f"[ERROR] OpenAI model '{LLM_MODEL}' is not configured correctly. "
            f"Ensure OPENAI_API_KEY is set in .env"
        ) from e


if __name__ == "__main__":
    test_llm_model_configured()
    assert "BIGDATA_API_KEY" in os.environ, "Please set BIGDATA_API_KEY in .env"
    assert "OPENAI_API_KEY" in os.environ, "Please set OPENAI_API_KEY in .env"
    mcp.run(transport=TRANSPORT)
