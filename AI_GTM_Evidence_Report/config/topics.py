"""
Topic templates and filter configuration for the AI-in-GTM evidence search.

Each topic is a Google-style keyword query combining explicit AI terms with one
Go-To-Market (GTM) activity cluster. Topics are formatted with a {company}
placeholder. topic_name = the GTM dimension (used for grouping/diagnostics).

OFFICIAL-SOURCES-ONLY strategy (two search passes per company, merged):
  Pass A -> category filter ["filings", "transcripts"]
            (regulatory filings, annual/interim reports, earnings-call transcripts)
  Pass B -> source filter ["DFF004"] (PubT Corporate Communications package:
            company-issued press releases / official publications)
No general news media, no sell-side research: the "news" and "research"
categories are deliberately NOT searched.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# AI x GTM topic set
# ---------------------------------------------------------------------------
# GTM scope per requirement: sales execution, lead gen, marketing/campaigns,
# customer acquisition/expansion, pricing/packaging, channel/partner motions,
# product launch/commercialization, segmentation/targeting/personalization,
# sales enablement, CRM/cadence optimization, pipeline/forecasting, quote-to-cash.
AI_GTM_TOPICS: List[Dict[str, str]] = [
    {
        "topic_name": "Sales Execution & Enablement",
        "topic_text": "{company} artificial intelligence AI sales force sales execution sales enablement sales productivity CRM customer relationship management",
    },
    {
        "topic_name": "Marketing & Personalization",
        "topic_text": "{company} AI generative AI marketing campaigns digital marketing personalization targeting customer segmentation",
    },
    {
        "topic_name": "Lead Gen & Customer Acquisition",
        "topic_text": "{company} AI machine learning lead generation customer acquisition new customers cross-selling upselling customer expansion win rate",
    },
    {
        "topic_name": "Pricing & Quote-to-Cash",
        "topic_text": "{company} AI machine learning pricing optimization dynamic pricing quotation configure price quote quote-to-cash order processing",
    },
    {
        "topic_name": "Channel, Partners & E-commerce",
        "topic_text": "{company} AI artificial intelligence distributors channel partners digital commerce e-commerce online sales platform",
    },
    {
        "topic_name": "Product Launch & Commercialization",
        "topic_text": "{company} launch AI-powered products artificial intelligence go-to-market commercialization new offering brought to market for customers",
    },
    {
        "topic_name": "Pipeline & Demand Forecasting",
        "topic_text": "{company} AI machine learning sales pipeline demand forecasting demand planning customer insights market intelligence",
    },
    {
        "topic_name": "GTM Strategy (broad)",
        "topic_text": "{company} artificial intelligence generative AI go-to-market strategy commercial excellence customer-facing digital sales transformation",
    },
]

# Backwards-compatibility alias: services/topic_search_service.py imports
# STANDARD_TOPICS from config.topics as its default topic set.
STANDARD_TOPICS = AI_GTM_TOPICS

# ---------------------------------------------------------------------------
# Official-sources-only filter criteria
# ---------------------------------------------------------------------------
# Pass A: Bigdata document categories that are primary-source by construction.
OFFICIAL_DOC_CATEGORIES = ["filings", "transcripts"]

# Pass B: PubT Corporate Communications package = company-issued press
# releases and official publications (same source id used by the
# Index_MA_Activity_Report corporate-actions tracker).
PRESS_RELEASE_SOURCE_IDS = ["DFF004"]

# Documents whose title matches any of these patterns are dropped BEFORE
# extraction (requirement: exclude investor / capital markets day decks).
EXCLUDED_TITLE_PATTERNS = [
    "capital markets day",
    "capital market day",
    "investor day",
    "investor conference",
]

# ---------------------------------------------------------------------------
# Explicit AI terms (per requirement) - single source of truth.
# ---------------------------------------------------------------------------
# Display list injected into the extraction prompt.
AI_TERMS = [
    "AI",
    "artificial intelligence",
    "GenAI",
    "generative AI",
    "LLM",
    "large language model",
    "machine learning",
]

# Regex patterns used for the deterministic post-extraction validation.
# Tuples of (pattern, case_sensitive). "AI"/"LLM" are case-sensitive so that
# words like "air" or "will" can never match; the phrases are case-insensitive.
AI_TERM_REGEXES = [
    (r"\bAI\b", True),
    (r"\bGenAI\b", False),
    (r"\bLLMs?\b", True),
    (r"artificial intelligence", False),
    (r"generative AI", False),
    (r"large language model", False),
    (r"machine learning", False),
]

# ---------------------------------------------------------------------------
# Fixed document-type vocabulary (injected into the extraction prompt and
# used to normalize the Document Type column).
# ---------------------------------------------------------------------------
DOCUMENT_TYPES = [
    "Regulatory Filing",
    "Annual Report",
    "Half-year / Interim Report",
    "ESG / Sustainability Report",
    "Press Release",
    "Earnings Call Transcript",
    "Other Official Publication",
]
