# Best Practices for Bigdata.com Search via Agent

1. # Use smart search mode

This mode is created for agents or humans that are not specialized in our sophisticated search tools. The agent only needs to formulate a query.text and the smart mode interprets the natural language request, extracting entities, temporal expressions and understanding intent in order to convert that textual query into a more structured search query, with entity filters, content and temporal constraints. This increases precision and reduces cost on trial and error by an agent that has not been trained on search generation.

Fast mode gives you full control over all filters for precise, deterministic, low-latency results. Smart mode handles query understanding for you, automatically inferring intent, applying filters, and running sub-queries for broader coverage. Ideal for natural-language questions without pre-processing.

| Request Parameter | Fast Mode | Smart Mode |
| :---- | :---- | :---- |
| Text | ✅ | ✅ |
| Temporal filter | ✅ Manual or Partially Automated | ✅ Manual or Fully Automated |
| Source filter | ✅ | ✅ Manual or Fully Automated Reporting |
| date filter | ✅ | Fully Automated |
| Document type filter | ✅ | Fully Automated |
| Entity filter | ✅ Manual or Partially Automated | Fully Automated |
| Reporting entity filter | ✅ | Fully Automated |
| Keyword filter | ✅ | Fully Automated |
| Sentiment filter | ✅ | Fully Automated (for directionality) |
| Ranking parameters | ✅ | Fully Automated |
| Content Diversification | ✅ | Fully Automated |

# 2\. Suggested Filter Criteria if using fast mode

A router agent or human that wants to take better advantage of our search tools customising the queries, not relying on the smart mode to do this automatically, can also take advantage of structured search requests using fast mode. 

Fast mode has a basic enrichment (when enabled) that also extracts entities and time periods from the query.text and applies them as filters, but it has no query understanding and can not restrict the query to specific content types. Therefore, it is recommended to use query filters when using smart mode: especially for company-specific or document-specific questions, if the intent is clear, precision can increase with structured filters.

These are some of the filters you can customize:

**Reminder: smart mode will interpret these for you**   
**TIP: you can use smart mode first to check the audit on the hints that are being extracted and later customize the query in fast mode in a similar way, if required.**

## 2.1 Entity Filters

Resolve any company name or ticker to a Bigdata.com entity ID before building the search.

There are two entity filters, and they are not interchangeable:

* filters.entity: documents that mention the company, such as third-party news, research notes, peer earnings calls, or other documents discussing the company.  
* filters.reporting\_entities: documents published or filed by the company itself, such as earnings calls, transcripts, SEC filings, annual reports, and presentations.

A simple routing heuristic:

* If the user asks for “Apple’s 10-K,” “Microsoft’s last earnings call,” or another company-authored document, use reporting\_entities.  
* If the user asks for “news about Apple,” “who is discussing Microsoft,” or third-party commentary, use entity.

The entity filter also supports all\_of, any\_of, and none\_of. The search\_in parameter accepts ALL, HEADLINE, or BODY.

## 2.2 Time Bounding

Use filters.timestamp.start and filters.timestamp.end with ISO-8601 timestamps.

We recommend defaulting to a 90-day rolling window if no timeframe is stated, and using a shorter 30-day window for “recent” or “latest” phrasing unless the user’s question implies otherwise.

## 2.3 Category Filter

Use category to narrow by source-type bucket. It accepts mode (INCLUDE or EXCLUDE) and a values array.

Supported values include:

news, news\_premium, news\_public, filings, transcripts, research, research\_investment\_research, research\_academic\_journals, regulatory, expert\_interviews, expert\_networks, podcasts, newsletters, and my\_files (own documents uploaded via Bigdata Connector).

## 2.4 Document Type Filter

Use document\_type when the user names a specific kind of document.

Supported document types include:

* NEWS  
* FILING  
* TRANSCRIPT  
* TRANSCRIPT-PRESENTATION  
* INVESTMENT-RESEARCH

Where appropriate, use structured document-type values with subtypes for finer control.

Filing subtypes include:

* SEC\_10\_K  
* SEC\_10\_Q  
* SEC\_8\_K  
* SEC\_20\_F  
* SEC\_S\_1  
* SEC\_S\_3  
* SEC\_6\_K  
* SEC\_DEF\_14A

Transcript subtypes include:

* EARNINGS\_CALL  
* GUIDANCE\_CALL  
* SALES\_REVENUE\_CALL  
* ANALYST\_INVESTOR\_SHAREHOLDER\_MEETING  
* CONFERENCE\_CALL  
* GENERAL\_PRESENTATION  
* SPECIAL\_SITUATION\_MA  
* SHAREHOLDERS\_MEETING  
* MANAGEMENT\_PLAN\_ANNOUNCEMENT  
* INVESTOR\_CONFERENCE\_CALL

Investment-research subtypes include:

* COMPANY\_REPORT  
* INDUSTRY\_REPORT  
* THEMATIC\_ANALYSIS  
* RATING\_REPORT  
* RESEARCH\_NOTE  
* ECONOMIC\_REPORT  
* FIXED\_INCOME\_REPORT  
* FUND\_REPORT  
* MARKET\_UPDATE  
* PORTFOLIO\_STRATEGY  
* PORTFOLIO\_SUMMARY  
* INDEX\_REPORT  
* COVERAGE\_ANALYSIS  
* FX\_AND\_DERIVATIVES\_REPORT  
* GENERIC\_REPORT

Use reporting\_periods to scope to a specific fiscal period. For example, for a query like “Apple’s Q4 2025 earnings call,” combine reporting\_entities, document\_type, the EARNINGS\_CALL subtype, and the relevant reporting period.

## 2.5 Keyword, Topic, Tag, Sentiment, and Source

Use these filters when they directly map to user intent:

* keyword: supports any\_of, all\_of, and none\_of, plus search\_in (ALL, HEADLINE, or BODY). Each term must be at least 3 characters. Keyword filters are strict string matches, meant to substitute entity filters when the entities are not supported (e.g. new drugs, new tech terms, etc)  
* topic.any\_of: use Knowledge Graph topic IDs when thematic narrowing is appropriate. IMPORTANT: This is not a semantic topic search. It is a strict topic matching based on templated event patterns, meant for high precision but lor recall. For broad topics, or to increase recall, use query.text.  
* tag.any\_of: use tag IDs when tag-based narrowing is appropriate (for private content only, on your own specified tags)  
* sentiment: supports sentiment labels (positive, negative, neutral) or numeric ranges between \-1 and \+1. This narrows down search results to chunks within that sentiment constraint.  
* source: accepts mode plus source IDs, useful for whitelisting curated sources, or blacklisting unwanted ones.

## 2.6 What Goes in query.text vs. Filters

Use query.text for semantic concepts that are not cleanly captured by structured filters, such as investment narratives, qualitative drivers, or thematic descriptions.

Query.text should be a natural language question or sentence, not keyword-only. It will be used to match results and to rerank them based on semantic proximity.

Query.text is NOT a prompt requesting a summary or a report, it is a search query that is used to retrieve relevant-related content. It is recommended to add details such as entities or specific terms in the query, or context that can make the semantics more clear.

Examples:

* “liquid cooling adoption in hyperscale data centers”, or even “How are hyperscale data centers adopting liquid cooling technology?”  
* “executive leadership transition in Revolut”, or “What are the recent executive leadership changes at Revolut?”  
* “FDA approval for novel oncology therapeutics”, or “What new cancer treatments have recently been approved by the FDA?”,  “Which novel oncology therapeutics recently got FDA approval?

Avoid putting content filters or document types in query.text when using fast mode, those can be represented with structured filters. (Smart mode would allow you to narrow down content with intent, such as requesting “news” or “earnings calls” in the query text itself).

query.text supports up to 1400 characters and does not need to carry every constraint if filters already narrow the request. There is no limit for smart mode.

## 2.7 auto\_enrich\_filters

If the router has already resolved entities and built precise filters, consider setting query.auto\_enrich\_filters to false so the API does not infer additional filters from query.text (entities and time periods)

If the agent is intentionally passing a raw or lightly processed user question, leaving auto-enrichment available may be useful, although we still recommend smart mode for better query enrichment and understanding.

## 2.8 max\_chunks

max\_chunks controls the number of chunks returned, with a maximum of 1000\.

Recommended ranges:

* 10–30 for direct question answering  
* 50–200 for retrieval that feeds a longer synthesis step  
* Higher values only when the downstream workflow can handle the added context volume

# 3\. Suggested Prompt for a Router Agent

*** Recommendation is to use smart mode for an agent. See the sample in [agent_to_search.ipynb](./agent_to_search.ipynb). ***

If not, you can experiment with prompts; in fast mode, you might have to play with it and tune it based on your need.

In fast mode, the router agent should perform entity resolution, filter construction, and date interpretation before calling search.

**Recommended system prompt:**

*You are a financial data retrieval routing agent. Translate the user's natural-language question into a structured request for the Bigdata.com /v1/search endpoint in fast mode.*

*Step 1 \-  Entity resolution: Extract company names, tickers, and unique identifiers. Resolve them to Bigdata.com entity IDs before building the search.*

*Step 2 \-  Choose the right entity filter: Use filters.reporting\_entities when the user wants documents authored or filed by the company, such as earnings calls, transcripts, 10-Ks, 10-Qs, 8-Ks, annual reports, or investor presentations. Use filters.entity.any\_of when the user wants documents that mention or discuss the company. Avoid using both for the same company in the same request unless there is a specific reason.*

*Step 3 \-  Decide what belongs in query.text: Put semantic concepts, investment themes, qualitative drivers, and narratives in query.text. Do not put tickers, document types, or generic financial nouns in query.text when they can be expressed with structured filters.*

*Step 4 \-  Time-bound the request: Compute filters.timestamp.start and filters.timestamp.end in ISO-8601 format from the user's phrasing. Use 30 days for “recent” or “latest,” quarter-to-date for “this quarter,” and 90 days when no timeframe is stated.*

*Step 5 \-  Apply structured filters: Use category for source-type bucketing, document\_type and subtypes when the user names a specific document, reporting\_periods when the user names a fiscal period, keyword for explicit must-have or exclude terms, topic.any\_of for topic IDs, sentiment only when sentiment is part of the user intent, and source for curated source lists.*

*Step 6 \-  Set ranking and budget: Set max\_chunks based on the downstream context budget. Use ranking\_params when the workflow needs stronger reranking, freshness weighting, source weighting, or content diversification.*

*Return only a valid JSON object matching the Bigdata.com Search API schema. Do not include commentary.*

# 4\. Ranking Parameters

ranking\_params can be used to tune retrieval quality for the downstream workflow. The available controls include:

* reranker.enabled: toggles cross-encoder semantic reranking.  
* reranker.threshold: minimum reranking\_relevance threshold for returned chunks.  
* freshness\_boost: increases preference for newer documents (default=1 is already giving preference to fresh results)  
* source\_boost: increases preference for higher-ranked or preferred sources (default=1 is already giving preference to authoritative sources)  
* content\_diversification.enabled: helps reduce redundant chunks and single-source overrepresentation (enabled by default)

Recommended tuning by query type:

* High-precision topic lookup: use a higher reranker threshold.  
* High-recall topic lookup/broad topic discovery: use a lower reranker threshold (or disable).  
* Very specific retrieval queries: may disable diversification

# 5\. Balancing Recall and Precision

Use an iterative fallback strategy rather than one hardcoded parameter set.

## Pass 1 \-  Precision First

Use smart mode with a query.text that describes the request, including entity names, periods and hints for content filtering. Note that smart mode also performs query expansion to increase recall, while maintaining precise filtering.

Alternatively, use fast mode, starting with some of those:

* Resolved entity IDs using either entity.any\_of or reporting\_entities  
* A tight timestamp window  
* Specific document\_type and subtype filters when applicable  
* keyword.all\_of for required terms, if the user provided explicit must-have entities or string-matched concepts not resolved with entity id’s  
* Specific natural language query.text, with moderate to higher reranker threshold

If the result set is sufficient for the downstream task, stop here.

## Pass 2 \-  Recall Expansion

If the first pass returns too little content, relax constraints in this order:

1. Remove topic or tag filters if they appear too restrictive.  
2. Widen the timestamp window (if that makes sense)  
3. Move strict keyword.all\_of constraints to keyword.any\_of, or remove them.  
4. Broaden document\_type if appropriate.  
5. Lower the reranker threshold or remove the override.  
6. Consider reformulating the query.text with semantic variations that may introduce different angles or concepts to the topic.  
7. Switch from reporting\_entities to entity.any\_of only if the task can accept third-party mentions instead of company-authored documents.

Keep the entity filter unless the user’s question is intentionally market-wide or sector-wide.

# 6\. Re-Ranking Before Synthesis

If the agent combines results from multiple Bigdata.com searches before passing them to a synthesis LLM, rank the candidate chunks using fields returned by the search response.

A practical default is:

1. Prioritize higher chunk relevance.  
2. Use document timestamp as a recency tiebreaker.  
3. Apply source preference only when source ranking or curated source lists are available.  
4. Use sentiment-based ranking only when the user’s task is explicitly sentiment-aware.  
5. Use a LLM for selecting the most relevant results