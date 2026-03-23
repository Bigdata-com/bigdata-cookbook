# Bigdata.com MCP Grounding & Attribution

_Adapted from the global `bigdata-mcp-grounding` skill for the Iran Geopolitical Risk Dashboard._

## Purpose

Ensure that every response using [Bigdata.com](https://bigdata.com) MCP data is **fully grounded** in the actual tool output, **properly attributed** with inline citations, and **transparent** about gaps where data was insufficient. When generating `src/dashboard.jsx`, every number in `GROUNDED_DATA` must trace to a specific MCP result.

## When This Skill Applies

This skill activates whenever the agent calls ANY of the following Bigdata.com MCP tools:

- `find_companies`
- `bigdata_company_tearsheet`
- `bigdata_country_tearsheet`
- `bigdata_events_calendar`
- `bigdata_search`
- `bigdata_market_tearsheet`

---

## Core Rules

### Rule 1 — Ground Every Claim

Every factual statement and every number in `GROUNDED_DATA` MUST be directly traceable to data returned by a Bigdata.com MCP tool in the current session. Do not:

- Invent figures, dates, percentages, or metrics not present in the MCP output
- Extrapolate trends beyond what the data shows
- Fill in missing data points from training knowledge without clearly flagging them as stale (see Rule 4)

### Rule 2 — Inline Citations

When writing analysis or commentary, cite sources inline immediately after the claim:

**For `bigdata_search` results:**
> [Source Name - MMM DD, YYYY](url)

Example:
> WTI rose 3.5% on Hormuz closure fears. [Reuters - Mar 5, 2026](https://example.com)

**For `bigdata_country_tearsheet` data:**
> *(Bigdata.com Country Tearsheet — [Country])*

**For `bigdata_events_calendar` data:**
> *(Bigdata.com Events Calendar)*

**For `bigdata_market_tearsheet` data:**
> *(Bigdata.com Market Tearsheet)*

### Rule 3 — `GROUNDED_DATA` Source Fields (Dashboard-Specific)

When populating `GROUNDED_DATA` in the JSX artifact, every data section must include a `source` field. Format:

```js
source: "Source Name, MMM DD YYYY; Source Name 2, MMM DD YYYY"
```

Examples from the existing component:
```js
source: "Goldman Sachs via Benzinga, Mar 2 2026; MT Newswires, Mar 5 2026"
source: "Reuters Mar 4 2026; Hapag-Lloyd via MT Newswires Mar 1 2026"
```

The `SourceTag` component parses this string by splitting on `;` and renders each source as a tag inside a collapsible footer. It also shows a hint to "hover underlined values above for attribution." The `source` field must never be empty or `undefined`. If multiple sources support a data block, join them with `;`.

### Rule 4 — Stale Data Handling

When `bigdata_search` returns no fresh results for a domain:

1. **Keep the previous value** from the existing `src/dashboard.jsx`
2. **Append a staleness flag** to the source field:
   ```js
   source: "Goldman Sachs via Benzinga, Mar 2 2026 [STALE — no fresh data Mar 6 2026]"
   ```
3. **Do NOT invent a replacement value** — stale is better than fabricated

For the dashboard header "last updated" timestamp, always use the actual session date/time.

### Rule 6 — Populate `src` Objects for Inline Hover Attribution

When populating `GROUNDED_DATA`, add `src` objects on key data points that will be wrapped by the `GroundedSpan` component for inline hover attribution. Each `src` object has:

```js
src: { source: "Source Name", date: "MMM DD YYYY", url: "https://..." }
```

If the MCP result includes a URL, populate `url`. Otherwise use `""` (never fabricate a URL).

**Required `src` locations** (minimum — add more when data supports it):

| Data path | Field(s) with `src` |
|---|---|
| `energyMarkets` | `brent.src`, `wti.src` (from `bigdata_market_tearsheet`) |
| `goldmanAnalysis` | `q2Forecast.src`, `upside.src`, `q4Forecast.src` |
| `hormuz` | `src` (status), `trafficSrc`, `offlineSrc` |
| `dualChokepoint` | `seaborneSrc`, `houthiSrc` |
| `chinaDeep` | `iranSrc`, `meSrc`, `sprSrc` |

Target **15–25** `GroundedSpan`-wrapped data points across the dashboard. The JSX component renders a **dashed blue underline** on hover-attributed values; the tooltip shows the source name (and a clickable link if `url` is populated).

---

### Rule 5 — No Fabricated Attribution

Never invent a source name, URL, date, or document title. If a `bigdata_search` result lacks a URL, cite it as:

> [Source Name - MMM DD, YYYY] (no URL available)

If source name or date is missing from the MCP output, use what is available:

> [Unknown Source - retrieved Mar 6, 2026]

---

## Tool Usage Map

| Domain | Primary Tool | Secondary Tool |
|---|---|---|
| Oil prices (WTI, Brent) + energy commodities | `bigdata_market_tearsheet` | `bigdata_search` |
| Goldman Sachs / analyst forecasts | `bigdata_search` | — |
| Hormuz / maritime status | `bigdata_search` | — |
| Military events | `bigdata_search` | — |
| Sanctions / diplomatic | `bigdata_search` | — |
| Financial markets, FX, equities | `bigdata_search` | — |
| Country macro indicators (GDP, CPI, trade) | `bigdata_country_tearsheet` | `bigdata_search` |
| Japan, Korea, China, India exposure data | `bigdata_country_tearsheet` | `bigdata_search` |
| Event timeline enrichment | `bigdata_events_calendar` | `bigdata_search` |
| Company / shipper intelligence (future) | `bigdata_company_tearsheet` | `find_companies` |

## Iterative Search Strategy

Run multiple targeted searches rather than one broad query. After initial results, identify gaps and refine:

1. Broad query first: `"WTI Brent crude oil prices today Iran conflict"`
2. Review results — if price data is stale, try: `"crude oil futures today March 2026"`
3. For analyst commentary: `"Goldman Sachs oil price forecast Iran Hormuz"`
4. For country-level macro: call `bigdata_country_tearsheet` for `JP`, `KR`, `CN`, `IN`, `US`

Recommended `max_chunks`: 20–50 per query. Prefer multiple focused calls over one mega-query.

---

## Grounding Checklist (Before Writing JSX)

- [ ] Every number in `GROUNDED_DATA` traces to a specific MCP result
- [ ] Every `source` field in `GROUNDED_DATA` is populated (never empty)
- [ ] Stale values are flagged with `[STALE — ...]` in the source string
- [ ] No fabricated source names, URLs, or dates appear anywhere
- [ ] `src` objects populated on all required locations (see Rule 6 table) — at least 12 `src` objects
- [ ] `SourceTag` `source` strings use `;`-separated format so the collapsible footer parses correctly
- [ ] The session date is correct in the dashboard header
