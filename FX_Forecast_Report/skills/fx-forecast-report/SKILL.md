---
name: bigdata-fx-forecast
description: >-
  Produce a short-horizon (default 5-day) FX forecast for any currency pair — directional
  call, driver breakdown, and key risks — driven entirely by Bigdata.com MCP tools
  (bigdata_country_tearsheet + bigdata_search). Use when the user asks for an FX forecast,
  currency-pair outlook, "where is USD/JPY / EUR/USD / GBP/JPY / USD/TWD heading", a
  rate-differential or carry view, intervention risk, or a 5-day directional read on a
  currency pair. Triggers: FX forecast, currency outlook, USD/JPY, EUR/USD, pair direction,
  carry trade, central bank rate differential, intervention risk.
---

# FX Forecast Report (Bigdata.com MCP)

Generate a repeatable, parameterized short-horizon FX forecast for any currency pair using
only [Bigdata.com](https://bigdata.com) MCP tools for data and your own synthesis for
scoring. No Python or SDK is required; a companion automated notebook lives in the
`FX_Forecast_Report/` cookbook if the user wants a coded pipeline.

## Inputs (confirm or infer, then state them back)

| Parameter | Meaning | Example |
|---|---|---|
| `pair` | Currency pair, BASE/QUOTE | `USD/JPY` |
| `base_country` / `quote_country` | 2-letter tearsheet codes | `US` / `JP` |
| `central_bank_base` / `central_bank_quote` | Central bank names | `Federal Reserve` / `Bank of Japan` |
| `horizon_days` | Forecast horizon | `5` |
| `sector_driver_terms` | Optional export/trade terms | `["semiconductor exports","TSMC"]` |
| `intervention_history` | Does a central bank in the pair manage the currency? | `true` |
| `weight_overrides` | Optional per-driver weights (override the defaults in Step 6; renormalized over active drivers) | `{"rate_differential":0.4,"risk_sentiment_carry":0.25}` |

`base_up` always means the **base currency appreciates** vs the quote (the pair rises);
`base_down` means it falls; `neutral` means no clear edge.

## Coverage caveat (read before pulling data)

`bigdata_country_tearsheet` accepts only these 2-letter codes and has **no pair argument**:
`AE AR AT AU BE BR CA CH CL CN CO CZ DE DK EG EMU ES FI FR GR HK HU ID IE IL IN IS IT JP KR
KW MX NL NO NZ PL PT QA RO RU SA SE SG SK TH TR UK US ZA`.

- A country outside this set (e.g. **Taiwan `TW`**) has **no tearsheet** → build that side of
  the pair from `bigdata_search` only, and say so in the report.
- The currency section lists majors vs USD; exotic-pair spot (e.g. TWD) is not returned
  structurally — treat exotic-pair pricing as qualitative.

## Workflow

Copy this checklist and track progress:

```
- [ ] 1. Confirm/infer parameters and map countries to 2-letter codes
- [ ] 2. Pull base + quote country tearsheets (skip unsupported; note it)
- [ ] 3. Central-bank feed: search both banks' latest policy signals
- [ ] 4. Driver evidence: focused searches for the remaining drivers
- [ ] 5. Score each active driver (lean / confidence / rationale / sources)
- [ ] 6. Weighted aggregation -> overall call + conviction
- [ ] 7. Write the report (summary, driver table, risk flags, appendix)
```

### Step 1 — Parameters
State the resolved parameters back to the user. Convert country names to codes (e.g.
Germany → `DE`, Eurozone → `EMU`). Use `find_securities` only if you must resolve a ticker
or entity id; it is **not** needed for tearsheets (they take a country code) or for smart
searches (names resolve internally).

### Step 2 — Data layer (tearsheets)
Call `bigdata_country_tearsheet` once for `base_country` and once for `quote_country`
(skip and note any unsupported country). From each tearsheet keep:
- **Economic Calendar — Upcoming Events**: list HIGH/MEDIUM-impact releases dated inside the
  horizon window (these become event-risk flags).
- **Currency** (spot, trend/momentum, cross-currency vs USD), **Treasury Yields** (US only),
  **Country Comparison** (G7 rates/CPI/unemployment), and relevant **Macroeconomic
  Overview** blocks (Trade & International, Central Bank & Monetary Policy).

### Step 3 — Central-bank feed (rate-differential evidence)
For **each** central bank, run 3–5 focused `bigdata_search` calls covering: latest policy-rate
decision and near-term expectations; forward guidance / tone (hawkish vs dovish); inflation
vs target; next meeting / recent official commentary. Phrase queries as market commentary and
add context `"search news and research"` (raw central-bank names alone can mis-route to
filings). This evidence feeds the **Rate Differential** driver.

### Step 4 — Driver evidence (search discipline: one focus per call)
Run focused searches per driver, built from the parameters. Split topics, periods, and
aspects into separate calls.

| Driver | Example queries (fill from params) |
|---|---|
| Trade & Capital Flows | `{quote} trade balance and export outlook`; `{quote} {sector_driver_terms} exports affecting the {quote_ccy}`; `foreign capital flows into {quote} assets` |
| Intervention Risk *(only if `intervention_history`)* | `{central_bank_quote} FX intervention, reserves, and verbal signaling on the {quote_ccy}` |
| Risk Sentiment / Carry | `broad {base_ccy} strength and dollar index direction`; `{pair} carry trade and global risk-on risk-off sentiment` |
| Geopolitical | `geopolitical and trade-policy tensions between {base} and {quote}`; `{quote} political risk affecting the {quote_ccy}` |
| Technical / Positioning | `{pair} technical analysis, realized volatility, and positioning` |

### Step 5 — Score each active driver
Drop **Intervention Risk** when `intervention_history` is false. For every remaining driver,
using the tearsheet context + retrieved evidence, decide:
- **lean**: `base_up` / `base_down` / `neutral`
- **confidence**: 0.0–1.0 from evidence volume, recency, and agreement (little/conflicting → low)
- **rationale**: one sentence
- **sources**: the documents used (headline, source name, date, URL)

### Step 6 — Aggregate
Map lean to a sign (`base_up`=+1, `base_down`=-1, `neutral`=0). For each driver compute
`contribution = sign × confidence × weight`. Sum to `net_score`.

Default weights (normalize over active drivers so they sum to 1):

| Driver | Default |
|---|---|
| Rate Differential | 0.30 |
| Trade & Capital Flows | 0.20 |
| Risk Sentiment / Carry | 0.20 |
| Intervention Risk | 0.10 |
| Geopolitical | 0.10 |
| Technical / Positioning | 0.10 |

If `weight_overrides` is supplied, apply those values on top of the defaults before
normalizing over active drivers. Otherwise tilt weights per pair: rate differential + trade
flows dominate for export-driven currencies (e.g. TWD, KRW); rate differential + intervention
+ carry dominate for JPY-type pairs.

- `net_score > +0.15` → **base_up** (pair rises); `< -0.15` → **base_down** (pair falls);
  else **neutral**.
- Conviction: `|net_score| ≥ 0.5` High, `≥ 0.25` Medium, else Low.

### Step 7 — Report
Use the template below.

## Report template

```markdown
# {pair} — {horizon_days}-Day FX Forecast

*Base: {base_name} ({central_bank_base}) · Quote: {quote_name} ({central_bank_quote})*

## Executive Summary
[One tight paragraph: directional call + conviction for {pair}, the 2–3 drivers doing the
most work, and the main risk. No invented numbers.]

**Overall call:** {direction, e.g. USD appreciates vs JPY (USD/JPY rises)}
**Conviction:** {High/Medium/Low} (net score {+/-0.NN})

## Driver Table
| Driver | Lean | Confidence | Weight | Rationale | Sources |
|---|---|---|---|---|---|
| Rate Differential | ▲ base up ({pair} ↑) | 0.80 | 35% | ... | 4 docs |
| ... | ... | ... | ... | ... | ... |

_"base up" means {base_ccy} strengthens vs {quote_ccy} (i.e. {pair} rises)._

## Risk Flags
- **Intervention risk** ({central_bank_quote}): ... (or "not flagged for this pair")
- **Geopolitical tail risk**: ...
- **Event risk (releases inside the horizon):** [table of HIGH/MEDIUM calendar events]

## Appendix — Sources
[Numbered list: [headline](url) — source (date)]
```

## Quality standards
- Cite sources inline as `[Source name - MMM DD, YYYY](url)` from `bigdata_search` results;
  every material claim is attributed.
- Separate **facts** (tearsheet/search) from **analysis** (your leans).
- Prefer a table or explicit series for numeric/trend points.
- Note reduced coverage whenever a country's tearsheet is unavailable.
- This report is a research aid, not investment advice.

## Footer (append verbatim)

> ---
> Powered by [Bigdata.com](https://bigdata.com). This report is for research purposes only
> and is not investment advice.
