# Update Dashboard — LLM Agent Runbook

You are a Cursor automation agent executing a scheduled update cycle for the **Iran Geopolitical Risk Intelligence Dashboard**. There is no traditional backend — you ARE the backend. Your job is to query live data, regenerate a self-contained React component, validate it, and push to GitHub.

**This document is your complete instruction set.** Read the referenced files before acting.

---

## Pre-Flight: Environment Setup

Before doing anything else:

1. Run `date -u '+%Y-%m-%d %H:%M UTC'` to get the exact current timestamp. Use this value for all time references throughout the cycle — header timestamp, commit message, and source freshness checks.
2. Run `npm install` to ensure all dependencies (including Vite) are available for the build step.

---

## Pre-Flight: Read These Files

After getting the timestamp, read these two files in parallel:

| File | Purpose |
|---|---|
| `src/dashboard.jsx` | Current live dashboard — your starting point and continuity reference |
| `skills/frontend-design.md` | Design guardrails: sizes, colors, component contracts, Size Budget |

All other rules (grounding, citation format, query list, GROUNDED_DATA schema, validation checklist) are inlined in this runbook below. The `skills/` files remain the source of truth for ad-hoc sessions but are **not read during automated cycles** to save tokens.

**Critical design constraints** (also in `skills/frontend-design.md`):
- **Inter** font for all display/body/UI text (not Syne, not Roboto, not system fonts)
- **JetBrains Mono** for all numeric data, prices, percentages, dates
- Inline `fontFamily` strings — no `FONTS` constant object
- The exact `COLORS` object (21 keys, `bg: "#0a0e1a"`) — see Step 3
- Flat card backgrounds (`COLORS.card`) — no gradients, no box shadows on cards
- Component contracts: `Badge`, `Metric`, `SourceTag`, `CardContainer`
- Fonts are loaded in `index.html`. Do NOT add a Google Fonts `<link>` tag inside the JSX component
- **Metric value max length: 15 characters.** Never put sentences, source names, or multi-clause text into the `value` prop. Use `sub` for context and `SourceTag` for attribution

---

## Step 1 — Query Bigdata.com MCP for Fresh Data

### 1A. Market Tearsheet (always first)

Call `bigdata_market_tearsheet` (no arguments) as the **first** MCP call every cycle. It returns authoritative real-time data for:

- **Energy commodities** (5): Brent Crude, WTI Crude, Natural Gas, Gasoline RBOB, Heating Oil — each with current price + 1D/5D/1M/3M/6M/YTD/1Y percentage changes
- **Major indexes** (38): S&P 500, Dow, Nasdaq, Nikkei, KOSPI, TAIEX, NIFTY, Hang Seng, etc.
- **Currencies** (49): USD/CNY, USD/INR, USD/JPY, USD/KRW, crypto, etc.
- **Global equity ETFs** (38): per-country performance

This is the **canonical price source** for all energy commodity data. Populate `GROUNDED_DATA.energyMarkets` prices and multi-period changes from this tool.

**Source attribution**: Add a Market Tearsheet entry to the `sources` array of every panel that uses price/index/currency data:
```js
{ headline: "Brent & WTI real-time prices and performance data", source: "Bigdata.com Market Tearsheet", ts: "{CYCLE_TIMESTAMP_ISO}", id: "MARKET_TEARSHEET", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-market-tearsheet" }
```
Must appear in: `energyMarkets.sources`, `timelineSources`, `countrySources`. Also add to `goldmanAnalysis.sources` if index/VIX data is referenced.

### 1B. Search Queries (10 consolidated)

Run all queries in parallel. Use `bigdata_search` with `max_chunks: 20` per query (use 30 only for query #1 if cross-validation of price commentary is needed).

```
1. "WTI Brent crude oil prices Iran Hormuz disruption today"
2. "Strait of Hormuz shipping closure carriers Maersk suspended attacks"
3. "Houthi Bab el-Mandeb Red Sea maritime chokepoint attacks"
4. "Iran military strikes US bases Israel nuclear operation latest"
5. "stock market equities Iran war Goldman Sachs oil forecast"
6. "Fed rate cut inflation oil price shock macro impact"
7. "China oil imports Iran crude SPR energy disruption"
8. "Japan South Korea India oil Middle East energy Iran exposure"
9. "Iran sanctions diplomacy negotiations ceasefire US"
10. "Iran escalation scenario analysis oil supply disruption causal chain"
```

### 1C. Adaptive Follow-Up Queries

After the initial 10 queries return, evaluate results against the `GROUNDED_DATA` schema and fire targeted follow-up queries for any gaps:

- For each section of `GROUNDED_DATA`, check if the initial results provide a fresh value
- If a field would fall back to `[STALE]`, construct a narrower follow-up query targeting that specific data point
- If mindmap reasoning lacks evidence for a causal link, query for that specific connection (e.g., `"Iran mine-laying Hormuz shipping insurance impact"`)
- If a major new event is detected but details are thin, query for that event specifically

**Constraints**:
- Maximum 3 follow-up queries per cycle (to cap cost/latency)
- Follow-ups use `max_chunks: 15` (narrow and targeted)
- Skip follow-ups entirely if all `GROUNDED_DATA` fields are populated with fresh data
- Log which follow-ups were triggered in the cycle feedback entry

### 1D. Country Tearsheets (follow-up only)

Do NOT call `bigdata_country_tearsheet` in the default batch. Call it as a follow-up only when search results lack country-level macro data for a specific economy (`JP`, `KR`, `CN`, `IN`, `US`). Re-enable in the default batch when conflict de-escalates and country macro data becomes the primary signal.

When a country tearsheet IS called, add a corresponding source entry to the relevant `sources` array:
```js
{ headline: "{Country} macro overview: GDP, CPI, indices, currency, economic calendar", source: "Bigdata.com Country Tearsheet — {CC}", ts: "{CYCLE_TIMESTAMP_ISO}", id: "COUNTRY_TEARSHEET_{CC}", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-country-tearsheet" }
```
Add to `countrySources` for the Country Exposure panel, and to `chinaDeep.sources` if the country is CN.

### 1E. Events Calendar (Monday only)

Call `bigdata_events_calendar` for energy sector events only on **Monday cycles**. Skip all other days.

---

## Step 2 — Synthesize Data into GROUNDED_DATA

Extract data from the MCP results and populate every field in the `GROUNDED_DATA` schema below.

### Grounding Rules (inlined from `skills/bigdata-mcp-grounding.md`)

- **Every value must trace to a specific MCP result** — do not invent figures
- **Every `sources` array must be populated** with structured source objects (see schema below)
- **If no fresh data found for a field**, retain the previous value from `src/dashboard.jsx` and append a source entry with `headline: "[STALE — no fresh data {TODAY}]"`
- **Never fabricate** a source name, URL, document ID, or date

### Source Object Schema

Each `sources` array contains objects with these fields:

```js
{
  headline: String,  // Article headline or data description
  source: String,    // Publisher name (e.g. "CNN", "Bigdata.com Market Tearsheet")
  ts: String,        // ISO 8601 timestamp (e.g. "2026-03-16T10:35:58")
  id: String,        // RP_DOCUMENT_ID from bigdata_search, or a descriptive ID for tearsheets
  url: String,       // Optional — article URL if available from the MCP result
}
```

### How to Populate Sources from MCP Results

**`bigdata_search`**: Each search result includes `id` (RP_DOCUMENT_ID), `headline`, `source.name`, `timestamp`, and often `url`. Map these directly:
```js
{ headline: result.headline, source: result.source.name, ts: result.timestamp, id: result.id, url: result.url || undefined }
```

**`bigdata_market_tearsheet`**: Add one entry per panel that uses price data:
```js
{ headline: "Brent & WTI real-time prices and performance data", source: "Bigdata.com Market Tearsheet", ts: "{CYCLE_TIMESTAMP_ISO}", id: "MARKET_TEARSHEET", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-market-tearsheet" }
```

**`bigdata_country_tearsheet`**: Add one entry per country queried, to the panel that uses that country's data:
```js
{ headline: "{Country} macro overview: GDP, CPI, indices, currency, economic calendar", source: "Bigdata.com Country Tearsheet — {CC}", ts: "{CYCLE_TIMESTAMP_ISO}", id: "COUNTRY_TEARSHEET_{CC}", url: "https://docs.bigdata.com/mcp-reference/tools/bigdata-country-tearsheet" }
```

**`bigdata_events_calendar`**: Add to the relevant panel (typically energyMarkets or timeline):
```js
{ headline: "Upcoming energy sector economic events", source: "Bigdata.com Events Calendar", ts: "{CYCLE_TIMESTAMP_ISO}", id: "EVENTS_CALENDAR", url: "https://bigdata.com" }
```

### Where to Include Tearsheet References

- **`energyMarkets.sources`**: Always include Market Tearsheet (canonical price source)
- **`goldmanAnalysis.sources`**: Include Market Tearsheet if index/VIX data is referenced
- **`countrySources`**: Include Market Tearsheet (ETF/currency data) + Country Tearsheet for each queried country (JP, CN, IN, US, etc.)
- **`chinaDeep.sources`**: Include Country Tearsheet — CN when China macro data is used
- **`timelineSources`**: Include Market Tearsheet (price history context)
- **`hormuz.sources` / `dualChokepoint.sources`**: Typically search-only (conflict news focus); add tearsheets only if macro/price data is directly cited

### GROUNDED_DATA Schema

```js
const GROUNDED_DATA = {
  energyMarkets: {
    brent:  { price: Number, d1: String, d5: String, m1: String, m3: String, ytd: String },
    wti:    { price: Number, d1: String, d5: String, m1: String, m3: String, ytd: String },
    rbob:   { price: Number, d1: String, d5: String, m1: String, m3: String, ytd: String },
    heat:   { price: Number, d1: String, d5: String, m1: String, m3: String, ytd: String },
    natgas: { price: Number, d1: String, d5: String, m1: String, m3: String, ytd: String },
    spread: String,           // WTI-Brent spread
    brentYearStart: String,   // Brent price on Jan 1
    timestamp: String,        // e.g. "Mar 16, 2026 06:03 UTC" — explicit data freshness
    drivers: [                // 3–5 key headlines driving the market — from bigdata_search, NOT the tearsheet
      { headline: String, detail: String, attribution: String }
    ],
    sources: [SourceObject],  // REQUIRED — include Market Tearsheet + key search results
  },
  goldmanAnalysis: {
    q2Forecast: { value: String, detail: String },  // value ≤15 chars
    upside: { value: String, detail: String },
    q4Forecast: { value: String, detail: String },
    riskPremium: String,
    riskPremiumPct: String,
    sources: [SourceObject],  // REQUIRED — include Market Tearsheet if index data used
  },
  hormuz: {
    globalOilTransitPct: String,
    status: String,
    statusDetail: String,
    carriersSuspended: String,
    rerouteVia: String,
    trafficDrop: String,
    accessLevel: String,  // "CLOSED" | "CONDITIONAL" | "SELECTIVE" | "OPEN"
                          // Derive from statusDetail: "CLOSED" = no transit; "CONDITIONAL" = selective
                          // passage for some flag/affiliation; "SELECTIVE" = specific corridors only;
                          // "OPEN" = normal transit resumed
    shipsStruck: Number,  // Running integer count of commercial ships struck since war began
                          // Extract from search results (e.g. "18 ships struck since war began")
                          // Carry forward from previous cycle if no new count reported
    alternativeRoutes: [
      { name: String, status: String, capacityMbd: String }
      // status: "ACTIVE" | "LIMITED" | "CLOSED"
      // Populate from Ceyhan/Jask/Yanbu news in the search batch
      // Carry forward entries, updating status only if a new development is mentioned
      // e.g. { name: "Iraq-Turkey Ceyhan", status: "ACTIVE", capacityMbd: "~0.4 mbd" }
      // e.g. { name: "Saudi Yanbu (Red Sea)", status: "ACTIVE", capacityMbd: "~3 mbd" }
      // e.g. { name: "Jask (Iran bypass)", status: "LIMITED", capacityMbd: "~0.3 mbd" }
    ],
    sources: [SourceObject],  // REQUIRED
  },
  dualChokepoint: {
    description: String,
    seaborneCrudeAffected: String,
    houthiStatus: String,
    qatarWarning: String,
    sources: [SourceObject],  // REQUIRED
  },
  countries: [
    {
      name: String, flag: String, meOilDep: String, hormuzDep: String,
      reserves: String, risk: Number (1–5), riskLabel: String, color: COLORS.xxx,
      cbPolicy: String,  // "HOLD" | "HIKE" | "CUT" | "N/A"
                         // Derive from country tearsheet + search results
                         // Pegged/producer economies always "N/A" (no search needed):
                         //   Qatar, UAE, Saudi Arabia → "N/A"
                         // Current known values:
                         //   Japan BOJ → "HOLD", South Korea BOK → "HOLD", Taiwan CBC → "HOLD"
                         //   India RBI → "HOLD", China PBOC → "CUT", Germany/EU ECB → "HOLD"
                         //   United States Fed → "HOLD"
    }
  ],
  countrySources: [SourceObject],  // REQUIRED — include Market Tearsheet + Country Tearsheets
  chinaDeep: {
    iranCrudeImportsMbd: String, iranShareOfImports: String,
    meShareOfSeaborne: { value: String, detail: String },  // value ≤15 chars
    meSeaborneMbd: String,
    sprBillionBbl: String, sprCoverDays: String,
    russianPivot: String, actions: String,
    sources: [SourceObject],  // REQUIRED — include Country Tearsheet — CN
  },
  timeline: [
    { date: String, event: String, oilImpact: String }
  ],
  timelineSources: [SourceObject],  // REQUIRED — include Market Tearsheet
  mindmapNodes: {
    root: { id: "root", label: String, type: "root" },
    military:   [{ id: String, label: String, detail: String, confidence: Number (0–1), timeHorizon: String }],
    energy:     [{ id: String, label: String, detail: String, confidence: Number (0–1), timeHorizon: String }],
    trade:      [{ id: String, label: String, detail: String, confidence: Number (0–1), timeHorizon: String }],
    diplomatic: [{ id: String, label: String, detail: String, confidence: Number (0–1), timeHorizon: String }],
    financial:  [{ id: String, label: String, detail: String, confidence: Number (0–1), timeHorizon: String }],
  },
};

// SourceObject schema (used in all `sources` arrays):
// { headline: String, source: String, ts: String (ISO 8601), id: String (RP_DOCUMENT_ID), url?: String }
```

**Note:** The old `oilPrices` section is replaced by `energyMarkets` (prices from market tearsheet) and `goldmanAnalysis` (analyst commentary from search results). The old `iranSupply` section is removed — its data is covered by `chinaDeep` and the `EnergyMarketsPanel`.

### Timeline Rules
- One entry per calendar day — the `date` field must be `"Mon DD, YYYY"` with no time-of-day suffix
- For past days: append a new entry at the end of the array if the date does not yet exist
- For the current day: find the existing entry for today and replace it entirely with the latest consolidated summary — do not add a second entry for the same day
- The current day's entry must always be the **last element** in the `timeline` array — the `TimelinePanel` uses last-element position (`i === events.length - 1`) to render the glowing current-day dot
- Replace `timelineSources` each cycle with sources from the current day + prior 2–3 days only (~8–12 entries) — do not accumulate across cycles
- Each entry needs a date, event description, and oil impact note

### Mindmap Rules
- Minimum 4 nodes per layer, maximum 6
- `confidence`: float 0–1 (0.9 = high confidence, 0.4 = speculative)
- `timeHorizon`: `"immediate"` (hours–days), `"medium"` (weeks), `"structural"` (months–quarters)
- Trace multi-hop causal chains (4–6 steps minimum)
- Identify at least 2 feedback loops across domains
- Propagate confidence downward: downstream effects ≤ triggering event confidence
- Base reasoning ONLY on retrieved data — reduce confidence for ungrounded links

---

## Step 3 — Regenerate `src/dashboard.jsx`

Always ensure `src/dashboard.jsx` reflects a **complete, fresh cycle** — every `GROUNDED_DATA`
field must be populated from current MCP results (no silent carryovers). Validate against the
checklist in Step 5.

**⚠ File size constraint — use StrReplace only:** `src/dashboard.jsx` is ~700 lines and
exceeds the Write tool's limit. **Never use the Write tool for this file.**

  **Edit strategy:**
 - Update only the sections that have changed in this cycle (new prices, new headlines, new
   timeline entries, revised mindmap nodes)
 - Use one StrReplace call per logical section: `energyMarkets`, `goldmanAnalysis`, `hormuz`,
   `dualChokepoint`, `countries`, `chinaDeep`, `timeline` (append only), `mindmapNodes`
 - Skip a section entirely if its data is identical to the previous cycle — do not rewrite it
   just to rewrite it
 - Only touch component code (panels, utility functions) if a structural change is required

### Structure (top to bottom)

```
1. import { useState } from "react";
2. const COLORS = { ... };          // exact 21-key object
3. const GROUNDED_DATA = { ... };   // freshly populated from Step 2
4. function Badge(...)              // utility components
5. function Metric(...)
6. function SourceTag(...)          // structured source list (headline, source, ts, id, url)
7. function CardContainer(...)
8. function EnergyMarketsPanel(...) // panel components
9. function HormuzStatusPanel(...)
10. function CountryExposurePanel(...)
11. function ChinaDeepDivePanel(...)
12. function MindmapPanel(...)
13. function TimelinePanel(...)
14. function GoldmanAnalysisPanel(...)
15. export default function IranGeopolDashboard()  // main dashboard
```

### Design Rules

**Typography**
- Body/UI font: `fontFamily: "'Inter', -apple-system, sans-serif"`
- Numeric font: `fontFamily: "'JetBrains Mono', monospace"`
- No `FONTS` constant — use inline strings directly
- Fonts are loaded in `index.html`. Do NOT add a Google Fonts `<link>` tag inside the JSX component

**Size Budget (ceilings — never exceed)**
- Dashboard title: 20px, weight 800, letterSpacing -0.3
- Panel headers: 16px, weight 700, uppercase, letterSpacing 1.2
- Metric values: 24px, weight 800, JetBrains Mono, letterSpacing -0.5
- Metric labels: 13px
- Body/detail text: 14px max
- Card padding: `"20px 24px"`
- Alert/info box padding: 10–12px
- Panel grid gap: 12–16px
- Badge font size: 13px
- Source tag font size: 12px
- Source tag metadata (source name, date, ID): 11px
- Minimum font size: 10px (no text smaller than this)

**Metric value max length: 15 characters.** The `value` prop of `Metric` must be a short numeric string (price, percentage, count). Never put sentences, source names, or multi-clause text into `value`. Use `sub` for context and `SourceTag` for attribution.

**Colors — use this exact 21-key object**
```js
const COLORS = {
  bg: "#0a0e1a",
  card: "#111827",
  cardHover: "#1a2237",
  border: "#1e293b",
  borderActive: "#3b82f6",
  accent: "#ef4444",
  accentGlow: "rgba(239,68,68,0.15)",
  blue: "#3b82f6",
  blueGlow: "rgba(59,130,246,0.12)",
  amber: "#f59e0b",
  amberGlow: "rgba(245,158,11,0.1)",
  emerald: "#10b981",
  emeraldGlow: "rgba(16,185,129,0.1)",
  purple: "#8b5cf6",
  text: "#f1f5f9",
  textMuted: "#94a3b8",
  textDim: "#64748b",
  critical: "#ef4444",
  high: "#f59e0b",
  medium: "#3b82f6",
  low: "#10b981",
};
```

**Cards**
- Flat background: `background: COLORS.card` — NO gradients, NO box shadows
- Border: `1px solid ${COLORS.border}`
- Border radius: 12
- Padding: `"20px 24px"`

**Accent strip**
- `width: 4, height: 20, borderRadius: 2`

**Pulse animation**
```css
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
```

**Component contracts**:
- `Badge({ children, color, glow })` — pill, 11px, uppercase, 700 weight
- `Metric({ label, value, sub, color })` — label dim/uppercase, value mono/large (max 15 chars), sub muted
- `SourceTag({ sources })` — accepts array of `{ headline, source, ts, id, url }` objects. Collapsed: "X sources cited" with expand arrow. Expanded: vertical list showing headline, source name (blue), formatted date (mono), document ID (dim mono), and link button (↗) when URL is present
- `CardContainer({ children, title, badge, accent, style })` — flat card, accent strip, uppercase title
- `GroundedSpan({ children, source, url })` — dotted underline, hover tooltip with source/link. Wrap 15–25 key data points (prices, Hormuz status, country risk levels). Do not wrap every token.

**EnergyMarketsPanel** — replaces the old OilPricePanel:
- Hero metric: Brent price + 1D change with arrow indicator
- Multi-period performance table: 5 rows (Brent, WTI, RBOB, Heating Oil, NatGas) × 6 columns (Price, 1D, 5D, 1M, 3M, YTD)
- Style: `fontSize: 12`, JetBrains Mono for numbers, alternating row backgrounds, color-coded percentages (green positive, red negative)
- Below table: WTI-Brent spread, Brent start-of-2026 reference
- **Key Drivers & News** section: 5–7 novel narratives with headline, detail and attribution (sourced from `bigdata_search`, not the tearsheet). Each driver has `headline` (bold, 11px), `detail` (1–2 sentences), and `attribution` (italic, source name + date)
- **Timestamp**: explicit data freshness label (e.g. "Mar 16, 2026 06:03 UTC") displayed top-right of the hero metric
- Source: "Bigdata.com Market Tearsheet; [search sources]"

**EnergyMarketsPanel** — replaces the old OilPricePanel:
- Hero metric: Brent price + 1D change with arrow indicator
- Multi-period performance table: 5 rows (Brent, WTI, RBOB, Heating Oil, NatGas) × 6 columns (Price, 1D, 5D, 1M, 3M, YTD)
- Style: `fontSize: 12`, JetBrains Mono for numbers, alternating row backgrounds, color-coded percentages (green positive, red negative)
- Below table: WTI-Brent spread, Brent start-of-2026 reference
- **Key Drivers & News** section: 5–7 novel narratives with headline, detail and attribution (sourced from `bigdata_search`, not the tearsheet). Each driver has `headline` (bold, 11px), `detail` (1–2 sentences), and `attribution` (italic, source name + date)
- **Timestamp**: explicit data freshness label (e.g. "Mar 16, 2026 06:03 UTC") displayed top-right of the hero metric
- Sources: `sources` array with Market Tearsheet entry + 5–8 key search results (each with headline, source, ts, id, url)

**Header**
- Pulsing red dot, dashboard title, subtitle with Bigdata.com link
- Right side: Badge with status, date in JetBrains Mono
- The date must reflect the timestamp obtained in the Pre-Flight step (from `date -u`)

**Footer**
- Left: grounding attribution with Bigdata.com link and last-cycle timestamp
- Right: "Powered by Bigdata.com" link

**Main layout**
- Root container: `fontFamily: "'Inter', -apple-system, sans-serif"`
- Tab navigation: 4 tabs (overview, countries, mindmap, timeline)
- Max width: 1300px centered
- Grid: `1fr 1fr` for paired panels, `1 / -1` for full-width

---

## Step 4 — Update `index.html` if Needed

Verify `index.html` loads the correct Google Fonts. It must import **Inter** and **JetBrains Mono**:

```html
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
```

If it currently loads a different font (e.g. Syne), replace the link.

---

## Step 5 — Validate Before Committing

Run through this checklist. If any item fails, fix it before proceeding.

- [ ] `src/dashboard.jsx` starts with `import { useState } from "react";`
- [ ] `COLORS` object has exactly 21 keys matching the values above
- [ ] No `FONTS` constant — font families are inline strings
- [ ] `fontFamily: "'Inter', -apple-system, sans-serif"` on the root container
- [ ] No Google Fonts `<link>` tag inside the JSX component (fonts loaded in `index.html` only)
- [ ] All 9 sections of `GROUNDED_DATA` are populated: `energyMarkets`, `goldmanAnalysis`, `hormuz`, `dualChokepoint`, `countries`, `countrySources`, `chinaDeep`, `timeline`, `timelineSources`, `mindmapNodes`
- [ ] Every `sources` array is non-empty and contains properly structured objects with `headline`, `source`, `ts`, `id` fields
- [ ] Market Tearsheet is referenced in: `energyMarkets.sources`, `goldmanAnalysis.sources` (if index data used), `countrySources`, `timelineSources`
- [ ] Country Tearsheet entries present in `countrySources` and `chinaDeep.sources` for each queried country
- [ ] Stale values have a source entry with `headline: "[STALE — ...]"`
- [ ] No `GroundedSpan` component or usages — removed in favor of `SourceTag` structured attribution
- [ ] All panel components present: EnergyMarkets, HormuzStatus, CountryExposure, ChinaDeepDive, Mindmap, Timeline, GoldmanAnalysis
- [ ] No `IranSupplyPanel` or `OilPricePanel` — these are removed/replaced
- [ ] Every `Metric` `value` prop is ≤15 characters (no sentences in value)
- [ ] All font sizes within Size Budget ceilings (see Step 3)
- [ ] Mindmap: minimum 4 nodes per layer, all have `confidence` and `timeHorizon`
- [ ] Dashboard header timestamp matches the `date -u` value from Pre-Flight
- [ ] No JSX syntax errors (balanced tags, valid JS expressions)
- [ ] No markdown fences wrapping the output
- [ ] The file ends with the closing brace of `export default function IranGeopolDashboard()`
- [ ] `index.html` loads Inter + JetBrains Mono (not Syne or other fonts)
- [ ] Run `npm run build` to confirm no build errors

---

## Step 6 — Commit and Push

Stage, commit, and push directly to `main`. Do NOT stop between these commands.

**Important:** `git push origin HEAD:main` pushes your commits to the remote `main` branch WITHOUT switching your local branch. You stay on whatever branch you are on. This does not violate any branch-switching rules — it is a remote-only operation.

```bash
git add src/dashboard.jsx index.html feedback/cycle-log.md
git commit -m "chore: regenerate dashboard — {TIMESTAMP_FROM_PREFLIGHT}

Updated GROUNDED_DATA via Bigdata.com MCP.
Key changes: {BRIEF_SUMMARY_OF_WHAT_CHANGED}"
git push origin HEAD:main
```

The push to `main` triggers `.github/workflows/deploy.yml`, which builds and deploys automatically.

---

## Error Handling

| Situation | Action |
|---|---|
| `bigdata_search` returns empty for a domain | Retain previous values, mark `[STALE]`, continue |
| Generated JSX fails `npm run build` | Fix syntax errors; do NOT push broken code |
| Mindmap reasoning produces < 4 nodes per layer | Supplement with retained nodes from previous cycle |
| `bigdata_country_tearsheet` unavailable | Fall back to `bigdata_search` for country data |
| All MCP tools unavailable | **Abort cycle entirely** — keep previous dashboard live |
| `index.html` font mismatch | Fix to load Inter + JetBrains Mono before committing |
| `git push origin HEAD:main` rejected (non-fast-forward) | Run `git pull --rebase origin main` then retry the push |
| Deploy is visually broken despite passing build | `git revert HEAD && git push origin HEAD:main` |
| `npm install` fails | Check network, clear `node_modules` and retry: `rm -rf node_modules && npm install` |
| Write tool returns "Invalid arguments" on `src/dashboard.jsx` | File is too large for Write tool (~700 lines). Switch to `StrReplace` immediately — do NOT retry Write. Update `GROUNDED_DATA` and components section by section with targeted StrReplace calls. |

---

## Step 7 — Write Cycle Feedback

After each cycle, append a timestamped entry to `feedback/cycle-log.md` documenting any friction, surprises, or suggestions for improving the runbook or tooling. This file is committed alongside the dashboard update.

**Create the file if it doesn't exist.** Use this format:

```markdown
## {TIMESTAMP_FROM_PREFLIGHT}

### What went well
- (e.g., all MCP queries returned fresh data, build passed first try)

### Issues encountered
- (e.g., had to install vite before `npm run build` would work, was on a feature branch instead of main)

### Suggestions
- (e.g., add `npm install` step to runbook before build, pin node version in CI)
```

**Rules:**
- Always append — never overwrite previous entries
- Be specific: include exact error messages, commands that failed, and what you did to fix them
- Flag anything that required improvisation not covered by this runbook
- Keep each entry concise (aim for 5–15 lines)
- `git add feedback/cycle-log.md` alongside `src/dashboard.jsx` in Step 6

---

## Reference: File Layout

```
llm_as_backend/
├── src/
│   ├── dashboard.jsx        ← YOU REGENERATE THIS
│   └── main.jsx             ← do not touch
├── index.html               ← verify font imports (fonts loaded here, NOT in JSX)
├── feedback/
│   └── cycle-log.md         ← append feedback each cycle (Step 7)
├── skills/
│   ├── frontend-design.md   ← design system rules (READ during pre-flight)
│   ├── bigdata-mcp-grounding.md  ← grounding rules (inlined in this runbook)
│   └── dashboard-gen.md     ← generation skill (inlined in this runbook)
├── spec/
│   └── dashboard-spec.md    ← product specification (not read during automated cycles)
├── .github/workflows/
│   ├── deploy.yml           ← push-to-main deploy
│   └── update.yml           ← cron-triggered update cycle
├── Dockerfile
├── fly.toml
└── nginx.conf
```
