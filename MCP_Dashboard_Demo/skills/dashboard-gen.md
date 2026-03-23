# Dashboard Generation Skill — Iran Geopolitical Risk Dashboard

## Role

You are a Cursor automation agent executing a scheduled LLM-as-backend update cycle for the Iran Geopolitical Risk Intelligence Dashboard. There is no traditional backend. You ARE the backend. Your job is to:

1. Query [Bigdata.com](https://bigdata.com) MCP tools to gather fresh geopolitical and market data
2. Synthesize that data into a grounded `GROUNDED_DATA` object
3. Regenerate `src/dashboard.jsx` as a complete, self-contained React component
4. Validate the output
5. Commit the file to the repo (the deploy pipeline handles the rest)

Apply `skills/bigdata-mcp-grounding.md` and `skills/frontend-design.md` throughout.

---

## System Context (Injected Each Cycle)

Before beginning, the agent prompt includes:

- Current date and time (UTC): `{CURRENT_DATETIME}`
- The full contents of `spec/dashboard-spec.md`
- The full contents of `skills/bigdata-mcp-grounding.md`
- The full contents of `skills/frontend-design.md`
- The full contents of the current `src/dashboard.jsx` (for continuity reference)

---

## Step 1 — Data Collection

### 1A. Market Tearsheet (always first)

Call `bigdata_market_tearsheet` (no arguments) as the **first** MCP call every cycle. It returns authoritative real-time energy commodity prices with multi-period changes, plus indexes, currencies, and ETFs. This is the **canonical price source** for `GROUNDED_DATA.energyMarkets`. Citation: `*(Bigdata.com Market Tearsheet)*`

### 1B. Search Queries (10 consolidated)

Run all queries in parallel. Use `bigdata_search` with `max_chunks: 20` per query (use 30 for query #1 only if cross-validation of price commentary is needed).

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

After the initial batch, evaluate results against `GROUNDED_DATA` schema. Fire targeted follow-ups for gaps:
- Maximum 3 follow-up queries per cycle (`max_chunks: 15`)
- Skip follow-ups entirely if all fields are populated with fresh data
- Log which follow-ups were triggered in the cycle feedback entry
- Call `bigdata_country_tearsheet` as a follow-up (not default batch) if search results lack country-level macro data

### 1D. Events Calendar (Monday only)

Call `bigdata_events_calendar` for energy sector events only on **Monday cycles**. Skip all other days.

---

## Step 2 — Data Synthesis

After all queries complete, extract and structure the following fields for `GROUNDED_DATA`. Apply Rule 1 from `skills/bigdata-mcp-grounding.md`: every value must trace to a specific MCP result. If no fresh data available, retain the previous value and append `[STALE — no fresh data {DATE}]` to the source field.

### Required `GROUNDED_DATA` Schema

```js
const GROUNDED_DATA = {
  energyMarkets: {
    brent:  { price: Number, d1: String, d5: String, m1: String, m3: String, ytd: String, src: { source: String, date: String, url: String } },
    wti:    { price: Number, d1: String, d5: String, m1: String, m3: String, ytd: String, src: { source: String, date: String, url: String } },
    rbob:   { price: Number, d1: String, d5: String, m1: String, m3: String, ytd: String },
    heat:   { price: Number, d1: String, d5: String, m1: String, m3: String, ytd: String },
    natgas: { price: Number, d1: String, d5: String, m1: String, m3: String, ytd: String },
    spread: String,           // WTI-Brent spread
    brentYearStart: String,   // Brent price on Jan 1
    timestamp: String,        // e.g. "Mar 16, 2026 06:03 UTC" — explicit data freshness
    drivers: [                // 3–5 key headlines from bigdata_search (not tearsheet)
      { headline: String, detail: String, attribution: String }
    ],
    source: String,           // "Bigdata.com Market Tearsheet; ..." — REQUIRED
  },

  goldmanAnalysis: {
    q2Forecast: { value: String, detail: String, src: { source, date, url } },  // value ≤15 chars
    upside: { value: String, detail: String, src: { source, date, url } },
    q4Forecast: { value: String, detail: String, src: { source, date, url } },
    riskPremium: String,
    riskPremiumPct: String,
    source: String,  // REQUIRED
  },

  hormuz: {
    globalOilTransitPct: String,
    status: String,
    statusDetail: String,
    carriersSuspended: String,
    rerouteVia: String,
    trafficSrc: { source, date, url },  // GroundedSpan on traffic metric
    offlineSrc: { source, date, url },  // GroundedSpan on offline output metric
    src: { source, date, url },         // GroundedSpan on status text
    source: String,            // REQUIRED
  },

  dualChokepoint: {
    description: String,
    seaborneCrudeAffected: String,
    seaborneSrc: { source, date, url },  // GroundedSpan
    houthiStatus: String,
    houthiSrc: { source, date, url },    // GroundedSpan
    source: String,            // REQUIRED
  },

  countries: [
    {
      name: String,
      flag: String,
      meOilDep: String,
      hormuzDep: String,
      reserves: String,
      risk: Number,            // 1–5 composite score
      riskLabel: String,       // "Critical" | "High" | "Medium" | "Low"
      color: String,
    }
  ],

  chinaDeep: {
    iranCrudeImportsMbd: String,
    iranShareOfImports: String,
    iranSrc: { source, date, url },      // GroundedSpan on imports metric
    meShareOfSeaborne: { value: String, detail: String },  // value ≤15 chars
    meSrc: { source, date, url },        // GroundedSpan on ME share metric
    meSeaborneMbd: String,
    sprBillionBbl: String,
    sprCoverDays: String,
    sprSrc: { source, date, url },       // GroundedSpan on SPR metric
    russianPivot: String,
    actions: String,
    source: String,            // REQUIRED
  },

  timeline: [
    { date: String, event: String, oilImpact: String }
  ],

  mindmapNodes: {
    root: { id: "root", label: String, type: "root" },
    military:   [{ id: String, label: String, detail: String, confidence: Number, timeHorizon: String }],
    energy:     [{ id: String, label: String, detail: String, confidence: Number, timeHorizon: String }],
    trade:      [{ id: String, label: String, detail: String, confidence: Number, timeHorizon: String }],
    diplomatic: [{ id: String, label: String, detail: String, confidence: Number, timeHorizon: String }],
    financial:  [{ id: String, label: String, detail: String, confidence: Number, timeHorizon: String }],
  },
};
```

**Note:** The old `oilPrices` section is replaced by `energyMarkets` (prices from market tearsheet) and `goldmanAnalysis` (analyst commentary from search results). The old `iranSupply` section is removed — its data is covered by `chinaDeep` and the `EnergyMarketsPanel`.

**`src` objects**: Populate `src` on key data points (prices, Hormuz status) for inline hover attribution via the `GroundedSpan` component.

Notes on mindmap nodes:
- `confidence`: float 0–1 representing certainty of the causal link (0.9 = high confidence, 0.4 = speculative)
- `timeHorizon`: `"immediate"` (hours–days), `"medium"` (weeks), `"structural"` (months–quarters)
- Minimum 4 nodes per layer, maximum 6

---

## Step 3 — Reasoning Mindmap (Extended Thinking)

The mindmap requires extended thinking / reasoning mode. Use an Opus-class or reasoning-capable model for this step.

### Mindmap Reasoning Prompt

```
You are analyzing the Iran geopolitical crisis as of {CURRENT_DATE}.

Using the search results gathered in this session, generate a comprehensive geopolitical causal mindmap with the following requirements:

ROOT NODE: "Iran Escalation — {CURRENT_DATE}"

For each of the five domain layers (Military/Security, Energy/Commodity, Trade/Maritime, Diplomatic/Political, Financial/Macro), generate 4–6 nodes. Each node must:
1. Have a short label (3–6 words)
2. Have a 1–2 sentence detail explaining the causal mechanism
3. Include a confidence score (0–1) based on available evidence
4. Include a time horizon tag: "immediate", "medium", or "structural"

Requirements for the reasoning process:
- Trace multi-hop causal chains (4–6 steps minimum). Example: Houthi attack → vessel avoidance → Cape rerouting → +15 day transit → inventory drawdown → price spike → inflation expectations
- Identify at least 2 feedback loops (reinforcing cycles) across domains
- Model conditional branches: what changes if Hormuz remains open vs. fully closed?
- Propagate confidence downward: if a triggering event has 0.7 confidence, downstream effects should be ≤ 0.7
- Flag which nodes represent base case vs. escalation scenario

Output format: structured JSON matching the `mindmapNodes` schema in `GROUNDED_DATA`.

Base your reasoning ONLY on the search results retrieved in this session. If a causal link is not grounded in retrieved data, reduce its confidence score accordingly.
```

---

## Step 4 — JSX Generation Rules

After synthesis, generate the complete `src/dashboard.jsx` following these strict rules:

### Output Format
- **Output ONLY the JSX file content**
- No markdown fences (no ` ```jsx ``` `)
- No explanation, no commentary before or after the code
- The file must begin with `import { useState } from "react";` and end with the `export default function` closing brace

### Content Rules
1. `GROUNDED_DATA` at the top — fully populated from Step 2
2. `COLORS` object (21 keys) immediately after — maintain existing values, may extend but not remove
3. All utility components: `Badge`, `Metric`, `SourceTag`, `CardContainer`, `GroundedSpan`
4. All panel components: `EnergyMarketsPanel`, `HormuzStatusPanel`, `CountryExposurePanel`, `ChinaDeepDivePanel`, `MindmapPanel`, `TimelinePanel`, `GoldmanAnalysisPanel`
5. Main export: `IranGeopolDashboard` with tab navigation and footer
6. `<style>` tag at end with `@keyframes pulse` and scrollbar styles
7. No Google Fonts `<link>` tag inside the JSX (fonts loaded in `index.html`)
8. Every `Metric` `value` prop ≤15 characters — short numeric string only

### Design Rules
Apply `skills/frontend-design.md` in full:
- Use the chosen distinctive display font (maintain whatever was chosen in the previous cycle for visual continuity)
- Maintain the dark crisis-monitor aesthetic
- No generic enterprise styling

### Continuity Rules
- Read the current `src/dashboard.jsx` before generating
- Maintain the same tab structure, panel layout, and color system
- The visual output should be recognizably the same dashboard, just with fresher data
- Only introduce design changes when the previous cycle explicitly flagged visual drift

---

## Step 5 — Validation Checklist

Before writing the output file, verify:

- [ ] All sections of `GROUNDED_DATA` are populated: `energyMarkets`, `goldmanAnalysis`, `hormuz`, `dualChokepoint`, `countries`, `chinaDeep`, `timeline`, `mindmapNodes`
- [ ] Every `source` field is non-empty
- [ ] Stale values have `[STALE — ...]` suffix
- [ ] All panel components present: EnergyMarkets, HormuzStatus, CountryExposure, ChinaDeepDive, Mindmap, Timeline, GoldmanAnalysis
- [ ] No `IranSupplyPanel` or `OilPricePanel` — these are removed/replaced
- [ ] `GroundedSpan` component present and used on 15–25 key data points
- [ ] Every `Metric` `value` prop ≤15 characters
- [ ] No Google Fonts `<link>` tag inside the JSX component
- [ ] `COLORS` object has exactly 21 keys
- [ ] Mindmap nodes: minimum 4 per layer, all have `confidence` and `timeHorizon`
- [ ] No JSX syntax errors (balanced tags, valid JS)
- [ ] No markdown fences in the output
- [ ] The file starts with `import { useState } from "react";`
- [ ] The file ends with `export default function IranGeopolDashboard()`
- [ ] The dashboard header timestamp reflects the current session date

---

## Step 6 — Commit

After generating and validating `src/dashboard.jsx`, commit it:

```bash
git add src/dashboard.jsx
git commit -m "chore: regenerate dashboard — {CURRENT_DATETIME}

Updated GROUNDED_DATA via Bigdata.com MCP.
Key changes: {BRIEF_SUMMARY_OF_WHAT_CHANGED}"
git push origin main
```

The push to `main` triggers `.github/workflows/deploy.yml`, which builds and deploys to Fly.io automatically.

---

## Error Handling

| Situation | Action |
|---|---|
| `bigdata_search` returns empty results for a domain | Retain previous values, mark as `[STALE]`, continue |
| Generated JSX fails validation | Do NOT commit; log the error; keep previous version live |
| Mindmap reasoning produces fewer than 4 nodes per layer | Supplement with retained nodes from previous cycle |
| `bigdata_country_tearsheet` unavailable | Fall back to `bigdata_search` for country-level data |
| All MCP tools unavailable | Abort cycle; do NOT generate; keep previous dashboard live |

---

## Cost Reference

| Component | Per Cycle |
|---|---|
| `bigdata_market_tearsheet` + `bigdata_search` (10 queries × 20 chunks + up to 3 follow-ups) | Included in Bigdata.com subscription |
| Sonnet-class model (data extraction + JSX gen) | ~$0.10–0.25 |
| Opus-class model (mindmap reasoning, extended thinking) | ~$0.50–1.00 |
| **Total per cycle** | **~$0.60–1.25** |

At hourly cadence: ~$430–900/month. At 4-hourly: ~$110–225/month.
Recommended: 4-hourly during active monitoring, 8-hourly in quieter periods.
