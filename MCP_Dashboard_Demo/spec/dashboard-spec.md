# Iran Geopolitical Risk Intelligence Dashboard

**Product Specification — DRAFT March 2026 — Confidential**

Geopolitical Mindmap & Country-Level Impact Analysis

---

## 1. Executive Summary

This document specifies a geopolitical intelligence dashboard focused on the Iran crisis, designed to surface country-level exposure rather than individual company risk. The core thesis: the Iran situation is fundamentally a geopolitical and macro event whose primary transmission channels run through sovereign actors, trade routes, energy flows, and military postures.

The dashboard is anchored by a reasoning-model-powered geopolitical mindmap that models the complex, multi-layered causal chains: Iranian nuclear escalation → Strait of Hormuz disruption → China oil import squeeze → SPR drawdowns → global commodity repricing. This level of causal depth exceeds what static rule-based systems can handle and requires an LLM reasoning layer.

---

## 2. Product Vision & Rationale

### 2.1 Why Countries, Not Companies

The Iran situation differs from, say, a tariff on a specific sector. Its effects propagate through sovereign-level channels: oil export dependencies, naval chokepoints, alliance structures, and military base vulnerabilities. While company-level analysis remains valuable downstream, the primary analytical frame should be: which countries are affected, how, and through what mechanism?

### 2.2 Why a Reasoning-Model Mindmap

A geopolitical mindmap of this complexity involves branching causal chains, feedback loops, and conditional scenarios (e.g., "if Strait of Hormuz is blocked AND China has less than 30 days of SPR, then..."). A standard LLM can enumerate factors but struggles with deep reasoning over interconnected nodes. The reasoning model version can:

- Trace multi-hop causal chains across domains (military → energy → trade → financial)
- Identify non-obvious second- and third-order effects
- Weight conditional probabilities across scenario branches
- Maintain coherence across a large, interconnected graph of geopolitical nodes

---

## 3. Dashboard Architecture

The dashboard is organized into five interconnected panels. Each panel can function independently but cross-links to others for drill-down context.

| Panel | Description | Key Data Sources |
|---|---|---|
| **Geopolitical Mindmap** | Interactive causal graph powered by reasoning model. Root node: Iran Escalation. Branches into military, energy, trade, financial, and diplomatic channels. | LLM reasoning engine, curated geopolitical event feeds, scenario libraries |
| **Country Exposure Matrix** | Heatmap of country-level risk across dimensions: energy dependency, trade route exposure, military proximity, diplomatic alignment, financial linkage. | IEA, EIA, UN Comtrade, military base databases, SWIFT data |
| **Maritime & Route Monitor** | Real-time and scenario-based view of critical waterways: Strait of Hormuz, Bab el-Mandeb, Suez Canal. Chokepoint closure simulation. | AIS shipping data, naval deployment feeds, Lloyd's intelligence |
| **Energy Flow Tracker** | Sankey diagram of global oil flows with Iran-origin barrels highlighted. Focus on China import dependency, SPR levels, alternative supplier capacity. | Kpler, Vortexa, IEA monthly reports, national SPR disclosures |
| **Military & Security Layer** | Map overlay of US/allied bases, Houthi/proxy attack vectors, recent strike activity, and asset redeployments. | OSINT feeds, ACLED, Pentagon releases, satellite imagery providers |

---

## 4. Geopolitical Mindmap — Deep Dive

### 4.1 Node Taxonomy

The mindmap is a directed acyclic graph (with some feedback edges) organized into five domain layers:

| Layer | Example Nodes | Connections |
|---|---|---|
| **Military / Security** | Houthi attacks on shipping, US base attacks in Iraq/Syria, Iranian proxy activation, Israeli strike scenarios | Drives route closures, insurance cost spikes, naval redeployments |
| **Energy / Commodity** | Strait of Hormuz closure, Iran oil export disruption, China crude import shortfall, SPR drawdown triggers | Feeds into trade flow disruptions, price spikes, inflationary pressure |
| **Trade / Maritime** | Suez rerouting, Bab el-Mandeb closure, container shipping delays, insurance premium surges | Cascades into supply chain disruption, port congestion, consumer prices |
| **Diplomatic / Political** | US–Iran negotiations, JCPOA status, China–Iran relationship, Gulf state neutrality, EU sanctions posture | Shapes scenario probabilities, sanctions effectiveness, coalition responses |
| **Financial / Macro** | Oil price shock, USD/CNY pressure, EM sovereign spreads, global growth downgrade, inflation expectations | End-state impact nodes that propagate to portfolio-level risk |

### 4.2 Reasoning Model Requirements

The mindmap generation engine must support the following capabilities, each of which pushes beyond standard LLM inference:

- **Multi-hop causal reasoning**: trace chains of 4–6 steps (e.g., proxy attack → route closure → rerouting → transit time increase → inventory drawdown → price spike)
- **Conditional branching**: model "if X then Y, else Z" logic across scenario trees, including probability-weighted branches
- **Cross-domain synthesis**: connect a military event to its energy, trade, and financial consequences in a single coherent graph
- **Feedback loop detection**: identify reinforcing cycles (e.g., price spike → Iran revenue increase → proxy funding → more attacks → further price spike)
- **Temporal sequencing**: model which effects are immediate (hours/days), medium-term (weeks), and structural (months/quarters)
- **Confidence scoring**: assign and propagate uncertainty through the graph

### 4.3 Interaction Model

Users interact with the mindmap via:

- **Click-to-expand**: collapsed by default, each node reveals children and causal annotations on click
- **Scenario toggle**: switch between base case, escalation, and de-escalation scenarios; the graph reconfigures accordingly
- **Country highlight**: select a country to illuminate all paths through the graph that affect it
- **Time horizon filter**: show only nodes relevant to a selected time window (1 week, 1 month, 1 quarter)

---

## 5. Country Exposure Matrix

The matrix scores each country across five risk dimensions on a 1–5 scale, producing a composite geopolitical exposure score. This replaces the company-level view for the Iran use case.

| Country | Energy Dep. | Route Exp. | Military Prox. | Diplo. Align. | Financial | Composite |
|---|---|---|---|---|---|---|
| China | 5 | 4 | 2 | 4 | 4 | 4.2 |
| India | 4 | 4 | 2 | 3 | 3 | 3.5 |
| Japan | 5 | 4 | 2 | 2 | 3 | 3.6 |
| South Korea | 5 | 4 | 2 | 2 | 3 | 3.6 |
| Saudi Arabia | 2 | 5 | 5 | 3 | 3 | 3.6 |
| UAE | 2 | 5 | 5 | 3 | 4 | 3.8 |
| Iraq | 3 | 3 | 5 | 5 | 4 | 4.0 |
| Turkey | 3 | 3 | 4 | 3 | 3 | 3.2 |
| Germany / EU | 3 | 3 | 1 | 2 | 3 | 2.6 |
| United States | 1 | 2 | 5 | 5 | 2 | 3.0 |

> Dimension key: 1 = minimal exposure, 5 = critical exposure. Scores are illustrative. Production version will use quantitative inputs (e.g., Iran crude as % of total imports) calibrated by the reasoning model.

---

## 6. Key Analytical Threads

### 6.1 China Oil Import Dependency

China imports approximately 1.5 million barrels per day of Iranian crude (roughly 10–15% of total crude imports, often via sanctioned "dark fleet" tankers). A disruption to Iranian supply — whether via tightened sanctions enforcement or physical route closure — would force China to either draw down strategic reserves, bid up alternatives from Saudi Arabia / Russia / Iraq, or accept demand destruction. The dashboard models each pathway and its cascading effects.

### 6.2 Maritime Chokepoint Risk

Three waterways are critical: the Strait of Hormuz (~20% of global oil transit), the Bab el-Mandeb Strait (gateway to Suez), and the Suez Canal itself. Houthi attacks have already demonstrated the vulnerability of the Bab el-Mandeb corridor. The dashboard simulates partial and full closure scenarios, computing rerouting costs, transit time additions, and insurance premium impacts.

### 6.3 Military Base & Proxy Attack Vectors

US military installations in Iraq, Syria, Bahrain, and Qatar face ongoing attack risk from Iranian-backed groups. The dashboard tracks attack frequency, severity escalation, and response patterns. Each base is mapped with its strategic function (logistics hub, naval command, air operations) so users can assess which capabilities are degraded by sustained attack.

### 6.4 Sanctions & Financial Plumbing

Secondary sanctions risk for countries and institutions transacting with Iran. The dashboard monitors SWIFT message volumes, sanctions designations, and enforcement actions to assess which countries have financial exposure that could be disrupted by a US sanctions escalation.

---

## 7. Technical Requirements

### 7.1 Reasoning Model Integration

The geopolitical mindmap requires a reasoning-capable LLM (e.g., Claude with extended thinking, or equivalent) that can:

- Accept a structured prompt containing current geopolitical state, scenario parameters, and node taxonomy
- Output a structured graph embedded as `GROUNDED_DATA.mindmapNodes` in the JSX artifact (JSON-compatible object with nodes, edges, causal annotations, confidence scores, and temporal tags)
- Support iterative refinement: user can challenge or modify a node, and the model re-reasons over the affected subgraph
- Operate within acceptable latency for interactive use (target: initial graph generation under 30 seconds, subgraph updates under 10 seconds)

### 7.2 Data Pipeline

Real-time and near-real-time ingestion from:

- Geopolitical event feeds (curated news, OSINT, wire services) via `bigdata_search`
- Energy market data (crude prices, tanker tracking, SPR levels, refinery runs) via `bigdata_search`
- Country macro indicators (GDP, CPI, trade balances) via `bigdata_country_tearsheet`
- Military activity feeds (ACLED, government releases) via `bigdata_search`
- Financial data (sovereign CDS spreads, FX, commodity futures) via `bigdata_search`
- Event timeline enrichment via `bigdata_events_calendar`

### 7.3 Frontend

The generated JSX component must be:

- **Self-contained**: single `.jsx` file with all data embedded as constants
- **Inline-styled**: uses the `COLORS` object, no external stylesheet dependencies
- **React-only**: no external UI library dependencies beyond React itself
- Graph visualization for mindmap: current panel uses expandable card taxonomy; future iterations may adopt D3.js force graph, Cytoscape.js, or React Flow for true graph rendering

---

## 8. Scope & Complexity Assessment

| Workstream | Complexity | Dependencies | Notes |
|---|---|---|---|
| Reasoning model mindmap engine | Very High | Reasoning model API, prompt engineering | Core differentiator. Start here. |
| Country exposure scoring model | High | Multi-source data pipeline | Can use static data for MVP. |
| Maritime route simulation | High | AIS data provider, route algorithms | Scenario engine is key value. |
| Energy flow Sankey diagram | Medium | Kpler / Vortexa integration | Visualization is standard; data is the challenge. |
| Military activity overlay | Medium | ACLED + OSINT feeds | Map layer with event markers. |
| Frontend graph visualization | High | D3/Cytoscape, UX design | Critical for mindmap usability. |
| Cross-panel linking | Medium | Shared state management | Country click → filter all panels. |

---

## 9. MVP Strategy

### Phase 1 — Reasoning Mindmap Prototype (Weeks 1–4)

- Build the reasoning model prompt + graph output pipeline
- Static geopolitical state input (manually curated, not real-time)
- Basic graph visualization with expand/collapse and scenario toggle
- Validate that the reasoning model produces genuinely insightful multi-hop analysis

### Phase 2 — Country Matrix + Energy Flows (Weeks 5–8)

- Country exposure heatmap with quantitative scoring
- Energy flow Sankey diagram (China focus)
- Cross-panel linking (click country → highlight in mindmap)

### Phase 3 — Live Data + Maritime Layer (Weeks 9–12)

- Real-time event feed integration
- Maritime route monitor with chokepoint simulation
- Military activity overlay
- Automated mindmap refresh when significant events are detected

---

## 10. Open Questions

- **Reasoning model selection**: Which model/endpoint provides the best balance of depth, structured output, and latency? Need to benchmark Claude extended thinking vs. alternatives on geopolitical reasoning tasks.
- **Company layer integration**: When and how to add company-level drill-down? Likely as an optional layer activated from the country matrix (e.g., click China → see Chinese refiners exposed to Iranian crude).
- **Scenario authoring**: Should users be able to define custom scenarios, or is a curated set (base / escalation / de-escalation) sufficient for v1?
- **Collaboration**: Is this a single-analyst tool or does it need shared scenario workspaces and annotations?
- **Update frequency**: How often should the reasoning model re-generate the mindmap? On-demand, daily, or event-triggered? Current implementation: every update cycle.
- **Data licensing**: Several critical data sources (Kpler, ACLED, Lloyd's) have commercial licensing requirements that affect cost and scope.

---

## Grounding Requirements (Addendum for Agent)

Every number in `GROUNDED_DATA` must:

1. Trace to a specific `bigdata_search` result or `bigdata_country_tearsheet` response
2. Include a `source` field: `"Source Name, Date"` format (e.g., `"Goldman Sachs via Yahoo Finance, Mar 4 2026"`)
3. If no fresh data found, retain the previous value and set a staleness indicator

The `SourceTag` component renders the `source` field — it must always be populated, never empty.

## Design Constraints (Addendum for Agent)

- Dark crisis-monitor aesthetic: deep navy/slate background (`#0a0e1a`), high-contrast red/amber/blue accent system
- Typography: Inter for all display/body/UI text; JetBrains Mono for all numeric data
- Self-contained inline styles using the `COLORS` constant object
- No Tailwind, no external CSS libraries in the generated component
- Utility components (`Badge`, `Metric`, `SourceTag`, `CardContainer`) must be preserved with their visual contract
