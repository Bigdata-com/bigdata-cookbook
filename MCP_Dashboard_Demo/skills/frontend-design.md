# Frontend Design Skill — Iran Geopolitical Risk Dashboard

## Design Philosophy

This dashboard is a **crisis intelligence terminal**, not a corporate analytics product. Every visual decision should reinforce that: high-stakes, real-time, authoritative. The aesthetic must feel like something a serious geopolitical analyst would trust — not a generic SaaS dashboard.

**Core aesthetic direction**: Dark editorial crisis-monitor. Stark, purposeful, dense with meaning.

---

## Typography

### Fonts
- **Inter** is the primary font for all display, heading, body, and UI text.
- **JetBrains Mono** is mandatory for ALL numeric data, prices, percentages, dates in data cells, and metric values.
- Both fonts are loaded via Google Fonts in `index.html`.

### Implementation
Fonts are referenced inline via `fontFamily` style props. There is no separate `FONTS` constant — just use the font-family strings directly:

```js
fontFamily: "'Inter', -apple-system, sans-serif"   // display, body, UI text
fontFamily: "'JetBrains Mono', monospace"           // numeric data, prices, dates
```

The Google Fonts link in `index.html`:
```html
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700;800&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
```

### Hierarchy
- Dashboard title: Inter, 18px, weight 800, letter-spacing -0.3
- Panel headers: Inter, 14px, weight 700, uppercase, letter-spacing 1.2
- Metric values: JetBrains Mono, 22px, weight 800, letter-spacing -0.5
- Metric labels: Inter, 11px, weight 600, uppercase, letter-spacing 1, `textDim` color
- Body text: Inter, 12px, regular weight
- Metadata/sources: Inter, 10px, italic, `textDim` color
- Badge text: Inter, 11px, weight 700, uppercase, letter-spacing 0.5

### Font Loading
Fonts are loaded via Google Fonts in `index.html` only. Do **NOT** add a Google Fonts `<link>` tag inside the JSX component — this causes double network requests and a flash of unstyled text.

---

## Size Budget (Ceilings — Never Exceed)

These are hard limits. The agent must not exceed them during regeneration.

| Element | Max Size |
|---|---|
| Metric values | `fontSize: 22` |
| Metric labels | `fontSize: 11` |
| Body/detail text | `fontSize: 12` |
| Card padding | `"20px 24px"` |
| Alert/info box padding | `10–12px` |
| Panel grid gap | `12–16px` |
| Badge font size | `11px` |
| Source tag font size | `10px` |

**Metric value max length: 15 characters.** The `value` prop of `Metric` must be a short numeric string (price, percentage, count). Never put sentences, source names, or multi-clause text into `value`. Use `sub` for context and `SourceTag` for attribution.

---

## Color System

Use this exact 21-key `COLORS` object. Do not add extra keys or change values — continuity across regen cycles depends on stability.

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

### Semantic Meaning
- `accent` / `critical` — danger, critical risk, active conflict
- `amber` / `high` — high risk, warning
- `blue` / `medium` — medium risk, info, active tab indicator
- `emerald` / `low` — low risk, positive
- `purple` — reasoning/AI layer (mindmap)
- `*Glow` variants — used as subtle tinted backgrounds (`background: COLORS.accentGlow`)

### Usage Patterns
- Tinted data cells: `background: COLORS.accentGlow` with `border: 1px solid ${COLORS.accent}22`
- Alert/info boxes: `background: ${COLORS.accent}11` with `border: 1px solid ${COLORS.accent}22`
- Alternating table rows: `background: ${COLORS.border}22` on even rows

---

## Layout Principles

### Grid
- Primary layout: CSS Grid, `gridTemplateColumns: "1fr 1fr"` for paired panels
- Full-width panels (tables, timeline, mindmap): `gridColumn: "1 / -1"`
- Gap: 16px between panels, 20–24px internal padding
- Max content width: 1300px centered

### Cards
- Cards use flat `COLORS.card` background — no gradients, no box shadows
- Border radius: 12px
- Border: `1px solid ${COLORS.border}`
- Padding: `20px 24px`

### Left Accent Strip
- Each `CardContainer` has a left accent strip: `width: 4, height: 20, borderRadius: 2`
- Color set by the `accent` prop

---

## Motion & Animation

### Pulsing Live Indicator
The live-status dot in the header must pulse using a simple opacity animation:
```css
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
```
Applied to the header dot: `animation: "pulse 2s infinite"`

### Hover States
- Buttons/tabs: border-color and color transition, `transition: "all 0.15s"`, no layout shift
- Mindmap nodes: `background` and `border` transition on expand/collapse, `transition: "all 0.2s"`

---

## Component Contracts

These utility components must be preserved exactly — they are the visual contract that the agent must maintain across regen cycles.

### `Badge({ children, color, glow })`
- Small inline label, pill shape (`borderRadius: 999`), uppercase, font-size 11px, font-weight 700, letter-spacing 0.5
- Default props: `color = COLORS.accent`, `glow = COLORS.accentGlow`
- Background is the `glow` prop, text color is the `color` prop, border is `1px solid ${color}33`
- Used for: status labels (ACTIVE CONFLICT, LIVE CRISIS, etc.)

### `Metric({ label, value, sub, color })`
- Three-line stack: label (dim, uppercase, spaced), value (mono, large, colored), sub (muted, small)
- Label: 11px, `COLORS.textDim`, uppercase, letter-spacing 1, weight 600
- Value: 22px, weight 800, `fontFamily: "'JetBrains Mono', monospace"`, letter-spacing -0.5
- Sub: 11px, `COLORS.textMuted`
- `color` prop (default `COLORS.text`) applies to the value only
- Used for: all numeric data points

### `SourceTag({ text })`
- Renders the `source` string from `GROUNDED_DATA`
- Must always be present when a data source exists
- Style: 10px, italic, `COLORS.textDim`, top border separator (`1px solid ${COLORS.border}`), paddingTop 6, marginTop 8
- **Never render an empty SourceTag** — if source is empty, omit the component

### `CardContainer({ children, title, badge, accent, style })`
- Flat dark card: `background: COLORS.card`, `borderRadius: 12`, `border: 1px solid ${COLORS.border}`, `padding: "20px 24px"`
- Header row: left accent strip (4px wide, 20px tall, rounded) + uppercase title (14px, weight 700, letter-spacing 1.2) + optional Badge
- `accent` prop (default `COLORS.blue`) sets the strip color and badge color
- `style` prop allows grid placement overrides (e.g., `gridColumn: "1 / -1"`)

### `GroundedSpan({ children, source, url })`
- Inline wrapper for any fact that needs hover attribution
- Style: `position: "relative"`, `borderBottom: "1px dotted ${COLORS.textDim}"`, `cursor: "help"`
- On hover: tooltip appears above with source name (linked if URL provided)
- Tooltip style: `COLORS.card` background, `COLORS.border` border, `borderRadius: 6`, `fontSize: 10`, `COLORS.textMuted` text, `boxShadow: "0 4px 12px rgba(0,0,0,0.4)"`
- Wrap 15–25 key data points across the dashboard (prices, Hormuz status, country risk levels). Do not wrap every token.

### `EnergyMarketsPanel`
- Replaces the old `OilPricePanel`
- **Hero metric**: Brent Crude price + 1D change with direction arrow
- **Multi-period performance table**: 5 rows (Brent, WTI, RBOB, Heating Oil, NatGas) × 6 columns (Price, 1D, 5D, 1M, 3M, YTD)
- Table style: `fontSize: 12`, JetBrains Mono for numbers, alternating row backgrounds (`${COLORS.border}22` on even rows), color-coded percentages (`COLORS.emerald` for positive, `COLORS.accent` for negative)
- Below table: WTI-Brent spread, Brent start-of-2026 reference
- **Key Drivers & News**: 3–5 headlines with detail and source attribution. Each entry: left border accent, headline bold 11px, detail 11px muted, attribution italic dim. Sourced from `bigdata_search` results (not the tearsheet)
- **Timestamp**: explicit data freshness (e.g. "Mar 16, 2026 06:03 UTC") displayed top-right next to the hero price, 10px JetBrains Mono dim
- Data source: `GROUNDED_DATA.energyMarkets` — prices from `bigdata_market_tearsheet`, drivers from `bigdata_search`

---

## Forbidden Patterns

- **No white cards, light backgrounds, or pastel colors**
- **No Tailwind classes** in the generated JSX (the component is self-contained with inline styles)
- **No external component libraries** (no MUI, no Ant Design, no shadcn)
- **No empty SourceTag components** — always populate or omit
- **No layout where every panel is the same size** — vary proportions

---

## Self-Contained JSX Model

The generated `src/dashboard.jsx` must be entirely self-contained:

1. All data in `GROUNDED_DATA` constant at the top
2. All colors in `COLORS` constant (exact values from the Color System section above)
3. All fonts referenced via `fontFamily` style prop (fonts loaded in `index.html`)
4. All animations in a `<style>` tag rendered at the bottom of the component
5. No imports beyond `{ useState }` from React
6. Single default export: `export default function IranGeopolDashboard()`

This portability means any future regeneration can produce a drop-in replacement — the entire component is one file.
