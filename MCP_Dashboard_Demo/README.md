# MCP Dashboard Demo (Bigdata.com)

**Static, source-attributed research dashboard pattern** built with [Bigdata.com](https://bigdata.com)’s MCP: market tearsheet, search, and country tearsheets feed a typed **`GROUNDED_DATA`** object in [`src/dashboard.jsx`](src/dashboard.jsx); the React UI reads only that object—**no in-browser API calls, database, or ETL**.

This folder is an **open-source illustration** bundled inside [bigdata-cookbook](../README.md). For API access and MCP setup, see [Build Your Own MCP](../Build_Your_Own_MCP/) and [Bigdata.com documentation](https://docs.bigdata.com).

## Snapshot & production split

| | |
|--|--|
| First live demo | **2026-03-07** |
| This source tree frozen | **2026-03-18** (`src/dashboard.jsx` matches that refresh) |
| Ongoing deploy & refresh | **Separate production repository** (not this cookbook) |

**Historical update-cycle handoff notes** for the frozen period are in [`feedback/cycle-log.md`](feedback/cycle-log.md). **Current production logs are not published here.**

## What this demonstrates

- **MCP → schema → UI:** An automation agent (historically **Cursor** with Bigdata.com MCP; **Claude Code** or any MCP-capable agent is equivalent) refreshes **`GROUNDED_DATA`**, applies **surgical edits** to `src/dashboard.jsx`, validates with `npm run build`, then commits. CI can optionally wrap the same flow ([`docs/reference-workflows/`](docs/reference-workflows/)).
- **Provenance:** Values are tied to structured sources (e.g. document id, timestamp, URL) as produced by MCP tools—not invented in the component layer.
- **MCP tools used in the live workflow** (see [MCP reference](https://docs.bigdata.com/mcp-reference/)): `bigdata_market_tearsheet`, `bigdata_search`, `bigdata_country_tearsheet`.

## What this repo is *not*

- Not a **live** ticker: between refreshes the UI is a **fixed snapshot**.
- Not the **deployment** repo: example Fly.io / GitHub Actions YAML is **reference-only** under [`docs/reference-workflows/`](docs/reference-workflows/) (`on:` triggers commented out so they are never active from this path).

## Prerequisites

- **Node.js 20+** (matches [`Dockerfile`](Dockerfile))
- **npm**

## Local development

```bash
cd MCP_Dashboard_Demo
npm ci
npm run dev
```

Production build:

```bash
npm run build
```

Preview the static output:

```bash
npm run preview
```

## Docker

```bash
docker build -t mcp-dashboard-demo .
docker run -p 8080:8080 mcp-dashboard-demo
```

Serves the Vite build via nginx on port **8080** ([`nginx.conf`](nginx.conf)).

## Data schema

The **authoritative contract** between refresh automation and UI is the `GROUNDED_DATA` object in [`src/dashboard.jsx`](src/dashboard.jsx) (sections such as `energyMarkets`, `goldmanAnalysis`, `hormuz`, `dualChokepoint`, `countries`, `chinaDeep`, `timeline`, `mindmapNodes`, plus `sources` arrays). When replicating on a new topic, define your own schema and mapping from MCP responses **in that file** (or a generated module)—the React components expect the shape already present there.

## Replicating on a new topic

1. Design a typed **`GROUNDED_DATA`** shape for your panels.
2. Run parallel **bigdata_search** (and tearsheet) queries aligned to that schema.
3. Map MCP responses into the object; mark gaps explicitly (e.g. stale markers)—do not silently reuse prior-cycle values in production automation.
4. Adjust JSX panels to read the new fields.
5. Host and automate refreshes from your **own** repository; copy [`docs/reference-workflows/`](docs/reference-workflows/) there if you use GitHub Actions.

## Disclaimer

This content is for **informational and technical illustration** only. It does not constitute investment advice. Figures reflect inputs available at the stated refresh time; citations are for verification and do not imply endorsement.

## License

Use of this example is subject to the [root repository license](../LICENSE).
