# People-Centric Career Network — Bigdata.com

Trace anyone's professional career by mining their **co-mention fingerprint** across news, filings, and transcripts via the [Bigdata.com](https://bigdata.com) API.

## What it does

| Step | What happens | API endpoint |
|------|-------------|--------------|
| **1 — Entity resolution** | Co-mentions hack: search the person's name with `entity_details=True`; their own entity surfaces in the `people` results → free `rp_entity_id` | `POST /v1/search/co-mentions/entities` |
| **2 — Full history** | Pull every company ever co-mentioned with the person (2010 → today) | same |
| **3 — Period slices** | Repeat per period (yearly / quarterly / monthly) to build a chunk-count matrix (company × period) | same |
| **4 — Outputs** | Console table · heatmap · network graph · JSON export | — |

**Focus: PERSON → COMPANY only.** No people-to-people edges, no organisations.
Node size and edge width in the network graph are proportional to co-mention chunk volume.

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Bigdata.com API key  →  platform.bigdata.com/api-keys
export BIGDATA_API_KEY=your_key_here

# 3. Run
python people_network_analysis.py "Dario Amodei"
```

---

## Usage

```
python people_network_analysis.py [person] [year_start] [year_end]
                                  [--frequency {yearly,quarterly,monthly}]
                                  [--threshold FLOAT]
                                  [--top N]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `person` | `Dario Amodei` | Full name to analyse |
| `year_start` | `2010` | First year of the analysis window |
| `year_end` | current year | Last year of the analysis window |
| `--frequency` | `yearly` | Time granularity: `yearly`, `quarterly`, or `monthly` |
| `--threshold` | `1.0` | Min % share of total co-mention chunks for a company to appear in results |
| `--top` | `20` | Top N companies included in all outputs |

### Examples

```bash
# Default — Dario Amodei, yearly, 2010 → today
python people_network_analysis.py

# Custom person
python people_network_analysis.py "Ray Dalio"

# Custom date range
python people_network_analysis.py "Ray Dalio" 2005 2024

# Quarterly granularity
python people_network_analysis.py "Ray Dalio" 2015 2024 --frequency quarterly

# Monthly (recommend narrow ranges — each year = 12 API calls)
python people_network_analysis.py "Jensen Huang" 2023 2025 --frequency monthly

# Stricter noise filter — only companies with ≥ 2.5% share
python people_network_analysis.py "Dario Amodei" --threshold 2.5

# Combine options
python people_network_analysis.py "Ray Dalio" --frequency quarterly --threshold 0.5
```

> **API call volume:** finer frequencies multiply calls per year (quarterly = 4×, monthly = 12×). For monthly analysis, keep the year range to 2–3 years.

---

## Outputs

All output files are written to the current directory, prefixed by the person's name slug (e.g. `dario_amodei_`).

| File | Description |
|------|-------------|
| `<slug>_heatmap.png` | Companies × periods heatmap — darker = more co-mentions |
| `<slug>_network.png` | Bipartite network graph — node size & edge width ∝ volume |
| `<slug>_career.json` | Structured JSON with per-company period breakdown |

### Example console output

```
=======================================================================
CAREER PATH — Dario Amodei
=======================================================================
Company                              First       Last   Total  Activity
-----------------------------------------------------------------------
Anthropic                             2021       2026   12430  ▁▂▄▆▇█
OpenAI                                2018       2021    4210  ▃▅▇▄▁
Google                                2015       2020    1890  ▁▂▃▂▁
=======================================================================
Coverage: 16 periods × 12 companies
```

### JSON structure

```json
{
  "person": "Dario Amodei",
  "entity_id": "RPE_XXX",
  "frequency": "yearly",
  "generated": "2026-03-18T...",
  "source": "Bigdata.com co-mentions API",
  "companies": [
    {
      "company": "Anthropic",
      "first_period": "2021",
      "last_period":  "2026",
      "total_chunks": 12430,
      "active_periods": 5,
      "period_chunks": {"2021": 800, "2022": 2100, ...}
    }
  ]
}
```

---

## File structure

```
career_network/
├── people_network_analysis.py   # standalone CLI script (run this)
├── requirements.txt             # Python dependencies
└── README.md                    # this file
```

---

## Extending the analysis

| Goal | How |
|------|-----|
| Different person | `python people_network_analysis.py "Jensen Huang"` |
| Wider history | Pass `2000` as `year_start` |
| More companies | `--top 30` |
| Finer granularity | `--frequency quarterly` or `--frequency monthly` |
| Reduce noise | `--threshold 2.0` (only companies with ≥ 2% share) |
| Batch multiple people | Loop `analyze_career_path()` when importing as a module |

---

## API reference

- Docs: [docs.bigdata.com/api-reference](https://docs.bigdata.com/api-reference/)
- Co-mentions: [docs.bigdata.com/api-reference/search/get-co-mentions](https://docs.bigdata.com/api-reference/search/get-co-mentions)
- Knowledge Graph: [docs.bigdata.com/api-reference/knowledge-graph](https://docs.bigdata.com/api-reference/knowledge-graph)
