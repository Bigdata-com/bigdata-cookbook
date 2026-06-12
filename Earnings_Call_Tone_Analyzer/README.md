# Earnings Call Tone Analyzer

Automated management tone scoring from earnings call transcripts. Uses [Bigdata.com](https://bigdata.com) for transcript retrieval and OpenAI (`gpt-5.4-nano`) for NLP-based sentiment analysis.

Built for portfolio managers and equity analysts who want a standardized, repeatable read on management tone across a large universe — currently the Russell 1000, with the ambition to scale to MSCI ACWI.

## How it works — step by step

### Step 1: Load the universe

The script reads a CSV of [Bigdata.com](https://bigdata.com) entity IDs (e.g. `us_top1000.csv`). Each row is one company. You can also pass specific entity IDs via `--entities`.

### Step 2: Resolve entity metadata

For each entity ID, the script calls the Bigdata.com Knowledge Graph (`autosuggest`) to resolve the company name and ticker. This ensures we have the correct mapping even if the CSV only contains raw IDs.

### Step 3: Search for earnings call transcripts

Using the Bigdata.com Search API, the script queries for documents where:
- The company is the **reporting entity** (i.e. it's their earnings call, not a mention)
- The document type is **EARNINGS_CALL** (filters out investor days, M&A calls, etc.)
- Results are sorted by date (most recent first)

Up to 20 raw results are fetched to account for duplicates (the same call may appear from multiple transcript providers like FactSet, Quartr, etc.).

### Step 4: De-duplicate transcripts

Multiple providers often cover the same earnings call. The script de-duplicates by parsing the quarter from the headline (e.g. "Q1 2026") and keeping only the first document per unique quarter. This yields ~5-6 unique quarters of coverage.

### Step 5: Download full transcript text

For each unique transcript, the script calls Bigdata.com's **Fetch Document** endpoint (`download_annotated_dict`), which returns the full annotated document including:
- Complete transcript text (all body blocks)
- Sentence-level sentiment scores
- Entity detections (speakers, companies mentioned)
- Event metadata (participants, reporting period)

The full text is extracted from the body blocks and capped at 80,000 characters to stay within LLM context limits.

### Step 6: Score management tone via LLM

Each transcript goes through a two-layer hybrid scoring process:

**Layer 1 — Deterministic Loughran-McDonald word counting.** The transcript is scanned against the [Loughran-McDonald Master Dictionary](https://sraf.nd.edu/loughranmcdonald-master-dictionary/) (~4,400 classified words), the standard academic lexicon for financial text analysis. This produces hard statistics:
- Counts of **positive**, **negative**, **uncertainty**, **litigious**, **constraining**, **strong modal**, and **weak modal** words
- **Net tone** ratio: `(positive - negative) / total_words`
- **Uncertainty ratio**: `uncertainty / total_words`
- **Modal ratio**: `(strong_modal - weak_modal) / (strong_modal + weak_modal)`

These numbers are objective, reproducible, and auditable — the same transcript always produces the same counts.

**Layer 2 — LLM contextual interpretation.** The transcript and the Layer 1 word-count statistics are passed together to OpenAI `gpt-5.4-nano`. The model's job is not to redo the counting but to interpret context that raw word counts miss:
1. **Contextual sentiment** — "revenue declined" in a risk disclaimer is different from "revenue declined" in the CEO's opening remarks
2. **Forward-looking language** strength (accelerating, expanding vs. cautious, uncertain)
3. **Hedging and uncertainty** patterns beyond single-word detection
4. **Confidence indicators** (definitive vs. tentative statements)
5. **Guidance tone** (raising, maintaining, or lowering)

The L-M statistics serve as a quantitative anchor — the LLM uses them as a starting point and adjusts based on contextual interpretation. This hybrid approach gives the best of both: a deterministic baseline that doesn't drift between runs, plus contextual judgment that a dictionary alone cannot provide.

The model returns a structured JSON with:
- **Tone category**: one of Very Bullish / Bullish / Neutral / Bearish / Very Bearish
- **Sentiment score**: a normalized float in [-1, +1]
- **Key NLP signals**: the most salient linguistic patterns detected
- **Investor insights**: actionable takeaways
- **Summary**: 2-3 sentence narrative

The response is forced into `json_object` mode for reliable parsing.

### Step 7: Compare Q/Q and Y/Y

Once all quarters are scored, the script identifies the **sequential** (prior quarter) and **year-ago** (same quarter last year) transcripts and runs a comparison prompt. The LLM calculates the delta in sentiment score and classifies the magnitude of change:

| Absolute Delta | Classification |
|----------------|----------------|
| 0.00 – 0.05 | Same / Slight change |
| 0.05 – 0.10 | Slightly More Bullish/Bearish |
| 0.10 – 0.18 | Notably More Bullish/Bearish |
| > 0.18 | Significantly More Bullish/Bearish |

When the change is within 0.00-0.05 ("Same"), the output also notes whether the prior tone was Bullish or Bearish (e.g. "Same (Bullish)").

### Step 8: Output

Results are written to a CSV (one row per company) and a summary scorecard is printed to the terminal. A cost summary with Bigdata API calls, OpenAI token usage, and dollar cost is logged at the end.

## Async pipeline & rate limiting

The entire pipeline runs asynchronously to maximize throughput:

- **Company-level concurrency**: up to 10 companies processed in parallel (configurable via `--concurrency`)
- **Bigdata.com**: token-bucket rate limiter at 500 requests/minute, semaphore capped at 15 concurrent calls
- **OpenAI**: semaphore capped at 20 concurrent calls
- **Retries**: all API calls use exponential backoff with jitter (up to 5 retries), with endpoint-specific retryable error detection (429s, 5xx, timeouts)

Within each company, transcript downloads and tone analyses run in parallel (all 6 quarters scored concurrently), and Q/Q + Y/Y comparisons also run in parallel.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install bigdata-client openai python-dotenv pandas
```

Create a `.env` file:

```
BIGDATA_API_KEY=your_bigdata_api_key
OPENAI_API_KEY=your_openai_api_key
```

Get your Bigdata API key at [platform.bigdata.com/api-keys](https://platform.bigdata.com/api-keys).

## Usage

```bash
# Single company by entity ID
.venv/bin/python3 earnings_tone_analyzer.py --entities 061366

# Small sample (first N from universe CSV)
.venv/bin/python3 earnings_tone_analyzer.py --limit 20 --output top20.csv

# Full Russell 1000
.venv/bin/python3 earnings_tone_analyzer.py --output full_universe.csv

# Custom concurrency (default: 10 parallel companies)
.venv/bin/python3 earnings_tone_analyzer.py --limit 100 --concurrency 15
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--universe` | `us_top1000.csv` | CSV with entity IDs (one column, header row) |
| `--limit` | all | Process only the first N companies |
| `--output` | `tone_scores.csv` | Output CSV path |
| `--entities` | — | Specific Bigdata.com entity IDs to process |
| `--concurrency` | 10 | Max companies processed in parallel |

## Output columns

| Column | Description |
|--------|-------------|
| `entity_id` | Bigdata.com entity ID |
| `company_name` | Resolved company name |
| `ticker` | Stock ticker |
| `current_quarter` | Latest earnings quarter (e.g. Q1 2026) |
| `earnings_date` | Date of the earnings call (YYYY-MM-DD) |
| `tone_category` | Very Bullish / Bullish / Neutral / Bearish / Very Bearish |
| `sentiment_score` | Float in [-1, +1] (Loughran-McDonald weighted) |
| `key_nlp_signals` | Semicolon-separated NLP signals |
| `investor_insights` | Semicolon-separated actionable insights |
| `summary` | 2-3 sentence tone summary |
| `qq_delta` | Score change vs prior quarter |
| `qq_change` | Q/Q change classification |
| `qq_prior_score` | Prior quarter's sentiment score |
| `yoy_delta` | Score change vs year-ago quarter |
| `yoy_change` | Y/Y change classification |
| `yoy_prior_score` | Year-ago quarter's sentiment score |

## Performance & cost

Measured on a 100-company sample:

| Metric | Per Company | 100 Companies | 1,000 Companies (est.) |
|--------|------------|---------------|------------------------|
| Wall time | ~2.1s | 3m 34s | ~35 min |
| Bigdata API calls | ~7.1 | 705 | ~7,100 |
| OpenAI API calls | ~6.9 | 687 | ~6,900 |
| OpenAI input tokens | ~54,700 | 5.4M | ~54.7M |
| OpenAI output tokens | ~2,500 | 246K | ~2.5M |
| OpenAI cost | **$0.014** | $1.39 | **~$14** |

## Why the hybrid approach matters

We validated the hybrid L-M + LLM scoring against a pure LLM-only baseline across a 100-company sample. The LLM-only approach exhibited a strong bullish bias — 38% of companies scored "Very Bullish" with a mean score of 0.43. The hybrid approach, anchored by deterministic word counts, produced more realistic and differentiated scores: mean 0.23, zero "Very Bullish", and a wider spread across Neutral and Bullish categories.

| Metric | LLM-only | Hybrid (L-M + LLM) |
|--------|----------|---------------------|
| Mean score | 0.43 | 0.23 |
| Very Bullish | 38% | 0% |
| Bullish | 57% | 85% |
| Neutral | 5% | 15% |

The relative ranking of companies is preserved (Pearson r = 0.72, Spearman ρ = 0.70) — the shift is a level correction, not a scramble. Companies that scored higher in the LLM-only run still score higher in the hybrid run. The L-M word counts act as a quantitative anchor that prevents the LLM from defaulting to optimistic readings when the transcript language is actually neutral or mixed.

## Data sources

- **Transcripts**: [Bigdata.com](https://bigdata.com) — earnings call transcripts via Search + Fetch Document APIs, with entity-level filtering and full annotated document retrieval
- **Tone scoring**: OpenAI `gpt-5.4-nano` with Loughran-McDonald financial sentiment framework
