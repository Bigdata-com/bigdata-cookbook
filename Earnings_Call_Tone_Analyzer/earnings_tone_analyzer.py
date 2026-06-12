"""
Earnings Call Management Tone Analyzer
Uses Bigdata.com for transcript retrieval and OpenAI GPT for tone scoring.
Produces a standardized sentiment scorecard with Q/Q and Y/Y comparisons.

Async pipeline with rate limiting, semaphore-based concurrency, exponential
backoff and retries for both Bigdata.com and OpenAI endpoints.
"""

import asyncio
import os
import json
import re
import csv
import time
import random
import logging
import functools
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

from bigdata_client import Bigdata
from bigdata_client.query import ReportingEntity, TranscriptTypes, All
from bigdata_client.models.search import DocumentType, SortBy

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

BIGDATA_API_KEY = os.environ["BIGDATA_API_KEY"]
OPENAI_KEY = os.environ["OPENAI_API_KEY"]

MODEL = "gpt-5.4-nano"

# ── Rate-limit / concurrency knobs ──────────────────────────────────────────
BIGDATA_RPM = 500            # Bigdata: 500 queries/min
BIGDATA_MAX_CONCURRENT = 15  # parallel Bigdata calls
OPENAI_MAX_CONCURRENT = 20   # parallel OpenAI calls
COMPANY_MAX_CONCURRENT = 10  # companies processed in parallel
MAX_RETRIES = 5
BACKOFF_BASE = 1.0           # first retry after ~1 s
BACKOFF_MAX = 60.0

# ── Prompts ─────────────────────────────────────────────────────────────────
TONE_PROMPT = """You are a senior equity analyst specializing in NLP-based earnings call analysis.

Analyze the following earnings call transcript for management tone using the Loughran-McDonald financial sentiment dictionary and tone weighting. Focus on:
- Word choice (positive/negative financial terms)
- Forward-looking language strength
- Hedging and uncertainty language
- Confidence indicators (definitive vs tentative)
- Guidance tone (raising/maintaining/lowering)

Company: {company_name} ({ticker})
Quarter: {quarter_label}

TRANSCRIPT:
{transcript_text}

Respond with ONLY a valid JSON object (no markdown, no code fences):
{{
  "quarter_label": "{quarter_label}",
  "tone_category": "<one of: Very Bullish, Bullish, Neutral, Bearish, Very Bearish>",
  "sentiment_score": <float between -1.0 and 1.0, using Loughran-McDonald weighted scoring>,
  "key_nlp_signals": [
    "<signal 1: e.g. 'Strong use of positive forward-looking terms (accelerating, expanding, confident)'>",
    "<signal 2>",
    "<signal 3>"
  ],
  "investor_insights": [
    "<insight 1: actionable observation for investors>",
    "<insight 2>"
  ],
  "summary": "<2-3 sentence summary of management tone>"
}}
"""

COMPARISON_PROMPT = """You are a senior equity analyst. Compare two earnings call tone assessments for the same company.

Company: {company_name} ({ticker})
Current Quarter: {current_label}
Comparison Quarter: {comparison_label}
Comparison Type: {comparison_type}

Current quarter analysis:
{current_json}

Comparison quarter analysis:
{comparison_json}

Calculate the delta in sentiment score and classify the change.

Scoring rubric for change classification:
- 0.00 – 0.05: Same / Slight change
- 0.05 – 0.10: Slightly More Bullish/Bearish
- 0.10 – 0.18: Notably More Bullish/Bearish
- >0.18: Significantly More Bullish/Bearish

If the score moved higher → "More Bullish" direction.
If the score moved lower → "More Bearish" direction.
If within 0.00-0.05, indicate "Same" and note if prior was Bullish or Bearish.

Respond with ONLY a valid JSON object (no markdown, no code fences):
{{
  "comparison_type": "{comparison_type}",
  "current_score": <float>,
  "comparison_score": <float>,
  "delta": <float, current minus comparison>,
  "abs_delta": <float>,
  "change_category": "<e.g. 'Notably More Bullish', 'Same (Bullish)', 'Slightly More Bearish', 'Significantly More Bullish'>",
  "explanation": "<1-2 sentences explaining the shift>"
}}
"""


# ── Rate limiter (token-bucket) ─────────────────────────────────────────────

class RateLimiter:
    """Async token-bucket rate limiter."""

    def __init__(self, rpm: float):
        self._interval = 60.0 / rpm   # seconds between tokens
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last + self._interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


# ── Retry with exponential backoff ──────────────────────────────────────────

async def retry_async(coro_fn, *args, label: str = "",
                      max_retries: int = MAX_RETRIES,
                      backoff_base: float = BACKOFF_BASE,
                      backoff_max: float = BACKOFF_MAX,
                      retryable=None):
    """Call an async function with exponential backoff + jitter.

    `retryable` is an optional callable(exception) -> bool to decide whether
    the error is transient.  Defaults to retrying on everything except
    KeyboardInterrupt and SystemExit.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return await coro_fn(*args)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            if retryable and not retryable(exc):
                raise
            if attempt == max_retries:
                log.error(f"[{label}] failed after {max_retries} attempts: {exc}")
                raise
            delay = min(backoff_base * (2 ** (attempt - 1)), backoff_max)
            jitter = delay * random.uniform(0.5, 1.0)
            log.warning(f"[{label}] attempt {attempt}/{max_retries} failed "
                        f"({type(exc).__name__}), retrying in {jitter:.1f}s")
            await asyncio.sleep(jitter)


def _is_retryable_openai(exc):
    from openai import RateLimitError, APITimeoutError, APIConnectionError, InternalServerError
    return isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError))


def _is_retryable_bigdata(exc):
    msg = str(exc).lower()
    return any(k in msg for k in ("timeout", "429", "500", "502", "503", "504",
                                   "rate", "connection", "reset"))


# ── Core helpers ────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Could not parse JSON from response", raw, 0)


def parse_quarter_from_headline(headline: str) -> tuple[str | None, int | None, int | None]:
    m = re.search(r"Q(\d)\s+(\d{4})", headline)
    if m:
        q, y = int(m.group(1)), int(m.group(2))
        return f"Q{q} {y}", q, y
    m = re.search(r"(\d{4}).*?Q(\d)", headline)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        return f"Q{q} {y}", q, y
    return headline, None, None


def find_sequential_and_yoy(analyses: list, current_q: int, current_y: int):
    prev_q = current_q - 1 if current_q > 1 else 4
    prev_y = current_y if current_q > 1 else current_y - 1
    yoy_q, yoy_y = current_q, current_y - 1
    sequential = yoy = None
    for t in analyses:
        q, y = t["q"], t["y"]
        if q == prev_q and y == prev_y:
            sequential = t
        if q == yoy_q and y == yoy_y:
            yoy = t
    return sequential, yoy


def load_universe(csv_path: str, limit: int | None = None) -> list[str]:
    entity_ids = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row and row[0].strip():
                entity_ids.append(row[0].strip())
    if limit:
        entity_ids = entity_ids[:limit]
    log.info(f"Loaded {len(entity_ids)} entity IDs from {csv_path}")
    return entity_ids


# ── Async pipeline ──────────────────────────────────────────────────────────

class ToneAnalyzer:
    """Orchestrates the async pipeline with proper rate limiting."""

    # OpenAI pricing (gpt-5.4-nano)
    OAI_INPUT_COST_PER_M = 0.20   # $/1M input tokens
    OAI_OUTPUT_COST_PER_M = 1.25  # $/1M output tokens

    def __init__(self, company_concurrency: int = COMPANY_MAX_CONCURRENT):
        self.bd = Bigdata(api_key=BIGDATA_API_KEY)
        self.oai = AsyncOpenAI(api_key=OPENAI_KEY)
        self.bd_limiter = RateLimiter(BIGDATA_RPM)
        self.bd_sem = asyncio.Semaphore(BIGDATA_MAX_CONCURRENT)
        self.oai_sem = asyncio.Semaphore(OPENAI_MAX_CONCURRENT)
        self.company_sem = asyncio.Semaphore(company_concurrency)
        self._completed = 0
        self._total = 0
        # Cost tracking
        self._oai_input_tokens = 0
        self._oai_output_tokens = 0
        self._oai_calls = 0
        self._bd_calls = 0
        self._cost_lock = asyncio.Lock()
        self._lock = asyncio.Lock()

    async def _tick_progress(self, ticker: str):
        async with self._lock:
            self._completed += 1
            log.info(f"[{self._completed}/{self._total}] Done: {ticker}")

    # ── Bigdata calls (sync library → thread pool, rate-limited) ────────

    async def _bd_call(self, fn, *args, label="bigdata"):
        """Wrap a sync Bigdata SDK call with semaphore + rate limiter + retry."""
        async def _inner():
            async with self.bd_sem:
                await self.bd_limiter.acquire()
                result = await asyncio.to_thread(fn, *args)
                async with self._cost_lock:
                    self._bd_calls += 1
                return result
        return await retry_async(_inner, label=label,
                                 retryable=_is_retryable_bigdata)

    async def resolve_entity(self, entity_id: str) -> dict | None:
        try:
            results = await self._bd_call(
                self.bd.knowledge_graph.autosuggest, entity_id, 1,
                label=f"resolve:{entity_id}")
            if results:
                c = results[0]
                return {"id": c.id, "name": c.name, "ticker": c.ticker}
        except Exception as e:
            log.warning(f"Could not resolve entity {entity_id}: {e}")
        return None

    async def fetch_transcripts(self, entity_id: str) -> list:
        try:
            def _search():
                query = All([
                    ReportingEntity(entity_id),
                    TranscriptTypes.EARNINGS_CALL,
                ])
                search = self.bd.search.new(
                    query, scope=DocumentType.TRANSCRIPTS, sortby=SortBy.DATE)
                return search.run(20)
            return await self._bd_call(_search, label=f"search:{entity_id}")
        except Exception as e:
            log.warning(f"Search failed for {entity_id}: {e}")
            return []

    async def download_transcript(self, doc, max_chars: int = 80000) -> str:
        try:
            annotated = await self._bd_call(
                doc.download_annotated_dict, label=f"fetch:{doc.id[:8]}")
            body = annotated.get("content", {}).get("body", [])
            text = "\n".join(b.get("text", "") for b in body if b.get("text"))
            if len(text) > max_chars:
                text = text[:max_chars] + "\n[... transcript truncated ...]"
            return text
        except Exception as e:
            log.warning(f"Failed to download document {doc.id}: {e}")
            return "\n".join(c.text for c in doc.chunks if c.text)

    # ── OpenAI calls (async native, rate-limited) ───────────────────────

    async def _oai_call(self, messages: list, label: str = "openai",
                        max_completion_tokens: int = 1500) -> str | None:
        async def _inner():
            async with self.oai_sem:
                resp = await self.oai.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_completion_tokens=max_completion_tokens,
                )
                if resp.usage:
                    async with self._cost_lock:
                        self._oai_input_tokens += resp.usage.prompt_tokens
                        self._oai_output_tokens += resp.usage.completion_tokens
                        self._oai_calls += 1
                return resp.choices[0].message.content
        raw = await retry_async(_inner, label=label,
                                retryable=_is_retryable_openai)
        return raw

    async def analyze_tone(self, company_name: str, ticker: str,
                           quarter_label: str, transcript_text: str) -> dict | None:
        prompt = TONE_PROMPT.format(
            company_name=company_name, ticker=ticker,
            quarter_label=quarter_label, transcript_text=transcript_text,
        )
        try:
            raw = await self._oai_call(
                [{"role": "user", "content": prompt}],
                label=f"tone:{ticker}:{quarter_label}", max_completion_tokens=1500)
            return _parse_json_response(raw)
        except Exception as e:
            log.error(f"Tone analysis failed for {ticker} {quarter_label}: {e}")
            return None

    async def compare_quarters(self, company_name: str, ticker: str,
                               current: dict, comparison: dict,
                               comparison_type: str) -> dict | None:
        prompt = COMPARISON_PROMPT.format(
            company_name=company_name, ticker=ticker,
            current_label=current["quarter_label"],
            comparison_label=comparison["quarter_label"],
            comparison_type=comparison_type,
            current_json=json.dumps(current, indent=2),
            comparison_json=json.dumps(comparison, indent=2),
        )
        try:
            raw = await self._oai_call(
                [{"role": "user", "content": prompt}],
                label=f"cmp:{ticker}:{comparison_type}", max_completion_tokens=800)
            return _parse_json_response(raw)
        except Exception as e:
            log.error(f"Comparison failed for {ticker}: {e}")
            return None

    # ── Per-company pipeline ────────────────────────────────────────────

    async def process_company(self, entity_id: str) -> dict | None:
        async with self.company_sem:
            return await self._process_company_inner(entity_id)

    async def _process_company_inner(self, entity_id: str) -> dict | None:
        info = await self.resolve_entity(entity_id)
        if not info:
            log.warning(f"Skipping unresolved entity: {entity_id}")
            await self._tick_progress(entity_id)
            return None

        company_name = info["name"]
        ticker = info["ticker"] or entity_id

        docs = await self.fetch_transcripts(entity_id)
        if not docs:
            log.warning(f"No transcripts found for {ticker}")
            await self._tick_progress(ticker)
            return None

        # De-duplicate by quarter
        seen = set()
        unique_docs = []
        for d in docs:
            ql, q, y = parse_quarter_from_headline(d.headline)
            key = (q, y) if q and y else d.id
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        if not unique_docs:
            await self._tick_progress(ticker)
            return None

        # Download transcripts in parallel (up to 6 quarters)
        to_process = unique_docs[:6]
        download_tasks = [self.download_transcript(d) for d in to_process]
        texts = await asyncio.gather(*download_tasks, return_exceptions=True)

        # Analyze tones in parallel
        tone_tasks = []
        tone_meta = []
        for d, text in zip(to_process, texts):
            if isinstance(text, Exception) or not text or len(text) < 500:
                continue
            ql, q, y = parse_quarter_from_headline(d.headline)
            tone_tasks.append(self.analyze_tone(company_name, ticker, ql, text))
            tone_meta.append((ql, q, y, d.id, str(d.timestamp)))

        if not tone_tasks:
            await self._tick_progress(ticker)
            return None

        tone_results = await asyncio.gather(*tone_tasks, return_exceptions=True)

        analyses = []
        for tone, (ql, q, y, doc_id, ts) in zip(tone_results, tone_meta):
            if isinstance(tone, Exception) or tone is None:
                continue
            tone["q"] = q
            tone["y"] = y
            tone["doc_id"] = doc_id
            tone["timestamp"] = ts
            analyses.append(tone)

        if not analyses:
            await self._tick_progress(ticker)
            return None

        current = analyses[0]
        result = {
            "entity_id": entity_id,
            "company_name": company_name,
            "ticker": ticker,
            "current_quarter": current["quarter_label"],
            "earnings_date": current.get("timestamp", "")[:10],
            "tone_category": current["tone_category"],
            "sentiment_score": current["sentiment_score"],
            "key_nlp_signals": "; ".join(current.get("key_nlp_signals", [])),
            "investor_insights": "; ".join(current.get("investor_insights", [])),
            "summary": current.get("summary", ""),
        }

        # Q/Q and Y/Y comparisons (parallel)
        if current.get("q") and current.get("y") and len(analyses) > 1:
            seq, yoy = find_sequential_and_yoy(analyses, current["q"], current["y"])
            cmp_tasks = {}
            if seq:
                cmp_tasks["qq"] = self.compare_quarters(
                    company_name, ticker, current, seq, "Q/Q Sequential")
            if yoy:
                cmp_tasks["yoy"] = self.compare_quarters(
                    company_name, ticker, current, yoy, "Y/Y Year-over-Year")

            if cmp_tasks:
                keys = list(cmp_tasks.keys())
                cmp_results = await asyncio.gather(
                    *cmp_tasks.values(), return_exceptions=True)
                for k, r in zip(keys, cmp_results):
                    if isinstance(r, Exception) or r is None:
                        continue
                    prefix = k
                    result[f"{prefix}_delta"] = r.get("delta")
                    result[f"{prefix}_change"] = r.get("change_category")
                    result[f"{prefix}_explanation"] = r.get("explanation")
                    result[f"{prefix}_prior_score"] = r.get("comparison_score")

        await self._tick_progress(ticker)
        return result

    # ── Orchestrator ────────────────────────────────────────────────────

    async def run(self, entity_ids: list[str]) -> list[dict]:
        self._total = len(entity_ids)
        self._completed = 0
        start = time.time()
        log.info(f"Starting analysis of {self._total} companies "
                 f"(concurrency: {COMPANY_MAX_CONCURRENT} companies, "
                 f"{BIGDATA_MAX_CONCURRENT} bigdata, "
                 f"{OPENAI_MAX_CONCURRENT} openai)")

        tasks = [self.process_company(eid) for eid in entity_ids]
        raw = await asyncio.gather(*tasks, return_exceptions=True)

        results = [r for r in raw if isinstance(r, dict)]
        elapsed = time.time() - start
        n = max(len(results), 1)

        oai_cost = (self._oai_input_tokens * self.OAI_INPUT_COST_PER_M
                    + self._oai_output_tokens * self.OAI_OUTPUT_COST_PER_M) / 1_000_000

        log.info(f"Completed {len(results)}/{self._total} companies "
                 f"in {elapsed:.1f}s ({elapsed/max(len(entity_ids),1):.1f}s/company)")
        log.info(f"── Cost summary ──")
        log.info(f"  Bigdata.com API calls: {self._bd_calls} total, "
                 f"~{self._bd_calls/n:.1f}/company")
        log.info(f"  OpenAI API calls:      {self._oai_calls} total, "
                 f"~{self._oai_calls/n:.1f}/company")
        log.info(f"  OpenAI tokens:         {self._oai_input_tokens:,} input + "
                 f"{self._oai_output_tokens:,} output = "
                 f"{self._oai_input_tokens + self._oai_output_tokens:,} total")
        log.info(f"  OpenAI cost:           ${oai_cost:.4f} total, "
                 f"${oai_cost/n:.4f}/company")
        log.info(f"  Est. 1000 companies:   ${oai_cost/n * 1000:.2f} OpenAI, "
                 f"~{self._bd_calls/n * 1000:.0f} Bigdata calls")

        return results


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Earnings Tone Analyzer (async)")
    parser.add_argument("--universe", default="us_top1000.csv",
                        help="CSV with entity IDs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of companies")
    parser.add_argument("--output", default="tone_scores.csv",
                        help="Output CSV path")
    parser.add_argument("--entities", nargs="*",
                        help="Specific entity IDs to process")
    parser.add_argument("--concurrency", type=int, default=COMPANY_MAX_CONCURRENT,
                        help=f"Max parallel companies (default: {COMPANY_MAX_CONCURRENT})")
    args = parser.parse_args()

    company_concurrency = args.concurrency

    if args.entities:
        entity_ids = args.entities
    else:
        csv_path = Path(__file__).parent / args.universe
        entity_ids = load_universe(str(csv_path), limit=args.limit)

    analyzer = ToneAnalyzer(company_concurrency=company_concurrency)
    results = asyncio.run(analyzer.run(entity_ids))

    if results:
        df = pd.DataFrame(results)
        output_path = Path(__file__).parent / args.output
        df.to_csv(output_path, index=False)
        log.info(f"Results saved to {output_path}")

        print("\n" + "=" * 100)
        print("EARNINGS TONE SCORECARD")
        print("=" * 100)
        cols = ["ticker", "current_quarter", "earnings_date", "tone_category",
                "sentiment_score", "qq_change", "qq_delta", "yoy_change", "yoy_delta"]
        display_cols = [c for c in cols if c in df.columns]
        print(df[display_cols].to_string(index=False))
        print("=" * 100)
    else:
        log.warning("No results produced.")


if __name__ == "__main__":
    main()
