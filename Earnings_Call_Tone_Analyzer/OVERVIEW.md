# Earnings Call Tone Analyzer — Overview

## What this delivers

A fully automated pipeline that reads the latest earnings call transcripts for any equity universe, scores management tone on a standardized scale, and tracks how that tone is shifting — both quarter-over-quarter and year-over-year.

The output is a single CSV with one row per company: a sentiment score, a categorical tone label, directional change flags, and a short narrative summary. It is designed to plug directly into existing investment workflows as an additional signal layer.

## Why it matters

Management tone on earnings calls carries forward-looking information that financial statements alone do not capture. Subtle shifts in language — more hedging, stronger conviction, softer guidance — often precede changes in fundamentals. Doing this manually is feasible for a concentrated portfolio but does not scale across hundreds of names.

This tool standardizes that analysis. Every company is evaluated against the same framework, using the same rubric, producing comparable scores. It removes the subjectivity of reading transcripts one at a time and makes tone a trackable, sortable, screenable metric.

## How it works

1. For each company, the system retrieves the most recent earnings call transcripts (typically 5-6 quarters of history) from Bigdata.com's transcript database.

2. Each transcript goes through a two-layer scoring process:

   **Layer 1 — Deterministic word counting.** The transcript is scanned against the Loughran-McDonald Master Dictionary (~4,400 classified words), the standard academic lexicon for financial text analysis. This produces hard statistics: counts of positive, negative, uncertainty, litigious, constraining, strong modal, and weak modal words, plus derived ratios (net tone, uncertainty ratio, etc.). These numbers are objective, reproducible, and auditable — the same transcript always produces the same counts.

   **Layer 2 — LLM interpretation.** The transcript and the word-count statistics are passed together to a language model. The model's job is not to redo the counting but to interpret context that raw word counts miss: "revenue declined" in a risk disclaimer is different from "revenue declined" in the CEO's opening remarks. The model evaluates forward-looking language strength, hedging patterns, confidence indicators, and guidance direction — dimensions where context and sentence structure matter more than individual word frequencies.

   This hybrid approach gives the best of both: a quantitative anchor that doesn't drift between runs, plus contextual judgment that a dictionary alone cannot provide.

3. The output for each quarter is a sentiment score normalized to a [-1, +1] scale and a categorical label (Very Bullish through Very Bearish).

4. The current quarter's score is compared against the prior quarter (Q/Q) and the same quarter last year (Y/Y). Changes are classified from "Same" through "Significantly More Bullish/Bearish" based on the magnitude of the shift.

## What the output looks like

| Ticker | Quarter | Date | Tone | Score | Q/Q | Y/Y |
|--------|---------|------|------|-------|-----|-----|
| FSLR | Q1 2026 | 2026-04-30 | Bullish | 0.42 | Notably More Bullish (+0.14) | Significantly More Bullish (+0.60) |
| AAPL | Q2 2026 | 2026-04-30 | Very Bullish | 0.62 | Same (Bullish) | Significantly More Bullish (+0.20) |
| PYPL | Q1 2026 | 2026-04-29 | Bullish | 0.23 | Notably More Bullish (+0.15) | Significantly More Bearish (-0.19) |

Each row also includes key NLP signals (the specific linguistic patterns driving the score), investor insights, and a short narrative summary.

## Scale and cost

The pipeline processes roughly 1,000 companies in under 45 minutes. LLM cost for the full Russell 1000 is approximately $15 per run. It is designed to run on a recurring basis — after each earnings season, or on demand when evaluating new positions.

## Validation: why the hybrid approach matters

We compared the hybrid scoring against a pure LLM-only baseline across 100 companies. Without the Loughran-McDonald anchor, the LLM exhibited a strong bullish bias — nearly 40% of companies were rated "Very Bullish" with an average score of 0.43. Earnings call transcripts are inherently optimistic (management controls the narrative), and the LLM tended to take that language at face value.

The hybrid approach corrects for this. The deterministic word counts ground the model in what the text actually contains — how many negative, uncertainty, and hedging words appear relative to positive ones. The result is a more realistic score distribution (average 0.23) with better differentiation between companies. Importantly, the relative ranking is preserved (correlation ~0.72): companies that genuinely sound more bullish still score higher. The improvement is in calibration, not ordering.

## Current scope and next steps

The tool currently covers the Russell 1000 universe using entity IDs from Bigdata.com. The natural extension is MSCI ACWI coverage, which requires only a broader universe file — the pipeline and scoring framework remain the same.
