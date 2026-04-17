# Smart Batching Search

A high-performance semantic search system that reduces API queries by **67-99%** (varies by topic specificity) through intelligent company grouping and parallel execution.

This module provides a two-step system for efficient semantic search:

1. **Planning**: Organize search using smart batching and return total expected chunks
2. **Execution**: Perform search with proportional sampling to preserve distribution

Find more details in the documentation [bigdata-smart-batching](https://pypi.org/project/bigdata-smart-batching/)

## Table of Contents

1. [Key Benefits](#key-benefits)
2. [Features](#features)
3. [Installation](#installation)
   - [Option 1: Using UV (Recommended)](#option-1-using-uv-recommended)
   - [Option 2: Using pip](#option-2-using-pip)
   - [Environment Setup](#environment-setup)
4. [Quick Start](#quick-start)
   - [Using the Jupyter Notebook](#using-the-jupyter-notebook)
   - [Basic Usage (Python Script)](#basic-usage-python-script)
   - [Advanced Usage](#advanced-usage)
     - [Save and Load Plans](#save-and-load-plans)
5. [How It Works](#how-it-works)
   - [Architecture Overview](#architecture-overview)
   - [Step 1: Planning](#step-1-planning)
   - [Step 2: Execution](#step-2-execution)
6. [Naive vs Smart Batching: Performance Comparison](#naive-vs-smart-batching-performance-comparison)
   - [The Problem with Naive Search](#the-problem-with-naive-search)
   - [Smart Batching Solution](#smart-batching-solution)
   - [Real-World Benchmarks](#real-world-benchmarks)
   - [Advantages of Smart Batching](#advantages-of-smart-batching)
   - [When to Use Each Approach](#when-to-use-each-approach)
7. [Examples](#examples)
   - [Example 1: Basic Search](#example-1-basic-search)
   - [Example 2: Multiple Percentage Runs](#example-2-multiple-percentage-runs)
   - [Example 3: Production Configuration](#example-3-production-configuration)
   - [Example 4: Large-Scale Search with Error Handling](#example-4-large-scale-search-with-error-handling)
   - [Rate limit strategies](#rate-limit-strategies)
   - [Performance Tuning](#performance-tuning)
     - [Optimize for Speed](#optimize-for-speed)
     - [Optimize for API Limits](#optimize-for-api-limits)
     - [Optimize for Memory](#optimize-for-memory)
8. [Configuration](#configuration)
   - [Environment Variables](#environment-variables)
   - [Default Settings](#default-settings)
9. [Quick Reference](#quick-reference)
   - [Performance Comparison Summary](#performance-comparison-summary)
   - [When to Use Smart Batching](#when-to-use-smart-batching)
10. [License](#license)

## Key Benefits

- **67-99% Query Reduction**: Search 4,732 companies with only 17-3,699 queries (varies by topic)
  - Niche topics: Up to 99.85% reduction (e.g., "Customer Trust Erosion": 17 queries)
  - Specialized topics: 96-97% reduction (e.g., "Higher ESG Compliance Costs": 435 queries)
  - Broad topics: 32-67% reduction (e.g., "Earnings": 3,699 queries)
- **Parallel Execution**: Rate-limited concurrent requests with semaphore control
- **Proportional Sampling**: Retrieve percentage of results while preserving distribution
- **Production Ready**: Comprehensive error handling, retries, and logging
- **Scalable**: Efficiently handles universes with 10,000+ companies
- **Topic-Optimized**: Most effective for specialized, niche topics with concentrated media coverage

## Features

- **Smart Batching Planning**: Organize searches using intelligent basket creation based on chunk volumes
- **Volume-Based Period Splitting**: When a company exceeds the chunk limit per query, split the date range into sub-periods so that chunk volume is balanced across periods using the volume time series (not just equal-length time splits)
- **Proportional Sampling**: Retrieve a percentage of total chunks while preserving distribution across baskets
- **Parallel Execution**: Efficient parallel search execution with rate limiting
- **Input Validation**: Comprehensive validation of dates, percentages, and file inputs
- **Plan Persistence**: Save and load search plans for reuse with different percentages
- **Comprehensive Testing**: Unit tests, validation tests, and integration tests

## Installation

### Option 1: Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

1. Install UV (if not already installed):

```bash
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

2. Install dependencies:

```bash
uv pip install -r requirements.txt
```

This includes **[bigdata-smart-batching](https://pypi.org/project/bigdata-smart-batching/)** (`from bigdata_smart_batching import ...`).

### Option 2: Using pip

1. Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment Setup

Set up environment variables:

```bash
export BIGDATA_API_KEY="your_api_key_here"
export BIGDATA_API_BASE_URL="https://api.bigdata.com"  # Optional, defaults to this
```

Or create a `.env` file in the `Smart_Batching` directory:

```
BIGDATA_API_KEY=your_api_key_here
BIGDATA_API_BASE_URL=https://api.bigdata.com
```

**Notebook / script:** Load `.env` and set `BIGDATA_API_BASE_URL` *before* importing the module (e.g. with `python-dotenv`), as the API base URL is read at import time. If you change it, restart the kernel or process and run from the beginning.

## Quick Start

### Using the Jupyter Notebook

The easiest way to test the functions is using the provided Jupyter notebook:

```bash
jupyter notebook test_smart_batching.ipynb
```

The notebook includes:

- Step-by-step testing of all functions (plan_search, execute_search, deduplicate_documents, save_plan, load_plan, load_universe_from_csv, convert_to_dataframe)
- Configuration examples
- Results analysis
- Multiple percentage testing

Install dependencies (including `bigdata-smart-batching` from `requirements.txt`) so that `from bigdata_smart_batching import ...` works in the notebook kernel.

### Basic Usage (Python Script)

```python
from bigdata_smart_batching import plan_search, execute_search, deduplicate_documents, convert_to_dataframe

# Step 1: Plan the search
plan = plan_search(
    text="earnings revenue profit",
    universe="id_name_mapping_us_top_3000.csv",  # or us_top3000.csv in test_data/
    start_date="2023-01-01",
    end_date="2023-12-31",
    api_key="your_api_key",  # or set BIGDATA_API_KEY in env
    api_base_url="https://api.bigdata.com"  # or set BIGDATA_API_BASE_URL before importing
)

print(f"Total expected chunks: {plan['total_expected_chunks']:,}")

# Step 2: Execute search with 10% of total chunks (preserves distribution)
results_raw = execute_search(
    search_plan=plan,
    chunk_percentage=0.1,  # 10% of total chunks
    requests_per_minute=100
)

# Step 3: Deduplicate and optionally convert to DataFrame
results = deduplicate_documents(results_raw)
print(f"Retrieved {len(results)} documents (deduplicated)")

df = convert_to_dataframe(results)  # one row per chunk
```

### Advanced Usage

#### Save and Load Plans

You can save a plan once and reuse it with different sampling percentages:

```python
from bigdata_smart_batching import plan_search, execute_search, save_plan, load_plan, deduplicate_documents

# 1. Create plan and save to disk
plan = plan_search(
    text="merger acquisition",
    universe="id_name_mapping_us_top_3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31",
)
save_plan(plan, "my_search_plan.json")

# 2. Later: load the same plan and run with different chunk_percentage
plan = load_plan("my_search_plan.json")

raw_10 = execute_search(plan, chunk_percentage=0.1)
raw_50 = execute_search(plan, chunk_percentage=0.5)

results_10pct = deduplicate_documents(raw_10)
results_50pct = deduplicate_documents(raw_50)
```

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Smart Batching Search Flow                    │
└─────────────────────────────────────────────────────────────────┘

Step 1: PLANNING
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Universe   │ --> │   Co-mention │ --> │   Basket     │
│   CSV File   │     │   API Query  │     │   Creation   │
│  (4,732      │     │  (Get chunk  │     │  (Group by   │
│  companies)  │     │   volumes)   │     │   volume)    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                      │
                                                      v
                                              ┌──────────────┐
                                              │  Search Plan │
                                              │  (435 baskets│
                                              │   vs 11,357  │
                                              │   queries)   │
                                              └──────────────┘

Step 2: EXECUTION
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Calculate  │ --> │   Parallel   │ --> │   Collect &  │
│  Proportional│     │   Search     │     │   Aggregate  │
│   Sampling   │     │  (Rate Limit)│     │   Results    │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Step 1: Planning

The `plan_search()` function:

1. **Loads the universe** of companies from CSV (e.g., 4,732 companies)
2. **Queries the comention endpoint** to get chunk volumes per company
3. **Splits date ranges by volume** when a company exceeds the chunk limit per query: fetches the volume time series and chooses sub-period boundaries so chunk volume is balanced across periods (see [Volume-Based Period Splitting](#volume-based-period-splitting)); optional via `apply_volume_splits` and `min_period_days`
4. **Creates optimized baskets** of companies grouped by volume:
  - High-volume companies → individual baskets
  - Medium-volume companies → grouped baskets
  - Low-volume companies → large grouped baskets
5. **Builds complete query structures** with search text embedded
6. **Returns a plan** with total expected chunks and basket configurations

**Basket Creation Process:**

```
Company Chunk Volumes (sorted):
┌─────────────────────────────────────────┐
│ Company A: 50,000 chunks  ──┐           │
│ Company B: 30,000 chunks  ──┤           │
│ Company C: 20,000 chunks  ──┼─> High   │ Individual baskets
│ Company D: 15,000 chunks  ──┤           │
│ Company E: 10,000 chunks  ──┘           │
│ Company F:  5,000 chunks  ──┐           │
│ Company G:  3,000 chunks  ──┼─> Medium │ Grouped baskets
│ Company H:  2,000 chunks  ──┤           │ (10-20 companies)
│ ... (100 more)              ┘           │
│ Company X:    50 chunks  ──┐           │
│ Company Y:    30 chunks  ──┼─> Low    │ Large grouped baskets
│ Company Z:    20 chunks  ──┤           │ (100+ companies)
│ ... (11,000 more)          ┘           │
└─────────────────────────────────────────┘

Result: 4,732 companies → ~17-3,699 baskets (varies by topic)
```

**Visual Comparison:**

```
Naive Approach:           Smart Batching:
┌─────────────┐          ┌─────────────┐
│ Company 1   │          │ Basket 1    │
│ Company 2   │          │ (10 high-   │
│ Company 3   │          │  volume)    │
│   ...       │   -->    │ Basket 2    │
│ Company     │          │ (100 medium)│
│ 11,357      │          │ Basket 3    │
└─────────────┘          │ (327 low)   │
11,357 monthly queries   └─────────────┘
by 10 company batch      435 queries
                         (96.2% reduction)
```

### Step 2: Execution

The `execute_search()` function:

1. **Calculates proportional chunks** per basket: `expected_chunks * chunk_percentage`
2. **Ensures minimum** of 1 chunk per basket (if expected > 0)
3. **Executes searches in parallel** with:
  - **Rate Limiting**: Sliding window algorithm (100 RPM default)
  - **Semaphore**: Limits concurrent connections (40 workers default)
  - **Error Handling**: Retries and graceful failure handling
4. **Collects** results (list of document dictionaries, each with a `chunks` array)
5. **Returns** the list. Use `deduplicate_documents()` to merge duplicate documents, then optionally `convert_to_dataframe()` for one row per chunk

## Naive vs Smart Batching: Performance Comparison

### The Problem with Naive Search

A naive approach searches each company individually:

- **4,732 companies** × time buckets × topics = **11,357 API queries** (naive approach)
- Each query is independent and cannot be optimized
- No knowledge of chunk volumes
- Inefficient for large universes

### Smart Batching Solution

Smart batching groups companies intelligently:

- **4,732 companies** → **17-3,699 baskets** (67-99% reduction, varies by topic specificity)
- Companies grouped by chunk volume
- High-volume companies get individual baskets
- Low-volume companies share baskets

### Real-World Benchmarks

The following table demonstrates the dramatic query reduction achievable with Smart Batching across different topic types (baseline: 4,732 companies, 24 monthly time buckets, 1 topic):


| Search Text                 | Naive Approach Queries | Smart Batching Queries | Query Reduction (%)         |
| --------------------------- | ---------------------- | ---------------------- | --------------------------- |
| Higher ESG Compliance Costs | 11,357                 | 435                    | **96.2%** (3.8% of naive)   |
| Customer Trust Erosion      | 11,357                 | 17                     | **99.85%** (0.15% of naive) |
| Decreased Consumer Demand   | 11,357                 | 504                    | **95.6%** (4.4% of naive)   |
| Increased Capex             | 11,357                 | 87                     | **99.2%** (0.77% of naive)  |
| Post-Covid Recovery         | 11,357                 | 466                    | **95.9%** (4.1% of naive)   |


**Key Insight**: Query reduction is most dramatic for niche or specialized topics where media coverage is concentrated among fewer companies. For example, "Customer Trust Erosion" achieves a 99.85% reduction (17 queries vs 11,357), while broader topics like "Earnings" show more modest but still significant 67% reductions. The more specialized the topic, the greater the query reduction benefit.

**Note**: Using a generic theme like "Earnings" for dataset generation is poor practice, as it will involve a massive number of chunks (millions) that will need to be post-processed. It is shown in the above table for comparison purposes only.

### Advantages of Smart Batching

1. **Massive Query Reduction**: 67-99% fewer API calls (varies by topic specificity)
  - Niche topics: Up to 99.85% reduction (e.g., "Customer Trust Erosion": 17 vs 11,357 queries)
  - Specialized topics: 96-97% reduction (e.g., "Higher ESG Compliance Costs": 435 vs 11,357 queries)
  - Broad topics: 32-67% reduction (e.g., "Earnings": 3,699 vs 11,357 queries)
2. **Faster Execution**: Parallel processing of fewer, optimized queries
3. **Cost Efficiency**: Significantly lower API usage costs
4. **Rate Limit Friendly**: Fewer queries = easier to stay within limits
5. **Scalable**: Works efficiently even with 10,000+ companies
6. **Volume-Based Period Splitting**: Sub-period boundaries follow actual chunk distribution (via volume time series), preserving distribution across spikes and sparse periods
7. **Proportional Sampling**: Maintains distribution when sampling subsets
8. **Topic-Specific Optimization**: Most effective for specialized, niche topics with concentrated coverage

### When to Use Each Approach

**Use Naive Search when:**

- Universe is small (< 100 companies)
- Need exact per-company results
- Companies have very different search requirements

**Use Smart Batching when:**

- Universe is large (> 500 companies)
- Companies can be grouped by volume
- Need efficient, scalable search
- Want to sample a percentage of results
- Working with rate limits

## Examples

### Example 1: Basic Search

```python
from bigdata_smart_batching import plan_search, execute_search, deduplicate_documents, convert_to_dataframe

# Plan the search (creates optimized baskets)
plan = plan_search(
    text="earnings revenue profit",
    universe="id_name_mapping_us_top_3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

print(f"Created {len(plan['baskets'])} baskets")
print(f"Expected chunks: {plan['total_expected_chunks']:,}")

# Execute with 10% sampling, then deduplicate
results_raw = execute_search(plan, chunk_percentage=0.1)
results = deduplicate_documents(results_raw)
print(f"Retrieved {len(results):,} documents (deduplicated)")

df = convert_to_dataframe(results)  # one row per chunk
```

### Example 2: Multiple Percentage Runs

```python
from bigdata_smart_batching import plan_search, execute_search, deduplicate_documents

plan = plan_search(
    text="merger acquisition",
    universe="id_name_mapping_us_top_3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Try different percentages without re-planning
for pct in [0.05, 0.1, 0.25, 0.5]:
    results_raw = execute_search(plan, chunk_percentage=pct)
    results = deduplicate_documents(results_raw)
    n_chunks = sum(len(d.get("chunks", [])) for d in results)
    print(f"{pct*100:3.0f}%: {len(results):,} documents, {n_chunks:,} chunks retrieved")
```

### Example 3: Production Configuration

```python
from bigdata_smart_batching import plan_search, execute_search, deduplicate_documents

# Conservative settings for production
plan = plan_search(
    text="ESG compliance costs",
    universe="id_name_mapping_us_top_3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Lower rate limit and workers for safety
results_raw = execute_search(
    plan,
    chunk_percentage=0.1,
    requests_per_minute=50,  # Conservative rate limit
    max_workers=20            # Fewer concurrent connections
)
results = deduplicate_documents(results_raw)
```

### Example 4: Large-Scale Search with Error Handling

```python
import logging
from bigdata_smart_batching import plan_search, execute_search, deduplicate_documents

logging.basicConfig(level=logging.INFO)

plan = plan_search(
    text="post-covid recovery",
    universe="id_name_mapping_us_top_3000.csv",
    start_date="2021-01-01",
    end_date="2023-12-31"
)

# Execute with full error handling
results_raw = execute_search(
    plan,
    chunk_percentage=0.1,
    requests_per_minute=100,
    max_workers=40
)
results = deduplicate_documents(results_raw)
n_chunks = sum(len(d.get("chunks", [])) for d in results)

print(f"Successfully retrieved {len(results):,} documents, {n_chunks:,} chunks")
print(f"Query reduction: {(1 - len(plan['baskets']) / 4731) * 100:.1f}%")
```

#### Rate limit strategies

**Conservative (Recommended for Production):**

```python
execute_search(plan, requests_per_minute=50, max_workers=20)
```

**Balanced (Default):**

```python
execute_search(plan, requests_per_minute=100, max_workers=40)
```

**Aggressive (Use with Caution):**

```python
execute_search(plan, requests_per_minute=200, max_workers=60)
```

**Note**: Always check your API rate limits and adjust accordingly.

### Performance Tuning

#### Optimize for Speed

```python
# More workers, higher rate limit
results = execute_search(
    plan,
    max_workers=60,
    requests_per_minute=150
)
```

#### Optimize for API Limits

```python
# Fewer workers, lower rate limit
results = execute_search(
    plan,
    max_workers=20,
    requests_per_minute=50
)
```

#### Optimize for Memory

```python
# Smaller sample size
results = execute_search(
    plan,
    chunk_percentage=0.05  # 5% instead of 10%
)
```

## Configuration

### Environment Variables

- `BIGDATA_API_KEY`: Required - Your Bigdata API key
- `BIGDATA_API_BASE_URL`: Optional - API base URL (default: [https://api.bigdata.com](https://api.bigdata.com))

### Default Settings

- `requests_per_minute`: 100 (configurable rate limit)
- `max_workers`: 40 (parallel workers / semaphore limit)
- `window_size_seconds`: 5 (rate limiter sliding window)
- `max_chunks_per_basket`: 1000 (basket size limit)
- `chunk_percentage`: Pass to `execute_search()` (e.g. 0.1 for 10% sampling)
- `volume_query_mode`: `"three_pass"` in `plan_search()`; use `"iterative"` for per-batch iterative discovery

## Quick Reference

### Performance Comparison Summary


| Metric              | Naive Search           | Smart Batching | Improvement                            |
| ------------------- | ---------------------- | -------------- | -------------------------------------- |
| **Queries**         | 11,357                 | 17-3,699       | **67-99% reduction** (varies by topic) |
| **Execution Time**  | ~20 minutes            | ~seconds       | **~1,000x faster**                     |
| **API Costs**       | High                   | Low            | **67-99% savings** (varies by topic)   |
| **Rate Limit Risk** | Very High              | Low            | **Much safer**                         |
| **Scalability**     | Poor (>1000 companies) | Excellent      | **Handles 10K+**                       |


### When to Use Smart Batching

✅ **Use Smart Batching for:**

- Large universes (> 500 companies)
- Production environments with rate limits
- Cost-sensitive applications
- When you need proportional sampling
- Time-sensitive searches

❌ **Consider Naive Search for:**

- Very small universes (< 100 companies)
- When exact per-company queries are required
- When companies have vastly different search needs

## License

This project is part of the Bigdata.com.

**Disclaimer**: This software is provided "as is" without warranty of any kind, express or implied. The authors
and contributors assume no responsibility for the accuracy, completeness, or usefulness of any information,
results, or processes provided. This software is for educational and research purposes only and is not
intended to be used as financial advice. Any use of this software for investment or trading decisions is at
your own risk. The authors and contributors shall not be liable for any damages arising from the use of
this software.