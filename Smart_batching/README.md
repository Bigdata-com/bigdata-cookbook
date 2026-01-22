# Smart Batching Search

A high-performance semantic search system that reduces API queries by **67-99%** (varies by topic specificity) through intelligent company grouping and parallel execution.

This module provides a two-step system for efficient semantic search:
1. **Planning**: Organize search using smart batching and return total expected chunks
2. **Execution**: Perform search with proportional sampling to preserve distribution

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
5. [How It Works](#how-it-works)
   - [Architecture Overview](#architecture-overview)
   - [Step 1: Planning](#step-1-planning)
   - [Step 2: Execution](#step-2-execution)
   - [Proportional Sampling](#proportional-sampling)
6. [Naive vs Smart Batching: Performance Comparison](#naive-vs-smart-batching-performance-comparison)
   - [The Problem with Naive Search](#the-problem-with-naive-search)
   - [Smart Batching Solution](#smart-batching-solution)
   - [Real-World Benchmarks](#real-world-benchmarks)
   - [Advantages of Smart Batching](#advantages-of-smart-batching)
   - [When to Use Each Approach](#when-to-use-each-approach)
7. [API Reference](#api-reference)
   - [`plan_search()`](#plan_search)
   - [`execute_search()`](#execute_search)
8. [Examples](#examples)
   - [Example 1: Basic Search](#example-1-basic-search)
   - [Example 2: Multiple Percentage Runs](#example-2-multiple-percentage-runs)
   - [Example 3: Production Configuration](#example-3-production-configuration)
   - [Example 4: Large-Scale Search with Error Handling](#example-4-large-scale-search-with-error-handling)
9. [Large-Scale Search & Performance](#large-scale-search--performance)
   - [Rate Limiting (Sliding Window Algorithm)](#rate-limiting-sliding-window-algorithm)
   - [Concurrency Control (Semaphore)](#concurrency-control-semaphore)
   - [Large-Scale Search Considerations](#large-scale-search-considerations)
   - [Performance Tuning](#performance-tuning)
10. [Configuration](#configuration)
    - [Environment Variables](#environment-variables)
    - [Default Settings](#default-settings)
11. [Quick Reference](#quick-reference)
    - [Performance Comparison Summary](#performance-comparison-summary)
    - [When to Use Smart Batching](#when-to-use-smart-batching)
12. [Testing](#testing)
13. [File Structure](#file-structure)
14. [Troubleshooting](#troubleshooting)
    - [Rate Limit Issues](#rate-limit-issues)
    - [Memory Issues](#memory-issues)
    - [Slow Execution](#slow-execution)
    - [Planning Takes Too Long](#planning-takes-too-long)
15. [License](#license)

## Key Benefits

- **67-99% Query Reduction**: Search 4,732 companies with only 16-3,699 queries (varies by topic)
  - Niche topics: Up to 99.86% reduction (e.g., "Customer Trust Erosion": 16 queries)
  - Specialized topics: 96-97% reduction (e.g., "Higher ESG Compliance Costs": 437 queries)
  - Broad topics: 32-67% reduction (e.g., "Earnings": 3,699 queries)
- **Parallel Execution**: Rate-limited concurrent requests with semaphore control
- **Proportional Sampling**: Retrieve percentage of results while preserving distribution
- **Production Ready**: Comprehensive error handling, retries, and logging
- **Scalable**: Efficiently handles universes with 10,000+ companies
- **Topic-Optimized**: Most effective for specialized, niche topics with concentrated media coverage

## Features

- **Smart Batching Planning**: Organize searches using intelligent basket creation based on chunk volumes
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

### Option 2: Using pip

1. Install dependencies:
```bash
pip install -r requirements.txt
```

### Environment Setup

2. Set up environment variables:
```bash
export BIGDATA_API_KEY="your_api_key_here"
export BIGDATA_API_BASE_URL="https://api.bigdata.com"  # Optional, defaults to this
```

Or create a `.env` file:
```
BIGDATA_API_KEY=your_api_key_here
BIGDATA_API_BASE_URL=https://api.bigdata.com
```

## Quick Start

### Using the Jupyter Notebook

The easiest way to test the functions is using the provided Jupyter notebook:

```bash
jupyter notebook test_smart_batching.ipynb
```

The notebook includes:
- Step-by-step testing of all functions
- Configuration examples
- Results analysis
- Multiple percentage testing

### Basic Usage (Python Script)

```python
from search_function import plan_search, execute_search

# Step 1: Plan the search
plan = plan_search(
    text="earnings revenue profit",
    universe_csv_path="us_top3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

print(f"Total expected chunks: {plan['total_expected_chunks']:,}")

# Step 2: Execute search with 10% of total chunks (preserves distribution)
results = execute_search(
    search_plan=plan,
    chunk_percentage=0.1,  # 10% of total chunks
    requests_per_minute=100
)

print(f"Retrieved {len(results)} chunks")
```

### Advanced Usage

#### Save and Load Plans

```python
from search_function import plan_search, execute_search, save_plan, load_plan

# Create and save a plan
plan = plan_search(
    text="merger acquisition",
    universe_csv_path="us_top3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

save_plan(plan, "my_search_plan.json")

# Later, load and execute with different percentages
plan = load_plan("my_search_plan.json")

# Try different percentages without re-planning
results_10pct = execute_search(plan, chunk_percentage=0.1)
results_50pct = execute_search(plan, chunk_percentage=0.5)
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
                                              │  (437 baskets│
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
3. **Creates optimized baskets** of companies grouped by volume:
   - High-volume companies → individual baskets
   - Medium-volume companies → grouped baskets
   - Low-volume companies → large grouped baskets
4. **Builds complete query structures** with search text embedded
5. **Returns a plan** with total expected chunks and basket configurations

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

Result: 4,732 companies → ~16-3,699 baskets (varies by topic)
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
by 10 company batch      437 queries
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
4. **Collects and optionally sorts/deduplicates** results
5. **Returns list of chunk dictionaries** with enriched metadata

**Execution Flow with Rate Limiting:**
```
Baskets Queue (437 baskets)
    │
    ├─> Worker 1 ──┐
    ├─> Worker 2 ──┤
    ├─> Worker 3 ──┼─> Semaphore (max 40)
    ├─> ...        ──┤
    └─> Worker 40 ──┘
                      │
                      v
              ┌───────────────┐
              │ Rate Limiter  │
              │ (Sliding Win) │
              │ 100 RPM       │
              └───────────────┘
                      │
                      v
              ┌───────────────┐
              │  API Requests │
              │  (Parallel)   │
              └───────────────┘
                      │
                      v
              ┌───────────────┐
              │   Results     │
              │  Collection   │
              └───────────────┘
```

**Rate Limiting Visualization:**
```
Time:  0s    1s    2s    3s    4s    5s    6s    7s
       │     │     │     │     │     │     │     │
Window:┌─────┐
       │ R1  │  R2  R3  R4  R5  R6  R7  R8
       └─────┘
            ┌─────┐
            │ R2  │  R3  R4  R5  R6  R7  R8  R9
            └─────┘
                 ┌─────┐
                 │ R3  │  R4  R5  R6  R7  R8  R9  R10
                 └─────┘
                      ┌─────┐
                      │ R4  │  R5  R6  R7  R8  R9  R10 R11
                      └─────┘

Window slides forward, keeping only last 5 seconds
Max requests in window = 100 RPM / 60 * 5 ≈ 8 requests
```

### Proportional Sampling

When `chunk_percentage=0.1` (10%):
- Basket with 1000 expected chunks → retrieves 100 chunks
- Basket with 500 expected chunks → retrieves 50 chunks
- Basket with 100 expected chunks → retrieves 10 chunks
- Basket with 10 expected chunks → retrieves 1 chunk (minimum)

This preserves the relative distribution while limiting total chunks.

**Visual Example:**
```
Basket Distribution (10% sampling):
┌─────────────────────────────────────────┐
│ Basket A: 1000 chunks → 100 chunks (10%)│
│ Basket B:  500 chunks →  50 chunks (10%)│
│ Basket C:  100 chunks →  10 chunks (10%)│
│ Basket D:   10 chunks →   1 chunk  (10%)│
└─────────────────────────────────────────┘
Total: 1610 expected → 161 retrieved
Distribution preserved: ✅
```

## Naive vs Smart Batching: Performance Comparison

### The Problem with Naive Search

A naive approach searches each company individually:
- **4,732 companies** × time buckets × topics = **11,357 API queries** (naive approach)
- Each query is independent and cannot be optimized
- No knowledge of chunk volumes
- Inefficient for large universes

### Smart Batching Solution

Smart batching groups companies intelligently:
- **4,732 companies** → **16-3,699 baskets** (67-99% reduction, varies by topic specificity)
- Companies grouped by chunk volume
- High-volume companies get individual baskets
- Low-volume companies share baskets

### Real-World Benchmarks

The following table demonstrates the dramatic query reduction achievable with Smart Batching across different topic types (baseline: 4,732 companies, 24 monthly time buckets, 1 topic):

| Search Text | Naive Approach Queries | Smart Batching Queries | Query Reduction (%) |
|-------------|----------------------|----------------------|---------------------|
| Earnings* | 11,357 | 3,699 | **67.4%** (32.6% of naive) |
| Higher ESG Compliance Costs | 11,357 | 437 | **96.2%** (3.8% of naive) |
| Customer Trust Erosion | 11,357 | 16 | **99.9%** (0.14% of naive) |
| Decreased Consumer Demand | 11,357 | 374 | **96.7%** (3.3% of naive) |
| Increased Capex | 11,357 | 25 | **99.8%** (0.22% of naive) |
| Post-Covid Recovery | 11,357 | 427 | **96.2%** (3.8% of naive) |

**Key Insight**: Query reduction is most dramatic for niche or specialized topics where media coverage is concentrated among fewer companies. For example, "Customer Trust Erosion" achieves a 99.86% reduction (16 queries vs 11,357), while broader topics like "Earnings" show more modest but still significant 67% reductions. The more specialized the topic, the greater the query reduction benefit.

**Note**: Using a generic theme like "Earnings" for dataset generation is poor practice, as it will involve a massive number of chunks (millions) that will need to be post-processed. It is shown in the above table for comparison purposes only.

### Advantages of Smart Batching

1. **Massive Query Reduction**: 67-99% fewer API calls (varies by topic specificity)
   - Niche topics: Up to 99.86% reduction (e.g., "Customer Trust Erosion": 16 vs 11,357 queries)
   - Specialized topics: 96-97% reduction (e.g., "Higher ESG Compliance Costs": 437 vs 11,357 queries)
   - Broad topics: 32-67% reduction (e.g., "Earnings": 3,699 vs 11,357 queries)
2. **Faster Execution**: Parallel processing of fewer, optimized queries
3. **Cost Efficiency**: Significantly lower API usage costs
4. **Rate Limit Friendly**: Fewer queries = easier to stay within limits
5. **Scalable**: Works efficiently even with 10,000+ companies
6. **Proportional Sampling**: Maintains distribution when sampling subsets
7. **Topic-Specific Optimization**: Most effective for specialized, niche topics with concentrated coverage

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

## API Reference

### `plan_search()`

Plan a search using smart batching.

**Parameters:**
- `text` (str): Search query text
- `universe_csv_path` (str): Path to CSV file with entity IDs (one per line)
- `start_date` (str): Start date in YYYY-MM-DD format
- `end_date` (str): End date in YYYY-MM-DD format
- `api_key` (str, optional): API key (defaults to BIGDATA_API_KEY env var)
- `api_base_url` (str, optional): API base URL

**Returns:**
- `Dict` with:
  - `total_expected_chunks`: Total chunks expected
  - `baskets`: List of basket configs with complete query structures
  - `planning_metadata`: Additional metadata

### `execute_search()`

Execute search with proportional sampling, rate limiting, and parallel execution.

**Parameters:**
- `search_plan` (Dict): Planning result from `plan_search()`
- `chunk_percentage` (float): Percentage of total chunks (0.0 to 1.0)
  - `1.0` = 100% of chunks (full retrieval)
  - `0.1` = 10% of chunks (sampling)
  - `0.05` = 5% of chunks (light sampling)
- `requests_per_minute` (int, optional): Rate limit for API calls (default: 100)
  - Uses sliding window algorithm
  - Lower values = more conservative, safer for strict rate limits
  - Higher values = faster but risk hitting rate limits
- `api_key` (str, optional): API key (defaults to BIGDATA_API_KEY env var)
- `api_base_url` (str, optional): API base URL
- `max_workers` (int, optional): Maximum parallel workers / semaphore limit (default: 40)
  - Controls concurrent API connections
  - Lower values = fewer connections, more sequential
  - Higher values = more parallel, faster but more resource-intensive
- `sort_results` (bool, optional): Sort by relevance score (default: True)
- `deduplicate_results` (bool, optional): Remove duplicate chunks (default: False)

**Returns:**
- `List[Dict]`: List of chunk dictionaries with:
  - `text`: Chunk text content
  - `relevance`: Relevance score (0.0 to 1.0)
  - `sentiment`: Sentiment score (if available)
  - `document_id`: Source document ID
  - `source_name`: Source name (e.g., "Benzinga")
  - `timestamp`: Document timestamp
  - `entity_ids`: List of entity IDs in chunk
  - `primary_entity_id`: Primary entity ID
  - Additional metadata fields

**Performance Characteristics:**
- **Time Complexity**: O(b / max_workers) where b = number of baskets
- **Space Complexity**: O(r) where r = number of result chunks
- **Rate Limiting**: Sliding window with configurable RPM
- **Concurrency**: Semaphore-controlled parallel execution

## Examples

### Example 1: Basic Search

```python
from search_function import plan_search, execute_search

# Plan the search (creates optimized baskets)
plan = plan_search(
    text="earnings revenue profit",
    universe_csv_path="us_top3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

print(f"Created {len(plan['baskets'])} baskets from 3,000 companies")
print(f"Expected chunks: {plan['total_expected_chunks']:,}")

# Execute with 10% sampling
results = execute_search(plan, chunk_percentage=0.1)
print(f"Retrieved {len(results):,} chunks")
```

### Example 2: Multiple Percentage Runs

```python
plan = plan_search(
    text="merger acquisition",
    universe_csv_path="us_top3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Try different percentages without re-planning
for pct in [0.05, 0.1, 0.25, 0.5]:
    results = execute_search(plan, chunk_percentage=pct)
    print(f"{pct*100:3.0f}%: {len(results):,} chunks retrieved")
```

### Example 3: Production Configuration

```python
# Conservative settings for production
plan = plan_search(
    text="ESG compliance costs",
    universe_csv_path="us_top3000.csv",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# Lower rate limit and workers for safety
results = execute_search(
    plan,
    chunk_percentage=0.1,
    requests_per_minute=50,  # Conservative rate limit
    max_workers=20            # Fewer concurrent connections
)
```

### Example 4: Large-Scale Search with Error Handling

```python
import logging

logging.basicConfig(level=logging.INFO)

plan = plan_search(
    text="post-covid recovery",
    universe_csv_path="us_top3000.csv",
    start_date="2021-01-01",
    end_date="2023-12-31"
)

# Execute with full error handling
results = execute_search(
    plan,
    chunk_percentage=0.1,
    requests_per_minute=100,
    max_workers=40,
    sort_results=True,
    deduplicate_results=False
)

print(f"Successfully retrieved {len(results):,} chunks")
print(f"Query reduction: {(1 - len(plan['baskets']) / 3000) * 100:.1f}%")
```

## Large-Scale Search & Performance

### Rate Limiting (Sliding Window Algorithm)

The system uses a **sliding window rate limiter** to ensure API compliance:

```
Time Window (5 seconds)
┌─────────────────────────────────────────┐
│ [Request 1] [Request 2] ... [Request N] │
│  <-- Window slides forward -->          │
└─────────────────────────────────────────┘
```

**How it works:**
- Maintains a deque of request timestamps
- Removes timestamps older than the window (5 seconds)
- Allows new request only if count < limit (100 RPM = ~1.67 requests/second)
- Thread-safe implementation for parallel execution

**Example:**
```python
# 100 requests per minute = ~1.67 requests per second
# With 5-second window, max 8-9 requests in window
# If window is full, request waits until oldest expires
```

### Concurrency Control (Semaphore)

A **semaphore** limits simultaneous API connections:

```
┌─────────────────────────────────────────┐
│     Semaphore (max_workers=40)          │
│  ┌────┐ ┌────┐ ┌────┐ ... ┌────┐      │
│  │ W1 │ │ W2 │ │ W3 │     │ W40 │      │
│  └────┘ └────┘ └────┘     └────┘      │
│     │      │      │          │          │
│     v      v      v          v          │
│  ┌──────────────────────────────┐      │
│  │    API Requests (Rate        │      │
│  │    Limited: 100 RPM)         │      │
│  └──────────────────────────────┘      │
└─────────────────────────────────────────┘
```

**Benefits:**
- Prevents overwhelming the API with too many concurrent connections
- Works in conjunction with rate limiting
- Configurable via `max_workers` parameter (default: 40)

### Large-Scale Search Considerations

#### 1. Memory Management

For very large result sets:
- Results are collected incrementally
- Consider using `chunk_percentage` to limit results
- Results can be streamed to disk if needed

#### 2. Time Complexity

**Planning Phase:**
- O(n) where n = number of companies
- Co-mention API calls: ~1 per company (can be batched)
- Basket creation: O(n log n) for sorting

**Execution Phase:**
- O(b) where b = number of baskets (typically 1-5% of companies)
- Parallel execution: O(b / max_workers)
- Rate limiting adds minimal overhead

#### 3. Network Efficiency

```
Naive: 11,357 sequential requests
Smart: 437 parallel requests (with rate limiting)

Time Savings:
- Naive: ~11,357 requests × 0.1s = 1,135 seconds (19 minutes)
- Smart: ~437 requests / 40 workers × 0.1s = ~1.1 seconds
- Speedup: ~1,000x faster
```

#### 4. Error Handling & Resilience

- **Retry Logic**: Failed requests are retried with exponential backoff
- **Graceful Degradation**: Partial results returned if some baskets fail
- **Error Reporting**: Detailed logging of failed baskets
- **Timeout Handling**: Configurable timeouts prevent hanging requests

#### 5. Rate Limit Strategies

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
- `BIGDATA_API_BASE_URL`: Optional - API base URL (default: https://api.bigdata.com)

### Default Settings

- `requests_per_minute`: 100 (configurable rate limit)
- `max_workers`: 40 (parallel workers / semaphore limit)
- `window_size_seconds`: 5 (rate limiter sliding window)
- `max_chunks_per_basket`: 1000 (basket size limit)
- `chunk_percentage`: 1.0 (100% by default, can be reduced for sampling)

## Quick Reference

### Performance Comparison Summary

| Metric | Naive Search | Smart Batching | Improvement |
|--------|-------------|----------------|-------------|
| **Queries** | 11,357 | 16-3,699 | **67-99% reduction** (varies by topic) |
| **Execution Time** | ~20 minutes | ~seconds | **~1,000x faster** |
| **API Costs** | High | Low | **67-99% savings** (varies by topic) |
| **Rate Limit Risk** | Very High | Low | **Much safer** |
| **Scalability** | Poor (>1000 companies) | Excellent | **Handles 10K+** |

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

## Testing

Run all tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=search_function --cov-report=html tests/
```

Run specific test file:
```bash
pytest tests/test_search_function.py -v
```

## File Structure

```
Smart_Batching/
├── search_function.py          # Main functions and classes
├── test_smart_batching.ipynb   # Jupyter notebook for testing
├── example_usage.py            # Example Python script
├── us_top3000.csv              # Universe file (company entity IDs)
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Pytest configuration
├── tests/
│   ├── __init__.py
│   ├── test_search_function.py    # Unit tests for main functions
│   ├── test_rate_limiter.py        # Rate limiter tests
│   ├── test_proportional_sampling.py  # Proportional sampling tests
│   └── test_validation.py          # Validation/integration tests
├── test_data/
│   ├── sample_universe.csv         # Small test universe
│   └── mock_api_responses.json     # Mock API responses
└── README.md                   # This file
```

## Troubleshooting

### Rate Limit Issues

**Problem**: Getting 429 (Too Many Requests) errors

**Solutions**:
1. Reduce `requests_per_minute`:
   ```python
   execute_search(plan, requests_per_minute=50)  # More conservative
   ```

2. Reduce `max_workers`:
   ```python
   execute_search(plan, max_workers=20)  # Fewer concurrent requests
   ```

3. Check your API rate limits and adjust accordingly

### Memory Issues

**Problem**: Running out of memory with large result sets

**Solutions**:
1. Use smaller `chunk_percentage`:
   ```python
   execute_search(plan, chunk_percentage=0.05)  # 5% instead of 10%
   ```

2. Process results in batches
3. Stream results to disk instead of keeping in memory

### Slow Execution

**Problem**: Search execution is taking too long

**Solutions**:
1. Increase `max_workers` (if rate limits allow):
   ```python
   execute_search(plan, max_workers=60)
   ```

2. Increase `requests_per_minute` (if API allows):
   ```python
   execute_search(plan, requests_per_minute=150)
   ```

3. Use smaller `chunk_percentage` for faster results

### Planning Takes Too Long

**Problem**: `plan_search()` is slow with large universes

**Note**: Planning queries the co-mention API for each company. This is necessary to create optimal baskets. Consider:
- Caching plans for reuse
- Using smaller universes if possible
- Running planning during off-peak hours

## License

This project is part of the Bigdata.com and WorldQuant Challenge.

**Disclaimer**: This software is provided "as is" without warranty of any kind, express or implied. The authors
and contributors assume no responsibility for the accuracy, completeness, or usefulness of any information,
results, or processes provided. This software is for educational and research purposes only and is not
intended to be used as financial advice. Any use of this software for investment or trading decisions is at
your own risk. The authors and contributors shall not be liable for any damages arising from the use of
this software.
