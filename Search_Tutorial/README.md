# How to Use Bigdata Research Tools for Advanced Scalable Search

A simple guide for performing advanced, scalable searches using the Bigdata API with Python.

## Quick Start

### 1. Install Dependencies
```bash
# Using UV (recommended)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Or using pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Up Credentials
```bash
# Create .env file with your Bigdata credentials
echo "BIGDATA_USERNAME=your_username" > .env
echo "BIGDATA_PASSWORD=your_password" >> .env
```

### 3. Run Advanced Search
```bash
python advanced_search.py
```

## What This Does

The `advanced_search.py` script demonstrates how to:

- **Search multiple companies** (Apple, Microsoft, Google, Amazon)
- **Use batched queries** for better performance
- **Process results in parallel** across date ranges
- **Handle errors gracefully** with comprehensive logging
- **Analyze results** with detailed breakdowns

## Key Features

### Scalable Search Architecture
- **Multi-threaded execution** across date ranges
- **Batched entity queries** for optimal API usage
- **Configurable batch sizes** and document limits
- **Progress monitoring** with real-time feedback

### Advanced Query Building
```python
# Companies to search
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

# Search sentences for filtering
sentences = [
    "AI artificial intelligence",
    "cloud computing services",
    "earnings revenue growth",
    "product launch announcement"
]
```

### Flexible Date Ranges
```python
# Monthly batches for optimal performance
date_ranges = create_date_ranges(
    start_date="2024-01-01",
    end_date="2024-12-31", 
    freq='M'  # Monthly batches
)
```

## Sample Output

```
🚀 Starting Advanced Bigdata Search Test
============================================================
✅ Successfully connected to Bigdata API

🔍 Step 1: Resolving companies by tickers...
✅ Found: Apple Inc. (AAPL)
✅ Found: Microsoft Corp. (MSFT)
✅ Found: Alphabet Inc. (GOOGL)
✅ Found: Amazon.com Inc. (AMZN)

📅 Step 2: Creating date ranges...
✅ Created 12 date range batches

🔧 Step 3: Building queries...
✅ Built 5 batched queries

🚀 Step 4: Executing search...
✅ Search completed successfully

📊 Step 5: Processing results...
✅ Processed results into DataFrame with 693 rows

📊 COMPREHENSIVE SEARCH ANALYSIS
================================================================================
📈 Total Documents Found: 693
🏢 Companies Mentioned: 4
📄 Document Types: {'all': 693}

🏢 Company Breakdown:
   Microsoft Corp.: 262 documents
   Alphabet Inc.: 250 documents
   Amazon.com Inc.: 94 documents
   Apple Inc.: 87 documents
```

## Customization

### Modify Search Parameters
Edit the configuration in `advanced_search.py`:

```python
# Change companies
tickers = ["TSLA", "NVDA", "META"]

# Adjust search sentences
sentences = [
    "electric vehicles",
    "semiconductor chips",
    "social media advertising"
]

# Modify date range
start_date = "2023-01-01"
end_date = "2024-12-31"
```

### Performance Tuning
```python
# Increase for more results
limit = 200

# Adjust for performance vs memory
batch_size = 15

# Change frequency (M=monthly, W=weekly, D=daily)
freq = 'W'
```

## Troubleshooting

### Common Issues

**Authentication Error**
```bash
❌ Failed to connect to Bigdata API
```
→ Check your credentials in `.env` file

**No Results Found**
```bash
❌ No documents found matching the search criteria
```
→ Try broader search terms or extend date range

**Import Errors**
```bash
ModuleNotFoundError: No module named 'bigdata_research_tools'
```
→ Install dependencies: `uv pip install -r requirements.txt`

## Performance Tips

- **Use monthly batches** (`freq='M'`) for optimal performance
- **Increase batch_size** for better throughput with large datasets
- **Limit search sentences** to focus on relevant topics
- **Monitor progress** to identify bottlenecks

## Files

- `advanced_search.py` - Main search script with all optimizations
- `requirements.txt` - Python dependencies
- `README.md` - This guide

## Learn More

- [Bigdata API Documentation](https://docs.bigdata.com)
- [Research Tools Reference](https://docs.bigdata.com/api-reference/research-tools)

---

**Ready to search! 🚀** 