"""
Advanced Bigdata Search with Improved Logic and Error Handling

This script demonstrates advanced search capabilities using:
- Batched query building for performance
- Multithreaded search execution
- Advanced post-processing and filtering
- Comprehensive error handling
"""

import os
import sys
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

from bigdata_client import Bigdata
from bigdata_client.models.entities import Company
from bigdata_client.models.search import DocumentType, SortBy
from bigdata_research_tools.search import (
    run_search, 
    build_batched_query, 
    create_date_ranges
)
from bigdata_research_tools.search.query_builder import EntitiesToSearch
from bigdata_research_tools.search.search_utils import filter_search_results
from bigdata_research_tools.search.screener_search import (
    filter_company_entities, 
    process_screener_search_results
)

def initialize_bigdata_client():
    """Initialize BigData client with error handling"""
    try:
        bigdata = Bigdata()
        print("✅ Successfully connected to BigData API")
        return bigdata
    except Exception as e:
        print(f"❌ Failed to connect to BigData API: {e}")
        print("Please check your credentials in the .env file")
        return None

def get_companies_by_tickers(bigdata, tickers):
    """Get company objects using autosuggest with error handling"""
    companies = []
    for ticker in tickers:
        try:
            suggestions = bigdata.knowledge_graph.autosuggest(ticker, limit=1)
            if suggestions:
                companies.extend(suggestions)
                print(f"✅ Found company for ticker {ticker}: {suggestions[0].name}")
            else:
                print(f"⚠️  No company found for ticker: {ticker}")
        except Exception as e:
            print(f"❌ Error finding company for ticker '{ticker}': {e}")
    
    return companies

def create_optimized_date_ranges(start_date=None, end_date=None, freq='M'):
    """Create optimized date ranges with sensible defaults"""
    if not start_date:
        # Default to 6 months ago
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    if not end_date:
        # Default to today
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"📅 Creating date ranges from {start_date} to {end_date} with frequency '{freq}'")
    
    try:
        date_ranges = create_date_ranges(
            start_date=start_date,
            end_date=end_date, 
            freq=freq
        )
        print(f"✅ Created {len(date_ranges)} date range batches")
        return date_ranges
    except Exception as e:
        print(f"❌ Error creating date ranges: {e}")
        return None

def build_optimized_queries(companies, sentences=None, batch_size=5, scope=DocumentType.ALL):
    """Build optimized batched queries with better configuration"""
    if not companies:
        print("❌ No companies provided for query building")
        return None
    
    try:
        # Extract company IDs
        entity_keys = [company.id for company in companies]
        print(f"🔍 Building queries for {len(entity_keys)} companies")
        
        # Create EntitiesToSearch configuration
        entities = EntitiesToSearch(companies=entity_keys)
        
        # Use sentences if provided, otherwise empty list
        search_sentences = sentences if sentences else []
        
        # Build batched queries with optimized parameters
        entity_queries = build_batched_query(
            sentences=search_sentences,
            entities=entities,
            batch_size=batch_size,
            scope=scope,
            keywords=None,  # Can be customized for keyword filtering
            control_entities=None,  # Can be used for co-mentions
            sources=None,  # Can filter by specific sources
            fiscal_year=None,  # Can filter by fiscal year
            custom_batches=None
        )
        
        print(f"✅ Built {len(entity_queries)} batched queries")
        return entity_queries
        
    except Exception as e:
        print(f"❌ Error building queries: {e}")
        return None

def execute_search_with_monitoring(queries, date_ranges, scope, limit=20):
    """Execute search with progress monitoring and error handling"""
    if not queries or not date_ranges:
        print("❌ Missing queries or date ranges for search execution")
        return None
    
    try:
        print(f"🚀 Executing search with {len(queries)} queries across {len(date_ranges)} date ranges")
        print(f"📊 Scope: {scope}, Document limit: {limit}")
        
        # Run multithreaded search
        results = run_search(
            queries=queries, 
            date_ranges=date_ranges, 
            scope=scope, 
            limit=limit
        )
        
        print(f"✅ Search completed successfully")
        return results
        
    except Exception as e:
        print(f"❌ Error during search execution: {e}")
        return None

def process_and_analyze_results(results, companies, document_type):
    """Process search results and provide comprehensive analysis"""
    if not results:
        print("❌ No results to process")
        return None
    
    try:
        print("🔧 Processing search results...")
        
        # Filter search results
        results, entities = filter_search_results(results)
        print(f"📈 Filtered to {len(results)} results with {len(entities)} entities")
        
        # Filter to only company entities
        entities = filter_company_entities(entities)
        print(f"🏢 Found {len(entities)} company entities")
        
        # Process results into structured DataFrame
        df_sentences = process_screener_search_results(
            results=results,
            entities=entities,
            companies=companies,
            document_type=document_type,
        )
        
        print(f"✅ Processed results into DataFrame with {len(df_sentences)} rows")
        return df_sentences
        
    except Exception as e:
        print(f"❌ Error processing results: {e}")
        return None

def print_comprehensive_analysis(df):
    """Print comprehensive analysis of search results"""
    if df is None or df.empty:
        print("❌ No data to analyze")
        return
    
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE SEARCH ANALYSIS")
    print("=" * 80)
    
    # Show DataFrame structure first
    print(f"📋 DataFrame Columns: {list(df.columns)}")
    print(f"📈 Total Documents Found: {len(df)}")
    
    # Basic statistics with safe column access
    if 'timestamp_utc' in df.columns:
        print(f"📅 Date Range: {df['timestamp_utc'].min()} to {df['timestamp_utc'].max()}")
    
    if 'entity_name' in df.columns:
        print(f"🏢 Companies Mentioned: {df['entity_name'].nunique()}")
    
    if 'source' in df.columns:
        print(f"📰 Sources: {df['source'].nunique()}")
    else:
        print("📰 Sources: Column not available")
    
    if 'document_type' in df.columns:
        print(f"📄 Document Types: {df['document_type'].value_counts().to_dict()}")
    else:
        print("📄 Document Types: Column not available")
    
    # Company breakdown
    if 'entity_name' in df.columns:
        print(f"\n🏢 Company Breakdown:")
        company_counts = df['entity_name'].value_counts()
        for company, count in company_counts.items():
            print(f"   {company}: {count} documents")
    
    # Source breakdown (if available)
    if 'source' in df.columns:
        print(f"\n📰 Top Sources:")
        source_counts = df['source'].value_counts().head(10)
        for source, count in source_counts.items():
            print(f"   {source}: {count} documents")
    
    # Sample results with safe column access
    print(f"\n📄 Sample Results (First 3):")
    for idx, row in df.head(3).iterrows():
        print(f"\n   Document {idx + 1}:")
        
        if 'headline' in df.columns:
            print(f"     Headline: {row.get('headline', 'N/A')}")
        
        if 'timestamp_utc' in df.columns:
            print(f"     Date: {row.get('timestamp_utc', 'N/A')}")
        
        if 'entity_name' in df.columns:
            print(f"     Company: {row.get('entity_name', 'N/A')}")
        
        if 'source' in df.columns:
            print(f"     Source: {row.get('source', 'N/A')}")
        
        if 'text' in df.columns:
            text_preview = str(row.get('text', 'N/A'))[:150]
            print(f"     Text Preview: {text_preview}...")
        
        # Show all available columns for this row
        print(f"     Available data: {list(row.index)}")

def main():
    """Main execution function with comprehensive error handling"""
    print("🚀 Starting Advanced BigData Search Test")
    print("=" * 60)
    
    # Initialize BigData client
    bigdata = initialize_bigdata_client()
    if not bigdata:
        return
    
    # Configuration
    tickers = ["AAPL", "MSFT"]  # Added more companies
    search_sentences = [
        "AI artificial intelligence",
        "cloud computing services"
    ]
    
    # Step 1: Get companies by tickers
    print("\n🔍 Step 1: Resolving companies by tickers...")
    companies = get_companies_by_tickers(bigdata, tickers)
    if not companies:
        print("❌ No companies found. Exiting.")
        return
    
    # Step 2: Create optimized date ranges
    print("\n📅 Step 2: Creating date ranges...")
    date_ranges = create_optimized_date_ranges(
        start_date="2025-06-01",
        end_date="2025-07-31", 
        freq='M'
    )
    if not date_ranges:
        print("❌ Failed to create date ranges. Exiting.")
        return
    
    # Step 3: Build optimized queries
    print("\n🔧 Step 3: Building optimized queries...")
    queries = build_optimized_queries(
        companies=companies,
        sentences=search_sentences,
        batch_size=5,
        scope=DocumentType.ALL  # Search all document types
    )
    if not queries:
        print("❌ Failed to build queries. Exiting.")
        return
    
    # Step 4: Execute search
    print("\n🚀 Step 4: Executing search...")
    results = execute_search_with_monitoring(
        queries=queries,
        date_ranges=date_ranges,
        scope=DocumentType.ALL,
        limit=50  # Increased limit for better coverage
    )
    if not results:
        print("❌ Search execution failed. Exiting.")
        return
    
    # Step 5: Process and analyze results
    print("\n📊 Step 5: Processing and analyzing results...")
    df_results = process_and_analyze_results(
        results=results,
        companies=companies,
        document_type=DocumentType.ALL
    )
    
    # Step 6: Print comprehensive analysis
    print_comprehensive_analysis(df_results)
    
    # Return results for further use
    return df_results

if __name__ == "__main__":
    main() 