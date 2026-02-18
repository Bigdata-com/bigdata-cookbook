"""
Helper functions for printing API results
"""

def print_search_results(response, data, max_results=2):
    """Print search API results"""
    if response.status_code == 200:
        results = data.get('results', [])
        print(f"✅ Status: {response.status_code}")
        print(f"📄 Found {len(results)} documents")
        print(f"💰 API Units Used: {data.get('usage', {}).get('api_query_units', 0)}\n")
        
        for i, doc in enumerate(results[:max_results], 1):
            print(f"--- Document {i} ---")
            print(f"Headline: {doc.get('headline', 'N/A')}")
            print(f"Source: {doc.get('source', {}).get('name', 'N/A')} (Rank: {doc.get('source', {}).get('rank', 'N/A')})")
            print(f"Date: {doc.get('timestamp', 'N/A')[:10]}")
            print(f"Chunks: {len(doc.get('chunks', []))}")
            if doc.get('chunks'):
                chunk = doc['chunks'][0]
                print(f"First chunk preview: {chunk.get('text', '')[:150]}...")
            print()
    else:
        print(f"❌ Error {response.status_code}: {data}")
