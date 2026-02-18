"""
Helper functions for printing API results
"""

def print_comention_results(response, data, max_per_type=5):
    """Print co-mentions API results"""
    if response.status_code == 200:
        results = data.get('results', {})
        print(f"✅ Status: {response.status_code}")
        print(f"💰 API Units Used: {data.get('usage', {}).get('api_query_units', 0)}\n")
        
        # Count total entities across all types
        total = sum(len(results.get(key, [])) for key in ['places', 'companies', 'organizations', 'people', 'products', 'concepts'])
        print(f"🔗 Found {total} co-mentioned entities across all types\n")
        
        # Display by type
        entity_types = ['companies', 'places', 'people', 'products', 'organizations', 'concepts']
        for entity_type in entity_types:
            entities = results.get(entity_type, [])
            if entities:
                print(f"📌 {entity_type.upper()} (showing top {max_per_type} of {len(entities)}):")
                for i, entity in enumerate(entities[:max_per_type], 1):
                    chunks = entity.get('total_chunks_count', 0)
                    headlines = entity.get('total_headlines_count', 0)
                    print(f"   {i}. ID: {entity['id']:8} | Chunks: {chunks:6} | Headlines: {headlines:5}")
                print()
    else:
        print(f"❌ Error {response.status_code}: {data}")
