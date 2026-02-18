"""
Helper functions for printing API results
"""

def print_volume_results(response, data, max_days=5):
    """Print volume API results"""
    if response.status_code == 200:
        total = data.get('results', {}).get('total', {})
        volume = data.get('results', {}).get('volume', [])
        print(f"✅ Status: {response.status_code}")
        print(f"📊 Total: {total.get('documents', 0)} documents, {total.get('chunks', 0)} chunks")
        print(f"💰 API Units Used: {data.get('usage', {}).get('api_query_units', 0)}\n")
        print(f"Top {max_days} days by volume:")
        for day_data in volume[:max_days]:
            print(f"  {day_data['date']}: {day_data['documents']} docs, {day_data['chunks']} chunks")
    else:
        print(f"❌ Error {response.status_code}: {data}")
