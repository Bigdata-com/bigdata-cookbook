"""
Helper functions for printing Knowledge Graph API results
"""

def print_companies(response, data, max_results=5):
    """Print companies search results and return list of IDs"""
    if response.status_code == 200:
        companies = data.get('results', [])
        print(f"✅ Status: {response.status_code}")
        print(f"🏢 Found {len(companies)} companies\n")
        company_ids = []
        for i, company in enumerate(companies[:max_results], 1):
            company_id = company.get('id')
            company_name = company.get('name')
            print(f"{i}. {company_name:40} (ID: {company_id})")
            company_ids.append(company_id)
        return company_ids
    else:
        print(f"❌ Error {response.status_code}: {data}")
        return []


def print_entity_details(response, data):
    """Print entity details"""
    if response.status_code == 200:
        results = data.get('results', {})
        
        # Results can be a dict (id -> entity) or a list
        if isinstance(results, dict):
            entities = list(results.values())
        else:
            entities = results
            
        print(f"✅ Status: {response.status_code}")
        print(f"📇 Found {len(entities)} entities\n")
        
        for i, entity in enumerate(entities, 1):
            print(f"{i}. {entity.get('name')}")
            print(f"   ID: {entity.get('id')} | Type: {entity.get('type')}")
            if 'ticker' in entity:
                print(f"   Ticker: {entity.get('ticker')}")
            print()
    else:
        print(f"❌ Error {response.status_code}: {data}")
