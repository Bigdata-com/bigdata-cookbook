"""Knowledge graph helpers using BigdataRestClient (no SDK)."""

from __future__ import annotations

from .bigdata_rest import BigdataRestClient


def get_entity_ids(entity_names: list[str]) -> tuple[list[str], list[str], list[dict]]:
    """Get entity IDs from company names using REST API.
    
    Args:
        entity_names: List of company names to search
        
    Returns:
        Tuple of (entity_ids, matched_names, company_objects)
    """
    client = BigdataRestClient()
    entity_name_to_keys = {}
    company_objects = []
    
    for name in entity_names:
        results = client.find_companies(name, limit=1)
        if not results:
            print(f'Could not find entity ID for {name}')
        else:
            updated = False
            for company in results:
                # Check if result matches the company name
                company_name = company.get('name', '')
                company_id = company.get('id', '')
                if company_id and name in company_name:
                    entity_name_to_keys[company_name] = company_id
                    company_objects.append(company)
                    updated = True
                    break
            if not updated:
                print(f'No matching entity ID found for {name} after checking all suggestions')
    
    return (
        list(entity_name_to_keys.values()),
        list(entity_name_to_keys.keys()),
        company_objects,
    )