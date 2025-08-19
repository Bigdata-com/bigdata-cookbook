import logging
from typing import Dict, List

from bigdata_client import Bigdata, Company
from bigdata_client.advanced_search_query import QueryComponent


def find_entity(suggestions: List[QueryComponent],
                ticker: str) -> Company:
    for suggestion in suggestions[ticker]:
        if isinstance(suggestion, Company) and suggestion.ticker == ticker:
            return suggestion


def get_entities(ticker_to_name: Dict[str, str]) -> List[Company]:
    bigdata = Bigdata()
    entities = []
    for ticker, name in ticker_to_name.items():
        suggestions = bigdata.knowledge_graph.autosuggest([ticker])
        if not suggestions:
            suggestions = bigdata.knowledge_graph.autosuggest(name)
        entity = find_entity(suggestions, ticker)
        if not entity:
            logging.error(f'Could not find entity for {ticker} ({name})')
        entities.append(entity)
    return entities


def extract_entity_keys(entities: List[Company]) -> List[str]:
    return [entity.id  if entity!= None else None for entity in entities]

def get_entity_ids(entity_names):
    bigdata = Bigdata()
    entity_name_to_keys = {}
    for name in entity_names:
        res = bigdata.knowledge_graph.autosuggest(name)
        if not res:
            print(f'Could not find entity ID for {name}')
        else:
            updated = False
            for i in res:
                if i.entity_type=='COMP' and name in i.name:
                    entity_name_to_keys[i.name] = i.id
                    updated = True  # Set flag to True if an update occurs
                    break
            if not updated:
                print(f'No matching entity ID found for {name} after checking all suggestions')

    return entity_name_to_keys