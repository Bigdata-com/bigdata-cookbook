from bigdata_client import Bigdata

def get_entity_ids(entity_names: list[str]):
    bigdata = Bigdata()
    entity_name_to_keys = {}
    company_objects = []
    for name in entity_names:
        res = bigdata.knowledge_graph.autosuggest(name, limit=1)
        if not res:
            print(f'Could not find entity ID for {name}')
        else:
            updated = False
            for i in res:
                if i.entity_type=='COMP' and name in i.name:
                    entity_name_to_keys[i.name] = i.id
                    company_objects.append(i)
                    updated = True  # Set flag to True if an update occurs
                    break
            if not updated:
                print(f'No matching entity ID found for {name} after checking all suggestions')

    return list(entity_name_to_keys.values()), list(entity_name_to_keys.keys()), company_objects